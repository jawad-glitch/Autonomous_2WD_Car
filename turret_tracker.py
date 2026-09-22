"""
Person-following turret car.

Turret (pan/tilt) tracks the person using the camera.
Car turns toward the turret in short pulses so the camera keeps up.
Once facing the person, the car holds TARGET_CM distance using the sonar.
When very close the camera can't recognise a person, so the car keeps
holding distance by sonar alone for a while.

pip install ultralytics opencv-python pyserial
"""

import math
import threading
import time

import cv2
import serial
from ultralytics import YOLO

SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CAM_URL = 'http://192.168.4.1:81/stream'
MODEL_PATH = 'yolov8n.pt'

# ================= DETECTION =================
ACQUIRE_CONF = 0.45       
TRACK_CONF = 0.20         
STICKY_RADIUS = 0.25      
LOST_HOLD_S = 3.0         

# ================= TURRET =================
PAN_CENTER = 90
PAN_START, TILT_START = 90, 70
PAN_LIMITS = (0, 180)
TILT_LIMITS = (30, 150)
PAN_GAIN = 0.035          
TILT_GAIN = 0.035
PIXEL_DEADZONE = 15
MAX_STEP_DEG = 5
PAN_DIR = 1               
TILT_DIR = 1              
AIM_Y = 0.40              
# ================= LASER =================
LASER_ENABLED = True
LASER_LOCK_PX = 40

# ================= CAR: TURNING =================
CAR_ENABLED = True
TURN_SIGN = 1             
TURN_START_DEG = 20
TURN_STOP_DEG = 10
MAX_OFFSET_DEG = 60
MIN_TURN = 130
MAX_TURN = 150
TURN_PULSE_S = 0.10
TURN_PAUSE_S = 0.35
TURN_LOCK_PX = 60
PAN_LIMIT_MARGIN = 2

# ================= CAR: DISTANCE (sonar) =================
TARGET_CM = 8
FWD_START_CM = 40         
FWD_STOP_CM = 38          
BACK_START_CM = 5         
BACK_STOP_CM = 7
DRIVE_SPEED = 160         
APPROACH_SPEED = 130      
SLOW_DOWN_CM = 40
CLOSE_TRUST_CM = 40       
CLOSE_HOLD_S = 10         
BLIND_CM = 20             

# ================= TIMING =================
SEND_HZ = 20
STALL_S = 1.0             # no new camera frame for this long -> stop the car
RECENTER_STEP_DEG = 1

STOP, FORWARD, BACKWARD, LEFT, RIGHT = 0, 1, 2, 3, 4
MOTOR_NAMES = {0: "STOP", 1: "FWD", 2: "BACK", 3: "LEFT", 4: "RIGHT"}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------- shared state ----------
state_lock = threading.Lock()
state = {'pan': PAN_START, 'tilt': TILT_START, 'laser': False, 'motor': STOP, 'speed': 0}
current_distance = 999
last_valid_distance = 999   # last reading that wasn't 999
running = True


def set_state(**kw):
    with state_lock:
        state.update(kw)


# ---------- serial ----------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
time.sleep(2)
ser.reset_input_buffer()


def serial_reader():
    global current_distance, last_valid_distance
    while running and ser.is_open:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            time.sleep(0.2)
            continue
        if line.startswith('DIST:'):
            try:
                d = int(line[5:])
            except ValueError:
                continue
            current_distance = d
            if d < 999:
                last_valid_distance = d


def sonar_blind():
    # No echo, but the last real reading was very close -> something is right
    # in front of the sensor, too close to measure. Treat as "don't move forward".
    return current_distance >= 999 and last_valid_distance < BLIND_CM


def serial_sender():
    period = 1.0 / SEND_HZ
    turn_dir, turn_t0 = None, 0.0
    while running:
        with state_lock:
            s = dict(state)
        motor, speed = s['motor'], s['speed']
        now = time.time()

        if motor in (LEFT, RIGHT):
            if motor != turn_dir:
                turn_dir, turn_t0 = motor, now
            if (now - turn_t0) % (TURN_PULSE_S + TURN_PAUSE_S) > TURN_PULSE_S:
                motor, speed = STOP, 0
        else:
            turn_dir = None

        cmd = (f"{int(round(s['pan']))},{int(round(s['tilt']))},"
               f"{1 if s['laser'] else 0},{motor},{speed}\n")
        try:
            ser.write(cmd.encode())
        except Exception:
            pass
        time.sleep(period)


# ---------- camera ----------
class Camera:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                continue
            with self.lock:
                self.frame = f
                self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, -1
            return self.frame.copy(), self.frame_id


# ---------- car decision ----------
class CarController:
    def __init__(self):
        self.turning = False
        self.drive = STOP

    def reset(self):
        self.turning = False
        self.drive = STOP

    def turn(self, pan, err_x):
        """Returns (motor, speed, reason) if the car should turn, otherwise None."""
        offset = (pan - PAN_CENTER) * TURN_SIGN
        threshold = TURN_STOP_DEG if self.turning else TURN_START_DEG
        if abs(offset) <= threshold:
            self.turning = False
            return None

        self.turning = True
        self.drive = STOP
        at_limit = (pan <= PAN_LIMITS[0] + PAN_LIMIT_MARGIN or
                    pan >= PAN_LIMITS[1] - PAN_LIMIT_MARGIN)
        if abs(err_x) > TURN_LOCK_PX and not at_limit:
            return STOP, 0, "waiting for turret"
        strength = clamp((abs(offset) - TURN_STOP_DEG) / (MAX_OFFSET_DEG - TURN_STOP_DEG), 0.0, 1.0)
        speed = int(MIN_TURN + strength * (MAX_TURN - MIN_TURN)) // 5 * 5
        return (LEFT if offset > 0 else RIGHT), speed, "turning toward you"

    def keep_distance(self, d):
        """Hold TARGET_CM using the sonar. Returns (motor, speed, reason)."""
        if d >= 999:
            if sonar_blind():
                self.drive = STOP
                return STOP, 0, "too close to measure"
            self.drive = FORWARD
            return FORWARD, DRIVE_SPEED, "far (no sonar echo)"

        # Hysteresis so it doesn't twitch around the target
        if self.drive == FORWARD and d <= FWD_STOP_CM:
            self.drive = STOP
        elif self.drive == BACKWARD and d >= BACK_STOP_CM:
            self.drive = STOP
        elif self.drive == STOP:
            if d > FWD_START_CM:
                self.drive = FORWARD
            elif d < BACK_START_CM:
                self.drive = BACKWARD

        if self.drive == FORWARD:
            speed = DRIVE_SPEED if d > SLOW_DOWN_CM else APPROACH_SPEED
            return FORWARD, speed, "approaching"
        if self.drive == BACKWARD:
            return BACKWARD, APPROACH_SPEED, "too close, backing off"
        return STOP, 0, "good spot"


def pick_target(boxes, confs, last_center, w):
    best, best_score = None, None
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if last_center is not None:
            dist = math.hypot(cx - last_center[0], cy - last_center[1])
            # weak detections only count if they're where you just were
            if conf < ACQUIRE_CONF and dist > STICKY_RADIUS * w:
                continue
            score = -dist
        else:
            if conf < ACQUIRE_CONF:
                continue
            score = (x2 - x1) * (y2 - y1)
        if best_score is None or score > best_score:
            best, best_score = (x1, y1, x2, y2), score
    return best


def main():
    global running
    model = YOLO(MODEL_PATH)
    cam = Camera(CAM_URL)
    car = CarController()
    threading.Thread(target=serial_reader, daemon=True).start()
    threading.Thread(target=serial_sender, daemon=True).start()

    pan, tilt = float(PAN_START), float(TILT_START)
    last_center = None
    last_seen = 0.0
    last_id = -1
    last_frame_time = time.time()
    last_log = None

    print("Tracking started - press Q to quit")

    try:
        while True:
            frame, fid = cam.read()
            if frame is None or fid == last_id:
                if time.time() - last_frame_time > STALL_S:
                    set_state(motor=STOP, speed=0, laser=False)
                time.sleep(0.005)
                continue
            last_id = fid
            last_frame_time = time.time()
            h, w = frame.shape[:2]

            result = model(frame, classes=[0], conf=TRACK_CONF, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
            else:
                boxes, confs = [], []
            target = pick_target(boxes, confs, last_center, w)

            now = time.time()
            dist = current_distance
            laser = False
            motor, speed, reason = STOP, 0, "car disabled"

            if target is not None:
                x1, y1, x2, y2 = target
                aim_x = (x1 + x2) / 2
                aim_y = y1 + (y2 - y1) * AIM_Y
                last_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                last_seen = now

                err_x = aim_x - w / 2
                err_y = aim_y - h / 2
                if abs(err_x) > PIXEL_DEADZONE:
                    pan -= PAN_DIR * clamp(err_x * PAN_GAIN, -MAX_STEP_DEG, MAX_STEP_DEG)
                if abs(err_y) > PIXEL_DEADZONE:
                    tilt += TILT_DIR * clamp(err_y * TILT_GAIN, -MAX_STEP_DEG, MAX_STEP_DEG)
                pan = clamp(pan, *PAN_LIMITS)
                tilt = clamp(tilt, *TILT_LIMITS)

                locked = abs(err_x) < LASER_LOCK_PX and abs(err_y) < LASER_LOCK_PX
                laser = LASER_ENABLED and locked

                if CAR_ENABLED:
                    turn_cmd = car.turn(pan, err_x)
                    motor, speed, reason = turn_cmd if turn_cmd else car.keep_distance(dist)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(aim_x), int(aim_y)), 5, (0, 0, 255), -1)
            else:
                lost_for = now - last_seen
                close = dist < CLOSE_TRUST_CM or sonar_blind()

                if CAR_ENABLED and last_seen > 0 and lost_for < CLOSE_HOLD_S and close:
                    # Too close for the camera to recognise you: hold distance by sonar,
                    # keep the turret where it is, no turning.
                    car.turning = False
                    motor, speed, reason = car.keep_distance(dist)
                    reason = "close, sonar only: " + reason
                else:
                    car.reset()
                    reason = "nobody seen"
                    if lost_for > LOST_HOLD_S:
                        last_center = None
                        pan += clamp(PAN_START - pan, -RECENTER_STEP_DEG, RECENTER_STEP_DEG)
                        tilt += clamp(TILT_START - tilt, -RECENTER_STEP_DEG, RECENTER_STEP_DEG)

            set_state(pan=pan, tilt=tilt, laser=laser, motor=motor, speed=speed)

            if (motor, reason) != last_log:
                last_log = (motor, reason)
                print(f"{time.strftime('%H:%M:%S')} {MOTOR_NAMES[motor]}: {reason} "
                      f"| sonar {dist}cm | pan {pan:.0f}")

            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            info = f"pan {pan:.0f}  sonar {dist}cm  {MOTOR_NAMES[motor]} {speed}"
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow('turret', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        time.sleep(0.15)
        try:
            ser.write(f"{PAN_START},{TILT_START},0,0,0\n".encode())
            time.sleep(0.1)
            ser.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Stopped: motors off, laser off, serial closed.")


if __name__ == '__main__':
    main()
