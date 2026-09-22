#include <ESP32Servo.h>

// Command from laptop, one per line:  pan,tilt,laser,motor[,speed]
// motor: 0=stop 1=forward 2=backward 3=left 4=right
// speed: 0-255, optional. Missing or 0 = use the defaults below.
// Replies: "DIST:<cm>" every 100 ms, "OBSTACLE" when forward is blocked.

// ---- PINS ----
const int PAN_PIN   = 4;
const int TILT_PIN  = 5;
const int LASER_PIN = 2;
const int IN1 = 27, IN2 = 26, IN3 = 25, IN4 = 33;
const int ENA = 14, ENB = 32;
const int TRIG_PIN = 12;
const int ECHO_PIN = 13;   // HC-SR04 echo is 5V -> use a voltage divider to 3.3V!

// ---- TUNING ----
const int DRIVE_SPEED = 170;                    // default forward/back speed
const int TURN_SPEED  = 200;                    // default turn speed (test script uses this)
const int STOP_CM     = 20;                     // block forward motion closer than this
const unsigned long CMD_TIMEOUT_MS     = 400;   // stop if the laptop goes quiet
const unsigned long PING_INTERVAL_MS   = 60;
const unsigned long REPORT_INTERVAL_MS = 100;

// ---- MOTOR DIRECTION FIX ----
// A wheel spins backwards when it should go forward? Flip its flag.
const bool INVERT_A = false;
const bool INVERT_B = false;
// LEFT and RIGHT mirrored in the test script? Set this to true.
const bool SWAP_TURN = false;

// ---- DRIFT FIX ----
// Curves RIGHT when driving straight? Try 20 (10-40). Worse? Use -20.
const int DRIFT_FIX = 0;

Servo panServo, tiltServo;

String rxBuf = "";
int requestedMotor = 0;
int requestedSpeed = 0;
int appliedMotor = -1;
int appliedSpeed = -1;
unsigned long lastCmdMs = 0, lastPingMs = 0, lastReportMs = 0, lastObstacleMsgMs = 0;

long distBuf[3] = {999, 999, 999};
int distIdx = 0;
long distanceCm = 999;

// ---- MOTORS ----
// dir: 1 = forward, -1 = backward, 0 = off
void drive(int dirA, int dirB, int spd) {
  if (INVERT_A) dirA = -dirA;
  if (INVERT_B) dirB = -dirB;
  int spdA = spd, spdB = spd;
  if (DRIFT_FIX > 0) spdA = max(0, spd - DRIFT_FIX);
  if (DRIFT_FIX < 0) spdB = max(0, spd + DRIFT_FIX);
  digitalWrite(IN1, dirA > 0 ? HIGH : LOW);
  digitalWrite(IN2, dirA < 0 ? HIGH : LOW);
  digitalWrite(IN3, dirB > 0 ? HIGH : LOW);
  digitalWrite(IN4, dirB < 0 ? HIGH : LOW);
  analogWrite(ENA, dirA != 0 ? spdA : 0);
  analogWrite(ENB, dirB != 0 ? spdB : 0);
}

void applyMotor(int m, int spd) {
  if (m == 0) spd = 0;
  if (m == appliedMotor && spd == appliedSpeed) return;   // only touch pins on change
  appliedMotor = m;
  appliedSpeed = spd;

  int leftCase = SWAP_TURN ? 4 : 3;
  if (m == 1)                drive( 1,  1, spd);   // forward
  else if (m == 2)           drive(-1, -1, spd);   // backward
  else if (m == leftCase)    drive( 1, -1, spd);   // left
  else if (m == 3 || m == 4) drive(-1,  1, spd);   // right
  else                       drive( 0,  0, 0);     // stop
}

// ---- ULTRASONIC ----
long pingOnce() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long us = pulseIn(ECHO_PIN, HIGH, 25000);  // ~4 m max
  if (us == 0) return 999;                            // no echo = nothing in range
  return us / 58;
}

long median3(long a, long b, long c) {
  return max(min(a, b), min(max(a, b), c));
}

void updateDistance() {
  distBuf[distIdx] = pingOnce();
  distIdx = (distIdx + 1) % 3;
  distanceCm = median3(distBuf[0], distBuf[1], distBuf[2]);
}

// ---- SERIAL ----
void processCommand(const String &line) {
  int pan, tilt, laser, motor, spd = 0;
  int n = sscanf(line.c_str(), "%d,%d,%d,%d,%d", &pan, &tilt, &laser, &motor, &spd);
  if (n < 4) return;

  panServo.write(constrain(pan, 0, 180));
  tiltServo.write(constrain(tilt, 0, 180));
  digitalWrite(LASER_PIN, laser == 1 ? HIGH : LOW);
  requestedMotor = (motor >= 0 && motor <= 4) ? motor : 0;
  requestedSpeed = (n == 5) ? constrain(spd, 0, 255) : 0;
  lastCmdMs = millis();
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(rxBuf);
      rxBuf = "";
    } else if (c != '\r') {
      rxBuf += c;
      if (rxBuf.length() > 32) rxBuf = "";   // garbage protection
    }
  }
}

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN, 500, 2400);
  tiltServo.attach(TILT_PIN, 500, 2400);
  panServo.write(90);
  tiltServo.write(70);

  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  applyMotor(0, 0);

  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  readSerial();
  unsigned long now = millis();

  if (now - lastPingMs >= PING_INTERVAL_MS) {
    lastPingMs = now;
    updateDistance();
  }

  int m = requestedMotor;

  // Watchdog: laptop went quiet -> stop everything
  if (millis() - lastCmdMs > CMD_TIMEOUT_MS) {
    m = 0;
    digitalWrite(LASER_PIN, LOW);
  }

  // Obstacle check every loop
  if (m == 1 && distanceCm <= STOP_CM) {
    m = 0;
    if (now - lastObstacleMsgMs > 200) {
      Serial.println("OBSTACLE");
      lastObstacleMsgMs = now;
    }
  }

  int spd = requestedSpeed;
  if (spd <= 0) spd = (m == 3 || m == 4) ? TURN_SPEED : DRIVE_SPEED;
  applyMotor(m, spd);

  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = now;
    Serial.print("DIST:");
    Serial.println(distanceCm);
  }
}