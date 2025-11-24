// Pines del ESP32 conectados al ULN2003
const int IN1 = 14;
const int IN2 = 27;
const int IN3 = 26;
const int IN4 = 25;

// Tiempo entre pasos (ms). Si el motor vibra, súbelo.
int stepDelay = 5;

// Secuencia estándar para 28BYJ-48
void stepMotor(int stepIndex) {
  switch (stepIndex) {
    case 0: digitalWrite(IN1,1); digitalWrite(IN2,1); digitalWrite(IN3,0); digitalWrite(IN4,0); break;
    case 1: digitalWrite(IN1,0); digitalWrite(IN2,1); digitalWrite(IN3,1); digitalWrite(IN4,0); break;
    case 2: digitalWrite(IN1,0); digitalWrite(IN2,0); digitalWrite(IN3,1); digitalWrite(IN4,1); break;
    case 3: digitalWrite(IN1,1); digitalWrite(IN2,0); digitalWrite(IN3,0); digitalWrite(IN4,1); break;
  }
}

void moverAdelante(int pasos) {
  for (int i = 0; i < pasos; i++) {
    for (int j = 0; j < 4; j++){
      stepMotor(j);
      delay(stepDelay);
    }
    delay(stepDelay);
  }
}

void moverAtras(int pasos) {
  for (int i = 0; i < pasos; i++) {
    for (int j = 3; j >= 0; j--){
      stepMotor(j);
      delay(stepDelay);
    }
    delay(stepDelay);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.println("Listo. Comandos: f100 = 100 pasos adelante, b200 = 200 pasos atrás.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() < 2) return;

    char dir = cmd.charAt(0);
    int pasos = cmd.substring(1).toInt();

    if (dir == '+') moverAdelante(pasos);
    if (dir == '-') moverAtras(pasos);
    if (dir == 's') {
      digitalWrite(IN1,0);
      digitalWrite(IN2,0);
      digitalWrite(IN3,0);
      digitalWrite(IN4,0);
    }
  }
}
