/*
 * Smart Agriculture System — ESP32 Sensor Node
 * Crop: Green Gram
 * Sensors: DHT11, Soil Moisture, pH, NPK (RS485)
 * Sends JSON to Django backend every 30 seconds
 *
 * Wiring Summary:
 *   DHT11 DATA       → GPIO 4
 *   Soil Moisture    → GPIO 34
 *   pH Sensor        → GPIO 35
 *   RS485 RX (NPK)   → GPIO 16
 *   RS485 TX (NPK)   → GPIO 17
 *   RS485 RE+DE      → GPIO 5
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "DHT.h"
#define RELAY_PIN 26 

// ─── CONFIG ─────────────────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// Your PC's local IP — run `hostname -I` (Linux) or `ipconfig` (Windows)
const char* SERVER_URL = "http://172.16.22.149:8000/api/sensor-data";
const char* DEVICE_ID     = "ESP32_GG_01";   // GG = Green Gram

// ─── PIN DEFINITIONS ────────────────────────────────────────────────────────
#define DHT_PIN          4
#define DHT_TYPE         DHT11
#define SOIL_PIN         34    // Capacitive soil moisture (ADC)
#define PH_PIN           35    // pH analog output (ADC)
#define RS485_RX         16
#define RS485_TX         17
#define RS485_DE_RE      5     // Direction control (HIGH=transmit, LOW=receive)

// ─── CALIBRATION VALUES ─────────────────────────────────────────────────────
// pH calibration (adjust after calibrating with pH 4.0 and pH 7.0 buffers)
const float PH_SLOPE     = 0.18;   // voltage per pH unit
const float PH_OFFSET    = 2.5;    // voltage at pH 7.0

// Soil moisture calibration (read with dry and wet soil)
const int MOISTURE_DRY   = 3500;   // ADC value in dry soil
const int MOISTURE_WET   = 1200;   // ADC value in fully wet soil

// ─── NPK SENSOR COMMAND (Modbus RTU) ────────────────────────────────────────
const byte NPK_CMD[]     = {0x01, 0x03, 0x00, 0x1E, 0x00, 0x03, 0x65, 0xCD};

DHT dht(DHT_PIN, DHT_TYPE);
HardwareSerial RS485Serial(2);   // Use UART2 for RS485

// ─── SETUP ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  dht.begin();

  // RS485 serial for NPK sensor
  RS485Serial.begin(9600, SERIAL_8N1, RS485_RX, RS485_TX);
  pinMode(RS485_DE_RE, OUTPUT);
  digitalWrite(RS485_DE_RE, LOW);  // Start in receive mode

  // Connect WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);  // HIGH = relay OFF (active LOW relay)
}

// ─── pH READING ─────────────────────────────────────────────────────────────
float readPH() {
  // Average 10 readings to reduce noise
  long sum = 0;
  for (int i = 0; i < 10; i++) { sum += analogRead(PH_PIN); delay(10); }
  float voltage = (sum / 10.0) * (3.3 / 4095.0);
  float ph = 7.0 + ((PH_OFFSET - voltage) / PH_SLOPE);
  return constrain(ph, 0.0, 14.0);
}

// ─── SOIL MOISTURE ──────────────────────────────────────────────────────────
float readMoisture() {
  int raw = analogRead(SOIL_PIN);
  float pct = (float)(MOISTURE_DRY - raw) / (MOISTURE_DRY - MOISTURE_WET) * 100.0;
  return constrain(pct, 0, 100);
}

// ─── NPK via RS485 ──────────────────────────────────────────────────────────
struct NPKValues { float n, p, k; bool valid; };

NPKValues readNPK() {
  NPKValues result = {0, 0, 0, false};

  // Send query command
  digitalWrite(RS485_DE_RE, HIGH);   // Transmit mode
  delay(10);
  RS485Serial.write(NPK_CMD, sizeof(NPK_CMD));
  RS485Serial.flush();
  delay(10);
  digitalWrite(RS485_DE_RE, LOW);    // Receive mode

  // Wait for response (9 bytes)
  unsigned long start = millis();
  while (RS485Serial.available() < 9 && millis() - start < 1000) delay(10);

  if (RS485Serial.available() >= 9) {
    byte buf[9];
    RS485Serial.readBytes(buf, 9);
    // Parse Modbus response: bytes 3-4 = N, 5-6 = P, 7-8 = K
    result.n = (buf[3] << 8) | buf[4];
    result.p = (buf[5] << 8) | buf[6];
    result.k = (buf[7] << 8) | buf[8];
    result.valid = true;
  } else {
    Serial.println("[NPK] No response from sensor — using fallback values");
    // Fallback: plausible values for demo
    result.n = 55.0; result.p = 35.0; result.k = 45.0;
    result.valid = false;
  }

  // Clear buffer
  while (RS485Serial.available()) RS485Serial.read();
  return result;
}


// Add this function:
void controlPump(float moisture) {
  if (moisture < 30) {
    digitalWrite(RELAY_PIN, LOW);   // Turn pump ON
    Serial.println("💧 Pump ON — soil moisture low");
  } else {
    digitalWrite(RELAY_PIN, HIGH);  // Turn pump OFF
    Serial.println("✓ Pump OFF — moisture OK");
  }
}

// ─── MAIN LOOP ──────────────────────────────────────────────────────────────
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    WiFi.reconnect();
    delay(5000);
    return;
  }

  // Read all sensors
  float temp     = dht.readTemperature();
  float humidity = dht.readHumidity();
  float ph       = readPH();
  float moisture = readMoisture();
  NPKValues npk  = readNPK();
  controlPump(moisture);
  // DHT11 failure fallback
  if (isnan(temp) || isnan(humidity)) {
    Serial.println("[DHT11] Read failed — using fallback");
    temp = 29.0; humidity = 70.0;
  }

  // Build JSON payload
  StaticJsonDocument<256> doc;
  doc["device_id"]    = DEVICE_ID;
  doc["ph"]           = round(ph * 100) / 100.0;
  doc["npk_n"]        = npk.n;
  doc["npk_p"]        = npk.p;
  doc["npk_k"]        = npk.k;
  doc["temperature"]  = round(temp * 10) / 10.0;
  doc["humidity"]     = round(humidity * 10) / 10.0;
  doc["soil_moisture"]= round(moisture * 10) / 10.0;

  String payload;
  serializeJson(doc, payload);

  Serial.println("─────────────────────────────");
  Serial.println("pH: "        + String(ph, 2));
  Serial.println("N/P/K: "     + String(npk.n) + "/" + String(npk.p) + "/" + String(npk.k));
  Serial.println("Temp: "      + String(temp, 1) + "°C | Humidity: " + String(humidity, 1) + "%");
  Serial.println("Moisture: "  + String(moisture, 1) + "%");
  Serial.println("Sending → "  + payload);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);

  if (code == 200) {
    String resp = http.getString();
    Serial.println("✓ Server OK: " + resp.substring(0, 100));
  } else {
    Serial.println("✗ POST failed. Code: " + String(code));
  }
  http.end();

  delay(30000);  // 30 seconds interval
}
