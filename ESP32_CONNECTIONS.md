# ESP32 Hardware Connections Guide
## Smart Agriculture System — Green Gram Monitoring

---

## Components Required
| Component | Purpose | Qty |
|-----------|---------|-----|
| ESP32 DevKit v1 | Microcontroller / WiFi | 1 |
| DHT11 | Temperature & Humidity | 1 |
| Soil Moisture Sensor (capacitive) | Soil moisture % | 1 |
| pH Sensor Module (SEN0161 or similar) | Soil pH | 1 |
| NPK Sensor (RS485, 7-in-1) | N, P, K values | 1 |
| RS485 to TTL converter (MAX485) | NPK sensor interface | 1 |
| 3.3V / 5V Power Supply | Powering sensors | 1 |
| Jumper wires, breadboard | Connections | — |

---

## Wiring Diagram

### 1. DHT11 (Temperature & Humidity)
```
DHT11 Pin    →   ESP32 Pin
─────────────────────────
VCC          →   3.3V
GND          →   GND
DATA         →   GPIO 4
(Add 10kΩ pull-up resistor between DATA and 3.3V)
```

### 2. Soil Moisture Sensor (Capacitive)
```
Sensor Pin   →   ESP32 Pin
──────────────────────────
VCC          →   3.3V
GND          →   GND
AOUT         →   GPIO 34 (ADC1_CH6 — input only pin)
```

### 3. pH Sensor Module (SEN0161 / analog output)
```
pH Module    →   ESP32 Pin
──────────────────────────
VCC          →   5V (use 5V pin on ESP32)
GND          →   GND
AOUT         →   GPIO 35 (ADC1_CH7 — input only pin)
```
> **Note:** pH sensor outputs 0–3V. ESP32 ADC reads 0–3.3V. Safe to connect directly.
> Calibrate with pH 4.0 and pH 7.0 buffer solutions.

### 4. NPK Sensor (RS485 UART) via MAX485 Module
```
MAX485 Pin   →   ESP32 Pin
──────────────────────────
VCC          →   5V
GND          →   GND
RO  (Recv)   →   GPIO 16 (RX2)
DI  (Data)   →   GPIO 17 (TX2)
RE  (RecvEn) →   GPIO 5  (active LOW)
DE  (DriveEn)→   GPIO 5  (active HIGH, tie RE+DE together)

NPK Sensor   →   MAX485
──────────────────────
A+           →   A (terminal)
B-           →   B (terminal)
VCC          →   12V (NPK sensors often need 9-24V — use separate adapter)
GND          →   GND (common ground)
```

---

## Full ESP32 Pin Summary
```
GPIO  4  — DHT11 Data
GPIO 16  — RS485 RX (NPK)
GPIO 17  — RS485 TX (NPK)
GPIO  5  — RS485 RE+DE (direction control)
GPIO 34  — Soil Moisture AOUT  [ADC input only]
GPIO 35  — pH Sensor AOUT      [ADC input only]
3.3V     — DHT11 VCC, Moisture VCC, pH VCC
5V       — MAX485 VCC
GND      — All GND (common)
```

---

## Arduino IDE Setup

### 1. Install ESP32 Board
- Arduino IDE → File → Preferences → Additional Boards URL:
  `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
- Tools → Board Manager → Search "esp32" → Install "ESP32 by Espressif"

### 2. Select Board
- Tools → Board → ESP32 Arduino → **ESP32 Dev Module**
- Tools → Port → Select your COM port
- Upload Speed: 115200

### 3. Install Libraries (Arduino Library Manager)
```
DHT sensor library     — by Adafruit
ArduinoJson            — by Benoit Blanchon
```
> HTTPClient and WiFi are built-in for ESP32.

---

## NPK Sensor RS485 Commands (Modbus RTU)
The 7-in-1 soil sensor uses Modbus. Send this hex command to read N, P, K:
```
01 03 00 1E 00 03 65 CD
```
Response is 9 bytes:
```
Byte[3–4] = Nitrogen (mg/kg)
Byte[5–6] = Phosphorus (mg/kg)
Byte[7–8] = Potassium (mg/kg)
```
The Arduino code in `esp32_sensor_node.ino` handles this with `readNPK()` function.

---

## Calibration Steps

### pH Calibration
1. Dip probe in pH 7.0 buffer → note ADC raw value (call it `raw7`)
2. Dip probe in pH 4.0 buffer → note ADC raw value (call it `raw4`)
3. In code: `ph = 7.0 + (raw7 - raw) * (3.0 / (raw7 - raw4));`

### Soil Moisture Calibration
1. Dry soil reading: note ADC value → this is 0%
2. Fully wet soil reading: note ADC value → this is 100%
3. In code: `moisture = map(raw, dryVal, wetVal, 0, 100);`

---

## Testing Connection
After uploading code, open Serial Monitor at 115200 baud.
You should see:
```
Connecting to WiFi........
Connected! IP: 192.168.1.105
Sending: {"device_id":"ESP32_01","ph":6.8,"npk_n":55,...}
Server response (200): {"status":"ok","predicted_yield_per_acre":512.3,...}
```
