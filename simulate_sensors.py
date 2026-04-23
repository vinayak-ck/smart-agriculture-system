"""
simulate_sensors.py
Run this on your PC to simulate ESP32 sensor data hitting your Django backend.
Usage: python simulate_sensors.py
"""
import requests
import random
import time
import math

SERVER = "http://localhost:8000/api/sensor-data"
DEVICE_ID = "ESP32_SIM_01"

t = 0
print(f"Sending simulated sensor data to {SERVER} every 10s... (Ctrl+C to stop)\n")

while True:
    t += 1
    # Simulate realistic slowly-changing sensor values
    ph           = round(6.5 + 0.5 * math.sin(t / 10) + random.uniform(-0.1, 0.1), 2)
    temperature  = round(27 + 3 * math.sin(t / 20) + random.uniform(-0.5, 0.5), 1)
    humidity     = round(65 + 10 * math.cos(t / 15) + random.uniform(-2, 2), 1)
    soil_moisture= round(45 + 8 * math.sin(t / 25) + random.uniform(-2, 2), 1)
    npk_n        = round(random.uniform(60, 90), 1)
    npk_p        = round(random.uniform(30, 55), 1)
    npk_k        = round(random.uniform(45, 70), 1)

    payload = {
        "device_id":    DEVICE_ID,
        "ph":           ph,
        "npk_n":        npk_n,
        "npk_p":        npk_p,
        "npk_k":        npk_k,
        "temperature":  temperature,
        "humidity":     humidity,
        "soil_moisture":soil_moisture,
    }

    try:
        res = requests.post(SERVER, json=payload, timeout=5)
        data = res.json()
        print(f"[{t:3d}] Sent → pH:{ph} T:{temperature}°C H:{humidity}% | "
              f"Yield:{data.get('predicted_yield_kg')}kg Health:{data.get('health_score')}%")
    except Exception as e:
        print(f"[{t:3d}] ERROR: {e}")

    time.sleep(10)
