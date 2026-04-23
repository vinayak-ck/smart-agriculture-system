"""
train_growth_model.py
─────────────────────
Train the Green Gram growth & yield regression model.

Usage:
    python train_growth_model.py                  ← uses included green_gram_growth_dataset.csv
    python train_growth_model.py --csv your.csv   ← use your own real field data CSV

Your CSV must have these columns:
    ph, npk_nitrogen, npk_phosphorus, npk_potassium,
    temperature, humidity, soil_moisture, day_number,
    height_cm, yield_per_acre_kg

Output:  ml_models/growth_model.pkl  (auto-loaded by Django on next request)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DEFAULT_CSV    = BASE_DIR / 'dataset/green_gram_growth_dataset.csv'
MODEL_OUT      = BASE_DIR / 'ml_models' / 'growth_model.pkl'
FEATURES       = ['ph', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium',
                  'temperature', 'humidity', 'soil_moisture', 'day_number']


def train(csv_path):
    print(f"\n[1/4] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=FEATURES + ['height_cm', 'yield_per_acre_kg'])
    print(f"      Rows: {len(df)}")

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}
    for target_col, key, label in [
        ('yield_per_acre_kg', 'yield_model', 'Yield (kg/acre)'),
        ('height_cm',          'height_model', 'Height (cm)'),
    ]:
        y = df[target_col].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        print(f"\n[2/4] Training {label} model...")
        model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_split=5, random_state=42
        )
        model.fit(X_tr, y_tr)

        r2  = round(r2_score(y_te, model.predict(X_te)) * 100, 1)
        mae = round(mean_absolute_error(y_te, model.predict(X_te)), 2)
        print(f"      R² Score : {r2}%")
        print(f"      MAE      : ±{mae} {label.split('(')[1].rstrip(')')}")

        # Feature importance
        fi = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
        print("      Feature importance:")
        for name, imp in fi:
            bar = '█' * int(imp * 40)
            print(f"        {name:<20} {bar} {round(imp*100, 1)}%")

        models[key] = (model, r2)

    print("\n[3/4] Saving model...")
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({
        'yield_model':     models['yield_model'][0],
        'height_model':    models['height_model'][0],
        'scaler':          scaler,
        'yield_accuracy':  models['yield_model'][1],
        'height_accuracy': models['height_model'][1],
    }, MODEL_OUT)

    print(f"[4/4] Done! Saved to: {MODEL_OUT}")
    print(f"\n      Yield R²  : {models['yield_model'][1]}%")
    print(f"      Height R² : {models['height_model'][1]}%")
    print("\n      Restart Django server to load the new model.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(DEFAULT_CSV), help='Path to training CSV')
    args = parser.parse_args()
    train(args.csv)
