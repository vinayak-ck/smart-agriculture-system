# train_random_forest.py
# ---------------------------------------------------------
# Train Random Forest Regressor for Green Gram Yield Prediction
# ---------------------------------------------------------

import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset/green_gram_growth_dataset.csv"
MODEL_OUT = BASE_DIR / "ml_models/random_forest_model.pkl"

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

print("\n[1/4] Loading Dataset...")

df = pd.read_csv(CSV_PATH)

FEATURES = [
    'ph',
    'npk_nitrogen',
    'npk_phosphorus',
    'npk_potassium',
    'temperature',
    'humidity',
    'soil_moisture',
    'day_number'
]

TARGET = 'yield_per_acre_kg'

X = df[FEATURES].values
y = df[TARGET].values

print(f"      Total Samples: {len(df)}")

# ---------------------------------------------------------
# Scaling
# ---------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# Train Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("\n[2/4] Training Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

preds = model.predict(X_test)

r2 = round(r2_score(y_test, preds) * 100, 1)
mae = round(mean_absolute_error(y_test, preds), 2)

print(f"      R² Score : {r2}%")
print(f"      MAE      : ±{mae} kg/acre")

# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

print("\n      Feature Importance:")

for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: -x[1]
):
    print(f"      {feature:<20} {round(importance * 100, 2)}%")

# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

print("\n[3/4] Saving Model...")

MODEL_OUT.parent.mkdir(exist_ok=True)

joblib.dump({
    'model': model,
    'scaler': scaler,
    'accuracy': r2
}, MODEL_OUT)

print(f"\n[4/4] Done!")
print(f"      Model Saved : {MODEL_OUT}")
print(f"      Accuracy    : {r2}%\n")