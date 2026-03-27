import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 🔹 Load dataset
df = pd.read_csv('../../data/Crop_recommendation.csv')

# 🔹 Create Growth Score (IMPORTANT LOGIC)
def calculate_growth(row):
    score = (
        (row['N'] + row['P'] + row['K']) / 3 +
        row['temperature'] * 2 +
        row['humidity'] * 0.5 -
        abs(row['ph'] - 7) * 10
    )
    return score

df['growth_score'] = df.apply(calculate_growth, axis=1)

# 🔹 Features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph']]
y = df['growth_score']

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🔹 Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 🔹 Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("Model trained successfully!")
print("MSE:", mse)

# 🔹 Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")