# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

# Generate synthetic training data
np.random.seed(42)
income = np.random.randint(5000, 50000, size=100)
internet_use = np.random.uniform(20, 100, size=100)
urban_rate = np.random.uniform(30, 100, size=100)

# Create a mock EV hotspot score based on weighted influence
ev_score = (
    0.3 * (income / 50000) +
    0.4 * (internet_use / 100) +
    0.3 * (urban_rate / 100)
) * 10  # Scale to range 0–10

df = pd.DataFrame({
    "incomeperperson": income,
    "internetuserate": internet_use,
    "urbanrate": urban_rate,
    "ev_hotspot_score": ev_score
})

X = df[["incomeperperson", "internetuserate", "urbanrate"]]
y = df["ev_hotspot_score"]

# Train and save Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)
joblib.dump(rf_model, "rf_model.pkl")

# Train and save XGBoost model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X, y)
joblib.dump(xgb_model, "xgb_model.pkl")

print("Models trained and saved successfully!")
