# train_dummy_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate dummy training data
df = pd.DataFrame({
    "incomeperperson": [10000, 20000, 30000, 40000],
    "internetuserate": [50, 60, 70, 80],
    "urbanrate": [60, 70, 80, 90],
    "ev_hotspot_score": [1, 2, 3, 4]  # arbitrary values for testing
})

X = df[["incomeperperson", "internetuserate", "urbanrate"]]
y = df["ev_hotspot_score"]

# Train a simple model
model = RandomForestRegressor()
model.fit(X, y)

# Save it
joblib.dump(model, "ev_model.pkl")
