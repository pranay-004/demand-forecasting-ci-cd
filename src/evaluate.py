import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
import os

# Load data
df = pd.read_csv("data/sales.csv")
df['date'] = pd.to_datetime(df['date'])

# Sort data
df = df.sort_values(by=['product_id', 'date'])

# Create lag features
df['lag_1'] = df.groupby('product_id')['sales'].shift(1)
df['lag_2'] = df.groupby('product_id')['sales'].shift(2)

df = df.dropna()

# Feature engineering
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Features and target
X = df[['product_id', 'day', 'month', 'day_of_week', 'lag_1', 'lag_2']]
y = df['sales']

# Split (same logic as training)
split = int(len(df) * 0.8)
X_test = X[split:]
y_test = y[split:]

# Load model
model = joblib.load("model/model.pkl")

# Predict
preds = model.predict(X_test)

# Calculate MAE
new_mae = mean_absolute_error(y_test, preds)

print("New MAE:", new_mae)

# Path for storing metrics
metrics_path = "model/metrics.txt"

# First run → no previous model
if not os.path.exists(metrics_path):
    print("No previous model found. Accepting first model ✅")

    with open(metrics_path, "w") as f:
        f.write(str(new_mae))

else:
    with open(metrics_path, "r") as f:
        old_mae = float(f.read())

    print("Old MAE:", old_mae)

    if new_mae < old_mae:
        print("New model is better ✅")

        with open(metrics_path, "w") as f:
            f.write(str(new_mae))
    else:
        print("Model is worse ❌ - Not updating model")
