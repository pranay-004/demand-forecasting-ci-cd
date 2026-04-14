import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/sales.csv")
df['date'] = pd.to_datetime(df['date'])

# Sort data (important for time series)
df = df.sort_values(by=['product_id', 'date'])

# Create lag features
df['lag_1'] = df.groupby('product_id')['sales'].shift(1)
df['lag_2'] = df.groupby('product_id')['sales'].shift(2)

# Drop missing values
df = df.dropna()

# Feature engineering
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Features and target
X = df[['product_id', 'day', 'month', 'day_of_week', 'lag_1', 'lag_2']]
y = df['sales']

# Train-test split (no shuffle for time-series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained successfully ✅")