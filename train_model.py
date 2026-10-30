
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("corrosion_dataset.csv")

# Separate inputs and target
X = df.drop(columns=["Corrosion rate"])
y = df["Corrosion rate"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train the AI model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("MODEL TRAINED SUCCESSFULLY!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R² score: {r2:.4f}")
print("Target unit: not yet verified")

