
import pandas as pd

df = pd.read_csv("corrosion_dataset.csv")

print("\n--- DATASET SIZE ---")
print("Rows and columns:", df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- CORROSION RATE RANGE ---")
print("Minimum:", df["Corrosion rate"].min())
print("Maximum:", df["Corrosion rate"].max())
print("Average:", df["Corrosion rate"].mean())
