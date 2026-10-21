import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "load_balancer_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape ---")
print(df.shape)

print("\n--- Column data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Summary statistics ---")
print(df.describe())

print("\n--- Samples per server ---")
print(df["server_id"].value_counts())

print("\n--- Average response time per server ---")
print(
    df.groupby("server_id")["response_time_ms"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)