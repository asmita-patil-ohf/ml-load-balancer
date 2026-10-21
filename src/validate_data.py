import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "load_balancer_dataset.csv"

EXPECTED_COLUMNS = [
    "server_id",
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
    "response_time_ms",
]

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nNumeric summary:")
print(df.describe())

print("\nUnexpected columns:")
print(set(df.columns) - set(EXPECTED_COLUMNS))

print("\nMissing expected columns:")
print(set(EXPECTED_COLUMNS) - set(df.columns))

# Check for invalid values
print("\nInvalid values:")
print("Invalid CPU:", (~df["cpu_percent"].between(0, 100)).sum())
print("Invalid memory:", (~df["memory_percent"].between(0, 100)).sum())
print("Non-positive latency:", (df["response_time_ms"] <= 0).sum())
print(
    "Invalid complexity:",
    (~df["request_complexity"].between(1, 5)).sum(),
)