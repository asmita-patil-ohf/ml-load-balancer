import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data" / "cleaned_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "splits"

TARGET = "response_time_ms"

FEATURES = [
    "server_id",
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) < 10:
        raise ValueError("Not enough observations to split.")

    # Keep chronological order: earlier rows train,
    # later rows test.
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining target summary:")
    print(y_train.describe().round(2))

    print("\nTesting target summary:")
    print(y_test.describe().round(2))

    print("\nFeature columns:")
    print(X_train.columns.tolist())

    print("\nTarget column:")
    print(y_train.name)


if __name__ == "__main__":
    main()