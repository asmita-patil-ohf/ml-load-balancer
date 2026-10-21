import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT_DIR / "data" / "load_balancer_dataset.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "cleaned_dataset.csv"

REQUIRED_COLUMNS = [
    "server_id",
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
    "response_time_ms",
]

NUMERIC_COLUMNS = [
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
    "response_time_ms",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original rows:", len(df))

    # Verify the expected dataset schema.
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Convert numeric fields; invalid strings become NaN.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    before = len(df)

    # Remove missing and non-finite values.
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[
        df[NUMERIC_COLUMNS].apply(
            lambda col: col.map(
                lambda value: pd.notna(value)
                and abs(value) != float("inf")
            )
        ).all(axis=1)
    ]

    print("Rows removed due to missing/invalid numbers:", before - len(df))

    # Validate logical ranges.
    valid = (
        df["server_id"].isin(
            ["server_1", "server_2", "server_3"]
        )
        & df["cpu_percent"].between(0, 100)
        & df["memory_percent"].between(0, 100)
        & (df["active_requests"] >= 0)
        & (df["queue_length"] >= 0)
        & (df["recent_avg_latency_ms"] > 0)
        & df["request_complexity"].between(1, 5)
        & (df["response_time_ms"] > 0)
    )

    invalid_count = int((~valid).sum())
    df = df.loc[valid].copy()

    print("Rows removed by range checks:", invalid_count)

    # Remove exact duplicate observations.
    before = len(df)
    df = df.drop_duplicates()
    print("Duplicate rows removed:", before - len(df))

    # Reset row numbering after cleaning.
    df = df.reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaned rows:", len(df))
    print("Saved to:", OUTPUT_FILE)

    print("\nFinal validation:")
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("\nRows per server:")
    print(df["server_id"].value_counts())


if __name__ == "__main__":
    main()