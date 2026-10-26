import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = ROOT_DIR / "models" / "latency_model.joblib"
TEST_FILE = ROOT_DIR / "data" / "splits" / "test.csv"

FEATURES = [
    "server_id",
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
]

TARGET = "response_time_ms"


def main():
    model = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "server_id": test_df["server_id"],
        "actual_latency_ms": y_test,
        "predicted_latency_ms": predictions,
    })

    print("\nSample predictions:")
    print(results.head(10).round(2).to_string(index=False))

    print(
        "\nMAE (ms):",
        round(mean_absolute_error(y_test, predictions), 3),
    )


if __name__ == "__main__":
    main() 
    