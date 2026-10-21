
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "splits"
MODEL_DIR = ROOT_DIR / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

TARGET = "response_time_ms"

NUMERIC_FEATURES = [
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
]

CATEGORICAL_FEATURES = ["server_id"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Check that the expected columns exist.
    required = FEATURES + [TARGET]

    for name, df in [("train", train_df), ("test", test_df)]:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name} data is missing: {missing}")

    X_train_full = train_df[FEATURES]
    y_train_full = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Use the final 20% of the training portion for validation.
    # Keep chronological order to approximate future predictions.
    split_idx = int(len(train_df) * 0.8)

    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full.iloc[:split_idx]

    X_val = X_train_full.iloc[split_idx:]
    y_val = y_train_full.iloc[split_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_pipeline(regressor):
    # Encode server names and impute missing feature values.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", regressor),
    ])


def evaluate(model, X, y):
    predictions = model.predict(X)

    return {
        "MAE_ms": round(float(mean_absolute_error(y, predictions)), 3),
        "RMSE_ms": round(
            float(np.sqrt(mean_squared_error(y, predictions))), 3
        ),
        "R2": round(float(r2_score(y, predictions)), 4),
    }


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    models = {
        "Dummy": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation_results = {}
    fitted_models = {}

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    # Train each candidate using training data only.
    for name, regressor in models.items():
        pipeline = build_pipeline(regressor)
        pipeline.fit(X_train, y_train)

        validation_results[name] = evaluate(
            pipeline, X_val, y_val
        )
        fitted_models[name] = pipeline

    print("\n--- Validation results ---")
    for name, result in validation_results.items():
        print(name, result)

    # Select the model with the lowest validation MAE.
    best_name = min(
        validation_results,
        key=lambda name: validation_results[name]["MAE_ms"],
    )

    best_model = fitted_models[best_name]

    # Evaluate the selected model on the untouched test set.
    test_results = evaluate(best_model, X_test, y_test)

    print("\nSelected model:", best_name)
    print("Final test results:", test_results)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "latency_model.joblib")

    metadata = {
        "selected_model": best_name,
        "target": TARGET,
        "features": FEATURES,
        "validation_results": validation_results,
        "test_results": test_results,
    }

    with (MODEL_DIR / "model_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved model:", MODEL_DIR / "latency_model.joblib")
    print("Saved metadata:", MODEL_DIR / "model_metadata.json")


if __name__ == "__main__":
    main()
