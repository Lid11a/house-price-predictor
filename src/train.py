from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from house_price_predictor.config import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, DEFAULT_TRAIN_PATH
from house_price_predictor.data import ensure_parent_dir, load_csv
from house_price_predictor.download import ensure_kaggle_house_prices_data
from house_price_predictor.logging_config import setup_logging
from house_price_predictor.modeling import ID_COLUMN, TARGET_COLUMN, build_training_pipeline, rmsle, split_features_target


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for the training script.
    parser = argparse.ArgumentParser(description="Train house price model.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--target-column", type=str, default=TARGET_COLUMN)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Load data, fit the pipeline, save the artifact and metrics; log milestones only.
    log = setup_logging("train")
    args = parse_args()
    log.info("Starting training; train_path=%s", args.train_path)

    ensure_kaggle_house_prices_data(
        train_path=args.train_path,
        test_path=args.train_path.parent / "test.csv",
        force_download=args.force_download,
    )
    log.info("Data ready (local files or downloaded).")

    df = load_csv(args.train_path)
    x, y = split_features_target(df, target_column=args.target_column)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state
    )

    pipeline = build_training_pipeline(x_train, alpha=args.alpha, random_state=args.random_state)
    pipeline.fit(x_train, np.log1p(y_train))
    log.info("Model fitted.")

    log_preds = pipeline.predict(x_valid)
    preds = np.expm1(log_preds)
    rmse_value = float(np.sqrt(mean_squared_error(y_valid, preds)))
    rmsle_value = rmsle(y_valid, preds)
    log.info("Validation RMSE=%.4f RMSLE=%.6f", rmse_value, rmsle_value)

    artifact = {
        "model": pipeline,
        "target_column": args.target_column,
        "id_column": ID_COLUMN,
        "sklearn_version": sklearn.__version__,
    }

    ensure_parent_dir(args.model_path)
    ensure_parent_dir(args.metrics_path)
    joblib.dump(artifact, args.model_path)

    metrics = {"rmse": rmse_value, "rmsle": rmsle_value}
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log.info("Model saved to %s", args.model_path)
    log.info("Metrics JSON saved to %s", args.metrics_path)


if __name__ == "__main__":
    main()
