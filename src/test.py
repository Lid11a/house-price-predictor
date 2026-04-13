from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from house_price_predictor.config import DEFAULT_MODEL_PATH, DEFAULT_PREDICTIONS_PATH, DEFAULT_TEST_PATH
from house_price_predictor.data import ensure_parent_dir, load_csv
from house_price_predictor.download import ensure_kaggle_house_prices_data
from house_price_predictor.logging_config import setup_logging
from house_price_predictor.modeling import ID_COLUMN
from house_price_predictor.predict import load_artifact


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for batch prediction on a CSV file.
    parser = argparse.ArgumentParser(description="Generate predictions for test dataset.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Load the model and test CSV (download if needed), write predictions CSV, log the result.
    log = setup_logging("predict_batch")
    args = parse_args()
    log.info("Batch predict; test_path=%s model_path=%s", args.test_path, args.model_path)

    ensure_kaggle_house_prices_data(
        train_path=args.test_path.parent / "train.csv",
        test_path=args.test_path,
        force_download=args.force_download,
    )

    artifact = load_artifact(args.model_path)
    model = artifact["model"]
    id_column = artifact.get("id_column", ID_COLUMN)

    test_df = load_csv(args.test_path)
    preds = np.expm1(model.predict(test_df))
    preds = np.clip(preds, a_min=0.0, a_max=None)

    if id_column in test_df.columns:
        submission = pd.DataFrame({id_column: test_df[id_column], "SalePrice": preds})
    else:
        submission = pd.DataFrame({"SalePrice": preds})

    ensure_parent_dir(args.output_path)
    submission.to_csv(args.output_path, index=False)
    log.info("Predictions saved to %s (%d rows)", args.output_path, len(submission))


if __name__ == "__main__":
    main()
