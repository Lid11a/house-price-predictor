from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from house_price_predictor.config import DEFAULT_MODEL_PATH
from house_price_predictor.sklearn_compat import patch_sklearn_simple_imputers


def load_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    # Load the joblib dict saved by train.py and patch estimators for sklearn version drift.
    artifact = joblib.load(model_path)
    model = artifact.get("model")
    if model is not None:
        patch_sklearn_simple_imputers(model)
    return artifact


def predict_records(records: list[dict[str, Any]], artifact: dict[str, Any]) -> list[int]:
    # Build a DataFrame aligned to training columns, run the pipeline, return sale prices as whole dollars.
    model = artifact["model"]
    df = pd.DataFrame.from_records(records)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        df = df.reindex(columns=list(feature_names))
    preds = np.expm1(model.predict(df))
    preds = np.clip(preds, a_min=0.0, a_max=None)
    return [int(round(float(x))) for x in preds]
