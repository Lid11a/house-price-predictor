import numpy as np
import pandas as pd

from house_price_predictor.modeling import build_training_pipeline, split_features_target


def test_training_pipeline_fits_and_predicts() -> None:
    # The training pipeline fits on toy data and produces one prediction per row.
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "GrLivArea": [900, 1100, 1200, 1600, 1800, 2100],
            "Neighborhood": ["NAmes", "NAmes", "CollgCr", "CollgCr", "OldTown", "OldTown"],
            "SalePrice": [120000, 140000, 150000, 200000, 220000, 260000],
        }
    )
    x, y = split_features_target(df)
    pipeline = build_training_pipeline(x, alpha=0.001, random_state=42)
    pipeline.fit(x, np.log1p(y))

    preds_log = pipeline.predict(x)
    assert preds_log.shape[0] == len(df)
