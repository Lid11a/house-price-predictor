from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    # Split a dataframe into feature matrix X and target vector y.
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    y = df[target_column].copy()
    x = df.drop(columns=[target_column], errors="ignore").copy()
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    # Build a ColumnTransformer: median imputation for numeric columns, mode + one-hot for the rest.
    num_cols = x.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [col for col in x.columns if col not in num_cols]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )


def build_training_pipeline(x: pd.DataFrame, alpha: float = 0.0005, random_state: int = 42) -> Pipeline:
    # Build a sklearn Pipeline: preprocessing plus Lasso (fit on log1p of price).
    preprocessor = build_preprocessor(x)
    model = Lasso(alpha=alpha, random_state=random_state, max_iter=30000)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def rmsle(y_true: pd.Series, y_pred: np.ndarray) -> float:
    # Root mean squared log error between true and predicted prices in original scale.
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
