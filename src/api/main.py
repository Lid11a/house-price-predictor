from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from house_price_predictor.config import (
    DATA_DESCRIPTION_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
)
from house_price_predictor.feature_schema import build_input_cheatsheet_markdown, build_prediction_input_schema
from house_price_predictor.predict import load_artifact, predict_records

app = FastAPI(
    title="House Price Predictor API",
    version="0.1.0",
    description=(
        "Predict Ames house sale prices from tabular features (Kaggle-style). "
        "**GET /cheatsheet** — Markdown memo: what each field means and which values are valid. "
        "**GET /schema** — machine-readable JSON for the same. "
        "**GET /docs** — Swagger UI with an example body for POST /predict."
    ),
)


PREDICT_EXAMPLE = {
    "records": [
        {
            "Id": 1461,
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotFrontage": 65.0,
            "LotArea": 8450,
            "Street": "Pave",
            "Alley": "NA",
            "LotShape": "Reg",
            "LandContour": "Lvl",
            "Utilities": "AllPub",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "Neighborhood": "CollgCr",
            "Condition1": "Norm",
            "Condition2": "Norm",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2003,
            "RoofStyle": "Gable",
            "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "None",
            "MasVnrArea": 0.0,
            "ExterQual": "Gd",
            "ExterCond": "TA",
            "Foundation": "PConc",
            "BsmtQual": "Gd",
            "BsmtCond": "TA",
            "BsmtExposure": "No",
            "BsmtFinType1": "GLQ",
            "BsmtFinSF1": 706,
            "BsmtFinType2": "Unf",
            "BsmtFinSF2": 0,
            "BsmtUnfSF": 0,
            "TotalBsmtSF": 706,
            "Heating": "GasA",
            "HeatingQC": "Ex",
            "CentralAir": "Y",
            "Electrical": "SBrkr",
            "1stFlrSF": 706,
            "2ndFlrSF": 676,
            "LowQualFinSF": 0,
            "GrLivArea": 1382,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "KitchenAbvGr": 1,
            "KitchenQual": "Gd",
            "TotRmsAbvGrd": 6,
            "Functional": "Typ",
            "Fireplaces": 1,
            "FireplaceQu": "Gd",
            "GarageType": "Attchd",
            "GarageYrBlt": 2003,
            "GarageFinish": "RFn",
            "GarageCars": 2,
            "GarageArea": 484,
            "GarageQual": "TA",
            "GarageCond": "TA",
            "PavedDrive": "Y",
            "WoodDeckSF": 192,
            "OpenPorchSF": 38,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "PoolArea": 0,
            "PoolQC": "NA",
            "Fence": "NA",
            "MiscFeature": "NA",
            "MiscVal": 0,
            "MoSold": 6,
            "YrSold": 2008,
            "SaleType": "WD",
            "SaleCondition": "Normal",
        }
    ]
}


class PredictRequest(BaseModel):
    # Request body: one JSON object per house; keys match training CSV columns (except the target).
    model_config = ConfigDict(json_schema_extra={"example": PREDICT_EXAMPLE})

    records: list[dict[str, Any]] = Field(
        min_length=1,
        description=(
            "List of houses. Each item is feature name → value, as in train.csv (omit SalePrice). "
            "See **GET /cheatsheet** for human-readable meanings and allowed category codes; "
            "**GET /schema** for the same data as JSON."
        ),
    )


class PredictResponse(BaseModel):
    # Response: predicted sale prices in whole US dollars, same order as records.
    predictions: list[int]


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    # Load the joblib artifact once per process (includes sklearn compatibility patches).
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_artifact(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    # Liveness probe for load balancers and manual checks.
    return {"status": "ok"}


@app.get("/schema")
def input_schema() -> dict[str, Any]:
    # JSON schema for POST /predict: feature names, numeric vs categorical, descriptions, and codes.
    return build_prediction_input_schema(
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        description_path=DATA_DESCRIPTION_PATH,
    )


@app.get("/cheatsheet")
def input_cheatsheet() -> Response:
    # Long-form Markdown guide: what each feature means and what strings/numbers to send.
    body = build_input_cheatsheet_markdown(
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        description_path=DATA_DESCRIPTION_PATH,
    )
    return Response(content=body, media_type="text/markdown; charset=utf-8")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    # Run the fitted pipeline on each record and return sale price predictions.
    try:
        artifact = get_artifact()
        preds = predict_records(payload.records, artifact)
        return PredictResponse(predictions=preds)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
