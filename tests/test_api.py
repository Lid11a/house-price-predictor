import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from house_price_predictor.modeling import build_training_pipeline, split_features_target


def _build_artifact() -> dict:
    # Build a tiny fitted pipeline in memory so API tests do not need a model file on disk.
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "GrLivArea": [900, 1100, 1200, 1600, 1800, 2100],
            "Neighborhood": ["NAmes", "NAmes", "CollgCr", "CollgCr", "OldTown", "OldTown"],
            "SalePrice": [120000, 140000, 150000, 200000, 220000, 260000],
        }
    )
    x, y = split_features_target(df)
    model = build_training_pipeline(x, alpha=0.001, random_state=42)
    model.fit(x, np.log1p(y))
    return {"model": model, "id_column": "Id", "target_column": "SalePrice"}


def test_predict_endpoint(monkeypatch) -> None:
    # POST /predict returns 200 and one prediction per input record.
    monkeypatch.setattr("api.main.get_artifact", lambda: _build_artifact())
    client = TestClient(app)

    payload = {
        "records": [
            {"Id": 10, "GrLivArea": 1500, "Neighborhood": "NAmes"},
            {"Id": 11, "GrLivArea": 1700, "Neighborhood": "CollgCr"},
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    assert all(isinstance(p, int) for p in body["predictions"])


def test_schema_endpoint() -> None:
    # GET /schema returns JSON with features and notes.
    client = TestClient(app)
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert "features" in body
    assert "notes" in body


def test_cheatsheet_endpoint() -> None:
    # GET /cheatsheet returns Markdown explaining inputs.
    client = TestClient(app)
    response = client.get("/cheatsheet")
    assert response.status_code == 200
    assert "markdown" in response.headers.get("content-type", "")
    assert b"House Price Predictor" in response.content
