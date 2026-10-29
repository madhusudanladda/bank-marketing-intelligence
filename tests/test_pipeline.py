# tests/test_pipeline.py
from fastapi.testclient import TestClient
from app.api import app
from src.predict import Predictor
from src.config import MODEL_PATH

def test_model_artifact_exists():
    assert MODEL_PATH.exists(), "Model artifact (.joblib) must be generated."

def test_inference_engine():
    predictor = Predictor()
    sample_data = [{
        "age": 35,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day_of_week": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }]
    results = predictor.predict(sample_data, threshold=0.35)
    assert len(results) == 1
    assert "subscribe" in results[0]
    assert 0.0 <= results[0]["probability"] <= 1.0

def test_api_health_and_predict():
    with TestClient(app) as client:
        # Health check
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "healthy"

        # Inference route
        payload = [{
            "age": 42,
            "job": "admin.",
            "marital": "single",
            "education": "tertiary",
            "default": "no",
            "balance": 2500,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day_of_week": 20,
            "month": "aug",
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }]
        pred_res = client.post("/predict?threshold=0.35", json=payload)
        assert pred_res.status_code == 200
        data = pred_res.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert 0.0 <= data["predictions"][0]["probability"] <= 1.0