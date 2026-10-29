# app/api.py
from contextlib import asynccontextmanager
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from src.predict import Predictor

predictor: Predictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor()
    yield

app = FastAPI(
    title="Bank Marketing Intelligence API",
    version="1.0.0",
    description="Predicts customer subscription probability for term deposits.",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {
        "message": "Bank Marketing Intelligence API is running!",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
def predict_endpoint(payload: List[Dict[str, Any]], threshold: float = 0.35):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model engine not ready.")
    try:
        results = predictor.predict(payload, threshold=threshold)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")