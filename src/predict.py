# src/predict.py
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_PATH, DEFAULT_THRESHOLD

class Predictor:
    def __init__(self, model_path: str = str(MODEL_PATH)):
        self.model = joblib.load(model_path)
        # Extract features expected by the ColumnTransformer pipeline
        preprocessor = self.model.named_steps.get('prep')
        if preprocessor and hasattr(preprocessor, 'feature_names_in_'):
            self.expected_features = list(preprocessor.feature_names_in_)
        else:
            self.expected_features = None

    def predict(self, records: list[dict], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
        df = pd.DataFrame(records)
        df.columns = [c.strip().lower() for c in df.columns]

        if self.expected_features:
            # Align features with training schema
            for col in self.expected_features:
                if col not in df.columns:
                    df[col] = 0

            df = df[self.expected_features].copy()

        # Type conversion based on ColumnTransformer step metadata
        preprocessor = self.model.named_steps.get('prep')
        if preprocessor:
            for name, trans, cols in preprocessor.transformers_:
                if name == 'num':
                    for col in cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                elif name == 'cat':
                    for col in cols:
                        if col in df.columns:
                            df[col] = df[col].astype(str)

        probs = self.model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)

        return [
            {
                "subscribe": bool(p),
                "probability": round(float(prob), 4),
                "decision_threshold": threshold
            }
            for p, prob in zip(preds, probs)
        ]

if __name__ == "__main__":
    predictor = Predictor()
    print("Predictor engine ready!")