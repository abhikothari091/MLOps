# FastAPI_Labs/src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import numpy as np

app = FastAPI(title="Iris Model API", version="2.0")

MODEL_PATH = Path(__file__).resolve().parent / "iris_model.pkl"
_model = None
_target_names = None

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float  = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float  = Field(..., description="Petal width in cm")

@app.on_event("startup")
def load_model():
    global _model, _target_names
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    bundle = joblib.load(MODEL_PATH)
    _model = bundle["pipeline"]
    _target_names = bundle["target_names"]

@app.get("/")
def root():
    return {"message": "Iris Model API is running."}

@app.get("/health")
def health():
    ok = _model is not None
    return {"status": "ok" if ok else "not_ready"}

@app.post("/predict")
def predict(features: IrisFeatures):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    x = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    pred = _model.predict(x)[0]
    label = _target_names[pred] if _target_names and 0 <= pred < len(_target_names) else str(pred)
    return {"response": int(pred), "label": label}