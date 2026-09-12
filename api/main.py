import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import List

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

app = FastAPI(
    title="Fashion-MNIST Refund Classifier",
    description="Batch-ready prediction service for returned fashion items.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "saved_model", "model.pkl"))
metrics_store = {
    "prediction_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "manual_reviews": 0,
    "total_latency_ms": 0.0
}
CONFIDENCE_THRESHOLD = 0.60
MODEL_VERSION = "fashion-mnist-rf-v1"


class ImageData(BaseModel):
    pixels: List[float] = Field(min_length=784, max_length=784)

    @field_validator("pixels")
    @classmethod
    def validate_pixel_range(cls, values):
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("Pixel values must be between 0 and 255")
        return values


@app.get("/")
def root():
    return {
        "message": "Fashion classifier API is running",
        "model_version": MODEL_VERSION,
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    successful = max(metrics_store["successful_predictions"], 1)
    return {
        **metrics_store,
        "average_prediction_latency_ms": round(
            metrics_store["total_latency_ms"] / successful, 2
        ),
        "model_version": MODEL_VERSION
    }



@app.post("/predict")
def predict(data: ImageData):
    started = time.perf_counter()
    metrics_store["prediction_requests"] += 1
    try:
        pixels = pd.DataFrame([data.pixels])
        normalized = pixels / 255.0
        prediction = model.predict(normalized)
        probabilities = model.predict_proba(normalized)[0].tolist()
        predicted_class = int(prediction[0])
        confidence = float(max(probabilities))
        manual_review = confidence < CONFIDENCE_THRESHOLD
        metrics_store["successful_predictions"] += 1
        metrics_store["manual_reviews"] += int(manual_review)
        metrics_store["total_latency_ms"] += (time.perf_counter() - started) * 1000
        return {
            "prediction": predicted_class,
            "label": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4),
            "manual_review_required": manual_review,
            "model_version": MODEL_VERSION,
            "probabilities": {
                CLASS_NAMES[i]: round(probabilities[i], 4)
                for i in range(len(CLASS_NAMES))
            },
        }
    except Exception:
        metrics_store["failed_predictions"] += 1
        raise
