import os
import json
import io
import numpy as np
import joblib
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

# Paths can be changed through environment variables on Render
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/product_classifier.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "saved_model/labels.json")

IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.60

app = FastAPI(
    title="Fashion Product Refund Classifier",
    description="Classifies returned fashion product images into refund item categories.",
    version="1.0"
)

model = None
labels = None


@app.on_event("startup")
def load_model():
    """
    Loads the trained machine learning model and label mapping when the API starts.
    """
    global model, labels

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(LABELS_PATH):
        raise RuntimeError(f"Labels file not found: {LABELS_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        raw_labels = json.load(f)

    # Convert label keys from strings to integers
    labels = {int(k): v for k, v in raw_labels.items()}

    print("Model and labels loaded successfully.")


def image_to_features(image_bytes):
    """
    Converts an uploaded image into the same flattened feature format
    used during model training.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image) / 255.0
    features = image_array.flatten().reshape(1, -1)

    return features


@app.get("/")
def root():
    """
    Simple endpoint to check whether the API is running.
    """
    return {
        "message": "Fashion Product Refund Classifier API is running",
        "model": "RandomForestClassifier",
        "classes": list(labels.values()) if labels else [],
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": labels is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives one image file and returns:
    - predicted class
    - confidence score
    - class probabilities
    - whether manual review is required
    """
    if model is None or labels is None:
        raise HTTPException(
            status_code=503,
            detail="Model or labels are not loaded yet."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        image_bytes = await file.read()
        features = image_to_features(image_bytes)

        predicted_label = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        # model.classes_ gives the correct class order for probabilities
        class_probabilities = {
            labels[int(class_id)]: float(probability)
            for class_id, probability in zip(model.classes_, probabilities)
        }

        predicted_class = labels[predicted_label]
        confidence = class_probabilities[predicted_class]

        manual_review_required = confidence < CONFIDENCE_THRESHOLD

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "manual_review_required": manual_review_required,
            "class_probabilities": class_probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))