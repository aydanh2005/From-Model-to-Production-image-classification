import os
import json
import io
import numpy as np
import joblib
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

MODEL_PATH = "saved_model/product_classifier.pkl"
LABELS_PATH = "saved_model/labels.json"

IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.60

app = FastAPI(
    title="Fashion Product Refund Classifier",
    description="Classifies returned fashion product images into 10 refund item categories.",
    version="1.0"
)

model = None
labels = None


@app.on_event("startup")
def load_model():
    global model, labels

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(LABELS_PATH):
        raise RuntimeError(f"Labels file not found: {LABELS_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        raw_labels = json.load(f)

    labels = {int(k): v for k, v in raw_labels.items()}

    print("Model and labels loaded successfully.")


def image_to_features(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return image_array.flatten().reshape(1, -1)


@app.get("/")
def root():
    return {
        "message": "Fashion Product Refund Classifier API is running",
        "model": "RandomForestClassifier",
        "classes": list(labels.values()) if labels else [],
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        image_bytes = await file.read()
        features = image_to_features(image_bytes)

        predicted_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        predicted_class = labels[predicted_index]
        confidence = float(probabilities[predicted_index])

        class_probabilities = {
            labels[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

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