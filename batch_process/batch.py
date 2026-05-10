import os
import shutil
import requests
import pandas as pd
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

INCOMING_DIR = "batch_process/incoming"
PROCESSED_DIR = "batch_process/processed"
FAILED_DIR = "batch_process/failed"
RESULTS_DIR = "results"

RESULTS_FILE = os.path.join(RESULTS_DIR, "batch_predictions.csv")

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def ensure_directories():
    """
    Creates required folders if they do not already exist.
    """
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def predict_image(image_path):
    """
    Sends one image to the FastAPI prediction endpoint.
    """
    with open(image_path, "rb") as image_file:
        files = {
            "file": (
                os.path.basename(image_path),
                image_file,
                "image/jpeg"
            )
        }

        response = requests.post(API_URL, files=files, timeout=30)
        response.raise_for_status()

        return response.json()


def run_batch_prediction():
    """
    Processes all new images in the incoming folder.
    Successful files are moved to processed.
    Failed files are moved to failed.
    Results are saved to a CSV file.
    """
    ensure_directories()

    batch_results = []

    image_files = [
        file_name for file_name in os.listdir(INCOMING_DIR)
        if file_name.lower().endswith(ALLOWED_EXTENSIONS)
    ]

    if not image_files:
        print("No new images found in incoming folder.")
        return

    print(f"Found {len(image_files)} image(s) for batch prediction.")

    for file_name in image_files:
        image_path = os.path.join(INCOMING_DIR, file_name)

        try:
            prediction = predict_image(image_path)

            result_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "filename": prediction["filename"],
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "manual_review_required": prediction["manual_review_required"]
            }

            batch_results.append(result_row)

            shutil.move(
                image_path,
                os.path.join(PROCESSED_DIR, file_name)
            )

            print(f"Processed: {file_name} -> {prediction['predicted_class']}")

        except Exception as error:
            shutil.move(
                image_path,
                os.path.join(FAILED_DIR, file_name)
            )

            print(f"Failed: {file_name}. Error: {error}")

    if batch_results:
        results_df = pd.DataFrame(batch_results)

        if os.path.exists(RESULTS_FILE):
            results_df.to_csv(
                RESULTS_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            results_df.to_csv(
                RESULTS_FILE,
                index=False
            )

        print(f"Batch results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_batch_prediction()