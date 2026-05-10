# Fashion Product Refund Classifier

## Project Overview

This project implements an image classification system for a refund department of an online shopping platform. The goal is to automatically classify returned fashion product images into item categories, such as shirts, handbags, watches, shoes, and sunglasses.

The system is designed as a small production-style machine learning pipeline. A trained model is served through a REST API, and a batch-processing script can process new refund images from an incoming folder and save the prediction results.

This project was created for the course **DLBDSMTP01 – Project: From Model to Production**, Task 2: Image classification for a refund department.

## Business Problem

As an online shopping platform grows, the number of returned items also increases. Manually sorting returned products becomes time-consuming and expensive. This project addresses that problem by using a machine learning model to classify product images automatically.

The system supports the refund department by predicting the product category from an image, returning a confidence score, flagging uncertain predictions for manual review, and processing multiple images in batches.

## Dataset

The project uses the **Fashion Product Images Small** dataset from Kaggle:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

For this prototype, selected categories were used to train a simple image classification model.

## Technologies Used

- Python
- FastAPI
- Uvicorn
- scikit-learn
- Pillow
- NumPy
- Pandas
- Joblib
- Requests
- Render configuration for deployment
- Local batch processing script

## Project Structure

```text
refund-item-image-classification-system-aydan/
│
├── api/
│   └── main.py
│
├── batch_process/
│   ├── batch.py
│   ├── incoming/
│   ├── processed/
│   └── failed/
│
├── model/
│
├── model_data/
│   └── fashion_products/
│       ├── images/
│       └── styles.csv
│
├── results/
│   └── batch_predictions.csv
│
├── saved_model/
│   ├── product_classifier.pkl
│   └── labels.json
│
├── requirements.txt
├── render.yaml
└── README.md
```

## System Architecture

The system follows this workflow:

```text
Returned item images
        ↓
Incoming image folder
        ↓
Batch processing script
        ↓
FastAPI prediction endpoint
        ↓
Trained classification model
        ↓
Prediction result with confidence score
        ↓
CSV results file + manual review flag
```

## API Endpoints

### Root Endpoint

```text
GET /
```

Returns a basic message showing that the API is running.

### Health Check Endpoint

```text
GET /health
```

Returns whether the API, model, and labels are loaded correctly.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "labels_loaded": true
}
```

### Prediction Endpoint

```text
POST /predict
```

Accepts an uploaded image and returns the predicted product class.

Example response:

```json
{
  "filename": "1526.jpg",
  "predicted_class": "Handbags",
  "confidence": 0.695,
  "manual_review_required": false,
  "class_probabilities": {
    "Casual Shoes": 0.015,
    "Handbags": 0.695,
    "Heels": 0.0,
    "Kurtas": 0.02,
    "Shirts": 0.01,
    "Sports Shoes": 0.005,
    "Sunglasses": 0.0,
    "Tops": 0.065,
    "Tshirts": 0.13,
    "Watches": 0.06
  }
}
```

## How to Run the Project Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Application

```bash
python -m uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

### 3. Open the API Documentation

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint to upload a product image and receive a prediction.

## Batch Processing

The batch-processing script simulates an overnight job. New images are placed in:

```text
batch_process/incoming
```

Run the batch script with:

```bash
python batch_process/batch.py
```

The script sends each image to the API, receives predictions, and saves the results to:

```text
results/batch_predictions.csv
```

Successfully processed images are moved to:

```text
batch_process/processed
```

Failed images are moved to:

```text
batch_process/failed
```

## Manual Review Logic

A confidence threshold of `0.60` is used.

If the model confidence is below `0.60`, the result is flagged as:

```text
manual_review_required = True
```

This means uncertain predictions can still be checked by a human employee.

## Example Batch Output

```csv
timestamp,filename,predicted_class,confidence,manual_review_required
2026-05-10T18:02:39,1569.jpg,Shirts,0.36,True
2026-05-10T18:02:39,1801.jpg,Watches,0.82,False
2026-05-10T18:02:39,1934.jpg,Shirts,0.28,True
2026-05-10T18:02:39,1163.jpg,Tshirts,0.775,False
```

## Deployment

The project includes a `render.yaml` file for deployment on Render.

The service start command is:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

This allows the FastAPI service to run as a web service.

## Limitations

This project is a prototype. The model is intentionally simple and uses resized image features with a traditional machine learning classifier. For a real production system, improvements could include:

- using a deep learning model such as CNN or transfer learning
- training on a larger and cleaner dataset
- adding a real database such as PostgreSQL
- adding authentication and access control
- adding automated monitoring and logging dashboards
- using scheduled cloud jobs for batch prediction
- retraining the model regularly with new refund data

## Conclusion

The project demonstrates how a machine learning model can be moved from a trained model into a small production-style service. The system can classify individual uploaded images through an API and can also process multiple refund images in batches. This supports the refund department by reducing manual sorting work while still keeping uncertain predictions available for human review.
