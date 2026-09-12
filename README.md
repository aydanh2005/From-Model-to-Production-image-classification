# Fashion-MNIST Refund Classifier

[![Tests](https://github.com/aydanh2005/From-Model-to-Production-image-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/aydanh2005/From-Model-to-Production-image-classification/actions/workflows/tests.yml)
[![Nightly verification](https://github.com/aydanh2005/From-Model-to-Production-image-classification/actions/workflows/nightly-batch.yml/badge.svg)](https://github.com/aydanh2005/From-Model-to-Production-image-classification/actions/workflows/nightly-batch.yml)

**Verified result:** 88.21% test accuracy and 0.8804 macro F1 on 10,000 held-out Fashion-MNIST images.

## Part 1: Train

1. Download the [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)
2. Unzip and place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` into `model_data/`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the training script:

```bash
python3 model/train.py
```

The trained model is saved to `saved_model/model.pkl`.

Evaluation evidence is saved to:

```text
results/metrics.json
results/classification_report.csv
results/confusion_matrix.png
```

---

## Part 2: API

Run the API:

```bash
uvicorn api.main:app --reload
```

The service exposes:

- `GET /health` for model availability
- `GET /metrics` for request, error, latency, review-rate, and model-version monitoring
- `POST /predict` for class probabilities, confidence, and manual-review routing

---

## Part 3: Samples and Prediction

Get a sample and send it to the API:

```bash
# Terminal 1 — start the API
uvicorn api.main:app --reload

# Terminal 2 — send a sample to /predict
python3 get_sample.py | curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @-
```

`get_sample.py` returns a JSON object with a `pixels` array of 784 integers (28×28 pixel values, range 0–255) representing one flattened Fashion-MNIST image.

---

## Part 4: Docker

> Make sure you have run `python3 model/train.py` first — the API container requires `saved_model/model.pkl` at build time.

Start all services (API + batch + PostgreSQL):

```bash
docker-compose up --build
```

Drop CSV files into `batch_process/upload/`. The Docker scheduler checks the folder every night at 02:00. Successful files move to `processed/`; unsuccessful files move to `failed/`.

---

## Training

`model/train.py`:

- Reads CSVs from `model_data/`
- Separates labels from pixel columns
- Normalizes pixel values (divides by 255)
- Trains a `RandomForestClassifier` on 60k images
- Evaluates accuracy on 10k test images
- Evaluates macro and weighted F1 and generates a confusion matrix
- Saves the model to `saved_model/model.pkl` using `joblib`

When the API starts, it loads the model once into memory:

```python
model = joblib.load(os.path.join(BASE_DIR, "saved_model", "model.pkl"))
```

---

## Batching

`batch_process/batch.py`:

- Looks in `upload/` for CSV files
- For each file, sends each row to `POST /predict` with the pixel data as JSON
- Saves results to PostgreSQL (`predictions` table)
- Runs automatically every night at 02:00 via the scheduler
- Uses a database uniqueness constraint to prevent duplicate source-file rows

To test manually, add a CSV to the upload folder:

```bash
cp model_data/fashion-mnist_test.csv batch_process/upload/
```

---

## Predictions Table

| column | type | example |
|---|---|---|
| `id` | auto-increment integer | 1, 2, 3... |
| `source_file` | text | `returns_batch_1.csv` |
| `row_index` | integer | 0, 1, 2... |
| `predicted_class` | integer | 9 |
| `predicted_label` | text | `Ankle boot` |
| `confidence` | float | 0.87 |
| `processed_at` | timestamp | `2026-04-20 02:00:00` |

## Verified Evaluation

| Measure | Result |
|---|---:|
| Training images | 60,000 |
| Test images | 10,000 |
| Test accuracy | 88.21% |
| Macro F1 score | 0.8804 |
| Random Forest trees | 50 |

![Fashion-MNIST confusion matrix](results/confusion_matrix.png)

---

## Dataset

`fashion-mnist_train.csv` contains 60,000 rows. Each row is a clothing image flattened to 784 pixel values (28×28 grayscale), plus a `label` column.

| label | class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The dataset does not contain refund data — it is used only to train the image classifier.

---

## Real-world Flow

1. A customer returns an item — staff photograph it
2. The photo is converted to pixel values and saved as a CSV in `batch_process/upload/`
3. Every night, the batch job calls `/predict` for each row
4. Predictions are stored in the `predictions` table in PostgreSQL

The prototype uses CSV files containing serialized 28 × 28 image pixels. A production version would convert warehouse photographs into the same validated representation before prediction or accept image uploads through a dedicated preprocessing endpoint.

## Monitoring and Reliability

Implemented monitoring includes `/health`, `/metrics`, HTTP validation, prediction confidence, a manual-review flag, batch console logs, processed/failed folders, database timestamps, model version, and GitHub Actions tests. Production monitoring should connect these signals to centralized dashboards and alerts and add formal data-drift thresholds.

The 0.60 confidence threshold demonstrates human-in-the-loop routing. Random Forest probabilities are not automatically calibrated, so production use would require probability calibration and a cost-based threshold analysis.

## Architecture

The editable system diagram is available in `docs/architecture-design.drawio`. The implemented design separates image storage, the nightly batch worker, REST API, saved model, PostgreSQL results, and manual review.

---

## How to Run

### Option A — Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python3 model/train.py

# 3. Start the API
uvicorn api.main:app --reload

# 4. Test with a sample (in a second terminal)
python3 get_sample.py | curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @-

# 5. Run the batch manually
python3 batch_process/batch.py
```

### Option B — Docker

```bash
# 1. Train the model (required before building)
python3 model/train.py

# 2. Start all services — API, batch scheduler, and PostgreSQL
docker-compose up --build

# API is available at http://localhost:8000
# Batch scheduler runs nightly at 02:00

# 3. Drop CSV files into the upload folder to be processed
cp model_data/fashion-mnist_test.csv batch_process/upload/
```
