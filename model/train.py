import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

train_file = (
    "model_data/fashion-mnist_train.csv"
    if os.path.exists("model_data/fashion-mnist_train.csv")
    else "model_data/fashion-mnist_train_sample.csv"
)
X_train = pd.read_csv(train_file)
y_train = X_train.pop("label").values
X_train = X_train.values / 255.0

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

if os.path.exists("model_data/fashion-mnist_test.csv"):
    X_test = pd.read_csv("model_data/fashion-mnist_test.csv")
    y_test = X_test.pop("label").values
    predictions = model.predict(X_test.values / 255.0)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as output:
        json.dump({
            "test_accuracy": float(accuracy),
            "macro_f1_score": float(report["macro avg"]["f1-score"]),
            "weighted_f1_score": float(report["weighted avg"]["f1-score"]),
            "training_images": int(len(X_train)),
            "test_images": int(len(X_test)),
            "classes": 10,
            "model": "RandomForestClassifier",
            "n_estimators": 50,
            "random_state": 42
        }, output, indent=2)
    pd.DataFrame(report).transpose().to_csv("results/classification_report.csv")
    matrix = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(matrix).plot(cmap="Blues", colorbar=False)
    plt.title("Fashion-MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=180)
    plt.close()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
else:
    print("Test CSV not found, skipping evaluation.")

os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/model.pkl")
