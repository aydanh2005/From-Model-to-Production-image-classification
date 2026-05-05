import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

CSV_PATH = "model_data/fashion_products/styles.csv"
IMAGE_DIR = "model_data/fashion_products/images"

MODEL_DIR = "saved_model"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(MODEL_DIR, "product_classifier.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

IMG_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 800

LABEL_COLUMN = "articleType"

SELECTED_CLASSES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Watches",
    "Sports Shoes",
    "Kurtas",
    "Tops",
    "Handbags",
    "Heels",
    "Sunglasses"
]


def prepare_dataframe():
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

    df = df.dropna(subset=["id", LABEL_COLUMN])
    df = df[df[LABEL_COLUMN].isin(SELECTED_CLASSES)]

    df["image_path"] = df["id"].astype(str).apply(
        lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg")
    )

    df = df[df["image_path"].apply(os.path.exists)]

    balanced_parts = []

    for category in SELECTED_CLASSES:
        category_df = df[df[LABEL_COLUMN] == category]
        sample_size = min(MAX_IMAGES_PER_CLASS, len(category_df))

        sampled_df = category_df.sample(
            n=sample_size,
            random_state=42
        )

        balanced_parts.append(sampled_df)

    df = pd.concat(balanced_parts)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Prepared dataset:")
    print(df[LABEL_COLUMN].value_counts())

    return df


def image_to_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return image_array.flatten()


def load_dataset(df, label_to_index):
    features = []
    labels = []

    for _, row in df.iterrows():
        try:
            x = image_to_features(row["image_path"])
            y = label_to_index[row[LABEL_COLUMN]]

            features.append(x)
            labels.append(y)

        except Exception as e:
            print(f"Skipping image {row['image_path']}: {e}")

    return np.array(features), np.array(labels)


def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = prepare_dataframe()

    labels = sorted(df[LABEL_COLUMN].unique().tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    with open(LABELS_PATH, "w") as f:
        json.dump(index_to_label, f, indent=4)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL_COLUMN]
    )

    print("Loading training images...")
    X_train, y_train = load_dataset(train_df, label_to_index)

    print("Loading test images...")
    X_test, y_test = load_dataset(test_df, label_to_index)

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=labels
    )

    print(report)
    print("Test accuracy:", accuracy)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "test_accuracy": float(accuracy),
            "classes": labels,
            "number_of_classes": len(labels),
            "training_images": int(len(X_train)),
            "test_images": int(len(X_test)),
            "image_size": IMG_SIZE,
            "label_column": LABEL_COLUMN,
            "model_type": "RandomForestClassifier on resized RGB image features"
        }, f, indent=4)

    save_confusion_matrix(y_test, y_pred, labels)

    print("Saving model...")
    joblib.dump(model, MODEL_PATH)

    print("Done.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")


if __name__ == "__main__":
    main()