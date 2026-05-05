import os
import pandas as pd

CSV_PATH = "model_data/fashion_products/styles.csv"
IMAGE_DIR = "model_data/fashion_products/images"

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

print("Dataset loaded successfully.")
print("CSV rows:", len(df))
print("Columns:", list(df.columns))

print("\nMaster category counts:")
print(df["masterCategory"].value_counts())

df["image_path"] = df["id"].astype(str).apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg")
)

df["image_exists"] = df["image_path"].apply(os.path.exists)

print("\nImages found:", df["image_exists"].sum())
print("Images missing:", (~df["image_exists"]).sum())

print("\nCategories with existing images:")
print(df[df["image_exists"]]["masterCategory"].value_counts())

print("\nTop 20 subcategories with existing images:")
print(df[df["image_exists"]]["subCategory"].value_counts().head(20))