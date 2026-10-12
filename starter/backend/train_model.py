"""Starter P3 — lengkapi training TF-IDF + classifier dan simpan artifact."""

import csv
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "training_samples.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


def main() -> None:
    texts = []
    labels = []

# 1. Baca dataset secara manual per baris agar aman dari masalah pemisah CSV
    with DATA.open(encoding="utf-8-sig") as file:
        lines = file.readlines()
        for line in lines[1:]:  # Lewati header
            line = line.strip()
            if not line:
                continue
            
            # Cek apakah menggunakan pemisah koma atau titik koma
            if ";" in line:
                parts = line.split(";")
            else:
                parts = line.rsplit(",", 1)
                
            if len(parts) >= 2:
                text = parts[0].strip().strip('"')
                label = parts[1].strip().strip('"')
                if text and label:
                    texts.append(text)
                    labels.append(label)

    # 2. Bentuk pipeline TF-IDF dan Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    # 3. Latih model (fitting) menggunakan data yang dibaca
    pipeline.fit(texts, labels)

    # 4. Buat direktori models jika belum ada
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Serialisasi dan simpan model ke file .pkl
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # 6. Simpan metadata (jumlah contoh data) ke metadata.json
    metadata = {
        "jumlah_contoh": len(texts)
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # 7. Cetak pesan sukses sesuai dengan ekspektasi modul
    print("Model demonstrasi berhasil dibuat.")
    print(f"Jumlah contoh: {len(texts)}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()