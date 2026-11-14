"""Train a linear sentiment classifier on a saved TF-IDF matrix."""

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression


def train_model(experiment: str = "unigram_bigram"):
    matrix_path = Path(f"data/processed/{experiment}_train.npz")
    labels_path = Path("data/processed/tfidf_train_labels.csv")
    if not matrix_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Run src/vectorizer.py before training.")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(load_npz(matrix_path), pd.read_csv(labels_path)["label"])
    Path("models").mkdir(exist_ok=True)
    output_path = Path(f"models/{experiment}_classifier.joblib")
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")
    return model


if __name__ == "__main__":
    train_model()