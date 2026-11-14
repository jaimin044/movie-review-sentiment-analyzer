"""Evaluate a saved sentiment classifier."""

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(experiment: str = "unigram_bigram") -> dict:
    model_path = Path(f"models/{experiment}_classifier.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Train a model before evaluation.")
    model = joblib.load(model_path)
    predictions = model.predict(load_npz(f"data/processed/{experiment}_test.npz"))
    labels = pd.read_csv("data/processed/tfidf_test_labels.csv")["label"]
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "report": classification_report(labels, predictions, output_dict=True),
    }
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()