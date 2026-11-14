"""Predict sentiment for a new review."""

from pathlib import Path

import joblib

from src.context_features import apply_context_features


def predict_sentiment(review: str, experiment: str = "unigram_bigram") -> str:
    model_path = Path(f"models/{experiment}_classifier.joblib")
    vectorizer_path = Path(f"models/{experiment}_vectorizer.joblib")
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Run vectorization and training before prediction.")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    features = vectorizer.transform([apply_context_features(review)])
    label = int(model.predict(features)[0])
    return "positive" if label == 1 else "negative"


if __name__ == "__main__":
    print(predict_sentiment(input("Review: ")))