import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
import joblib
from src.context_features import apply_context_features
from src.preprocessing import run_preprocessing

def run_tfidf_pipeline(use_context_features=True):
    print("--- Starting Phase 4: TF-IDF Feature Extraction ---")

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    dataset_path = "data/processed/imdb_cleaned.csv"
    if not os.path.exists(dataset_path):
        run_preprocessing(output_path=dataset_path)

    df = pd.read_csv(dataset_path)
    df["review"] = df["review"].fillna("").astype(str)
    if use_context_features:
        df["review"] = df["review"].map(apply_context_features)

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'],
        df['label'],
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )
    pd.DataFrame({"label": y_train}).to_csv(
        "data/processed/tfidf_train_labels.csv", index=False
    )
    pd.DataFrame({"label": y_test}).to_csv(
        "data/processed/tfidf_test_labels.csv", index=False
    )

    experiments = {
        "unigram": (1, 1),
        "unigram_bigram": (1, 2),
        "unigram_bigram_trigram": (1, 3)
    }

    comparison_data = []

    for name, ngram_range in experiments.items():
        print(f"\nRunning TF-IDF experiment: {name} (ngram_range={ngram_range})...")
        start_time = time.time()

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=50000,
            sublinear_tf=True
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        elapsed_time = time.time() - start_time

        save_npz(f"data/processed/{name}_train.npz", X_train_vec)
        save_npz(f"data/processed/{name}_test.npz", X_test_vec)
        joblib.dump(vectorizer, f"models/{name}_vectorizer.joblib")

        comparison_data.append({
            "Experiment": name,
            "Ngram Range": str(ngram_range),
            "Feature Count": X_train_vec.shape[1],
            "Train Shape": str(X_train_vec.shape),
            "Test Shape": str(X_test_vec.shape),
            "Time Taken (s)": round(elapsed_time, 2)
        })
        print(f"Saved matrices and vectorizer for {name}.")

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("data/processed/experiment_comparison.csv", index=False)
    print("\n--- Phase 4 Completed Successfully ---")

if __name__ == "__main__":
    run_tfidf_pipeline()