# Movie Review Sentiment Analyzer

An end-to-end NLP project for movie review sentiment analysis with context-aware features and machine learning models.

## Project Structure

```
movie-review-sentiment-analyzer/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_context_aware_features.ipynb
│   ├── 04_tfidf_experiments.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── context_features.py
│   ├── vectorizer.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_context_features.py
│   └── test_prediction.py
│
├── reports/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
