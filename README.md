# Context-Aware Movie Review Sentiment Analyzer

An academic natural language processing (NLP) application comparing conventional baseline TF-IDF classification against rule-based context-aware feature engineering (negation scope and contrastive clause marking) on the IMDb 50,000 Movie Reviews dataset.

---

## 📌 Project Overview

Sentiment analysis often fails on nuanced syntax such as negation (*"not good"*) and contrastive shifts (*"the visuals were great, but the plot was boring"*). Standard bag-of-words pipelines frequently invert polarity by stripping negation words as stopwords and ignoring clause boundaries.

This project investigates whether explicitly marking syntactic context improves sentiment classification across two distinct linear architectures: **Multinomial Naive Bayes** and **Logistic Regression**.

### Key Highlights
- **Dataset**: IMDb 50K Movie Reviews (49,582 clean unique reviews, 80:20 stratified split).
- **Core Technology Stack**: Python, Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Streamlit, Joblib, Pytest.
- **End-to-End Pipeline**: Modular data cleaning, context feature extraction, TF-IDF vectorization, model training, evaluation, and an interactive Streamlit web dashboard.
- **Test Coverage**: 120 automated unit and integration tests passing with 100% success.

---

## 🏛️ System Architecture

```text
movie-review-sentiment-analyzer/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & benchmark analysis
│
├── src/
│   ├── __init__.py            # Package root & public exports
│   ├── preprocessing.py       # Cleaning, contraction expansion, lemmatization, splitting
│   ├── context_features.py    # Negation scope (NOT_) & contrast marking (BEFORE_BUT_/AFTER_BUT_)
│   ├── vectorizer.py          # TF-IDF vectorizers & scikit-learn transformer pipelines
│   ├── train.py               # Model training, serialization, & pipeline persistence
│   ├── evaluate.py            # Comprehensive evaluation metrics & dark-themed plots
│   └── predict.py             # Inference API with in-memory pipeline caching
│
├── models/                    # Serialized .joblib model pipelines (E1–E4)
│   ├── baseline_nb.joblib
│   ├── baseline_logistic.joblib
│   ├── context_nb.joblib
│   └── context_logistic.joblib
│
├── reports/                   # Evaluation artifacts & visualization figures
│   ├── evaluation_results.csv
│   ├── classification_reports.txt
│   ├── confusion_matrix_baseline_nb.png
│   ├── confusion_matrix_baseline_logistic.png
│   ├── confusion_matrix_context_nb.png
│   ├── confusion_matrix_context_logistic.png
│   └── model_comparison.png
│
├── notebooks/                 # Exploratory data analysis & preprocessing notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_text_preprocessing.ipynb
│
└── tests/                     # 120 automated test cases
    ├── test_app.py
    ├── test_preprocessing.py
    ├── test_context_features.py
    ├── test_vectorizer.py
    ├── test_train.py
    ├── test_evaluate.py
    └── test_prediction.py
```

---

## 🔬 The Four Classification Experiments

All models were evaluated on the exact same held-out test partition (**9,917 reviews**, 50.19% positive, 49.81% negative).

| Experiment | Model ID | Feature Engineering | Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **E1** | `baseline_nb` | Baseline Unigram TF-IDF (46,864 terms) | Multinomial Naive Bayes ($\alpha=1.0$) | 86.13% | 87.26% | 84.75% | 85.99% |
| **E2** | `baseline_logistic` | Baseline Unigram TF-IDF (46,864 terms) | Logistic Regression ($C=1.0$, L2) | **89.29%** | **88.48%** | **90.44%** | **89.45%** |
| **E3** | `context_nb` | Context Unigrams + Bigrams (102,568 terms) | Multinomial Naive Bayes ($\alpha=1.0$) | 85.80% | 87.05% | 84.25% | 85.62% |
| **E4** | `context_logistic` | Context Unigrams + Bigrams (102,568 terms) | Logistic Regression ($C=1.0$, L2) | 88.99% | 88.14% | 90.19% | 89.16% |

> **🏆 Best Performing Model**: **Experiment 2 (Baseline Logistic Regression)** achieved the highest test accuracy (**89.29%**) and F1-score (**89.45%**).

---

## 📊 Honest Academic Discussion & Insights

In accordance with rigorous empirical standards, we document that **context-aware models achieved slightly lower overall test accuracy than baseline models** (E2: 89.29% vs. E4: 88.99%; E1: 86.13% vs. E3: 85.80%).

### Why Context Features Did Not Outperform Baselines Overall:
1. **Feature Sparsity & Vocabulary Explosion**:
   Prefixing tokens with `NOT_`, `BEFORE_BUT_`, and `AFTER_BUT_` expanded the vocabulary from **46,864 features to 102,568 features** (+118%). Linear classifiers like Naive Bayes and Logistic Regression rely on direct word frequency counts. Splitting a common token (e.g. `good` into `good`, `NOT_good`, `BEFORE_BUT_good`, `AFTER_BUT_good`) dilutes term frequency and introduces severe parameter sparsity without dense continuous embeddings.
2. **Document-Level Polarity vs. Clause-Level Polarity**:
   IMDb reviews average **231 words** in length. Across multi-paragraph reviews, overall sentiment is largely determined by the cumulative density of sentiment words rather than single isolated negations.
3. **Local Nuance vs. Global Regularization**:
   While context features excel at classifying targeted negation phrases (e.g., *"not good"*, *"never liked"*), applying rule-based prefixes across entire documents adds minor noise to bag-of-words weights.

---

## 🚀 Interactive Streamlit Web Application

The Streamlit web application (`app.py`) provides an interactive interface for model exploration:

### Features
1. **Interactive Sentiment Analyzer**:
   - Single-review classification across all 4 experiments.
   - Quick-load curated test cases (Clearly Positive, Clearly Negative, Negation-Heavy, Contrastive, Very Short).
   - Real-time word and character counters.
   - Informational warning for short reviews (< 5 words).
   - Prominent sentiment outcome cards with estimated posterior confidence.
   - Explicit probability disclaimers (clarifying that confidence is model probability, not guaranteed accuracy).
2. **Session Prediction History & CSV Export**:
   - In-memory tracking of all analyzed reviews during the browser session.
   - Automatic prevention of duplicate entries caused by Streamlit reruns.
   - Tabular history review preview.
   - One-click CSV export via `st.download_button`.
   - Clear history button.
3. **Model Comparison Dashboard**:
   - Dynamic loading of metrics directly from `reports/evaluation_results.csv`.
   - Standout banner highlighting the best-performing model (E2).
   - Grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score on a standard 0–100% scale.
   - 2x2 grid displaying dark-themed confusion matrix heatmaps matching the UI.
4. **Comprehensive Methodology & Academic About Pages**:
   - Detailed documentation of dataset preparation, preprocessing, context engineering, and model architectures.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jaimin044/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Commands Reference

### Model Training
Trains, evaluates, and serializes all 4 models (`E1` through `E4`) into `models/`:
```bash
python -m src.train
```

### Model Evaluation & Report Generation
Generates evaluation tables, classification reports, and dark-themed confusion matrix plots into `reports/`:
```bash
python -m src.evaluate
```

### Launch Streamlit Application
Starts the local interactive Streamlit server:
```bash
streamlit run app.py
```

### Run Test Suite
Executes all 120 automated unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## 🧪 Automated Test Suite Summary

The test suite covers backend modules, context transformers, model persistence, and Streamlit helper functions:

```text
tests/
├── test_app.py                 # 19 tests: validation, history, deduplication, CSV, dashboard data
├── test_context_features.py    #  4 tests: negation scope & contrast marking logic
├── test_evaluate.py            #  6 tests: metric calculation, plotting, pipeline discovery
├── test_prediction.py          # 15 tests: input validation, pipeline caching, batch prediction
├── test_preprocessing.py       # 34 tests: cleaning, contractions, negations, lemmatization, splits
├── test_train.py               # 15 tests: schema validation, training loops, pipeline construction
└── test_vectorizer.py          # 18 tests: TF-IDF transformers, n-grams, serialization

Total: 120 passed in ~4.0s (100% pass rate)
```
