# Movie Review Sentiment Analyzer

An end-to-end NLP project for movie review sentiment analysis with context-aware features and machine learning models.

## Project Structure

```
movie-review-sentiment-analyzer/
│
├── data/
│   ├── raw/
│   │   └── IMDB_Dataset.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
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

---

## NLP Data Preprocessing Architecture & Decisions

### 1. The Sentiment Analysis Preprocessing Dilemma
Standard, off-the-shelf NLP preprocessing pipelines typically apply aggressive transformations:
- Stripping all punctuation
- Stripping all stopwords using generic stopword lists (e.g., standard NLTK stopwords)
- Stripping or naive splitting of apostrophes (`didn't` -> `didnt` or `did` + `nt`)

**Why this ruins sentiment analysis:**
1. **Sentiment Inversion via Naive Stopword Removal**:
   Standard NLTK English stopwords list contains `not`, `no`, `nor`, `neither`, `wasn't`, `didn't`, `wouldn't`, etc.
   If applied naively:
   > *"The movie was not good and I did not enjoy it."*  
   > **Naive Stopword Output:** `['movie', 'good', 'enjoy']`  
   > The negative sentiment is flipped to positive, directly damaging model accuracy.
2. **Clause Boundary Loss via Punctuation Removal**:
   Context-aware feature engineering requires knowing where a negation clause starts and where it ends. For instance, in *"It was not good, but I loved the music!"*, the negation `not` applies only up to the comma/conjunction. If all punctuation is erased, negation scope leaks across clauses.
3. **Contraction Degradation**:
   Contractions like `wasn't`, `couldn't`, `won't` hide negation words. Expanding them to `was not`, `could not`, `will not` reveals the canonical negation word `not` so subsequent modules can process them consistently.

---

### 2. Context-Aware Preprocessing Decisions

| Preprocessing Step | Traditional Approach | Our Context-Aware Approach | Motivation & Impact |
| :--- | :--- | :--- | :--- |
| **HTML Cleaning** | Strip tags or ignore | Regex replacement with a single space (`' '`) | Replaces `<br />`, `<div>`, etc., with spaces to prevent words on either side from merging together. |
| **Contraction Expansion** | Strip apostrophes (`didnt`) | Dictionary + regex expansion (`didn't` -> `did not`) | Explicitly extracts canonical negation tokens (`not`) needed for negation scope analysis. |
| **Punctuation** | Strip all punctuation | Retain sentence & clause delimiters (`.`, `!`, `?`, `,`, `;`, `:`) as discrete tokens | Preserves clause boundaries for negation scoping and emotional intensity cues (`!`, `?`). |
| **Stopword Handling** | Remove all 198 NLTK stopwords | Remove stopwords while strictly protecting `NEGATION_WORDS` (`not`, `no`, `never`, `neither`, `nor`, `hardly`, etc.) | Prevents sentiment inversion and retains negative polarity markers. |
| **Lemmatization** | Simple noun-only lemmatizer | WordNet lemmatizer with Penn Treebank POS tag mapping | Properly lemmatizes verbs (`running` -> `run`, `was` -> `be`), adjectives (`better` -> `good`), and nouns (`movies` -> `movie`). |

---

### 3. Collaboration Contract & API Reference

All preprocessing routines are exposed via `src/preprocessing.py` and `src/__init__.py`. Downstream team members can import them directly:

```python
from src.preprocessing import (
    clean_text,
    tokenize_text,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    expand_contractions,
    split_data,
)
```

#### Shared Functions

- **`clean_text(text: str, lowercase: bool = True, expand_contractions_flag: bool = True, preserve_punctuation: bool = True) -> str`**
  - **Input**: Raw text string.
  - **Output**: Cleaned string with HTML stripped, contractions expanded, and punctuation normalized.
- **`tokenize_text(text: str, preserve_punctuation: bool = True) -> list[str]`**
  - **Input**: Cleaned or raw text string.
  - **Output**: List of string tokens. Punctuation marks are kept as standalone tokens by default.
- **`remove_stopwords(tokens: list[str], preserve_negations: bool = True, preserve_punctuation: bool = True) -> list[str]`**
  - **Input**: List of string tokens.
  - **Output**: List of tokens with standard stopwords removed, but negation words (`not`, `no`, `never`, etc.) strictly preserved.
- **`lemmatize_tokens(tokens: list[str], use_pos: bool = True) -> list[str]`**
  - **Input**: List of string tokens.
  - **Output**: List of lemmatized tokens with punctuation tokens untouched.
- **`preprocess_text(text: str, return_tokens: bool = False) -> str | list[str]`**
  - **Input**: Raw text string.
  - **Output**: Fully preprocessed string (default, ideal for TF-IDF / vectorizers) or list of tokens (when `return_tokens=True`).
- **`split_data(df: pd.DataFrame, target_col: str = "sentiment", test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]`**
  - **Input**: DataFrame containing `review` and `sentiment` columns.
  - **Output**: Tuple of `(train_df, test_df)` with identical stratified class balance.

---

### 4. Shared Dataset Splits

The raw IMDb 50,000 dataset was deduplicated (418 exact duplicate reviews removed) leaving **49,582 unique reviews**.
A stratified 80:20 train-test split was executed using `random_state=42` and saved to `data/processed/`:

- **Train Split**: [`data/processed/train.csv`](file:///Users/jaiminkoriya/Documents/movie-review-sentiment-analyzer/data/processed/train.csv)
  - **Rows**: 39,665
  - **Positive**: 19,907 (50.19%)
  - **Negative**: 19,758 (49.81%)
- **Test Split**: [`data/processed/test.csv`](file:///Users/jaiminkoriya/Documents/movie-review-sentiment-analyzer/data/processed/test.csv)
  - **Rows**: 9,917
  - **Positive**: 4,977 (50.19%)
  - **Negative**: 4,940 (49.81%)

All subsequent modules (feature engineering, vectorization, model training, evaluation) must use these exact splits to prevent data leakage and guarantee comparability.

---

## Setup & Running Unit Tests

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run preprocessing unit tests:
   ```bash
   python -m pytest tests/test_preprocessing.py -v
   ```

3. Launch Jupyter to view exploration and preprocessing notebooks:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   jupyter notebook notebooks/02_text_preprocessing.ipynb
   ```
