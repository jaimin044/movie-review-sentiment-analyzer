# Development Handoff: Movie Review Sentiment Analyzer

## Project Overview

Repository: `https://github.com/jaimin044/movie-review-sentiment-analyzer`

Project: Context-Aware Movie Review Sentiment Analyzer

Dataset: IMDb 50K Movie Reviews

Technology stack:
- Python
- Scikit-learn
- NLTK
- Pandas and NumPy
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Pytest

The core machine-learning implementation is complete. The remaining task is to improve and finalize the Streamlit frontend, dashboard, user experience, testing, and documentation.

## 1. Existing Machine Learning Implementation

The project contains four trained sentiment classification experiments:

| Experiment | Feature Extraction | Classifier |
|---|---|---|
| E1 | Baseline TF-IDF | MultinomialNB |
| E2 | Baseline TF-IDF | Logistic Regression |
| E3 | Context-Aware TF-IDF | MultinomialNB |
| E4 | Context-Aware TF-IDF | Logistic Regression |

Context-aware feature engineering includes:
- Negation scope detection using NOT_ prefixes.
- Contrastive clause marking using BEFORE_BUT_ and AFTER_BUT_ prefixes.

Existing backend modules:

- `src/preprocessing.py`
- `src/context_features.py`
- `src/vectorizer.py`
- `src/train.py`
- `src/evaluate.py`
- `src/predict.py`

The existing prediction API is:

```python
from src.predict import predict_sentiment, list_available_models

result = predict_sentiment(
    review="The movie was excellent.",
    model_name="baseline_logistic",
    models_dir="models"
)
```

The returned dictionary contains:
- `sentiment`
- `confidence`
- `model`

Available model IDs:
- `baseline_nb`
- `baseline_logistic`
- `context_nb`
- `context_logistic`

Use the existing prediction API. Do not implement another preprocessing or prediction pipeline.

## 2. Current Experimental Results

The models were evaluated on the same 9,917-review test dataset.

| Experiment | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| E1 | 86.13% | 87.26% | 84.75% | 85.99% |
| E2 | 89.29% | 88.48% | 90.44% | 89.45% |
| E3 | 85.80% | 87.05% | 84.25% | 85.62% |
| E4 | 88.99% | 88.14% | 90.19% | 89.16% |

E2 currently has the highest reported accuracy: 89.29%.

These values must be loaded from the generated evaluation results rather than manually hardcoded in the dashboard.

## 3. Existing Project Structure

Important files:

```text
movie-review-sentiment-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── context_features.py
│   ├── vectorizer.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── baseline_nb.joblib
│   ├── baseline_logistic.joblib
│   ├── context_nb.joblib
│   └── context_logistic.joblib
│
├── reports/
│   ├── evaluation_results.csv
│   ├── classification_reports.txt
│   ├── confusion_matrix_baseline_nb.png
│   ├── confusion_matrix_baseline_logistic.png
│   ├── confusion_matrix_context_nb.png
│   ├── confusion_matrix_context_logistic.png
│   └── model_comparison.png
│
└── tests/
```

Model artifacts, datasets, and generated reports may be excluded by `.gitignore`. Do not change these exclusions without a clear reason.

## 4. Streamlit Frontend Improvements

The existing `app.py` already supports:

- Sentiment prediction.
- Selection between all four models.
- Confidence display.
- Model availability indicators.
- Model comparison dashboard.
- Evaluation metrics.
- Accuracy charts.
- Confusion matrix display.
- Methodology page.
- About Project page.

Improve the existing implementation instead of unnecessarily rewriting the entire application.

### A. Improve the Model Comparison Dashboard

1. Display all four experiments with their full descriptive names.
2. Display accuracy, precision, recall, and F1-score.
3. Replace the stacked detailed metric visualization with a grouped bar chart.
4. Use a consistent percentage scale from 0 to 100.
5. Clearly identify the best-performing model.
6. Show the model ID only as secondary information, not as the main model name.
7. Load metrics dynamically from `reports/evaluation_results.csv`.
8. Handle missing or incomplete evaluation reports gracefully.

### B. Improve Confusion Matrices

The current confusion matrix images have white backgrounds and some titles are clipped.

Improve the plotting implementation in `src/evaluate.py`.

Requirements:

- Use a consistent dark theme matching the Streamlit application.
- Use readable annotation colors and font sizes.
- Increase figure dimensions appropriately.
- Prevent title and axis-label clipping.
- Maintain the existing negative/positive label ordering.
- Keep all numerical confusion matrix values unchanged.
- Regenerate all four confusion matrix images.
- Verify that the dashboard displays all four images correctly.

### C. Implement Prediction History

Add a prediction history section to the Sentiment Analyzer.

For every successful prediction, store:

- Timestamp
- Original review
- Model ID
- Full model name
- Predicted sentiment
- Confidence

Requirements:

- Use `st.session_state` so history persists during the current Streamlit session.
- Avoid adding duplicate entries caused by unrelated Streamlit reruns.
- Show history in a readable table.
- Provide a clear-history button.
- Handle an empty history gracefully.

### D. CSV Export

Add a download button to export prediction history as CSV.

Requirements:

- Include all history fields.
- Use pandas for CSV generation.
- Use Streamlit's `st.download_button`.
- Handle empty history.
- Do not write private user reviews to persistent files automatically.

### E. Input Quality and Validation

Improve the review input experience.

Requirements:

- Reject empty or whitespace-only input.
- Preserve the existing backend validation.
- Display a word count.
- Display a warning for very short reviews because they may provide insufficient context.
- Make clear that short-review warnings are informational, not guaranteed measures of prediction reliability.
- Do not artificially modify user review text.
- Keep the existing maximum input length or document any change.

### F. Improve Prediction Results

Display:

- Predicted sentiment.
- Selected model name.
- Confidence percentage.
- Visual positive/negative indicators.
- An explanation that confidence is the model's estimated probability for the predicted class and may not be calibrated.

Do not describe confidence as guaranteed correctness.

### G. Improve Overall UI

Create a polished, consistent Streamlit interface.

Requirements:

- Consistent dark visual theme.
- Clear visual hierarchy.
- Professional typography.
- Consistent spacing and alignment.
- Responsive layout.
- Readable charts and tables.
- Consistent colors for positive and negative predictions.
- Clear navigation.
- Avoid unnecessary animations or excessive visual effects.

Keep the application suitable for an academic project demonstration.

### H. Improve Methodology and About Pages

Provide clear explanations of:

- IMDb 50K dataset.
- Dataset cleaning and stratified splitting.
- Text preprocessing.
- Stopword handling and negation preservation.
- POS-guided lemmatization.
- Negation scope detection.
- Contrastive clause marking.
- TF-IDF feature extraction.
- Multinomial Naive Bayes.
- Logistic Regression.
- The four experiments and their differences.
- Evaluation metrics.

Explain honestly that context-aware models achieved slightly lower overall accuracy in the current experiments.

Do not claim context-aware features improved overall performance.

## 5. Testing Requirements

Run the existing test suite before and after implementation.

Current reported baseline: 101 passing tests.

Add tests where practical for:

- Prediction history insertion.
- Duplicate prevention.
- History clearing.
- CSV export data.
- Empty input handling.
- Short-review warnings.
- Missing model artifacts.
- Missing evaluation CSV.
- Missing confusion matrix images.
- Dashboard data preparation.

Do not delete or weaken existing tests.

Run:

```bash
python -m pytest tests/ -v
```

Also manually verify the Streamlit application with all four models.

Test these review categories:

1. Clearly positive review.
2. Clearly negative review.
3. Negation-heavy review.
4. Contrastive review.
5. Very short review.
6. Empty review.

## 6. Backend and Git Safety Requirements

IMPORTANT:

Do not modify the following files unless a genuine bug is found and explained:

- `src/preprocessing.py`
- `src/context_features.py`
- `src/vectorizer.py`
- `src/train.py`
- `src/predict.py`

The training and prediction pipelines have already been integrated and tested.

Changes to `src/evaluate.py` should be limited to visualization improvements unless additional changes are necessary.

Do not:

- Retrain models unnecessarily.
- Change training configurations.
- Change the train/test split.
- Change random_state=42.
- Change preprocessing behavior.
- Modify the trained model artifacts.
- Commit datasets or model artifacts.
- Push directly to main.
- Remove existing project functionality.

Work on a dedicated frontend branch:

`feature/streamlit-app`

If that branch already exists remotely, fetch and inspect it before making changes. Do not overwrite another developer's work.

## 7. Documentation

Update README.md with:

- Project overview.
- Architecture.
- Four experiments.
- Installation instructions.
- Virtual environment setup.
- Model training command.
- Evaluation command.
- Streamlit launch command.
- Test execution command.
- Application screenshots, if available.

Application launch:

```bash
streamlit run app.py
```

## 8. Final Verification

Before declaring the work complete:

1. Run all existing tests.
2. Verify all four model artifacts are detected.
3. Verify predictions work for all four models.
4. Verify the comparison dashboard.
5. Verify all confusion matrices.
6. Verify prediction history.
7. Verify CSV download.
8. Verify empty input handling.
9. Verify short-review warnings.
10. Verify the README instructions.

Run:

```bash
python -m pytest tests/ -v
streamlit run app.py
```

## 9. Final Deliverables

Provide:

- List of modified files.
- Explanation of changes.
- Screenshots of the finished application.
- Test results.
- Confirmation that all four models work.
- Confirmation that the dashboard metrics match evaluation_results.csv.
- Git commit hash.
- Git branch name.
- Confirmation that changes were pushed to the feature branch.

Do not claim a feature is complete unless it has actually been implemented and tested.

The objective is to deliver a polished, fully functional academic NLP application while preserving the already working machine-learning backend.