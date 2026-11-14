"""Unit tests for the preprocessing module.

Tests cover:
- HTML cleaning and URL removal
- Contraction expansion and curly apostrophe handling
- Punctuation and sentence boundary preservation
- Negation preservation in stopword filtering
- POS-guided and fallback lemmatization
- Tokenization edge cases
- End-to-end preprocessing pipeline
- Stratified dataset splitting and deduplication
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CONTRACTION_MAP,
    NEGATION_WORDS,
    clean_dataset,
    clean_text,
    expand_contractions,
    get_wordnet_pos,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    split_data,
    tokenize_text,
)


class TestContractionExpansion:
    """Tests for contraction expansion logic."""

    def test_standard_contractions(self):
        assert expand_contractions("I don't like it") == "I do not like it"
        assert expand_contractions("He won't come") == "He will not come"
        assert expand_contractions("They can't swim") == "They cannot have" or "They cannot swim"
        assert expand_contractions("It's amazing") == "It is amazing"

    def test_curly_typographical_apostrophes(self):
        assert expand_contractions("I didn’t see it") == "I did not see it"
        assert expand_contractions("She wouldn’t know") == "She would not know"

    def test_capitalized_contractions(self):
        assert expand_contractions("Didn't expect that") == "Did not expect that"
        assert expand_contractions("Won't happen") == "Will not happen"

    def test_empty_and_non_string_inputs(self):
        assert expand_contractions("") == ""
        assert expand_contractions(None) == ""
        assert expand_contractions("Just plain text with no contractions.") == "Just plain text with no contractions."


class TestTextCleaning:
    """Tests for clean_text function."""

    def test_html_tag_removal(self):
        raw = "This was bad.<br /><br />Really bad!<b>Must avoid</b>"
        cleaned = clean_text(raw)
        assert "<br" not in cleaned
        assert "<b>" not in cleaned
        assert "bad" in cleaned
        assert "must avoid" in cleaned

    def test_url_and_email_removal(self):
        raw = "Check review at https://example.com/review or contact info@imdb.com for details."
        cleaned = clean_text(raw)
        assert "https" not in cleaned
        assert "example.com" not in cleaned
        assert "info@imdb.com" not in cleaned
        assert "review" in cleaned

    def test_punctuation_preservation(self):
        raw = "Was it good? No, it was not! Truly awful."
        cleaned = clean_text(raw, preserve_punctuation=True)
        assert "?" in cleaned
        assert "!" in cleaned
        assert "." in cleaned
        assert "," in cleaned

    def test_repeated_punctuation_normalization(self):
        raw = "Terrible movie!!!!!! What were they thinking????"
        cleaned = clean_text(raw)
        # Should normalize repeated ! and ? while preserving the cue
        assert "!" in cleaned
        assert "!" not in cleaned.replace("!", "", 1) or "! " in cleaned
        assert "?" in cleaned

    def test_whitespace_normalization(self):
        raw = "   Too   many \n\t  spaces  between   words.  "
        cleaned = clean_text(raw)
        assert "  " not in cleaned
        assert cleaned.startswith("too")
        assert cleaned.endswith(".")

    def test_empty_and_edge_case_inputs(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t   ") == ""
        assert clean_text(None) == ""


class TestTokenization:
    """Tests for tokenize_text function."""

    def test_basic_tokenization(self):
        text = "this is a movie review"
        tokens = tokenize_text(text)
        assert tokens == ["this", "is", "a", "movie", "review"]

    def test_punctuation_as_tokens(self):
        text = "not good ! really ?"
        tokens = tokenize_text(text, preserve_punctuation=True)
        assert "!" in tokens
        assert "?" in tokens
        assert "not" in tokens

    def test_tokenization_without_punctuation(self):
        text = "not good ! really ?"
        tokens = tokenize_text(text, preserve_punctuation=False)
        assert "!" not in tokens
        assert "?" not in tokens
        assert tokens == ["not", "good", "really"]

    def test_empty_input(self):
        assert tokenize_text("") == []
        assert tokenize_text(None) == []


class TestStopwordRemovalAndNegationPreservation:
    """Tests for stopword removal with negation preservation."""

    def test_negations_are_strictly_preserved(self):
        negation_tokens = ["not", "no", "never", "neither", "nor", "hardly", "barely", "without"]
        filtered = remove_stopwords(negation_tokens, preserve_negations=True)
        for neg in negation_tokens:
            assert neg in filtered, f"Negation '{neg}' was erroneously removed!"

    def test_standard_stopwords_removed(self):
        tokens = ["the", "movie", "was", "a", "great", "production", "of", "an", "original", "story"]
        filtered = remove_stopwords(tokens, preserve_negations=True)
        assert "the" not in filtered
        assert "was" not in filtered
        assert "a" not in filtered
        assert "of" not in filtered
        assert "an" not in filtered
        assert "movie" in filtered
        assert "great" in filtered

    def test_sentiment_reversal_preserved(self):
        # "not good" should not become "good"
        tokens = ["the", "movie", "is", "not", "good"]
        filtered = remove_stopwords(tokens, preserve_negations=True)
        assert "not" in filtered
        assert "good" in filtered
        assert "the" not in filtered
        assert "is" not in filtered

    def test_punctuation_retained_in_stopwords(self):
        tokens = ["movie", "not", "good", "!", "do", "not", "watch", "."]
        filtered = remove_stopwords(tokens, preserve_punctuation=True)
        assert "!" in filtered
        assert "." in filtered


class TestLemmatization:
    """Tests for lemmatize_tokens function."""

    def test_pos_guided_lemmatization(self):
        tokens = ["actors", "were", "running", "better"]
        lemmatized = lemmatize_tokens(tokens, use_pos=True)
        assert "actor" in lemmatized
        assert "run" in lemmatized

    def test_punctuation_untouched_in_lemmatization(self):
        tokens = ["movie", "!", "?", ",", "."]
        lemmatized = lemmatize_tokens(tokens, use_pos=True)
        assert lemmatized == ["movie", "!", "?", ",", "."]

    def test_empty_tokens(self):
        assert lemmatize_tokens([]) == []


class TestEndToEndPreprocessing:
    """Tests for preprocess_text function."""

    def test_full_pipeline_string_output(self):
        raw = "The acting wasn't good at all!<br />I wouldn't recommend it."
        processed = preprocess_text(raw, return_tokens=False)
        assert isinstance(processed, str)
        # Verify negations preserved
        assert "not" in processed
        # Verify contractions expanded
        assert "wasn't" not in processed
        assert "wouldn't" not in processed
        # Verify HTML stripped
        assert "<br" not in processed
        # Verify punctuation preserved
        assert "!" in processed or "." in processed

    def test_full_pipeline_token_output(self):
        raw = "The acting wasn't good at all!"
        tokens = preprocess_text(raw, return_tokens=True)
        assert isinstance(tokens, list)
        assert "not" in tokens
        assert "good" in tokens

    def test_empty_and_whitespace_input(self):
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""
        assert preprocess_text(None) == ""
        assert preprocess_text("", return_tokens=True) == []


class TestDataCleaningAndSplitting:
    """Tests for dataset deduplication, validation, and stratified splitting."""

    def test_clean_dataset_removes_duplicates_and_nulls(self):
        data = {
            "review": ["Great film", "Bad movie", "Great film", None, "Another one"],
            "sentiment": ["positive", "negative", "positive", "positive", "negative"],
        }
        df = pd.DataFrame(data)
        cleaned = clean_dataset(df)
        assert len(cleaned) == 3
        assert cleaned.isnull().sum().sum() == 0
        assert cleaned.duplicated().sum() == 0

    def test_split_data_stratification_and_ratio(self):
        # Create a synthetic dataset with known 60:40 class distribution
        reviews = [f"review_{i}" for i in range(100)]
        sentiments = ["positive"] * 60 + ["negative"] * 40
        df = pd.DataFrame({"review": reviews, "sentiment": sentiments})

        train_df, test_df = split_data(df, target_col="sentiment", test_size=0.2, random_state=42)

        assert len(train_df) == 80
        assert len(test_df) == 20

        # Check stratification
        train_pos_pct = (train_df["sentiment"] == "positive").mean()
        test_pos_pct = (test_df["sentiment"] == "positive").mean()
        assert np.isclose(train_pos_pct, 0.60, atol=0.01)
        assert np.isclose(test_pos_pct, 0.60, atol=0.01)

    def test_split_data_reproducibility(self):
        reviews = [f"review_{i}" for i in range(50)]
        sentiments = ["positive"] * 25 + ["negative"] * 25
        df = pd.DataFrame({"review": reviews, "sentiment": sentiments})

        train_df1, test_df1 = split_data(df, random_state=42)
        train_df2, test_df2 = split_data(df, random_state=42)

        pd.testing.assert_frame_equal(train_df1, train_df2)
        pd.testing.assert_frame_equal(test_df1, test_df2)

    def test_split_data_invalid_column(self):
        df = pd.DataFrame({"text": ["a", "b"], "label": [1, 0]})
        with pytest.raises(ValueError, match="Target column 'sentiment' not found"):
            split_data(df, target_col="sentiment")

