"""Preprocessing module for Context-Aware Movie Review Sentiment Analyzer.

This module provides data cleaning, text normalization, tokenization,
negation-aware stopword removal, lemmatization, and stratified dataset splitting.
All functions are modular and designed to be imported by downstream feature
engineering and modeling pipelines.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Set, Tuple, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# NLTK resources are optional; the module has deterministic fallbacks for
# offline environments and does not download data during import.

# Comprehensive dictionary of English contractions
CONTRACTION_MAP: dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

# Negation words preserved for context-aware sentiment analysis
NEGATION_WORDS: set[str] = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nobody",
    "nowhere",
    "nothing",
    "hardly",
    "scarcely",
    "barely",
    "without",
    "cannot",
    "n't",
}

# Contrastive conjunctions (optional preservation for discourse shift)
CONTRAST_WORDS: set[str] = {
    "but",
    "however",
    "although",
    "though",
    "yet",
    "despite",
    "in spite of",
    "nevertheless",
}

# Sentence boundary and clause punctuation marks
PUNCTUATION_MARKS: set[str] = {".", "!", "?", ",", ";", ":", "-", "..."}

# Compiled regex patterns for efficiency
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEAT_EXCLAMATION_RE = re.compile(r"!{2,}")
_REPEAT_QUESTION_RE = re.compile(r"\?{2,}")
_REPEAT_PERIOD_RE = re.compile(r"\.{4,}")

# Contraction regex pattern
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
    flags=re.IGNORECASE,
)
_FALLBACK_NT_RE = re.compile(r"(\w+)n't\b", flags=re.IGNORECASE)

_LEMMATIZER = WordNetLemmatizer()
try:
    _BASE_STOPWORDS: set[str] = set(stopwords.words("english"))
except LookupError:
    _BASE_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
        "from", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "was", "were", "with", "do", "does", "did", "i", "he", "she",
        "they", "you",
    }


def expand_contractions(text: str) -> str:
    """Expands standard English contractions in text.

    Replaces typographical quotes ('’', '‘') with standard ASCII single quotes,
    then expands contractions (e.g., "didn't" -> "did not", "won't" -> "will not").
    Preserves case (uppercase or capitalized) when expanding.

    Args:
        text (str): Input text containing potential contractions.

    Returns:
        str: Text with contractions expanded.

    Example:
        >>> expand_contractions("I didn't like it; won't watch again.")
        'I did not like it; will not watch again.'
    """
    if not isinstance(text, str) or not text:
        return ""

    # Normalize typographical apostrophes
    normalized = text.replace("’", "'").replace("‘", "'")

    def _replace_match(match: re.Match[str]) -> str:
        matched_str = match.group(0)
        lower_str = matched_str.lower()
        expanded = CONTRACTION_MAP.get(lower_str, matched_str)
        if matched_str.isupper():
            return expanded.upper()
        if matched_str[0].isupper():
            return expanded.capitalize()
        return expanded

    expanded = _CONTRACTION_RE.sub(_replace_match, normalized)
    # Handle remaining arbitrary words ending with n't (e.g., mightn't)
    expanded = _FALLBACK_NT_RE.sub(r"\1 not", expanded)
    return expanded


def clean_text(
    text: str,
    lowercase: bool = True,
    expand_contractions_flag: bool = True,
    preserve_punctuation: bool = True,
) -> str:
    """Cleans and normalizes raw text review data.

    Steps performed:
    1. Validates and converts input to string (handles None/empty strings).
    2. Strips HTML markup (e.g., '<br />', '<b>'), replacing with a single space
       to prevent accidental word concatenations.
    3. Removes URLs and email addresses.
    4. Expands English contractions to reveal negation tokens ('did not', 'will not').
    5. Normalizes whitespace and excessive repeated punctuation ('!!!!' -> '!', '????' -> '?').
    6. Preserves sentence boundaries and punctuation needed for context features.
    7. Optionally lowercases text.

    Args:
        text (str): The raw review text.
        lowercase (bool, optional): Whether to convert text to lowercase. Defaults to True.
        expand_contractions_flag (bool, optional): Whether to expand contractions. Defaults to True.
        preserve_punctuation (bool, optional): Whether to preserve clause/sentence punctuation
            (.,!?;:) needed for context-aware feature engineering. Defaults to True.

    Returns:
        str: Cleaned and normalized text string.

    Example:
        >>> clean_text("Great movie!<br /><br />Didn't regret it.")
        'great movie! did not regret it.'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text

    # Remove HTML tags (replace with space to avoid word merging)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)

    # Remove URLs and email addresses
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EMAIL_RE.sub(" ", cleaned)

    # Expand contractions before lowercasing or punctuation stripping
    if expand_contractions_flag:
        cleaned = expand_contractions(cleaned)

    # Normalize repeated punctuation intensity while retaining sentiment cues
    cleaned = _REPEAT_EXCLAMATION_RE.sub(" !", cleaned)
    cleaned = _REPEAT_QUESTION_RE.sub(" ?", cleaned)
    cleaned = _REPEAT_PERIOD_RE.sub(" ... ", cleaned)

    if preserve_punctuation:
        # Keep alphanumeric, whitespace, and key punctuation characters
        cleaned = re.sub(r"[^\w\s.,!?;:\-']", " ", cleaned)
        # Ensure punctuation marks have spacing so tokenizers cleanly isolate them
        cleaned = re.sub(r"([.,!?;:])", r" \1 ", cleaned)
    else:
        # Remove all punctuation, keeping only letters, digits, and spaces
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    if lowercase:
        cleaned = cleaned.lower()

    # Collapse multiple whitespaces
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def tokenize_text(
    text: str,
    preserve_punctuation: bool = True,
) -> list[str]:
    """Tokenizes text into a list of word and punctuation tokens.

    Preserves sentence boundaries and punctuation tokens (., !, ?, etc.)
    which are required by the context-aware feature engineering module to
    delimit clauses and negation scopes.

    Args:
        text (str): Cleaned or raw input text.
        preserve_punctuation (bool, optional): Whether to retain punctuation tokens.
            Defaults to True.

    Returns:
        list[str]: List of string tokens.

    Example:
        >>> tokenize_text("not good !")
        ['not', 'good', '!']
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"\w+|[^\w\s]", text)

    if not preserve_punctuation:
        tokens = [t for t in tokens if t.isalnum()]

    return tokens


def get_wordnet_pos(treebank_tag: str) -> Optional[str]:
    """Maps Penn Treebank POS tags to WordNet POS constants.

    Args:
        treebank_tag (str): Penn Treebank POS tag (e.g., 'JJ', 'VBD', 'NN').

    Returns:
        Optional[str]: WordNet POS constant ('a', 'v', 'n', 'r') or None.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def lemmatize_tokens(
    tokens: list[str],
    use_pos: bool = True,
) -> list[str]:
    """Lemmatizes tokens to their root dictionary form using WordNet.

    If use_pos is True, applies part-of-speech tagging to correctly lemmatize
    verbs ('running' -> 'run'), adjectives ('better' -> 'good'), and nouns
    ('movies' -> 'movie'). Punctuation marks and non-alphanumeric tokens are
    preserved untouched.

    Args:
        tokens (list[str]): List of word and punctuation tokens.
        use_pos (bool, optional): Whether to use POS tags for accurate lemmatization.
            Defaults to True.

    Returns:
        list[str]: List of lemmatized tokens.

    Example:
        >>> lemmatize_tokens(['actors', 'were', 'running', '!'], use_pos=True)
        ['actor', 'be', 'run', '!']
    """
    if not tokens:
        return []

    if not use_pos:
        return [_LEMMATIZER.lemmatize(t) if t.isalnum() else t for t in tokens]

    try:
        pos_tagged = nltk.pos_tag(tokens)
    except LookupError:
        return [
            token if not token.isalnum() else _fallback_lemma(token)
            for token in tokens
        ]
    lemmatized: list[str] = []

    for word, tag in pos_tagged:
        # Preserve punctuation tokens as-is
        if not word.isalnum():
            lemmatized.append(word)
            continue

        wn_tag = get_wordnet_pos(tag)
        if wn_tag:
            lemma = _LEMMATIZER.lemmatize(word, wn_tag)
        else:
            lemma = _LEMMATIZER.lemmatize(word)
        lemmatized.append(lemma)

    return lemmatized


def _fallback_lemma(word: str) -> str:
    """Apply minimal morphology when WordNet is unavailable."""

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def remove_stopwords(
    tokens: list[str],
    preserve_negations: bool = True,
    preserve_contrast: bool = False,
    preserve_punctuation: bool = True,
    custom_exclude_stopwords: Optional[Set[str]] = None,
) -> list[str]:
    """Removes standard English stopwords while protecting critical context tokens.

    Critical context tokens include negation words ('not', 'no', 'never', 'neither',
    'nor', 'hardly', etc.) and punctuation boundaries (., !, ?, etc.).
    Preserving these tokens prevents sentiment inversion and enables downstream
    negation scope modeling.

    Args:
        tokens (list[str]): List of tokens to filter.
        preserve_negations (bool, optional): Whether to keep negation words. Defaults to True.
        preserve_contrast (bool, optional): Whether to keep contrast words ('but', 'however').
            Defaults to False.
        preserve_punctuation (bool, optional): Whether to keep punctuation tokens. Defaults to True.
        custom_exclude_stopwords (set[str], optional): Additional words to protect from removal.

    Returns:
        list[str]: Filtered token list.

    Example:
        >>> remove_stopwords(['the', 'movie', 'was', 'not', 'good', '!'])
        ['movie', 'not', 'good', '!']
    """
    if not tokens:
        return []

    # Build effective stopword set
    effective_stopwords = set(_BASE_STOPWORDS)

    if preserve_negations:
        effective_stopwords -= NEGATION_WORDS
        # Exclude any contraction pieces containing n't from stopwords
        effective_stopwords = {
            sw for sw in effective_stopwords
            if not ("n't" in sw or sw in NEGATION_WORDS)
        }

    if preserve_contrast:
        effective_stopwords -= CONTRAST_WORDS

    if custom_exclude_stopwords:
        effective_stopwords -= custom_exclude_stopwords

    filtered: list[str] = []
    for token in tokens:
        lower_token = token.lower()
        if not token.isalnum():
            if preserve_punctuation:
                filtered.append(token)
            continue

        if lower_token not in effective_stopwords:
            filtered.append(token)

    return filtered


def preprocess_text(
    text: str,
    lowercase: bool = True,
    expand_contractions_flag: bool = True,
    remove_stopwords_flag: bool = True,
    preserve_negations: bool = True,
    preserve_punctuation: bool = True,
    use_pos_lemmatization: bool = True,
    return_tokens: bool = False,
) -> Union[str, list[str]]:
    """Complete end-to-end preprocessing pipeline for movie review text.

    Executes:
    1. clean_text: HTML removal, URL removal, contraction expansion, whitespace normalization.
    2. tokenize_text: Word and punctuation tokenization preserving clause boundaries.
    3. remove_stopwords: Negation-preserving stopword elimination.
    4. lemmatize_tokens: POS-guided WordNet lemmatization.

    Args:
        text (str): Raw input review text.
        lowercase (bool, optional): Convert text to lowercase. Defaults to True.
        expand_contractions_flag (bool, optional): Expand contractions like "didn't". Defaults to True.
        remove_stopwords_flag (bool, optional): Filter stopwords. Defaults to True.
        preserve_negations (bool, optional): Protect negation words from stopword filter. Defaults to True.
        preserve_punctuation (bool, optional): Retain punctuation marks. Defaults to True.
        use_pos_lemmatization (bool, optional): Use POS tags for lemmatization. Defaults to True.
        return_tokens (bool, optional): If True, returns list of tokens. If False, returns
            space-delimited string (ideal for TF-IDF / vectorizers). Defaults to False.

    Returns:
        Union[str, list[str]]: Preprocessed text string or list of tokens.

    Example:
        >>> preprocess_text("The movie wasn't good at all!")
        'movie not good !'
        >>> preprocess_text("The movie wasn't good at all!", return_tokens=True)
        ['movie', 'not', 'good', '!']
    """
    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""

    # 1. Clean text
    cleaned = clean_text(
        text=text,
        lowercase=lowercase,
        expand_contractions_flag=expand_contractions_flag,
        preserve_punctuation=preserve_punctuation,
    )

    # 2. Tokenize
    tokens = tokenize_text(
        text=cleaned,
        preserve_punctuation=preserve_punctuation,
    )

    # 3. Stopword removal (protecting negations and punctuation)
    if remove_stopwords_flag:
        tokens = remove_stopwords(
            tokens=tokens,
            preserve_negations=preserve_negations,
            preserve_punctuation=preserve_punctuation,
        )

    # 4. Lemmatize
    tokens = lemmatize_tokens(
        tokens=tokens,
        use_pos=use_pos_lemmatization,
    )

    if return_tokens:
        return tokens

    return " ".join(tokens)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads dataset from CSV file.

    Args:
        filepath (str): Path to CSV dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is empty or missing required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")

    return df


def clean_dataset(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Cleans dataset by dropping missing values and duplicate rows.

    Does not modify labels or drop non-duplicate valid reviews.

    Args:
        df (pd.DataFrame): Raw DataFrame.
        subset (list[str], optional): Columns to check for duplicates.
            Defaults to None (all columns).

    Returns:
        pd.DataFrame: Cleaned DataFrame with reset index.
    """
    initial_count = len(df)
    # 1. Handle missing values
    df_cleaned = df.dropna().copy()
    null_dropped = initial_count - len(df_cleaned)

    # 2. Handle duplicates
    before_dup = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(subset=subset).reset_index(drop=True)
    dup_dropped = before_dup - len(df_cleaned)

    print(f"Data cleaning summary: {null_dropped} null rows dropped, {dup_dropped} duplicate rows dropped.")
    print(f"Cleaned dataset rows: {len(df_cleaned)} (from initial {initial_count}).")

    return df_cleaned


def split_data(
    df: pd.DataFrame,
    target_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs stratified train-test split.

    Ensures identical class proportions in train and test splits, and deterministic
    reproducibility across experiments using random_state=42.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str, optional): Target class column for stratification. Defaults to 'sentiment'.
        test_size (float, optional): Proportion of test data. Defaults to 0.2 (80:20 split).
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

    Raises:
        ValueError: If target_col is not present in df.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def prepare_and_save_splits(
    raw_csv_path: str = "data/raw/IMDB_Dataset.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw IMDb dataset, cleans missing/duplicate rows, splits 80:20, and saves to disk.

    Saves:
    - {output_dir}/train.csv
    - {output_dir}/test.csv

    Args:
        raw_csv_path (str, optional): Path to raw CSV file. Defaults to 'data/raw/IMDB_Dataset.csv'.
        output_dir (str, optional): Directory to save processed splits. Defaults to 'data/processed'.
        test_size (float, optional): Test split ratio. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = load_dataset(raw_csv_path)

    print("Cleaning dataset (removing nulls and duplicates)...")
    df_clean = clean_dataset(df)

    print(f"Performing stratified {int((1-test_size)*100)}:{int(test_size*100)} split (random_state={random_state})...")
    train_df, test_df = split_data(
        df=df_clean,
        target_col="sentiment",
        test_size=test_size,
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train split ({len(train_df)} rows) to: {train_path}")
    print(f"Saved test split ({len(test_df)} rows) to: {test_path}")

    return train_df, test_df


def run_preprocessing(
    input_path: str = "data/raw/IMDB Dataset.csv",
    output_path: str = "data/processed/imdb_cleaned.csv",
) -> pd.DataFrame:
    """Create the normalized dataset consumed by the vectorizer."""

    df = clean_dataset(load_dataset(input_path), subset=["review"])
    df["review"] = df["review"].map(preprocess_text)
    df = df[df["review"].ne("")].copy()
    df["label"] = df["sentiment"].str.lower().eq("positive").astype(int)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    # Self-test when run directly
    sample = "The movie was not good! In fact, it wasn't worth watching, but the music was nice."
    print("Sample:", sample)
    print("Cleaned:", clean_text(sample))
    print("Tokens:", tokenize_text(clean_text(sample)))
    print("Preprocessed (str):", preprocess_text(sample))
    print("Preprocessed (tokens):", preprocess_text(sample, return_tokens=True))
