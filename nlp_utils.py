
"""
nlp_utils.py
Module B1 - Text Preprocessing

Reusable text-cleaning utilities shared by the sentiment model (B2) and the
chatbot's ML fallback (B3).

Pipeline covers the Week 6 syllabus points:
    - lowercasing
    - punctuation removal
    - stopword removal
    - lemmatization (NLTK WordNetLemmatizer)
    - tokenization
"""

from __future__ import annotations

import re
import string
from functools import lru_cache

import nltk


# --------------------------------------------------------------------------- #
# NLTK data setup — downloads quietly once, then reuses cached data.
# --------------------------------------------------------------------------- #
_REQUIRED_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_data() -> None:
    for path, package in _REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

from nltk.corpus import stopwords as _stopwords_corpus
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()
_STOPWORDS = set(_stopwords_corpus.words("english"))

# Retail-domain negation words matter a lot for sentiment ("not good" != "good"),
# so we deliberately keep common negations even though they're in the default
# NLTK stopword list.
_NEGATIONS_TO_KEEP = {"no", "not", "nor", "never", "none", "n't"}
_STOPWORDS -= _NEGATIONS_TO_KEEP

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# --------------------------------------------------------------------------- #
# Individual pipeline steps (exposed separately so notebooks can inspect
# each stage, per the B1 syllabus requirement)
# --------------------------------------------------------------------------- #
def lowercase(text: str) -> str:
    return text.lower()


def remove_urls_and_html(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text


def remove_punctuation(text: str) -> str:
    return text.translate(_PUNCT_TABLE)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def remove_short_tokens(tokens: list[str], min_len: int = 2) -> list[str]:
    return [t for t in tokens if len(t) >= min_len]


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess(text: str, return_tokens: bool = False) -> str | list[str]:
    """
    Full B1 cleaning pipeline: lowercase -> strip URLs/HTML -> remove
    punctuation/numbers -> tokenize -> drop stopwords -> lemmatize.

    Returns a cleaned string by default, or the token list if return_tokens=True.
    """
    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""

    text = lowercase(text)
    text = remove_urls_and_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_short_tokens(tokens)
    tokens = lemmatize_tokens(tokens)

    return tokens if return_tokens else " ".join(tokens)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Vectorized-friendly batch cleaning for a pandas Series / list of raw texts."""
    return [preprocess(t) for t in texts]


# --------------------------------------------------------------------------- #
# Quick manual test: `python nlp_utils.py "Some review text!!"`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else (
        "This dress is NOT great at all... I did NOT enjoy the fit, "
        "and it arrived 3 days late! Visit https://example.com for a refund."
    )

    print("Original: ", sample)
    print("Cleaned:  ", preprocess(sample))
    print("Tokens:   ", preprocess(sample, return_tokens=True))
