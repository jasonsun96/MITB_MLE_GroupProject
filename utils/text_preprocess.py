"""NLTK text preprocessing: lowercase, stopword removal, lemmatization."""
from __future__ import annotations

import re

_STOPWORDS: set[str] | None = None
_LEMMATIZER = None


def ensure_nltk_data() -> None:
    import nltk

    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            else:
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download(resource, quiet=True)


def _stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_data()
        from nltk.corpus import stopwords

        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        ensure_nltk_data()
        from nltk.stem import WordNetLemmatizer

        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def raw_tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens before stopword removal / lemmatization."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def preprocess_tokens(text: str) -> list[str]:
    """
    Lowercase, remove English stopwords, lemmatize.
    Returns a list of tokens ready for n-gram / TF-IDF extraction.
    """
    stops = _stopwords()
    lemmatizer = _lemmatizer()
    out: list[str] = []

    for token in raw_tokenize(text):
        if token in stops:
            continue

        # Drop 1-char letter tokens (e, n, i, ...) — usually noise
        if len(token) == 1 and token.isalpha():
            continue

        # Drop pure numbers except 4-digit years (keeps 2019, drops 000, 5, 276)
        if token.isdigit():
            if len(token) != 4:
                continue
            out.append(token)
            continue

        out.append(lemmatizer.lemmatize(token))

    return out


def preprocess_text(text: str) -> str:
    """Space-joined preprocessed tokens (for silver `tokens` column)."""
    return " ".join(preprocess_tokens(text))


simple_tokenize = preprocess_tokens
