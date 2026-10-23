"""
Feature Extraction and Engineering Module for Spam Email Classification.
Implements:
1. TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization with N-Grams
2. Domain-Specific Heuristic Feature Extraction (Capitals, Punctuations, Currency symbols, URLs)
3. Combined Feature Extraction Pipeline
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from src.preprocessor import TextPreprocessor


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts structural domain heuristics from raw email text:
    - Uppercase ratio (shouting)
    - Exclamation mark count & frequency
    - Question mark count
    - Currency symbol count ($, €, £)
    - URL count
    - Total character and word length
    - Numeric digit density
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def _extract_features_single(self, text: str) -> List[float]:
        if not isinstance(text, str) or len(text) == 0:
            return [0.0] * 8

        total_chars = max(len(text), 1)
        words = text.split()
        total_words = max(len(words), 1)

        # 1. Capital letter ratio
        cap_chars = sum(1 for c in text if c.isupper())
        cap_ratio = cap_chars / total_chars

        # 2. Exclamation mark count & ratio
        exclamation_count = text.count("!")
        exclamation_ratio = exclamation_count / total_chars

        # 3. Question mark count
        question_count = text.count("?")

        # 4. Currency symbols count
        currency_count = sum(text.count(sym) for sym in ["$", "€", "£", "¥", "₹"])

        # 5. URL count heuristic
        url_count = len(re.findall(r"https?://|www\.|\.(?:com|org|net|biz|xyz|info)", text, re.I))

        # 6. Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / total_chars

        # 7. Word count normalized (log scale)
        norm_word_count = np.log1p(total_words)

        # 8. All-caps word ratio (e.g. "FREE", "WIN", "URGENT")
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        all_caps_ratio = all_caps_words / total_words

        return [
            cap_ratio,
            exclamation_ratio,
            float(question_count),
            float(currency_count),
            float(url_count),
            digit_ratio,
            norm_word_count,
            all_caps_ratio,
        ]

    def transform(self, X):
        features = [self._extract_features_single(text) for text in X]
        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "cap_ratio",
            "exclamation_ratio",
            "question_count",
            "currency_count",
            "url_count",
            "digit_ratio",
            "norm_word_count",
            "all_caps_ratio",
        ])


def build_tfidf_vectorizer(
    max_features: int = 3000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Creates a pre-configured TF-IDF Vectorizer optimized for spam classification.
    Sublinear TF applies 1 + log(tf) scaling to diminish the impact of repetitive spam tokens.
    """
    preprocessor = TextPreprocessor(use_stemming=True)
    return TfidfVectorizer(
        preprocessor=preprocessor.process,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        token_pattern=r"(?u)\b\w+\b",
    )


def extract_single_email_features(text: str) -> Dict[str, Any]:
    """
    Extracts a human-readable dictionary of structural features for the UI/API.
    """
    extractor = DomainFeatureExtractor()
    raw_feats = extractor._extract_features_single(text)
    names = extractor.get_feature_names_out()
    return {name: round(val, 4) for name, val in zip(names, raw_feats)}

