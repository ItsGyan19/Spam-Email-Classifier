"""
Unit Tests for Feature Extraction and Domain Heuristics.
"""

import numpy as np
import pytest
from src.features import DomainFeatureExtractor, build_tfidf_vectorizer, extract_single_email_features


def test_domain_feature_extractor():
    extractor = DomainFeatureExtractor()
    sample_text = "URGENT WINNER!!! Click http://scam.biz to claim $500 cash now!"

    feats = extractor.transform([sample_text])
    assert feats.shape == (1, 8)

    # Check heuristics
    h_dict = extract_single_email_features(sample_text)
    assert h_dict["currency_count"] >= 1.0
    assert h_dict["url_count"] >= 1.0
    assert h_dict["cap_ratio"] > 0.05
    assert h_dict["exclamation_ratio"] > 0.0


def test_tfidf_vectorizer_pipeline():
    corpus = [
        "Congratulations you won the lottery and free money!",
        "Quarterly budget review meeting notes attached.",
        "Win free cash prizes and gifts today only!",
        "Please review the pull request for code changes.",
    ]
    vec = build_tfidf_vectorizer(max_features=50, ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)

    assert matrix.shape[0] == 4
    assert matrix.shape[1] > 0
    feature_names = vec.get_feature_names_out()
    assert len(feature_names) == matrix.shape[1]

