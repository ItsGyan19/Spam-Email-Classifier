"""
Unit Tests for NLP Text Preprocessor.
"""

import pytest
from src.preprocessor import TextPreprocessor, preprocess_corpus


def test_clean_text_normalizations():
    preprocessor = TextPreprocessor()

    # 1. URLs
    text_with_url = "Check this link: https://verify-bank.com/login for updates."
    cleaned = preprocessor.clean_text(text_with_url)
    assert "token_url" in cleaned
    assert "https://" not in cleaned

    # 2. Emails
    text_with_email = "Contact us at support@example.org immediately."
    cleaned_email = preprocessor.clean_text(text_with_email)
    assert "token_email" in cleaned_email
    assert "@" not in cleaned_email

    # 3. Currency
    text_with_currency = "You won $5,000 cash prize!"
    cleaned_curr = preprocessor.clean_text(text_with_currency)
    assert "token_money" in cleaned_curr


def test_stopwords_and_stemming():
    preprocessor = TextPreprocessor(use_stemming=True)
    # 'running' should be stemmed to 'run'
    # 'is' and 'the' should be stripped as stopwords
    processed = preprocessor.process("The user is running tests")
    tokens = processed.split()

    assert "run" in tokens
    assert "test" in tokens
    assert "is" not in tokens
    assert "the" not in tokens


def test_empty_and_edge_cases():
    preprocessor = TextPreprocessor()
    assert preprocessor.process("") == ""
    assert preprocessor.process(None) == ""
    assert preprocessor.clean_text("   \n\t  ") == ""

