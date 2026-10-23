"""
Unit Tests for Machine Learning Models, Persistence, and Explainability.
"""

from pathlib import Path
import pytest
from src.features import build_tfidf_vectorizer
from src.model import SpamClassifierPipeline, TopicClassifierPipeline
from src.classifier import EmailClassifierService


def test_spam_classifier_train_and_predict(tmp_path):
    X = [
        "Congratulations! You won $1,000,000 lottery cash prize! Claim your lottery win now at http://win.com",
        "Hi team, attached is the Q3 sprint review presentation for tomorrow team meeting.",
        "URGENT: Your bank account is locked! Click link to verify immediately win cash.",
        "Can we reschedule our project sync to 2 PM this afternoon for team meeting?",
    ]
    y = ["spam", "ham", "spam", "ham"]

    # Use min_df=1 for small toy test fixture
    vec = build_tfidf_vectorizer(min_df=1)
    model = SpamClassifierPipeline(vectorizer=vec)
    model.fit(X, y)

    preds = model.predict(["Congratulations win cash lottery", "Hi team meeting tomorrow sync"])
    assert preds[0] == "spam"
    assert preds[1] == "ham"

    # Test explain_text
    expl = model.explain_text("Claim your free $500 lottery gift today!")
    assert "top_spam_words" in expl
    assert "top_ham_words" in expl

    # Test persistence (save/load)
    save_file = tmp_path / "test_model.joblib"
    model.save(save_file)
    assert save_file.exists()

    loaded = SpamClassifierPipeline.load(save_file)
    loaded_preds = loaded.predict(["Claim your free $500 lottery gift today!"])
    assert loaded_preds[0] == "spam"


def test_classifier_service():
    service = EmailClassifierService()
    res = service.classify_email("Hi Sarah, let's meet tomorrow at 10 AM to discuss code reviews.")
    assert res["is_spam"] is False
    assert res["label"] == "ham"
    assert res["confidence"] > 0.5
    assert "explanation" in res
    assert "heuristics" in res
