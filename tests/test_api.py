"""
Unit Tests for Flask REST API Endpoints.
"""

import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Email Spam &amp; Topic Classifier" in response.data or b"Email Spam & Topic Classifier" in response.data


def test_api_classify_valid(client):
    response = client.post(
        "/api/classify",
        json={"text": "CONGRATULATIONS! You won $2,000,000 wire transfer! Send details now!"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "label" in data
    assert "confidence" in data
    assert data["is_spam"] is True
    assert "explanation" in data


def test_api_classify_empty(client):
    response = client.post("/api/classify", json={"text": ""})
    assert response.status_code == 400


def test_api_batch(client):
    response = client.post(
        "/api/batch",
        json={
            "emails": [
                "URGENT: Verify your PayPal account now: http://phish.com",
                "Hi team, code review is complete.",
            ]
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_api_metrics(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "best_algorithm" in data
    assert "benchmark_results" in data


def test_api_classify_oversized_text(client):
    """Verifies that an email exceeding 50,000 characters is rejected with 413."""
    oversized_text = "Free lottery prize! " * 3000  # ~60,000 characters
    response = client.post("/api/classify", json={"text": oversized_text})
    assert response.status_code == 413
    data = response.get_json()
    assert "error" in data
    assert "exceeds maximum allowed limit" in data["error"]


def test_api_batch_oversized_limit(client):
    """Verifies that a batch exceeding 100 emails is rejected with 400."""
    too_many_emails = ["Meeting reminder"] * 105
    response = client.post("/api/batch", json={"emails": too_many_emails})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Batch size exceeds maximum limit" in data["error"]


def test_api_sanitize_null_byte(client):
    """Verifies that strings containing null bytes do not crash the parser."""
    text_with_null = "Meeting\x00 reminder for sprint\x00 retro"
    response = client.post("/api/classify", json={"text": text_with_null})
    assert response.status_code == 200
    data = response.get_json()
    assert data["label"] == "ham"


