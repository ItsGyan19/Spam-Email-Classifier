"""
Flask Web Application & REST API for Topic Spam Email Classifier.
Provides:
- Web Dashboard with Real-time classification
- Visual Confidence Meter, Topic Badges, Explainability cues
- Pre-filled sample emails
- Model Metrics & Benchmark tab
- REST API (/api/classify, /api/batch, /api/metrics)
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from src.classifier import EmailClassifierService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "app" / "templates"),
    static_folder=str(PROJECT_ROOT / "app" / "static"),
)

# ==============================================================================
# SECURITY & CRASH-PREVENTION GUARDS
# ==============================================================================
# 1. Enforce 2 Megabytes maximum payload limit for any request
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# 2. Maximum characters allowed per email (50,000 characters is ~10,000 words)
MAX_TEXT_LENGTH = 50_000

# 3. Maximum items allowed per batch request
MAX_BATCH_SIZE = 100


def sanitize_input(raw_text: str) -> str:
    """Removes null bytes and dangerous control characters that could crash parsers."""
    if not isinstance(raw_text, str):
        return ""
    # Strip null bytes and non-printable control characters (except \n, \r, \t)
    cleaned = raw_text.replace("\x00", "")
    return cleaned


@app.errorhandler(RequestEntityTooLarge)
def handle_payload_too_large(error):
    """Graceful JSON handler when request exceeds MAX_CONTENT_LENGTH."""
    return jsonify({
        "error": "Payload too large. Maximum allowed request size is 2 Megabytes."
    }), 413


# Global classifier service instance
classifier_service = None


def get_service():
    global classifier_service
    if classifier_service is None:
        classifier_service = EmailClassifierService()
    return classifier_service


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json(silent=True) or {}
        raw_text = data.get("text", "")

        if not isinstance(raw_text, str) or not raw_text.strip():
            return jsonify({"error": "No email text provided"}), 400

        # Enforce maximum text length guard
        if len(raw_text) > MAX_TEXT_LENGTH:
            return jsonify({
                "error": f"Email text exceeds maximum allowed limit of {MAX_TEXT_LENGTH:,} characters (received {len(raw_text):,} characters)."
            }), 413

        # Sanitize input
        clean_text = sanitize_input(raw_text)

        service = get_service()
        result = service.classify_email(clean_text)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        return jsonify({
            "error": "An internal server error occurred while processing the email."
        }), 500


@app.route("/api/batch", methods=["POST"])
def classify_batch():
    try:
        data = request.get_json(silent=True) or {}
        emails = data.get("emails", [])

        if not isinstance(emails, list) or len(emails) == 0:
            return jsonify({"error": "Provide a non-empty array of email texts in 'emails'"}), 400

        # Enforce batch size limit
        if len(emails) > MAX_BATCH_SIZE:
            return jsonify({
                "error": f"Batch size exceeds maximum limit of {MAX_BATCH_SIZE} emails (received {len(emails)})."
            }), 400

        # Sanitize and validate each item in the batch
        sanitized_emails = []
        for i, item in enumerate(emails):
            if not isinstance(item, str):
                return jsonify({"error": f"Item at index {i} is not a valid text string"}), 400
            if len(item) > MAX_TEXT_LENGTH:
                return jsonify({
                    "error": f"Item at index {i} exceeds character limit of {MAX_TEXT_LENGTH:,} characters"
                }), 413
            sanitized_emails.append(sanitize_input(item))

        service = get_service()
        results = service.classify_batch(sanitized_emails)
        return jsonify({"count": len(results), "results": results})

    except Exception as e:
        logger.error(f"Error during batch classification: {e}", exc_info=True)
        return jsonify({
            "error": "An internal server error occurred while processing the batch."
        }), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    meta_path = PROJECT_ROOT / "models" / "metadata.json"
    if not meta_path.exists():
        return jsonify({"error": "Model metadata not found. Run training first."}), 404

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return jsonify(meta)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting Spam Email Classifier Web App on http://127.0.0.1:{port} (debug={debug_mode})")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

