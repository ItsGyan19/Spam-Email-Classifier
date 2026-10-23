"""
High-Level Inference and Prediction Engine for Spam & Topic Email Classification.
Combines:
- Spam vs Ham binary classification with confidence probabilities
- Fine-grained Topic prediction (Phishing, Lottery, Marketing, Crypto, Work, Personal, etc.)
- Explainability (top influential words per email)
- Structural heuristic analysis (links, caps, currency, punctuation)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from src.features import extract_single_email_features
from src.model import SpamClassifierPipeline, TopicClassifierPipeline


TOPIC_DISPLAY_NAMES = {
    "phishing": "Phishing & Credential Alert",
    "lottery": "Lottery / Sweepstakes Scam",
    "marketing": "Commercial Marketing / Promo",
    "crypto_scam": "Crypto / Get-Rich-Quick Scam",
    "work": "Work & Team Communication",
    "personal": "Personal & Social Email",
    "notification": "Order & Transaction Alert",
    "newsletter": "Newsletter & Educational",
}


class EmailClassifierService:
    """
    Unified Service to perform complete classification, topic identification,
    explainability, and structural metric extraction on emails.
    """

    def __init__(
        self,
        spam_model_path: str | Path = "models/spam_classifier_model.joblib",
        topic_model_path: str | Path = "models/topic_classifier_model.joblib",
    ):
        self.spam_model_path = Path(spam_model_path)
        self.topic_model_path = Path(topic_model_path)

        if not self.spam_model_path.exists():
            raise FileNotFoundError(
                f"Model file {self.spam_model_path} not found. Run 'python src/train.py' first."
            )

        self.spam_model = SpamClassifierPipeline.load(self.spam_model_path)
        
        self.topic_model = None
        if self.topic_model_path.exists():
            self.topic_model = TopicClassifierPipeline.load(self.topic_model_path)

    def classify_email(self, text: str) -> Dict[str, Any]:
        """
        Processes a single email text and returns full analysis:
        - prediction ('spam' or 'ham')
        - confidence score (0.0 to 1.0)
        - predicted topic
        - explainability cues
        - structural domain heuristics
        """
        if not text or not text.strip():
            return {
                "label": "ham",
                "is_spam": False,
                "confidence": 0.5,
                "confidence_percent": 50.0,
                "topic": "unknown",
                "topic_display": "Unknown",
                "topic_confidence": 0.0,
                "explanation": {"top_spam_words": [], "top_ham_words": [], "all_analyzed_tokens": 0},
                "heuristics": {},
            }

        # 1. Binary Spam Prediction & Probability
        probs = self.spam_model.predict_proba([text])[0]
        classes = list(self.spam_model.classes_)
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_idx = classes.index("ham") if "ham" in classes else 0

        spam_prob = float(probs[spam_idx])
        ham_prob = float(probs[ham_idx])

        is_spam = spam_prob >= 0.50
        predicted_label = "spam" if is_spam else "ham"
        confidence = spam_prob if is_spam else ham_prob

        # 2. Topic Prediction
        topic_raw = "general"
        topic_prob = 0.0
        if self.topic_model:
            topic_classes, t_probs = self.topic_model.predict_proba([text])
            top_t_idx = int(np.argmax(t_probs[0]))
            topic_raw = topic_classes[top_t_idx]
            topic_prob = float(t_probs[0][top_t_idx])

        topic_display = TOPIC_DISPLAY_NAMES.get(topic_raw, topic_raw.replace("_", " ").title())

        # 3. Explainability (Feature contribution)
        explanation = self.spam_model.explain_text(text, top_k=6)

        # 4. Domain Heuristics
        heuristics = extract_single_email_features(text)

        return {
            "label": predicted_label,
            "is_spam": is_spam,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "topic": topic_raw,
            "topic_display": topic_display,
            "topic_confidence": round(topic_prob * 100, 2),
            "explanation": explanation,
            "heuristics": heuristics,
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classifies a list of emails."""
        return [self.classify_email(t) for t in texts]

