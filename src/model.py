"""
Machine Learning Models and Evaluation for Spam & Topic Classification.
Implements:
1. Multi-model training and automated evaluation (MNB, CNB, Logistic Regression, LinearSVC, Random Forest)
2. Stratified Cross-Validation & Metric Computation (Accuracy, Precision, Recall, F1, ROC-AUC)
3. Topic Classifier (Multi-class topic identification)
4. Model Explainability (Feature contributions / Word importance)
5. Model Persistence (Save / Load)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import build_tfidf_vectorizer


MODEL_ZOO = {
    "MultinomialNB": MultinomialNB(alpha=0.5),
    "ComplementNB": ComplementNB(alpha=0.5),
    "LogisticRegression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
    "LinearSVC": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
}


class SpamClassifierPipeline:
    """
    End-to-end Pipeline combining NLP TF-IDF vectorization with a Classifier.
    Includes persistence, prediction with calibrated confidence, and word-level explainability.
    """

    def __init__(self, classifier=None, vectorizer=None):
        self.vectorizer = vectorizer if vectorizer is not None else build_tfidf_vectorizer()
        self.classifier = classifier if classifier is not None else LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])
        self.classes_ = np.array(["ham", "spam"])

    def fit(self, X: pd.Series | List[str], y: pd.Series | List[str]):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            X_vec = self.pipeline.named_steps["tfidf"].transform(X)
            return clf.predict_proba(X_vec)
        elif hasattr(clf, "decision_function"):
            X_vec = self.pipeline.named_steps["tfidf"].transform(X)
            decision = clf.decision_function(X_vec)
            prob = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - prob, prob])
        else:
            preds = self.predict(X)
            return np.array([[0.0, 1.0] if p == "spam" else [1.0, 0.0] for p in preds])

    def explain_text(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Explain the decision for an individual email by extracting which words
        had the highest positive (spam) or negative (ham) impact.
        """
        tfidf_step = self.pipeline.named_steps["tfidf"]
        clf_step = self.pipeline.named_steps["clf"]

        # Transform single text
        X_vec = tfidf_step.transform([text])
        feature_names = tfidf_step.get_feature_names_out()
        nonzero_indices = X_vec.nonzero()[1]

        # Extract weights based on model type
        weights = None
        if hasattr(clf_step, "coef_"):
            weights = clf_step.coef_[0]
        elif hasattr(clf_step, "feature_log_prob_"):
            # For Naive Bayes: difference between spam and ham log-probabilities
            spam_idx = list(self.classes_).index("spam")
            ham_idx = list(self.classes_).index("ham")
            weights = clf_step.feature_log_prob_[spam_idx] - clf_step.feature_log_prob_[ham_idx]
        elif hasattr(clf_step, "calibrated_classifiers_"):
            # For CalibratedClassifierCV
            coefs = [cc.estimator.coef_[0] for cc in clf_step.calibrated_classifiers_ if hasattr(cc.estimator, "coef_")]
            if coefs:
                weights = np.mean(coefs, axis=0)

        word_scores = []
        if weights is not None and len(nonzero_indices) > 0:
            for idx in nonzero_indices:
                word = feature_names[idx]
                tf_val = X_vec[0, idx]
                impact = float(weights[idx] * tf_val)
                word_scores.append({"word": word, "impact": round(impact, 4), "tfidf": round(float(tf_val), 4)})

        word_scores.sort(key=lambda x: x["impact"], reverse=True)
        top_spam_cues = [w for w in word_scores if w["impact"] > 0][:top_k]
        top_ham_cues = [w for w in sorted(word_scores, key=lambda x: x["impact"]) if w["impact"] < 0][:top_k]

        return {
            "top_spam_words": top_spam_cues,
            "top_ham_words": top_ham_cues,
            "all_analyzed_tokens": len(nonzero_indices),
        }

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "SpamClassifierPipeline":
        pipeline = joblib.load(filepath)
        instance = cls()
        instance.pipeline = pipeline
        instance.vectorizer = pipeline.named_steps["tfidf"]
        instance.classifier = pipeline.named_steps["clf"]
        instance.classes_ = pipeline.named_steps["clf"].classes_
        return instance


class TopicClassifierPipeline:
    """
    Multi-class NLP classifier for specific email topics
    (phishing, lottery, marketing, crypto_scam, work, personal, notification, newsletter).
    """

    def __init__(self):
        self.vectorizer = build_tfidf_vectorizer(max_features=2500)
        self.classifier = LogisticRegression(C=3.0, max_iter=1000, random_state=42)
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])

    def fit(self, X: pd.Series | List[str], y: pd.Series | List[str]):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> Tuple[List[str], np.ndarray]:
        probs = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.named_steps["clf"].classes_)
        return classes, probs

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "TopicClassifierPipeline":
        pipeline = joblib.load(filepath)
        instance = cls()
        instance.pipeline = pipeline
        instance.vectorizer = pipeline.named_steps["tfidf"]
        instance.classifier = pipeline.named_steps["clf"]
        return instance
