"""
Model Training and Benchmarking Pipeline.
Trains multiple NLP classification algorithms, runs stratified 5-fold cross-validation,
evaluates precision/recall/F1/ROC-AUC, selects the best model, trains the topic classifier,
and persists artifacts to disk.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.features import build_tfidf_vectorizer
from src.model import MODEL_ZOO, SpamClassifierPipeline, TopicClassifierPipeline


def load_data(filepath: str | Path = "data/emails.csv") -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run data/generate_dataset.py first.")
    df = pd.read_csv(path)
    return df


def benchmark_models(X_train: pd.Series, y_train: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluates each model in MODEL_ZOO using Stratified 5-Fold Cross Validation.
    """
    print("\n" + "=" * 80)
    print("RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION ACROSS CANDIDATE ALGORITHMS")
    print("=" * 80)
    
    # Binary mapping for scoring
    y_binary = (y_train == "spam").astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    
    results = {}
    
    for name, clf in MODEL_ZOO.items():
        print(f"--> Training & Cross-Validating: {name}...")
        pipeline = SpamClassifierPipeline(classifier=clf)
        cv_scores = cross_validate(
            pipeline.pipeline,
            X_train,
            y_binary,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        
        metrics = {
            "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
            "precision_mean": float(np.mean(cv_scores["test_precision"])),
            "precision_std": float(np.std(cv_scores["test_precision"])),
            "recall_mean": float(np.mean(cv_scores["test_recall"])),
            "recall_std": float(np.std(cv_scores["test_recall"])),
            "f1_mean": float(np.mean(cv_scores["test_f1"])),
            "f1_std": float(np.std(cv_scores["test_f1"])),
            "roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        }
        results[name] = metrics

    # Print summary table
    print("\n" + "-" * 80)
    header = f"{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}"
    print(header)
    print("-" * 80)
    for name, m in results.items():
        print(
            f"{name:<22} | {m['accuracy_mean']*100:6.2f}%   | {m['precision_mean']*100:6.2f}%   | {m['recall_mean']*100:6.2f}%   | {m['f1_mean']*100:6.2f}%   | {m['roc_auc_mean']*100:6.2f}%"
        )
    print("-" * 80 + "\n")
    return results


def train_and_save_best_model(
    df: pd.DataFrame,
    benchmark_results: Dict[str, Dict[str, float]],
    models_dir: str | Path = "models",
) -> Dict[str, Any]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X = df["content"]
    y_spam = df["label"]
    y_topic = df["topic"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test, topic_train, topic_test = train_test_split(
        X, y_spam, y_topic, test_size=0.2, random_state=42, stratify=y_spam
    )

    # Pick the algorithm with the highest F1-score (harmonic mean of precision and recall)
    best_algo_name = max(benchmark_results.keys(), key=lambda k: benchmark_results[k]["f1_mean"])
    print(f"Selected Best Algorithm: {best_algo_name} (F1: {benchmark_results[best_algo_name]['f1_mean']*100:.2f}%)")

    # Train final binary spam classifier
    best_clf = MODEL_ZOO[best_algo_name]
    final_spam_pipeline = SpamClassifierPipeline(classifier=best_clf)
    final_spam_pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_pred = final_spam_pipeline.predict(X_test)
    y_prob = final_spam_pipeline.predict_proba(X_test)
    spam_idx = list(final_spam_pipeline.classes_).index("spam")
    y_prob_spam = y_prob[:, spam_idx]
    y_test_binary = (y_test == "spam").astype(int)

    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_precision = float(precision_score(y_test, y_pred, pos_label="spam"))
    test_recall = float(recall_score(y_test, y_pred, pos_label="spam"))
    test_f1 = float(f1_score(y_test, y_pred, pos_label="spam"))
    test_roc_auc = float(roc_auc_score(y_test_binary, y_prob_spam))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel()

    print("Held-out Test Set Evaluation:")
    print(f"  Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"  Precision: {test_precision * 100:.2f}% (False Positives: {fp})")
    print(f"  Recall:    {test_recall * 100:.2f}% (False Negatives: {fn})")
    print(f"  F1-Score:  {test_f1 * 100:.2f}%")
    print(f"  ROC-AUC:   {test_roc_auc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Ham (TN):  {tn} | False Spam (FP): {fp}")
    print(f"  False Ham (FN): {fn}  | True Spam (TP):  {tp}")

    # Train Topic Classifier (Multi-class)
    print("\nTraining Fine-Grained Topic Classifier...")
    topic_pipeline = TopicClassifierPipeline()
    topic_pipeline.fit(X_train, topic_train)
    topic_preds = topic_pipeline.predict(X_test)
    topic_acc = float(accuracy_score(topic_test, topic_preds))
    print(f"Topic Classifier Test Accuracy: {topic_acc * 100:.2f}%")

    # Save models
    spam_model_path = models_dir / "spam_classifier_model.joblib"
    topic_model_path = models_dir / "topic_classifier_model.joblib"
    final_spam_pipeline.save(spam_model_path)
    topic_pipeline.save(topic_model_path)
    print(f"Saved Spam Classifier to: {spam_model_path}")
    print(f"Saved Topic Classifier to: {topic_model_path}")

    # Save metadata
    metadata = {
        "best_algorithm": best_algo_name,
        "classes": list(final_spam_pipeline.classes_),
        "topics": list(np.unique(df["topic"])),
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_metrics": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "topic_test_accuracy": topic_acc,
        "benchmark_results": benchmark_results,
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Metadata & Metrics to: {metadata_path}")

    return metadata


def main():
    df = load_data()
    X = df["content"]
    y = df["label"]
    
    # 1. Benchmark all algorithms
    benchmark_results = benchmark_models(X, y)
    
    # 2. Select, train, evaluate, and save best model
    metadata = train_and_save_best_model(df, benchmark_results)
    print("\nTraining pipeline finished successfully!")


if __name__ == "__main__":
    main()
