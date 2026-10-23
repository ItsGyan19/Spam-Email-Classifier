"""
Educational Walkthrough: NLP & Supervised Classification for Email Spam Detection.
Demonstrates step-by-step AI and ML concepts:
1. NLP Preprocessing Pipeline (Tokenization, Normalization, Stopwords, Stemming)
2. TF-IDF Vectorization & Mathematical Mechanics
3. Model Training & Comparison across Algorithms
4. Precision-Recall Trade-offs in Spam Filtering
5. Explainability (Local feature attribution)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.preprocessor import TextPreprocessor
from src.features import build_tfidf_vectorizer, DomainFeatureExtractor
from src.model import SpamClassifierPipeline
from src.classifier import EmailClassifierService


def section(title: str):
    print("\n" + "=" * 80)
    print(f"STEP: {title}")
    print("=" * 80)


def step_1_nlp_preprocessing():
    section("1. NATURAL LANGUAGE PROCESSING (NLP) PREPROCESSING")
    raw_sample = (
        "URGENT: Your Bank of America account was logged in from Russia! "
        "Visit http://secure-boa-login-verify.net/reset or email security@bank-alert.com "
        "to stop unauthorized $1,250 wire transfer immediately!!!"
    )
    print("Original Raw Email:")
    print(f"  \"{raw_sample}\"\n")

    preprocessor = TextPreprocessor(use_stemming=True)

    # Step-by-step breakdown
    cleaned = preprocessor.clean_text(raw_sample)
    print("Stage 1 - Normalization & Token Replacement:")
    print("  • URLs normalized to 'token_url'")
    print("  • Emails normalized to 'token_email'")
    print("  • Currency normalized to 'token_money'")
    print("  • Punctuations removed and lowercased")
    print(f"Result:\n  \"{cleaned}\"\n")

    processed = preprocessor.process(raw_sample)
    print("Stage 2 - Stopword Removal & Porter Stemming:")
    print("  • Common non-informative English stopwords stripped")
    print("  • Inflected words reduced to root stems (e.g., 'unauthorized' -> 'unauthor')")
    print(f"Result:\n  \"{processed}\"")


def step_2_tfidf_mechanics():
    section("2. TF-IDF VECTORIZATION & N-GRAM EXTRACTION")
    corpus = [
        "Congratulations! You won the cash lottery prize of $1,000,000! Claim now.",
        "Reminder: The weekly project sprint retro is scheduled for 3 PM today.",
        "URGENT: Account locked. Click link to verify your identity.",
        "Can we reschedule our lunch meeting to tomorrow afternoon?",
    ]
    print(f"Corpus size: {len(corpus)} documents\n")

    vectorizer = build_tfidf_vectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Total extracted n-grams (unigrams + bigrams): {len(feature_names)}")
    print(f"Sample vocabulary tokens: {feature_names[:15]}")
    print(f"Sparse matrix shape: {matrix.shape} (documents x features)")

    # Print top weights for doc 0 (spam)
    doc_0_weights = matrix.toarray()[0]
    top_indices = np.argsort(doc_0_weights)[::-1][:6]
    print("\nTop TF-IDF weights for Document 0 ('Lottery winner'):")
    for idx in top_indices:
        if doc_0_weights[idx] > 0:
            print(f"  • Token: '{feature_names[idx]}': TF-IDF score = {doc_0_weights[idx]:.4f}")


def step_3_classification_and_metrics():
    section("3. SUPERVISED CLASSIFICATION & MODEL COMPARISON")
    print("In spam filtering, class evaluation metrics have specific real-world consequences:")
    print("  • False Positive (FP): A legitimate email is classified as SPAM.")
    print("    -> CRITICAL RISK: User misses an urgent work email or flight confirmation!")
    print("  • False Negative (FN): A spam email reaches the inbox.")
    print("    -> ANNOYANCE: User has to delete an unwanted message.")
    print("  • CONCLUSION: PRECISION must be prioritized near 100% so legitimate mail is never lost.")

    meta_path = PROJECT_ROOT / "models" / "metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        print("\nCross-Validation Benchmark Results across Algorithms:")
        bench = meta.get("benchmark_results", {})
        for name, m in bench.items():
            print(
                f"  [{name:<18}] Accuracy: {m['accuracy_mean']*100:5.2f}% | Precision: {m['precision_mean']*100:5.2f}% | Recall: {m['recall_mean']*100:5.2f}% | F1: {m['f1_mean']*100:5.2f}%"
            )


def step_4_inference_and_explainability():
    section("4. INFERENCE & WORD-LEVEL LOCAL EXPLAINABILITY")
    service = EmailClassifierService()

    test_emails = [
        "Congratulations! Your name was chosen in our lottery sweepstakes! Claim your $500,000 wire transfer.",
        "Hi Alex, could you review the latest pull request when you have time today? Thanks!",
    ]

    for email in test_emails:
        res = service.classify_email(email)
        print(f"\nEmail: \"{email[:80]}...\"")
        print(f"  Verdict:           {'[ SPAM ]' if res['is_spam'] else '[ HAM ]'} ({res['confidence_percent']:.2f}% confidence)")
        print(f"  Predicted Topic:   {res['topic_display']} ({res['topic_confidence']:.1f}%)")
        
        expl = res["explanation"]
        if res["is_spam"] and expl["top_spam_words"]:
            words = ", ".join([f"{w['word']} (+{w['impact']:.2f})" for w in expl["top_spam_words"][:4]])
            print(f"  Strongest Cues:    {words}")
        elif not res["is_spam"] and expl["top_ham_words"]:
            words = ", ".join([f"{w['word']} ({w['impact']:.2f})" for w in expl["top_ham_words"][:4]])
            print(f"  Strongest Cues:    {words}")


def main():
    print("=" * 80)
    print("EDUCATIONAL WALKTHROUGH: TOPIC SPAM EMAIL CLASSIFIER")
    print("=" * 80)
    step_1_nlp_preprocessing()
    step_2_tfidf_mechanics()
    step_3_classification_and_metrics()
    step_4_inference_and_explainability()
    print("\n" + "=" * 80)
    print("Walkthrough completed successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

