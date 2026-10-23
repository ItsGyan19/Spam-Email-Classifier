"""
Interactive Command Line Interface (CLI) for Topic Spam Email Classifier.
Commands:
  python cli.py predict --text "Your email text here"
  python cli.py interactive
  python cli.py metrics
  python cli.py test-samples
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import EmailClassifierService


SAMPLE_EMAILS = [
    (
        "Spam: Urgent Bank Phishing",
        "URGENT: Your Wells Fargo online account access is locked due to 3 failed attempts! "
        "Log in immediately to restore access: http://wellsfargo-verify-security-now.com or call support."
    ),
    (
        "Spam: Mega Lottery Prize",
        "CONGRATULATIONS! You have been selected as the winner of $1,000,000 in the International "
        "MegaMillions Sweepstakes! Wire transfer ready. Send full name, phone number, and bank account now."
    ),
    (
        "Spam: Crypto AI Scam",
        "EARN $1,200 EVERY DAY FROM HOME! Our Automated Bitcoin Trading Bot guarantees 500% profit per week! "
        "Claim your free registration bonus today at: http://crypto-wealth-bot.biz"
    ),
    (
        "Ham: Team Sprint Planning",
        "Hi team, just a reminder that our sprint retro is scheduled for 3 PM today in Conference Room C. "
        "Please update your Jira tickets with remaining story points before the sync. Thanks!"
    ),
    (
        "Ham: Flight Booking Confirmation",
        "Your flight reservation for flight UA 492 to San Francisco (SFO) is confirmed for Nov 14. "
        "E-ticket and baggage details are attached. Thank you for flying United Airlines."
    ),
]


def print_result_card(title: str, text: str, result: dict):
    print("=" * 80)
    if title:
        print(f"EMAIL SAMPLE: {title}")
        print("-" * 80)
    print("CONTENT:")
    preview = text.strip() if len(text.strip()) < 220 else text.strip()[:220] + "..."
    print(f"  \"{preview}\"\n")
    
    label = result["label"].upper()
    conf = result["confidence_percent"]
    topic = result["topic_display"]
    
    tag = "[ SPAM ]" if result["is_spam"] else "[ HAM (Legitimate) ]"
    print(f"PREDICTION:      {tag}  (Confidence: {conf:.2f}%)")
    print(f"PREDICTED TOPIC: {topic} (Confidence: {result['topic_confidence']:.1f}%)")
    print(f"SPAM PROBABILITY: {result['spam_probability'] * 100:.2f}% | HAM PROBABILITY: {result['ham_probability'] * 100:.2f}%")
    
    # Structural features
    h = result.get("heuristics", {})
    print("\nSTRUCTURAL SIGNALS:")
    print(f"  • Capital Letter Ratio: {h.get('cap_ratio', 0)*100:.1f}%")
    print(f"  • Currency Symbols:     {int(h.get('currency_count', 0))}")
    print(f"  • URL / Link Count:     {int(h.get('url_count', 0))}")
    print(f"  • Exclamation Marks:    {int(h.get('exclamation_ratio', 0)*100):.1f}% frequency")

    # Word-level explainability
    expl = result.get("explanation", {})
    if result["is_spam"] and expl.get("top_spam_words"):
        cues = ", ".join([f"'{item['word']}' (+{item['impact']:.2f})" for item in expl["top_spam_words"][:4]])
        print(f"\nTOP SPAM SIGNALS DETECTED:\n  {cues}")
    elif not result["is_spam"] and expl.get("top_ham_words"):
        cues = ", ".join([f"'{item['word']}' ({item['impact']:.2f})" for item in expl["top_ham_words"][:4]])
        print(f"\nTOP HAM SIGNALS DETECTED:\n  {cues}")
        
    print("=" * 80 + "\n")


def cmd_predict(args, service: EmailClassifierService):
    if not args.text:
        print("Error: Please provide email text using --text '...'")
        return
    res = service.classify_email(args.text)
    print_result_card("Manual Input", args.text, res)


def cmd_interactive(service: EmailClassifierService):
    print("=" * 80)
    print("SPAM EMAIL CLASSIFIER - INTERACTIVE REPL")
    print("Type or paste an email message below, then press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 80)
    while True:
        try:
            user_input = input("\nEnter Email Text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive mode. Goodbye!")
                break
            res = service.classify_email(user_input)
            print_result_card("Live Test", user_input, res)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


def cmd_test_samples(service: EmailClassifierService):
    print("\nRUNNING PRE-DEFINED BENCHMARK SAMPLES:\n")
    for title, text in SAMPLE_EMAILS:
        res = service.classify_email(text)
        print_result_card(title, text, res)


def cmd_metrics():
    meta_path = Path("models/metadata.json")
    if not meta_path.exists():
        print("Metadata not found. Please run 'python src/train.py' first.")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    print("\n" + "=" * 80)
    print("MODEL METRICS & BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Selected Best Algorithm: {meta['best_algorithm']}")
    print(f"Total Dataset Samples:   {meta['dataset_size']} (Train: {meta['train_size']}, Test: {meta['test_size']})")
    print(f"Supported Topics:        {', '.join(meta['topics'])}")
    
    print("\nHELD-OUT TEST SET METRICS:")
    tm = meta["test_metrics"]
    print(f"  • Accuracy:  {tm['accuracy']*100:.2f}%")
    print(f"  • Precision: {tm['precision']*100:.2f}%")
    print(f"  • Recall:    {tm['recall']*100:.2f}%")
    print(f"  • F1-Score:  {tm['f1']*100:.2f}%")
    print(f"  • ROC-AUC:   {tm['roc_auc']*100:.2f}%")
    print(f"  • Topic Accuracy: {meta.get('topic_test_accuracy', 0)*100:.2f}%")
    
    cm = meta["confusion_matrix"]
    print("\nCONFUSION MATRIX:")
    print(f"  True Negatives (Ham):  {cm['tn']}  |  False Positives (False Spam): {cm['fp']}")
    print(f"  False Negatives (Missed): {cm['fn']} |  True Positives (Spam):      {cm['tp']}")

    print("\nCROSS-VALIDATION COMPARISON (5-Fold Stratified):")
    bench = meta.get("benchmark_results", {})
    print(f"{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 72)
    for algo, scores in bench.items():
        print(f"{algo:<22} | {scores['accuracy_mean']*100:6.2f}%   | {scores['precision_mean']*100:6.2f}%   | {scores['recall_mean']*100:6.2f}%   | {scores['f1_mean']*100:6.2f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Spam Email Classifier NLP & ML CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Predict
    predict_p = subparsers.add_parser("predict", help="Classify a single email string")
    predict_p.add_argument("--text", type=str, required=True, help="Raw email text to classify")

    # Interactive
    subparsers.add_parser("interactive", help="Start interactive classification session")

    # Test Samples
    subparsers.add_parser("test-samples", help="Run preset test samples")

    # Metrics
    subparsers.add_parser("metrics", help="Display model evaluation metrics and comparison table")

    args = parser.parse_args()

    if args.command == "metrics":
        cmd_metrics()
        return

    try:
        service = EmailClassifierService()
    except FileNotFoundError:
        print("Models not found. Training model now...")
        import subprocess
        subprocess.run([sys.executable, "src/train.py"], check=True)
        service = EmailClassifierService()

    if args.command == "predict":
        cmd_predict(args, service)
    elif args.command == "interactive":
        cmd_interactive(service)
    elif args.command == "test-samples" or args.command is None:
        cmd_test_samples(service)


if __name__ == "__main__":
    main()

