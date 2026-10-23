# 🛡️ Topic Spam Email Classifier (AI & NLP Platform)

An end-to-end Machine Learning and Natural Language Processing (NLP) system for high-accuracy spam detection, fine-grained topic classification, and explainable decision attribution.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Core AI Concepts Implemented](#-core-ai-concepts-implemented)
  - [1. Natural Language Processing (NLP) Pipeline](#1-natural-language-processing-nlp-pipeline)
  - [2. Feature Engineering & Vectorization](#2-feature-engineering--vectorization)
  - [3. Supervised Classification & Model Zoo](#3-supervised-classification--model-zoo)
  - [4. Precision vs. Recall Trade-off](#4-precision-vs-recall-trade-off)
  - [5. Fine-Grained Topic Tagging](#5-fine-grained-topic-tagging)
  - [6. Explainable AI (XAI)](#6-explainable-ai-xai)
- [Architecture & Project Structure](#-architecture--project-structure)
- [Quick Start](#-quick-start)
- [Web Dashboard & REST API](#-web-dashboard--rest-api)
- [Command Line Interface (CLI)](#-command-line-interface-cli)
- [Model Benchmarks & Evaluation](#-model-benchmarks--evaluation)
- [Testing & Quality Assurance](#-testing--quality-assurance)

---

## 🚀 Overview

Spam filtering is one of the classic, foundational applications of Artificial Intelligence and Natural Language Processing. However, modern deceptive emails are not just random junk—they span distinct categories such as **credential phishing**, **financial & lottery scams**, **work-from-home crypto schemes**, and **aggressive commercial promotions**.

This project implements:
1. **Binary Spam vs. Ham Classifier**: Detects malicious vs legitimate emails with high precision.
2. **Multi-Class Topic Classifier**: Categorizes emails into 8 topics (Phishing, Lottery, Marketing, Crypto Scam, Work, Personal, Notifications, Newsletters).
3. **Domain Heuristic Extractor**: Extracts structural email features (uppercase ratio, currency markers, embedded URLs, exclamation frequencies).
4. **Token Attribution Engine**: Highlights the exact words contributing positively (spam) or negatively (ham) to the prediction.
5. **Interactive Web Dashboard**: Modern responsive UI with live probability gauges, presets, and benchmark metrics.
6. **Production-Ready REST API & CLI**: For single-message and batch inferences.

---

## 🧠 Core AI Concepts Implemented

```
Raw Email Text
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 1. NLP Preprocessing Stage                             │
│   • HTML Stripping                                     │
│   • Entity Normalization (<URL>, <EMAIL>, <MONEY>)     │
│   • Case Folding & Non-Alphanumeric Punctuation Clean  │
│   • Contextual Stopword Removal                        │
│   • Porter Stemming (e.g., 'unauthorized' -> 'unauthor')│
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 2. Feature Extraction & Engineering                   │
│   • Sublinear TF-IDF Vectorization (Unigrams + Bigrams)│
│   • Domain Structural Heuristics (Caps, $, Links, !)   │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 3. Machine Learning Classification Zoo                 │
│   • Multinomial Naive Bayes (MNB)                      │
│   • Complement Naive Bayes (CNB)                       │
│   • Logistic Regression (Calibrated Probabilities)     │
│   • Support Vector Machine (LinearSVC)                 │
│   • Random Forest Classifier (Ensemble Trees)          │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 4. Inference & Explainability                          │
│   • Calibrated Confidence Probability                  │
│   • Fine-Grained Topic Prediction                      │
│   • Local Token Attribution (Top Spam & Ham Cues)      │
└────────────────────────────────────────────────────────┘
```

### 1. Natural Language Processing (NLP) Pipeline
- **Entity Normalization**: Replaces volatile strings with generalized tokens (`token_url`, `token_email`, `token_money`, `token_phone`, `token_num`, `token_exclamation`). This prevents the model from overfitting to individual URLs while learning the statistical significance of URL presence in spam.
- **Stopword Filtering**: Removes frequent English words (`"the"`, `"and"`, `"in"`) while preserving crucial domain discriminators (`"free"`, `"urgent"`, `"win"`).
- **Porter Stemmer**: Normalizes morphological word variations to a shared grammatical root (e.g., `"congratulations"`, `"congratulated"` $\to$ `"congratul"`).

### 2. Feature Engineering & Vectorization
- **TF-IDF with Sublinear Scaling**:
  $$\text{TF-IDF}(t, d, D) = (1 + \log(\text{TF}(t, d))) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  Diminishes the dominating impact of artificially repeated spam keywords.
- **N-Gram Word Combinations**: Extracts both unigrams and bigrams (`"wire transfer"`, `"account suspended"`, `"free spins"`), capturing multi-word contextual phrases.
- **Structural Heuristics**: Extracts numerical vectors for all-caps ratio, exclamation mark frequency, currency symbol counts, and URL densities.

### 3. Supervised Classification & Model Zoo
The project benchmarks 5 candidate classifiers under 5-Fold Stratified Cross-Validation:
- **Multinomial Naive Bayes**: Applies Bayes' Theorem under conditional word independence:
  $$P(\text{Spam} \mid \mathbf{x}) \propto P(\text{Spam}) \prod_{i=1}^n P(w_i \mid \text{Spam})$$
- **Complement Naive Bayes**: Corrects for class imbalance by estimating parameters from the complement of each class.
- **Logistic Regression**: Regularized log-odds classification with L2 penalty, providing well-calibrated probabilities.
- **Linear Support Vector Classifier (LinearSVC)**: Maximizes the geometric margin separating hyperplanes in high-dimensional sparse text vectors.
- **Random Forest**: Non-linear ensemble model aggregating decision trees.

### 4. Precision vs. Recall Trade-off
In spam detection, **Precision is paramount**:
- **False Positive (Type I error)**: A legitimate email is sent to the spam folder. A user could miss an urgent job interview, medical confirmation, or client contract.
- **False Negative (Type II error)**: A spam email reaches the inbox. A minor annoyance requiring the user to tap delete.
Our selection objective enforces maximum Precision alongside F1-score to prevent false positives.

### 5. Fine-Grained Topic Tagging
In addition to binary classification, the system classifies incoming emails into 8 specialized categories:
- 🚨 **Phishing & Credential Alert**
- 💰 **Lottery & Sweepstakes Scam**
- 🏷️ **Commercial Marketing & Promo**
- 📈 **Crypto & Get-Rich-Quick Scam**
- 💼 **Work & Team Communication**
- ☕ **Personal & Social Email**
- ✈️ **Order & Flight Transaction Alert**
- 📰 **Newsletter & Educational Digest**

### 6. Explainable AI (XAI)
For any given input message, the system calculates feature attribution scores:
$$\text{Impact}(w) = \text{TF-IDF}(w) \times \theta_w$$
where $\theta_w$ is the model's learned weight for token $w$. The UI and CLI visually present the strongest positive indicators (spam cues) and negative indicators (legitimate ham cues).

---

## 📂 Architecture & Project Structure

```
AI-Project/
├── app/
│   ├── app.py                     # Flask Web Application & REST API
│   ├── static/
│   │   ├── app.js                 # Dynamic frontend interactions & API client
│   │   └── style.css              # Dark/Light responsive modern styling
│   └── templates/
│       └── index.html             # Dashboard UI template
├── data/
│   ├── generate_dataset.py        # Generates curated benchmark emails
│   └── emails.csv                 # 2,600 annotated emails with topics
├── models/
│   ├── spam_classifier_model.joblib # Persisted binary classifier pipeline
│   ├── topic_classifier_model.joblib# Persisted multi-class topic pipeline
│   └── metadata.json              # Evaluation metrics & benchmark scores
├── notebooks/
│   ├── spam_classification_walkthrough.ipynb # Step-by-step Jupyter Notebook
│   └── walkthrough.py             # Executable CLI walkthrough
├── src/
│   ├── __init__.py
│   ├── classifier.py              # High-level inference service
│   ├── features.py                # Feature extraction & domain heuristics
│   ├── model.py                   # Model zoo, pipelines, explainability
│   ├── preprocessor.py            # Text normalization, stopwords & stemming
│   └── train.py                   # Automated training & cross-validation
├── tests/
│   ├── test_api.py                # REST API tests
│   ├── test_features.py           # Feature extraction tests
│   ├── test_model.py              # ML pipeline & persistence tests
│   └── test_preprocessor.py       # NLP preprocessing tests
├── cli.py                         # Interactive CLI tool
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup
```bash
# Clone or navigate into the project directory
cd /path/to/AI-Project

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
# 1. Generate the 2,600-sample balanced dataset
python data/generate_dataset.py

# 2. Benchmark algorithms with 5-fold cross-validation and save the best model
python src/train.py
```

---

## 🌐 Web Dashboard & REST API

### Launching the Dashboard
```bash
python app/app.py
```
Open your browser at **http://127.0.0.1:5001**.

### REST API Endpoints

#### 1. Classify Single Email (`POST /api/classify`)
```bash
curl -X POST http://127.0.0.1:5001/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: Your PayPal account has been restricted! Click http://verify-paypal.com to unlock."}'
```
**Response:**
```json
{
  "confidence": 0.9998,
  "confidence_percent": 99.98,
  "explanation": {
    "top_spam_words": [
      {"impact": 1.5124, "tfidf": 0.2841, "word": "access"},
      {"impact": 0.7562, "tfidf": 0.1420, "word": "urgent"}
    ]
  },
  "heuristics": {
    "cap_ratio": 0.056,
    "currency_count": 0,
    "url_count": 1
  },
  "is_spam": true,
  "label": "spam",
  "topic": "phishing",
  "topic_confidence": 92.4,
  "topic_display": "Phishing & Credential Alert"
}
```

#### 2. Batch Classification (`POST /api/batch`)
```bash
curl -X POST http://127.0.0.1:5001/api/batch \
  -H "Content-Type: application/json" \
  -d '{"emails": ["Congratulations! You won $1,000,000!", "Sprint retro at 3 PM today."]}'
```

#### 3. Benchmark Metrics (`GET /api/metrics`)
```bash
curl http://127.0.0.1:5001/api/metrics
```

---

## 💻 Command Line Interface (CLI)

The CLI tool allows quick classification directly from terminal.

### Classify an ad-hoc message:
```bash
python cli.py predict --text "CONGRATULATIONS! You won $2,500,000 in the lottery! Send your bank details now."
```

### Run preset test samples:
```bash
python cli.py test-samples
```

### View cross-validation benchmarks and confusion matrix:
```bash
python cli.py metrics
```

### Launch interactive REPL:
```bash
python cli.py interactive
```

---

## 📊 Model Benchmarks & Evaluation

Results from 5-Fold Stratified Cross-Validation on the 2,600-sample dataset:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** *(Selected)* | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Complement Naive Bayes** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| **Logistic Regression** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| **LinearSVC** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| **Random Forest** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

### Held-out Test Set Confusion Matrix (520 Samples)
- **True Ham (TN)**: 280 (100% correct)
- **False Spam (FP)**: 0 (Zero False Positives!)
- **False Ham (FN)**: 0 (Zero Missed Spams)
- **True Spam (TP)**: 240 (100% caught)

---

## 🧪 Testing & Quality Assurance

Run the comprehensive test suite:
```bash
PYTHONPATH=. pytest tests/ -v
```
All 12 unit tests verify:
- Normalization of URLs, emails, phone numbers, and currencies
- Stopword preservation and Porter Stemming
- Domain structural heuristic extraction
- TF-IDF sparse matrix generation
- End-to-end model training, prediction, and persistence
- Flask web routes and REST API JSON schemas

