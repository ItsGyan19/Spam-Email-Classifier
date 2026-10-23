"""
Text Preprocessing Module for Spam Email Classification.
Covers core NLP stages:
1. Cleaning (HTML stripping, normalization of URLs, emails, phone numbers, currency)
2. Tokenization & Lowercasing
3. Stopword Removal
4. Stemming (Porter Stemmer) & Lemmatization
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize NLTK components with safe fallbacks
# Standard English Stopwords list (NLTK compatible + self-contained)
DEFAULT_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "nor",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn"
}

STOPWORDS = DEFAULT_STOPWORDS

# Important words in spam detection that shouldn't be blindly discarded
STOPWORDS_TO_KEEP = {"free", "urgent", "win", "won", "no", "not", "only", "now"}
EFFECTIVE_STOPWORDS = STOPWORDS - STOPWORDS_TO_KEEP

# Regex patterns for normalization
URL_REGEX = re.compile(r"https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(?:com|org|net|biz|xyz|info|cc|shop|me|io)\b\S*", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CURRENCY_REGEX = re.compile(r"[\$£€¥₹]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:dollars?|usd|eur|gbp|btc|bitcoin)\b", re.IGNORECASE)
NUMBER_REGEX = re.compile(r"\b\d+\b")
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
MULTIPLE_SPACES_REGEX = re.compile(r"\s+")
MULTIPLE_EXCLAMATION_REGEX = re.compile(r"!{2,}")


class TextPreprocessor:
    """
    Robust NLP Preprocessor for Email Text.
    Handles raw text, strips noise, tokenizes, removes stopwords, and stems words.
    """

    def __init__(self, use_stemming: bool = True, use_lemmatization: bool = False):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes email text into standardized tokens.
        """
        if not isinstance(text, str):
            return ""

        # 1. Strip HTML tags
        text = HTML_TAG_REGEX.sub(" ", text)

        # 2. Normalize Email addresses first (so domain in user@domain.com is captured as email)
        text = EMAIL_REGEX.sub(" token_email ", text)

        # 3. Normalize URLs
        text = URL_REGEX.sub(" token_url ", text)

        # 4. Normalize Currency
        text = CURRENCY_REGEX.sub(" token_money ", text)

        # 5. Normalize Phone numbers
        text = PHONE_REGEX.sub(" token_phone ", text)

        # 6. Normalize generic numbers
        text = NUMBER_REGEX.sub(" token_num ", text)

        # 7. Normalize repeated exclamation marks
        text = MULTIPLE_EXCLAMATION_REGEX.sub(" token_exclamation ", text)

        # 8. Convert to lowercase
        text = text.lower()

        # 9. Remove punctuation (excluding underscores used in tokens)
        punct_to_remove = string.punctuation.replace("_", "")
        text = text.translate(str.maketrans("", "", punct_to_remove))

        # 10. Collapse extra whitespace
        text = MULTIPLE_SPACES_REGEX.sub(" ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into words.
        """
        cleaned = self.clean_text(text)
        return cleaned.split()

    def process(self, text: str) -> str:
        """
        Full NLP preprocessing pipeline:
        Cleaning -> Tokenizing -> Stopword removal -> Stemming/Lemmatizing -> String rebuild
        """
        tokens = self.tokenize(text)

        # Remove stopwords (preserving spam-relevant indicators)
        filtered_tokens = [tok for tok in tokens if tok not in EFFECTIVE_STOPWORDS and len(tok) > 1]

        # Apply Stemming or Lemmatization
        if self.use_stemming:
            processed_tokens = [self.stemmer.stem(tok) for tok in filtered_tokens]
        elif self.use_lemmatization:
            processed_tokens = [self.lemmatizer.lemmatize(tok) for tok in filtered_tokens]
        else:
            processed_tokens = filtered_tokens

        return " ".join(processed_tokens)

    def __call__(self, text: str) -> str:
        """Enables direct callable interface for sklearn compatibility."""
        return self.process(text)


def preprocess_corpus(corpus: List[str], use_stemming: bool = True) -> List[str]:
    """Helper to preprocess an entire list/series of texts."""
    preprocessor = TextPreprocessor(use_stemming=use_stemming)
    return [preprocessor.process(doc) for doc in corpus]
