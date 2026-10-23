"""
Spam Email Classifier NLP & ML Package.
"""

from src.preprocessor import TextPreprocessor
from src.features import DomainFeatureExtractor, build_tfidf_vectorizer
from src.model import SpamClassifierPipeline, TopicClassifierPipeline

__all__ = [
    "TextPreprocessor",
    "DomainFeatureExtractor",
    "build_tfidf_vectorizer",
    "SpamClassifierPipeline",
    "TopicClassifierPipeline",
]

