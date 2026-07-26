"""
Aspect-Based Sentiment Analysis Module
=======================================
Uses Zero-Shot Classification with facebook/bart-large-mnli to extract
aspect-specific sentiments from product reviews.

Candidate aspects:
- Product Quality
- Shipping
- Packaging
- Battery
- Price
- Customer Service
- Performance
- Durability

Returns JSON with aspect -> sentiment mapping.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

import torch
from transformers import pipeline

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
MODEL_NAME = 'facebook/bart-large-mnli'
MAX_CHUNK_LENGTH = 512

# Default aspect candidates for product reviews
DEFAULT_ASPECTS = [
    "Product Quality",
    "Shipping",
    "Packaging",
    "Battery",
    "Price",
    "Customer Service",
    "Performance",
    "Durability"
]

# Sentiment labels for zero-shot classification
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


class AspectService:
    """
    Aspect-based sentiment analysis using zero-shot classification.
    Singleton pattern - loads the model once.
    """

    _instance = None
    _classifier = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load the zero-shot classification model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading zero-shot model: {MODEL_NAME} on device {device}")

            self._classifier = pipeline(
                'zero-shot-classification',
                model=MODEL_NAME,
                device=device
            )

            self._is_loaded = True
            logger.info("Zero-shot model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load zero-shot model: {e}")
            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and self._classifier is not None

    def extract_aspects(
        self,
        text: str,
        aspects: Optional[List[str]] = None,
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Extract aspect-based sentiments from a review text.

        Args:
            text: Review text to analyze
            aspects: List of aspect candidates (uses defaults if None)
            confidence_threshold: Minimum confidence for aspect detection

        Returns:
            Dictionary with aspect sentiments and metadata
        """
        start_time = time.time()

        if aspects is None:
            aspects = DEFAULT_ASPECTS

        if not self._is_loaded or not text.strip():
            return self._rule_based_aspects(text, aspects)

        try:
            # Truncate text to model's max length
            truncated_text = text[:MAX_CHUNK_LENGTH]

            # Classify each aspect against the text
            aspect_results = {}
            aspect_scores = {}

            # Batch process aspects for efficiency
            for aspect in aspects:
                result = self._classifier(
                    truncated_text,
                    candidate_labels=SENTIMENT_LABELS,
                    hypothesis_template=f"This aspect is {{}}."
                )

                # Get the sentiment with highest score
                scores = result['scores']
                labels = result['labels']
                max_idx = scores.index(max(scores))
                sentiment = labels[max_idx]
                confidence = scores[max_idx]

                if confidence >= confidence_threshold:
                    aspect_results[aspect.lower().replace(' ', '_')] = sentiment
                    aspect_scores[aspect.lower().replace(' ', '_')] = round(confidence, 4)

            inference_time = int((time.time() - start_time) * 1000)

            return {
                'aspects': aspect_results if aspect_results else self._rule_based_aspects(text, aspects)['aspects'],
                'aspect_scores': aspect_scores,
                'inference_time_ms': inference_time,
                'model_loaded': self._is_loaded,
                'num_aspects_detected': len(aspect_results),
                'total_aspects': len(aspects)
            }

        except Exception as e:
            logger.error(f"Aspect extraction error: {e}")
            return self._rule_based_aspects(text, aspects)

    def extract_aspects_batch(
        self,
        texts: List[str],
        aspects: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract aspect sentiments for multiple texts.

        Args:
            texts: List of review texts
            aspects: List of aspect candidates

        Returns:
            List of aspect extraction results
        """
        return [self.extract_aspects(text, aspects) for text in texts]

    def _rule_based_aspects(
        self,
        text: str,
        aspects: List[str]
    ) -> Dict[str, Any]:
        """
        Fallback rule-based aspect extraction when model is not available.

        Args:
            text: Review text
            aspects: List of aspect candidates

        Returns:
            Dictionary with rule-based aspect sentiments
        """
        text_lower = text.lower()
        aspect_results = {}

        # Positive and negative keywords for each aspect
        aspect_keywords = {
            'product_quality': {
                'positive': ['high quality', 'well made', 'durable', 'sturdy', 'solid'],
                'negative': ['poor quality', 'cheap', 'flimsy', 'broken', 'defective']
            },
            'shipping': {
                'positive': ['fast shipping', 'quick delivery', 'arrived early', 'well packaged'],
                'negative': ['slow shipping', 'late delivery', 'damaged', 'lost package']
            },
            'packaging': {
                'positive': ['well packaged', 'secure', 'protected', 'nice packaging'],
                'negative': ['poor packaging', 'damaged box', 'broken package']
            },
            'battery': {
                'positive': ['long battery', 'good battery', 'battery lasts'],
                'negative': ['short battery', 'battery dies', 'poor battery']
            },
            'price': {
                'positive': ['great price', 'good value', 'affordable', 'worth'],
                'negative': ['overpriced', 'too expensive', 'not worth', 'overpaid']
            },
            'customer_service': {
                'positive': ['great service', 'helpful support', 'excellent customer'],
                'negative': ['poor service', 'bad support', 'terrible customer', 'rude']
            },
            'performance': {
                'positive': ['works great', 'excellent performance', 'fast', 'smooth'],
                'negative': ['poor performance', 'slow', 'laggy', 'does not work']
            },
            'durability': {
                'positive': ['long lasting', 'durable', 'built to last', 'strong'],
                'negative': ['broke quickly', 'falls apart', 'cheap material', 'worn out']
            }
        }

        for aspect_key, keywords in aspect_keywords.items():
            pos_count = sum(1 for kw in keywords['positive'] if kw in text_lower)
            neg_count = sum(1 for kw in keywords['negative'] if kw in text_lower)

            if pos_count > neg_count:
                aspect_results[aspect_key] = 'positive'
            elif neg_count > pos_count:
                aspect_results[aspect_key] = 'negative'
            elif pos_count > 0 or neg_count > 0:
                aspect_results[aspect_key] = 'neutral'

        # If no aspects detected, mark first few as neutral
        if not aspect_results:
            for aspect in aspects[:3]:
                key = aspect.lower().replace(' ', '_')
                aspect_results[key] = 'neutral'

        return {
            'aspects': aspect_results,
            'aspect_scores': {k: 0.5 for k in aspect_results},
            'inference_time_ms': 0,
            'model_loaded': False,
            'num_aspects_detected': len(aspect_results),
            'total_aspects': len(aspects)
        }

    def get_aggregate_aspects(
        self,
        texts: List[str],
        aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate aspect sentiments across multiple reviews.

        Args:
            texts: List of review texts
            aspects: List of aspect candidates

        Returns:
            Dictionary with aggregated aspect statistics
        """
        if aspects is None:
            aspects = DEFAULT_ASPECTS

        aspect_keys = [a.lower().replace(' ', '_') for a in aspects]

        # Count sentiments per aspect
        sentiment_counts = {
            key: {'positive': 0, 'negative': 0, 'neutral': 0}
            for key in aspect_keys
        }

        all_results = self.extract_aspects_batch(texts, aspects)

        for result in all_results:
            for aspect_key, sentiment in result['aspects'].items():
                if aspect_key in sentiment_counts:
                    if sentiment in sentiment_counts[aspect_key]:
                        sentiment_counts[aspect_key][sentiment] += 1

        # Calculate percentages
        total_reviews = len(texts)
        aggregate = {}
        for aspect_key, counts in sentiment_counts.items():
            total_detected = sum(counts.values())
            if total_detected > 0:
                aggregate[aspect_key] = {
                    'positive_pct': round(counts['positive'] / total_detected * 100, 1),
                    'negative_pct': round(counts['negative'] / total_detected * 100, 1),
                    'neutral_pct': round(counts['neutral'] / total_detected * 100, 1),
                    'total_mentions': total_detected,
                    'coverage': round(total_detected / total_reviews * 100, 1)
                }

        return {
            'aggregate_aspects': aggregate,
            'total_reviews': total_reviews,
            'total_aspects': len(aspect_keys)
        }


# Module-level singleton accessor
_aspect_service_instance = None


def get_aspect_service() -> AspectService:
    """Get or create the singleton AspectService instance."""
    global _aspect_service_instance
    if _aspect_service_instance is None:
        _aspect_service_instance = AspectService()
    return _aspect_service_instance
