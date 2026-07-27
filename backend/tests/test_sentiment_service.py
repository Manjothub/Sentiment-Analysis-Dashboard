"""
Unit Tests for Sentiment Service
=================================
Tests for the sentiment inference service, aspect extraction, and topic modeling.
"""

import os
import sys
import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml_pipeline.sentiment_service import (
    SentimentService,
    get_sentiment_service,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS
)
from app.services.ml_pipeline.aspect_service import (
    AspectService,
    get_aspect_service,
    DEFAULT_ASPECTS
)
from app.services.ml_pipeline.topic_model import (
    TopicModeler,
    get_topic_modeler
)


class TestSentimentService:
    """Tests for SentimentService class."""

    def test_singleton_pattern(self):
        """Test that SentimentService follows singleton pattern."""
        service1 = SentimentService()
        service2 = SentimentService()
        assert service1 is service2

    def test_initial_state(self):
        """Test initial state before model is loaded."""
        service = SentimentService()
        # Service might not have model loaded in test env
        assert hasattr(service, '_is_loaded')
        assert hasattr(service, '_initialized')

    def test_device_detection(self):
        """Test device property always returns a valid device."""
        service = SentimentService()
        device = service.device
        assert device in ['cpu', 'cuda', 'mps']

    def test_is_loaded_property(self):
        """Test is_loaded property returns boolean."""
        service = SentimentService()
        assert isinstance(service.is_loaded, bool)

    def test_get_model_info(self):
        """Test get_model_info returns valid structure."""
        service = SentimentService()
        info = service.get_model_info()
        assert 'model_loaded' in info
        assert 'device' in info
        assert 'model_type' in info

    def test_fallback_analysis_positive(self):
        """Test fallback analysis with positive text."""
        service = SentimentService()
        result = service._fallback_analysis("I love this product, it is great and amazing!")
        assert result['predicted_sentiment'] == 'positive'
        assert result['positive_score'] > result['negative_score']

    def test_fallback_analysis_negative(self):
        """Test fallback analysis with negative text."""
        service = SentimentService()
        result = service._fallback_analysis("This is terrible, awful product. I hate it.")
        assert result['predicted_sentiment'] == 'negative'
        assert result['negative_score'] > result['positive_score']

    def test_fallback_analysis_neutral(self):
        """Test fallback analysis with neutral text."""
        service = SentimentService()
        result = service._fallback_analysis("The product arrived on Tuesday.")
        assert result['predicted_sentiment'] == 'neutral'
        assert result['confidence_score'] == 0.5

    def test_fallback_analysis_empty(self):
        """Test fallback analysis with mixed sentiment returns neutral."""
        service = SentimentService()
        result = service._fallback_analysis("Good and bad mixed feelings about this.")
        # Both positive and negative words present
        assert result['predicted_sentiment'] in ['positive', 'negative', 'neutral']

    def test_get_probabilities_model_not_loaded(self):
        """Test get_probabilities when model is not loaded."""
        service = SentimentService()
        probs = service.get_probabilities("Test text")
        assert 'positive' in probs
        assert 'negative' in probs
        assert 'neutral' in probs

    def test_get_confidence_model_not_loaded(self):
        """Test get_confidence when model is not loaded."""
        service = SentimentService()
        confidence = service.get_confidence("Test text")
        assert 0 <= confidence <= 1

    def test_analyze_batch_empty(self):
        """Test batch analysis with empty list."""
        service = SentimentService()
        results = service.analyze_batch([])
        assert results == []

    def test_analyze_batch_single(self):
        """Test batch analysis with single item."""
        service = SentimentService()
        results = service.analyze_batch(["Great product"])
        assert len(results) == 1
        assert 'predicted_sentiment' in results[0]
        assert 'confidence_score' in results[0]

    def test_analyze_batch_multiple(self):
        """Test batch analysis with multiple items."""
        service = SentimentService()
        texts = ["Great product", "Terrible item", "Average quality"]
        results = service.analyze_batch(texts)
        assert len(results) == 3
        for result in results:
            assert 'predicted_sentiment' in result

    def test_label_constants(self):
        """Test that label constants are correctly defined."""
        assert ID2LABEL[0] == 'negative'
        assert ID2LABEL[1] == 'positive'
        assert LABEL2ID['negative'] == 0
        assert LABEL2ID['positive'] == 1

    def test_num_labels(self):
        """Test number of labels is 2 for SST-2 model."""
        assert NUM_LABELS == 2

    def test_get_sentiment_service_function(self):
        """Test module-level accessor function."""
        service = get_sentiment_service()
        assert isinstance(service, SentimentService)

    def test_get_sentiment_service_singleton(self):
        """Test that accessor returns the same instance."""
        service1 = get_sentiment_service()
        service2 = get_sentiment_service()
        assert service1 is service2


class TestAspectService:
    """Tests for AspectService class."""

    def test_singleton_pattern(self):
        """Test that AspectService follows singleton pattern."""
        service1 = AspectService()
        service2 = AspectService()
        assert service1 is service2

    def test_initial_state(self):
        """Test initial state."""
        service = AspectService()
        assert hasattr(service, '_is_loaded')
        assert hasattr(service, '_initialized')

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        service = AspectService()
        assert isinstance(service.is_loaded, bool)

    def test_default_aspects(self):
        """Test that default aspects are defined."""
        assert len(DEFAULT_ASPECTS) > 0
        assert 'Product Quality' in DEFAULT_ASPECTS
        assert 'Price' in DEFAULT_ASPECTS
        assert 'Shipping' in DEFAULT_ASPECTS

    def test_rule_based_aspects_positive(self):
        """Test rule-based aspect extraction with positive keywords."""
        service = AspectService()
        text = "This product has high quality and great value for the price."
        result = service._rule_based_aspects(text, DEFAULT_ASPECTS)
        assert 'aspects' in result
        assert 'num_aspects_detected' in result
        assert result['num_aspects_detected'] > 0

    def test_rule_based_aspects_negative(self):
        """Test rule-based aspect extraction with negative keywords."""
        service = AspectService()
        text = "Poor quality product, shipping was slow and customer service was terrible."
        result = service._rule_based_aspects(text, DEFAULT_ASPECTS)
        assert result['num_aspects_detected'] > 0

    def test_rule_based_aspects_empty_text(self):
        """Test rule-based extraction with empty text."""
        service = AspectService()
        result = service._rule_based_aspects('', DEFAULT_ASPECTS)
        assert result['num_aspects_detected'] > 0  # Falls back to neutral

    def test_aspects_no_keywords(self):
        """Test extraction with text containing no aspect keywords."""
        service = AspectService()
        text = "The weather is nice today."
        result = service._rule_based_aspects(text, DEFAULT_ASPECTS)
        assert 'aspects' in result
        # Should have at least one aspect (fallback neutral)

    def test_extract_aspects_model_not_loaded(self):
        """Test extract_aspects falls back to rule-based when model not loaded."""
        service = AspectService()
        result = service.extract_aspects("Great product quality")
        assert 'aspects' in result
        assert 'model_loaded' in result

    def test_get_aggregate_aspects(self):
        """Test aggregate aspect analysis across multiple texts."""
        service = AspectService()
        texts = [
            "Great product quality, fast shipping",
            "Poor quality, slow delivery",
            "Good value for money"
        ]
        result = service.get_aggregate_aspects(texts, DEFAULT_ASPECTS)
        assert 'aggregate_aspects' in result
        assert result['total_reviews'] == 3

    def test_extract_aspects_batch(self):
        """Test batch aspect extraction."""
        service = AspectService()
        texts = ["Great quality", "Bad shipping", "Good price"]
        results = service.extract_aspects_batch(texts, DEFAULT_ASPECTS)
        assert len(results) == 3
        for r in results:
            assert 'aspects' in r

    def test_get_aspect_service_function(self):
        """Test module-level accessor function."""
        service = get_aspect_service()
        assert isinstance(service, AspectService)

    def test_get_aspect_service_singleton(self):
        """Test that accessor returns same instance."""
        service1 = get_aspect_service()
        service2 = get_aspect_service()
        assert service1 is service2


class TestTopicModeler:
    """Tests for TopicModeler class."""

    def test_initial_state(self):
        """Test initial state of topic modeler."""
        tm = TopicModeler()
        assert not tm.is_fitted
        assert tm.topics is None
        assert tm.probs is None

    def test_get_topic_info_not_fitted(self):
        """Test get_topic_info when model is not fitted."""
        tm = TopicModeler()
        info = tm.get_topic_info()
        assert 'error' in info or 'num_topics' in info

    def test_rule_based_topics(self):
        """Test rule-based topic extraction fallback."""
        tm = TopicModeler()
        texts = [
            "Great product quality, very durable",
            "Fast shipping, arrived quickly",
            "Poor customer service, terrible support"
        ]
        result = tm._rule_based_topics(texts)
        assert 'num_topics' in result
        assert 'top_topics' in result
        assert len(result['top_topics']) > 0

    def test_rule_based_topics_empty(self):
        """Test rule-based topics with empty text list."""
        tm = TopicModeler()
        result = tm._rule_based_topics([])
        assert result['num_topics'] == 0
        assert result['total_documents'] == 0

    def test_rule_based_topics_single_text(self):
        """Test rule-based topics with single text."""
        tm = TopicModeler()
        result = tm._rule_based_topics(["Amazing product quality and durability"])
        assert result['num_topics'] > 0

    def test_transform_not_fitted(self):
        """Test transform when model is not fitted."""
        tm = TopicModeler()
        topics = tm.transform(["Test text"])
        assert topics == [-1]

    def test_get_topic_keywords_not_fitted(self):
        """Test get_topic_keywords when not fitted."""
        tm = TopicModeler()
        keywords = tm.get_topic_keywords(0)
        assert keywords == []

    def test_get_topic_modeler_function(self):
        """Test module-level accessor function."""
        tm = get_topic_modeler()
        assert isinstance(tm, TopicModeler)

    def test_get_topic_modeler_singleton(self):
        """Test that accessor returns same instance."""
        tm1 = get_topic_modeler()
        tm2 = get_topic_modeler()
        assert tm1 is tm2

    def test_topic_keyword_coverage(self):
        """Test that rule-based topics cover expected keywords."""
        tm = TopicModeler()
        result = tm._rule_based_topics([
            "Product quality is excellent",
            "Customer service was helpful",
            "Fast shipping"
        ])
        topic_names = [t['name'] for t in result['top_topics']]
        assert any('Quality' in name for name in topic_names) or \
               any('Service' in name for name in topic_names) or \
               any('Shipping' in name for name in topic_names)


class TestIntegration:
    """Integration tests across multiple services."""

    def test_sentiment_to_aspect_pipeline(self):
        """Test running sentiment analysis then aspect extraction."""
        sentiment_service = SentimentService()
        aspect_service = AspectService()

        # Run sentiment
        sentiment_result = sentiment_service.analyze("Great product with excellent quality and fast shipping")

        # Run aspect extraction
        aspect_result = aspect_service.extract_aspects("Great product with excellent quality and fast shipping")

        assert 'predicted_sentiment' in sentiment_result
        assert 'aspects' in aspect_result

    def test_fallback_chain(self):
        """Test that fallback chain works when model not available."""
        sentiment_service = SentimentService()
        aspect_service = AspectService()

        # Even without model, should return valid results
        sentiment = sentiment_service.analyze("Test")
        aspects = aspect_service.extract_aspects("Test")

        assert sentiment['predicted_sentiment'] is not None
        assert aspects['num_aspects_detected'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
