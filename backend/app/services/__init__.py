"""
Services Package
================
Exports all service modules for the application.

Services:
    - nlp_service: Original NLP service with sentiment analysis
    - ml_pipeline: ML pipeline components (training, evaluation, inference)
    - alert_service: Alert generation for sentiment changes
"""

from app.services.nlp_service import SentimentAnalyzer

# ML Pipeline services
from app.services.ml_pipeline import (
    prepare_dataset,
    create_dataset_dict,
    train_distilbert,
    evaluate_model,
    SentimentService,
    AspectService,
    TopicModeler,
)

__all__ = [
    'SentimentAnalyzer',
    'prepare_dataset',
    'create_dataset_dict',
    'train_distilbert',
    'evaluate_model',
    'SentimentService',
    'AspectService',
    'TopicModeler',
]
