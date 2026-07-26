"""
ML Pipeline Package
===================
Provides machine learning pipeline components for DistilBERT fine-tuning,
sentiment inference, aspect extraction, and topic modeling.
"""

from .prepare_dataset import prepare_dataset, create_dataset_dict
from .train_model import train_distilbert
from .evaluate_model import evaluate_model
from .sentiment_service import SentimentService
from .aspect_service import AspectService
from .topic_model import TopicModeler

__all__ = [
    'prepare_dataset',
    'create_dataset_dict',
    'train_distilbert',
    'evaluate_model',
    'SentimentService',
    'AspectService',
    'TopicModeler',
]
