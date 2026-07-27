"""
Sentiment Inference Service
============================
Singleton service for loading a trained DistilBERT model and running inference.
Supports single review prediction, batch prediction, probability scores,
confidence scores, latency measurement, GPU detection, and CPU fallback.

The service is registered as a Flask extension via current_app.extensions['model_service'].
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Pipeline,
    pipeline
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
NUM_LABELS = 2
ID2LABEL = {0: 'negative', 1: 'positive'}
LABEL2ID = {'negative': 0, 'positive': 1}

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_MODEL_DIR = DEFAULT_MODEL_NAME


class SentimentService:
    """
    Singleton sentiment analysis service using pretrained Hugging Face Transformers.
    Loads the model once at startup and reuses it across all requests.
    """

    _instance = None
    _model = None
    _tokenizer = None
    _classifier = None
    _device = None
    _is_loaded = False
    _model_path = None

    def __new__(cls, model_path: Optional[str] = None):
        """Singleton pattern - only one instance ever created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the sentiment service.

        Args:
            model_path: Path or HuggingFace model ID for sentiment model.
        """
        if self._initialized:
            return

        self._initialized = True
        self._model_path = model_path or DEFAULT_MODEL_NAME

        # Detect device
        self._detect_device()

        # Load model
        self.load_model()

        logger.info("SentimentService initialized")

    def _detect_device(self) -> None:
        """Detect available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = torch.device('mps')
            logger.info("Using Apple MPS")
        else:
            self._device = torch.device('cpu')
            logger.info("Using CPU")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the pretrained model and tokenizer.

        Args:
            model_path: Path or Hugging Face model identifier.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if model_path:
            self._model_path = model_path

        # If model_path is a local path that doesn't exist, fall back to DEFAULT_MODEL_NAME
        if self._model_path and not os.path.exists(self._model_path) and '/' not in self._model_path and '\\' in self._model_path:
            self._model_path = DEFAULT_MODEL_NAME

        try:
            logger.info(f"Loading pretrained model: {self._model_path}")

            # Load tokenizer and model from Hugging Face hub or local path
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path)

            # Move model to device
            self._model = self._model.to(self._device)
            self._model.eval()

            self._is_loaded = True
            logger.info("Model loaded successfully into memory.")
            return True

        except Exception as e:
            logger.error(f"Failed to load model '{self._model_path}': {e}")
            # Attempt fallback to default HF model if custom path failed
            if self._model_path != DEFAULT_MODEL_NAME:
                try:
                    logger.info(f"Attempting fallback to default Hugging Face model: {DEFAULT_MODEL_NAME}")
                    self._model_path = DEFAULT_MODEL_NAME
                    self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
                    self._model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME)
                    self._model = self._model.to(self._device)
                    self._model.eval()
                    self._is_loaded = True
                    logger.info("Fallback model loaded successfully.")
                    return True
                except Exception as fallback_err:
                    logger.error(f"Fallback model loading also failed: {fallback_err}")

            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded and self._model is not None

    @property
    def device(self) -> str:
        """Get current device string."""
        return self._device.type if self._device else 'cpu'

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using the pretrained Transformer model.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with prediction results (predicted_sentiment, positive_score, negative_score, neutral_score, confidence_score)
        """
        start_time = time.time()

        if not self.is_loaded:
            return self._fallback_analysis(text)

        if not text or not text.strip():
            return {
                'predicted_sentiment': 'neutral',
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'confidence_score': 1.0,
                'inference_time_ms': int((time.time() - start_time) * 1000),
                'model_loaded': True
            }

        try:
            inputs = self._tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

            # Handle 2-class output (SST-2: 0=NEGATIVE, 1=POSITIVE)
            if len(probabilities) == 2:
                neg_score = float(probabilities[0])
                pos_score = float(probabilities[1])
                diff = abs(pos_score - neg_score)

                if diff < 0.15:
                    predicted_sentiment = 'neutral'
                    confidence = round(1.0 - diff, 4)
                elif pos_score > neg_score:
                    predicted_sentiment = 'positive'
                    confidence = round(pos_score, 4)
                else:
                    predicted_sentiment = 'negative'
                    confidence = round(neg_score, 4)

                neu_score = round(max(0.0, 1.0 - diff), 4)
            else:
                pred_class = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_class])
                predicted_sentiment = ID2LABEL.get(pred_class, 'neutral')
                pos_score = float(probabilities[LABEL2ID.get('positive', 1)])
                neg_score = float(probabilities[LABEL2ID.get('negative', 0)])
                neu_score = 0.0

            result = {
                'predicted_sentiment': predicted_sentiment,
                'positive_score': round(pos_score, 4),
                'negative_score': round(neg_score, 4),
                'neutral_score': neu_score,
                'confidence_score': confidence,
                'inference_time_ms': int((time.time() - start_time) * 1000),
                'model_loaded': True
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            result = self._fallback_analysis(text)
            result['inference_error'] = str(e)

        return result

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of prediction result dictionaries
        """
        if not self.is_loaded:
            return [self._fallback_analysis(t) for t in texts]

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._analyze_batch_internal(batch)
            results.extend(batch_results)

        return results

    def _analyze_batch_internal(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Internal batch analysis method.

        Args:
            texts: Batch of texts

        Returns:
            List of prediction result dictionaries
        """
        start_time = time.time()

        try:
            # Tokenize batch
            inputs = self._tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding=True
            ).to(self._device)

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

            probs = probabilities.cpu().numpy()
            batch_time = (time.time() - start_time) * 1000
            per_item_time = batch_time / max(len(texts), 1)

            results = []
            for i, text in enumerate(texts):
                if len(probs[i]) == 2:
                    neg_score = float(probs[i][0])
                    pos_score = float(probs[i][1])
                    diff = abs(pos_score - neg_score)

                    if diff < 0.15:
                        predicted_sentiment = 'neutral'
                        confidence = round(1.0 - diff, 4)
                    elif pos_score > neg_score:
                        predicted_sentiment = 'positive'
                        confidence = round(pos_score, 4)
                    else:
                        predicted_sentiment = 'negative'
                        confidence = round(neg_score, 4)

                    neu_score = round(max(0.0, 1.0 - diff), 4)
                else:
                    pred_class = int(np.argmax(probs[i]))
                    confidence = float(probs[i][pred_class])
                    predicted_sentiment = ID2LABEL.get(pred_class, 'neutral')
                    pos_score = float(probs[i][LABEL2ID.get('positive', 1)])
                    neg_score = float(probs[i][LABEL2ID.get('negative', 0)])
                    neu_score = 0.0

                results.append({
                    'predicted_sentiment': predicted_sentiment,
                    'positive_score': round(pos_score, 4),
                    'negative_score': round(neg_score, 4),
                    'neutral_score': neu_score,
                    'confidence_score': round(confidence, 4),
                    'inference_time_ms': round(per_item_time, 2),
                    'model_loaded': True
                })

            return results

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return [self._fallback_analysis(t) for t in texts]

    def get_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get probability scores for each sentiment class.

        Args:
            text: Input text

        Returns:
            Dictionary with probability for each class
        """
        result = self.analyze(text)
        return {
            'positive': result['positive_score'],
            'negative': result['negative_score'],
            'neutral': result['neutral_score']
        }

    def get_confidence(self, text: str) -> float:
        """
        Get confidence score for a prediction.

        Args:
            text: Input text

        Returns:
            Confidence score between 0 and 1
        """
        result = self.analyze(text)
        return result['confidence_score']

    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback rule-based sentiment analysis when model is not available.

        Args:
            text: Input text

        Returns:
            Dictionary with fallback prediction
        """
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'perfect', 'best', 'awesome', 'happy', 'satisfied',
            'beautiful', 'outstanding', 'superb', 'brilliant', 'delightful'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor',
            'hate', 'disappointed', 'useless', 'waste', 'regret',
            'dreadful', 'appalling', 'abysmal', 'atrocious', 'dismal'
        }

        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        total = pos_count + neg_count

        if total == 0:
            return {
                'predicted_sentiment': 'neutral',
                'positive_score': 0.33,
                'negative_score': 0.33,
                'neutral_score': 0.34,
                'confidence_score': 0.5,
                'inference_time_ms': 0,
                'model_loaded': False
            }

        positive_ratio = pos_count / total
        if positive_ratio > 0.6:
            label = 'positive'
        elif positive_ratio < 0.4:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'predicted_sentiment': label,
            'positive_score': round(positive_ratio, 4),
            'negative_score': round(1 - positive_ratio, 4),
            'neutral_score': 0.0,
            'confidence_score': round(abs(positive_ratio - 0.5) * 2, 4),
            'inference_time_ms': 0,
            'model_loaded': False
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {
                'model_loaded': False,
                'model_path': self._model_path,
                'device': self.device,
                'model_type': 'DistilBERT'
            }

        return {
            'model_loaded': True,
            'model_path': self._model_path,
            'device': self.device,
            'model_type': 'DistilBERT',
            'num_labels': NUM_LABELS,
            'classes': list(ID2LABEL.values()),
            'id2label': ID2LABEL,
            'label2id': LABEL2ID,
            'max_length': 256
        }


# Module-level singleton accessor
_service_instance = None


def get_sentiment_service(model_path: Optional[str] = None) -> SentimentService:
    """
    Get or create the singleton SentimentService instance.

    Args:
        model_path: Path to trained model

    Returns:
        SentimentService singleton instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SentimentService(model_path)
    return _service_instance
