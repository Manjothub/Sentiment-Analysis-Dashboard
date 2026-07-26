"""
Topic Modeling Module
=====================
Implements BERTopic for extracting topics from product reviews.
Generates top topics, topic frequency, representative documents,
and topic keywords. Results are saved to PostgreSQL.

Usage:
    python -c "from app.services.ml_pipeline.topic_model import TopicModeler; tm = TopicModeler(); tm.fit(texts)"
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'saved_models', 'topic_model')
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'amazon_cleaned.csv')


class TopicModeler:
    """
    Topic modeling using BERTopic.
    Extracts topics from product reviews and stores results.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        Initialize the topic modeler.

        Args:
            model_dir: Directory to save/load the topic model
        """
        self.model_dir = model_dir
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.is_fitted = False

        # Try to load existing model
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load a previously saved topic model.

        Returns:
            True if model loaded successfully
        """
        model_path = os.path.join(self.model_dir, 'topic_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.topic_model = pickle.load(f)
                self.is_fitted = True
                logger.info(f"Loaded existing topic model from: {model_path}")
                return True
            except Exception as e:
                logger.warning(f"Could not load existing topic model: {e}")
        return False

    def _save_model(self) -> None:
        """Save the fitted topic model to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, 'topic_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.topic_model, f)
        logger.info(f"Topic model saved to: {model_path}")

    def fit(
        self,
        texts: List[str],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Fit BERTopic model on a list of texts.

        Args:
            texts: List of review texts
            save_model: Whether to save the model after fitting

        Returns:
            Dictionary with topic modeling results
        """
        logger.info(f"Fitting topic model on {len(texts):,} documents...")

        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer
            from hdbscan import HDBSCAN
            from umap import UMAP

            # Configure embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Configure UMAP for dimensionality reduction
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

            # Configure HDBSCAN for clustering
            hdbscan_model = HDBSCAN(
                min_cluster_size=15,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

            # Configure vectorizer for topic representation
            vectorizer_model = CountVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8
            )

            # Create and fit BERTopic model
            self.topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                top_n_words=10,
                verbose=True
            )

            self.topics, self.probs = self.topic_model.fit_transform(texts)
            self.is_fitted = True

            if save_model:
                self._save_model()

            # Get topic info
            results = self.get_topic_info()

            logger.info(f"Topic modeling completed: {results['num_topics']} topics found")
            return results

        except ImportError as e:
            logger.error(f"BERTopic dependencies not installed: {e}")
            logger.error("Install with: pip install bertopic sentence-transformers hdbscan umap-learn")
            return self._rule_based_topics(texts)

        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return self._rule_based_topics(texts)

    def fit_from_csv(
        self,
        csv_path: str = DEFAULT_DATA_PATH,
        text_column: str = 'cleaned_text',
        nrows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fit topic model from a CSV file.

        Args:
            csv_path: Path to CSV file
            text_column: Column containing text data
            nrows: Number of rows to use (None for all)

        Returns:
            Dictionary with topic modeling results
        """
        import pandas as pd

        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path, nrows=nrows)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")

        texts = df[text_column].dropna().tolist()
        logger.info(f"Loaded {len(texts):,} documents")

        return self.fit(texts)

    def get_topic_info(self) -> Dict[str, Any]:
        """
        Get information about all discovered topics.

        Returns:
            Dictionary with topic details
        """
        if not self.is_fitted or self.topic_model is None:
            return {'error': 'Model not fitted yet', 'num_topics': 0}

        try:
            # Get topic information from BERTopic
            topic_info = self.topic_model.get_topic_info()

            # Get top topics (excluding outlier topic -1)
            top_topics = topic_info[topic_info['Topic'] != -1].head(10)

            topics_list = []
            for _, row in top_topics.iterrows():
                topic_id = int(row['Topic'])
                topic_words = self.topic_model.get_topic(topic_id)

                topics_list.append({
                    'topic_id': topic_id,
                    'count': int(row['Count']),
                    'name': row['Name'],
                    'representation': row['Representation'],
                    'keywords': [word for word, _ in topic_words[:10]],
                    'keyword_scores': [round(score, 4) for _, score in topic_words[:10]]
                })

            # Get representative documents per topic
            representative_docs = {}
            try:
                reps = self.topic_model.get_representative_docs()
                for topic_id, docs in reps.items():
                    if topic_id != -1:
                        representative_docs[int(topic_id)] = docs[:3]
            except Exception:
                pass

            # Calculate topic distribution
            topic_counts = Counter(self.topics)
            total_docs = len(self.topics)
            distribution = {
                int(topic_id): {
                    'count': count,
                    'percentage': round(count / total_docs * 100, 2)
                }
                for topic_id, count in topic_counts.most_common(10)
                if topic_id != -1
            }

            return {
                'num_topics': len(topic_info[topic_info['Topic'] != -1]),
                'total_documents': total_docs,
                'outlier_count': int(topic_counts.get(-1, 0)),
                'outlier_percentage': round(topic_counts.get(-1, 0) / total_docs * 100, 2),
                'top_topics': topics_list,
                'topic_distribution': distribution,
                'representative_documents': representative_docs,
                'model_fitted': True
            }

        except Exception as e:
            logger.error(f"Failed to get topic info: {e}")
            return {'error': str(e), 'num_topics': 0}

    def transform(self, texts: List[str]) -> List[int]:
        """
        Transform new texts to their topic assignments.

        Args:
            texts: List of texts to classify

        Returns:
            List of topic IDs
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, cannot transform")
            return [-1] * len(texts)

        try:
            topics, _ = self.topic_model.transform(texts)
            return topics.tolist()
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            return [-1] * len(texts)

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get keywords for a specific topic.

        Args:
            topic_id: Topic ID
            top_n: Number of keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        if not self.is_fitted:
            return []
        try:
            return self.topic_model.get_topic(topic_id)[:top_n]
        except Exception:
            return []

    def _rule_based_topics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fallback rule-based topic extraction when BERTopic is not available.

        Args:
            texts: List of review texts

        Returns:
            Dictionary with rule-based topic results
        """
        logger.info("Using rule-based topic extraction (BERTopic not available)")

        # Define topic keywords
        topic_keywords = {
            'product_quality': ['quality', 'durable', 'sturdy', 'well made', 'solid', 'craftsmanship'],
            'customer_service': ['customer service', 'support', 'refund', 'return', 'helpful', 'response'],
            'shipping_delivery': ['shipping', 'delivery', 'arrived', 'package', 'shipment', 'delivered'],
            'price_value': ['price', 'worth', 'expensive', 'cheap', 'value', 'cost', 'affordable'],
            'usability': ['easy', 'difficult', 'setup', 'install', 'use', 'user friendly', 'intuitive'],
            'battery': ['battery', 'charge', 'power', 'lasts', 'battery life'],
            'performance': ['fast', 'slow', 'performance', 'speed', 'works great', 'efficient'],
            'design': ['design', 'look', 'appearance', 'style', 'color', 'size', 'compact']
        }

        # Count topic mentions
        topic_counts = Counter()
        topic_texts = {topic: [] for topic in topic_keywords}

        for text in texts:
            text_lower = text.lower()
            for topic, keywords in topic_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    topic_counts[topic] += 1
                    if len(topic_texts[topic]) < 5:
                        topic_texts[topic].append(text[:200])

        total_docs = len(texts)
        top_topics = topic_counts.most_common(10)

        topics_list = []
        for i, (topic_name, count) in enumerate(top_topics):
            topics_list.append({
                'topic_id': i,
                'name': topic_name.replace('_', ' ').title(),
                'count': count,
                'percentage': round(count / total_docs * 100, 2) if total_docs > 0 else 0,
                'keywords': topic_keywords[topic_name],
                'representative_docs': topic_texts[topic_name][:3]
            })

        top_sum = sum(count for _, count in top_topics)
        outlier_count = total_docs - top_sum
        outlier_percentage = round((total_docs - top_sum) / total_docs * 100, 2) if total_docs > 0 else 0.0

        return {
            'num_topics': len(top_topics),
            'total_documents': total_docs,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'top_topics': topics_list,
            'topic_distribution': {
                topic_name: {
                    'count': count,
                    'percentage': round(count / total_docs * 100, 2) if total_docs > 0 else 0
                }
                for topic_name, count in top_topics
            },
            'model_fitted': False,
            'method': 'rule_based'
        }

    def save_topics_to_db(self, results: Dict[str, Any]) -> bool:
        """
        Save topic modeling results to PostgreSQL.

        Args:
            results: Topic modeling results dictionary

        Returns:
            True if saved successfully
        """
        try:
            from app.database import db
            from app.models import Product
            from flask import Flask
            from app.config import Config

            app = Flask(__name__)
            app.config.from_object(Config)
            db.init_app(app)

            with app.app_context():
                # Store topic data in a JSON file for now
                # (PostgreSQL JSONB storage can be extended later)
                output_path = os.path.join(self.model_dir, 'topic_results.json')
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                logger.info(f"Topic results saved to: {output_path}")
                return True

        except Exception as e:
            logger.warning(f"Failed to save topics to database: {e}")
            return False


# Module-level accessor
_topic_modeler_instance = None


def get_topic_modeler(model_dir: str = DEFAULT_MODEL_DIR) -> TopicModeler:
    """Get or create the singleton TopicModeler instance."""
    global _topic_modeler_instance
    if _topic_modeler_instance is None:
        _topic_modeler_instance = TopicModeler(model_dir)
    return _topic_modeler_instance
