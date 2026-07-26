"""
Model API Blueprint
====================
Provides endpoints for model management, sentiment prediction, and topic modeling.

Endpoints:
    GET  /api/model/status       - Model loading status
    POST /api/predict            - Single review prediction
    POST /api/batch_predict      - Batch prediction
    GET  /api/topics             - Topic modeling results
    GET  /api/model/info         - Model metadata
    GET  /api/aspects            - Aggregate aspect analysis
    GET  /api/dashboard/overview - Combined dashboard data

Does NOT remove or modify existing endpoints.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request, current_app

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.utils.logger import get_logger
from app.database import db
from app.models import SentimentResult, Review, Product, ModelVersion

logger = get_logger(__name__)

model_bp = Blueprint('model', __name__)


def get_sentiment_service():
    """Get the sentiment service from Flask extensions."""
    return current_app.extensions.get('model_service')


def get_aspect_service():
    """Get the aspect service from Flask extensions."""
    return current_app.extensions.get('aspect_service')


def get_topic_modeler():
    """Get the topic modeler from Flask extensions."""
    return current_app.extensions.get('topic_modeler')


@model_bp.route('/model/status', methods=['GET'])
def model_status():
    """
    Get the current status of all ML models.

    Returns:
        JSON response with model loading status and device information
    """
    sentiment_service = get_sentiment_service()
    aspect_service = get_aspect_service()
    topic_modeler = get_topic_modeler()

    status = {
        'application': 'running',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'sentiment': {
                'loaded': sentiment_service.is_loaded if sentiment_service else False,
                'device': sentiment_service.device if sentiment_service else 'unknown',
                'model_type': 'DistilBERT',
                'classes': ['positive', 'neutral', 'negative'],
                'model_path': sentiment_service._model_path if sentiment_service else None
            },
            'aspect_extraction': {
                'loaded': aspect_service.is_loaded if aspect_service else False,
                'model_type': 'facebook/bart-large-mnli',
            },
            'topic_modeling': {
                'loaded': topic_modeler.is_fitted if topic_modeler else False,
                'num_topics': len(topic_modeler.get_topic_info().get('top_topics', [])) if topic_modeler and topic_modeler.is_fitted else 0
            }
        },
        'overall_status': 'healthy' if (sentiment_service and sentiment_service.is_loaded) else 'degraded'
    }

    # Get latest model version from database
    try:
        latest_model = ModelVersion.query.order_by(
            ModelVersion.created_at.desc()
        ).first()
        if latest_model:
            status['models']['sentiment']['version'] = latest_model.version
            status['models']['sentiment']['trained_at'] = latest_model.training_date.isoformat() if latest_model.training_date else None
            status['models']['sentiment']['metrics'] = {
                'accuracy': latest_model.accuracy,
                'f1_score': latest_model.f1_score,
                'precision': latest_model.precision,
                'recall': latest_model.recall
            }
    except Exception as e:
        logger.warning(f"Could not fetch model version from DB: {e}")

    status_code = 200 if status['overall_status'] in ['healthy', 'degraded'] else 503
    return jsonify(status), status_code


@model_bp.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Predict sentiment for a single review text.

    Request Body:
        {
            "text": "Review text to analyze",
            "include_aspects": true,
            "store_result": true,
            "product_id": "optional_product_id"
        }

    Returns:
        JSON response with sentiment prediction, confidence scores,
        aspect analysis (optional), and inference metadata
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    include_aspects = data.get('include_aspects', True)
    store_result = data.get('store_result', False)
    product_id = data.get('product_id')

    try:
        sentiment_service = get_sentiment_service()
        if not sentiment_service or not sentiment_service.is_loaded:
            return jsonify({'error': 'Model not loaded', 'message': 'Sentiment model is not available'}), 503

        start_time = time.time()

        # Get sentiment prediction
        sentiment_result = sentiment_service.analyze(text)

        # Get aspect extraction if requested
        aspect_data = None
        if include_aspects:
            aspect_service = get_aspect_service()
            if aspect_service and aspect_service.is_loaded:
                aspect_data = aspect_service.extract_aspects(text)

        total_time_ms = int((time.time() - start_time) * 1000)

        # Store result in database if requested
        if store_result:
            review_id = store_sentiment_result(text, sentiment_result, aspect_data, product_id)
        else:
            review_id = None

        response = {
            'success': True,
            'text': text[:200],  # Truncated for response
            'sentiment': sentiment_result,
            'aspects': aspect_data['aspects'] if aspect_data else None,
            'aspect_scores': aspect_data['aspect_scores'] if aspect_data else None,
            'metadata': {
                'total_inference_time_ms': total_time_ms,
                'model_loaded': sentiment_result.get('model_loaded', True),
                'aspects_included': include_aspects and aspect_data is not None,
                'stored_in_db': store_result,
                'product_id': product_id,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Emit WebSocket event for real-time updates
        try:
            from models import socketio
            socketio.emit('new_review', {
                'review': {
                    'review_text': text,
                    'product_id': product_id,
                    'source': 'api_prediction',
                },
                'sentiment': sentiment_result
            })
            if store_result and review_id:
                socketio.emit('review_processed', {
                    'review_id': review_id,
                    'sentiment': sentiment_result,
                })
        except Exception as socket_err:
            logger.warning(f"Socket emit failed: {socket_err}")

        logger.info(f"Prediction completed in {total_time_ms}ms: {sentiment_result['predicted_sentiment']}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500


@model_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict sentiment for multiple review texts in batch.

    Request Body:
        {
            "texts": ["Review 1", "Review 2", ...],
            "include_aspects": false,
            "store_results": false,
            "product_id": "optional_product_id"
        }

    Returns:
        JSON response with batch prediction results and statistics
    """
    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({'error': 'Texts array is required'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'Texts must be a non-empty array'}), 400

    if len(texts) > 100:
        return jsonify({'error': 'Maximum batch size is 100'}), 400

    include_aspects = data.get('include_aspects', False)
    store_results = data.get('store_results', False)
    product_id = data.get('product_id')

    try:
        sentiment_service = get_sentiment_service()
        if not sentiment_service or not sentiment_service.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 503

        start_time = time.time()

        # Clean and validate texts
        valid_texts = [t.strip() for t in texts if t.strip()]
        invalid_count = len(texts) - len(valid_texts)

        if not valid_texts:
            return jsonify({'error': 'No valid texts to analyze'}), 400

        # Batch sentiment prediction
        batch_results = sentiment_service.analyze_batch(valid_texts)

        # Optionally store results
        stored_ids = []
        if store_results:
            for i, (text, result) in enumerate(zip(valid_texts, batch_results)):
                review_id = store_sentiment_result(
                    text, result,
                    aspect_data=None,
                    product_id=product_id
                )
                if review_id:
                    stored_ids.append(review_id)

        total_time_ms = int((time.time() - start_time) * 1000)

        # Calculate batch statistics
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        avg_confidence = 0.0
        for result in batch_results:
            sentiment = result['predicted_sentiment']
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            avg_confidence += result['confidence_score']

        avg_confidence = round(avg_confidence / len(batch_results), 4) if batch_results else 0

        response = {
            'success': True,
            'total_texts': len(texts),
            'valid_texts': len(valid_texts),
            'invalid_texts': invalid_count,
            'results': batch_results,
            'statistics': {
                'sentiment_distribution': sentiment_counts,
                'average_confidence': avg_confidence,
                'total_inference_time_ms': total_time_ms,
                'avg_inference_time_per_item_ms': round(total_time_ms / len(valid_texts), 2) if valid_texts else 0
            },
            'metadata': {
                'stored_in_db': store_results,
                'stored_review_ids': stored_ids if stored_ids else None,
                'product_id': product_id,
                'timestamp': datetime.now().isoformat()
            }
        }

        logger.info(f"Batch prediction completed: {len(valid_texts)} items in {total_time_ms}ms")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({'error': 'Batch prediction failed', 'message': str(e)}), 500


@model_bp.route('/topics', methods=['GET'])
def get_topics():
    """
    Get topic modeling results.

    Query Parameters:
        n_topics (int): Number of topics to return (default: 10)
        include_docs (bool): Include representative documents (default: false)

    Returns:
        JSON response with topic modeling results
    """
    n_topics = request.args.get('n_topics', 10, type=int)
    include_docs = request.args.get('include_docs', False, type=bool)

    try:
        topic_modeler = get_topic_modeler()

        if not topic_modeler or not topic_modeler.is_fitted:
            # Attempt to fit topic model on existing data
            try:
                import pandas as pd

                csv_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'dataset', 'processed', 'amazon_cleaned.csv'
                )

                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, nrows=1000)
                    texts = df['cleaned_text'].dropna().tolist()
                    topic_modeler.fit(texts)
                else:
                    return jsonify({
                        'error': 'Topic model not fitted',
                        'message': 'No preprocessed dataset available for training'
                    }), 503

            except Exception as e:
                logger.error(f"Failed to fit topic model on demand: {e}")
                return jsonify({
                    'error': 'Topic model not available',
                    'message': 'Topic model has not been trained yet'
                }), 503

        # Get topic info
        topic_info = topic_modeler.get_topic_info()

        # Limit number of topics
        if 'top_topics' in topic_info:
            topic_info['top_topics'] = topic_info['top_topics'][:n_topics]

        if not include_docs and 'representative_documents' in topic_info:
            del topic_info['representative_documents']

        # Get latest topics from database
        try:
            from app.models import TopicResult
            db_topics = TopicResult.query.order_by(
                TopicResult.created_at.desc()
            ).limit(n_topics).all()
            if db_topics:
                topic_info['stored_topics'] = [t.to_dict() for t in db_topics]
        except Exception:
            pass

        return jsonify({
            'success': True,
            'topics': topic_info,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Failed to get topics: {e}")
        return jsonify({'error': 'Failed to get topics', 'message': str(e)}), 500


@model_bp.route('/model/info', methods=['GET'])
def model_info():
    """
    Get detailed information about the trained sentiment model.

    Returns:
        JSON response with model architecture, hyperparameters, metrics
    """
    try:
        sentiment_service = get_sentiment_service()

        if not sentiment_service:
            return jsonify({'error': 'Model service not initialized'}), 503

        model_info_data = sentiment_service.get_model_info()

        # Add training history from database
        try:
            versions = ModelVersion.query.order_by(
                ModelVersion.created_at.desc()
            ).limit(5).all()

            model_info_data['training_history'] = [
                {
                    'version': v.version,
                    'accuracy': v.accuracy,
                    'f1_score': v.f1_score,
                    'precision': v.precision,
                    'recall': v.recall,
                    'training_date': v.training_date.isoformat() if v.training_date else None,
                    'parameters': v.parameters
                }
                for v in versions
            ]
        except Exception as e:
            logger.warning(f"Could not fetch training history: {e}")
            model_info_data['training_history'] = []

        # Add model statistics
        try:
            total_predictions = SentimentResult.query.count()
            model_info_data['total_predictions'] = total_predictions

            # Distribution of predicted sentiments
            from sqlalchemy import func
            dist = db.session.query(
                SentimentResult.predicted_sentiment,
                func.count(SentimentResult.result_id)
            ).group_by(SentimentResult.predicted_sentiment).all()

            model_info_data['prediction_distribution'] = {
                sentiment: count for sentiment, count in dist
            }
        except Exception as e:
            logger.warning(f"Could not fetch model statistics: {e}")

        return jsonify({
            'success': True,
            'model_info': model_info_data,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return jsonify({'error': 'Failed to get model info', 'message': str(e)}), 500


@model_bp.route('/aspects', methods=['GET'])
def get_aspects():
    """
    Get aggregate aspect analysis across all reviews.
    
    Query Parameters:
        product_id (optional): Filter by product
    
    Returns:
        JSON response with aspect statistics
    """
    product_id = request.args.get('product_id')
    
    try:
        query = Review.query
        if product_id:
            query = query.filter_by(product_id=product_id)
        
        reviews = query.all()
        
        aspects_summary = {
            'quality': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
            'shipping': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
            'customer_service': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
            'value': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
            'usability': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
        }
        
        for review in reviews:
            if review.sentiment_result and review.sentiment_result.aspects:
                for aspect, sentiment in review.sentiment_result.aspects.items():
                    if aspect in aspects_summary:
                        aspects_summary[aspect]['mentions'] += 1
                        if sentiment in aspects_summary[aspect]:
                            aspects_summary[aspect][sentiment] += 1
        
        result = {}
        for aspect, counts in aspects_summary.items():
            total = counts['mentions']
            if total > 0:
                result[aspect] = {
                    'mentions': total,
                    'positive_pct': round(counts['positive'] / total * 100, 1),
                    'negative_pct': round(counts['negative'] / total * 100, 1),
                    'neutral_pct': round(counts['neutral'] / total * 100, 1)
                }
            else:
                result[aspect] = {
                    'mentions': 0,
                    'positive_pct': 0,
                    'negative_pct': 0,
                    'neutral_pct': 0
                }
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Failed to get aspect analysis: {e}")
        return jsonify({'error': str(e)}), 500


@model_bp.route('/dashboard/overview', methods=['GET'])
def dashboard_overview():
    """
    Get combined dashboard overview data.
    
    Query Parameters:
        product_id (optional): Filter by product
        days (int): Number of days for trends (default: 30)
    
    Returns:
        JSON response with combined dashboard data
    """
    product_id = request.args.get('product_id')
    days = request.args.get('days', 30, type=int)
    
    try:
        # Stats
        stats_query = Review.query
        if product_id:
            stats_query = stats_query.filter_by(product_id=product_id)
        total_reviews = stats_query.count()
        
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        if total_reviews > 0:
            reviews_with_sentiment = stats_query.join(SentimentResult).all()
            for review in reviews_with_sentiment:
                if review.sentiment_result:
                    label = review.sentiment_result.predicted_sentiment
                    if label in sentiment_counts:
                        sentiment_counts[label] += 1
        
        ratings = [r.raw_rating for r in stats_query.all() if r.raw_rating]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0
        
        stats = {
            'total_reviews': total_reviews,
            'sentiment_distribution': sentiment_counts,
            'avg_rating': avg_rating,
            'product_id': product_id
        }
        
        # Trends
        since = datetime.now(timezone.utc) - timedelta(days=days)
        trend_query = Review.query.filter(Review.ingested_at >= since)\
            .join(SentimentResult)\
            .order_by(Review.ingested_at.asc())
        if product_id:
            trend_query = trend_query.filter_by(product_id=product_id)
        
        daily_stats = {}
        for review in trend_query.all():
            day_key = review.ingested_at.strftime('%Y-%m-%d')
            if day_key not in daily_stats:
                daily_stats[day_key] = {
                    'date': day_key,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'total': 0
                }
            if review.sentiment_result:
                label = review.sentiment_result.predicted_sentiment
                if label in daily_stats[day_key]:
                    daily_stats[day_key][label] += 1
                daily_stats[day_key]['total'] += 1
        trends = sorted(daily_stats.values(), key=lambda x: x['date'])
        
        # Aspects
        aspects = get_dashboard_aspects(product_id)
        
        # Recent reviews
        recent_query = Review.query
        if product_id:
            recent_query = recent_query.filter_by(product_id=product_id)
        recent_reviews = recent_query.order_by(Review.ingested_at.desc()).limit(10).all()
        
        # Alert summary
        alert_summary = {
            'total_alerts': 0,
            'unacknowledged': 0,
            'by_severity': {'critical': 0, 'warning': 0, 'info': 0}
        }
        
        return jsonify({
            'stats': stats,
            'trends': trends,
            'aspects': aspects,
            'recent_reviews': [r.to_dict() for r in recent_reviews],
            'alert_summary': alert_summary,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        return jsonify({'error': str(e)}), 500


def get_dashboard_aspects(product_id=None):
    """Helper to get aggregate aspects."""
    query = Review.query
    if product_id:
        query = query.filter_by(product_id=product_id)
    
    reviews = query.all()
    
    aspects_summary = {
        'quality': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
        'shipping': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
        'customer_service': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
        'value': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
        'usability': {'positive': 0, 'negative': 0, 'neutral': 0, 'mentions': 0},
    }
    
    for review in reviews:
        if review.sentiment_result and review.sentiment_result.aspects:
            for aspect, sentiment in review.sentiment_result.aspects.items():
                if aspect in aspects_summary:
                    aspects_summary[aspect]['mentions'] += 1
                    if sentiment in aspects_summary[aspect]:
                        aspects_summary[aspect][sentiment] += 1
    
    result = {}
    for aspect, counts in aspects_summary.items():
        total = counts['mentions']
        if total > 0:
            result[aspect] = {
                'mentions': total,
                'positive_pct': round(counts['positive'] / total * 100, 1),
                'negative_pct': round(counts['negative'] / total * 100, 1),
                'neutral_pct': round(counts['neutral'] / total * 100, 1)
            }
        else:
            result[aspect] = {
                'mentions': 0,
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 0
            }
    
    return result


def store_sentiment_result(
    text: str,
    sentiment_result: Dict[str, Any],
    aspect_data: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None
) -> Optional[int]:
    """
    Store a sentiment prediction result in the database.

    Args:
        text: Original review text
        sentiment_result: Sentiment analysis result
        aspect_data: Optional aspect extraction result
        product_id: Optional product ID

    Returns:
        review_id if stored successfully, None otherwise
    """
    try:
        # Create or get product
        if product_id:
            product = Product.query.get(product_id)
            if not product:
                product = Product(
                    product_id=product_id,
                    product_name=f"Product-{product_id[:8]}",
                    category='general'
                )
                db.session.add(product)
                db.session.flush()
        else:
            # Use a default product for standalone predictions
            default_product = Product.query.filter_by(product_id='default').first()
            if not default_product:
                default_product = Product(
                    product_id='default',
                    product_name='Standalone Prediction',
                    category='general'
                )
                db.session.add(default_product)
                db.session.flush()
            product_id = 'default'

        # Create review
        review = Review(
            product_id=product_id,
            review_text=text,
            cleaned_text=text,
            source='api_prediction'
        )
        db.session.add(review)
        db.session.flush()

        # Create sentiment result
        sentiment_db = SentimentResult(
            review_id=review.review_id,
            predicted_sentiment=sentiment_result['predicted_sentiment'],
            positive_score=sentiment_result.get('positive_score', 0.0),
            negative_score=sentiment_result.get('negative_score', 0.0),
            neutral_score=sentiment_result.get('neutral_score', 0.0),
            confidence_score=sentiment_result.get('confidence_score', 0.0),
            aspects=aspect_data.get('aspects', {}) if aspect_data else {},
            aspect_scores=aspect_data.get('aspect_scores', {}) if aspect_data else {},
            model_version='latest',
            model_name='distilbert-sentiment',
            inference_time_ms=sentiment_result.get('inference_time_ms', 0)
        )
        db.session.add(sentiment_db)
        db.session.commit()

        logger.info(f"Stored sentiment result for review {review.review_id}")
        return review.review_id

    except Exception as e:
        logger.error(f"Failed to store sentiment result: {e}")
        db.session.rollback()
        return None
