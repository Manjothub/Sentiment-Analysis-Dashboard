"""
Sentiment Analysis API Blueprint
===============================
Provides endpoints for analyzing sentiment of reviews and retrieving sentiment data.
"""

from flask import Blueprint, jsonify, request, current_app
from app.models import Review, SentimentResult, Product
from app.database import db
from app.utils.logger import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)

sentiment_bp = Blueprint('sentiment', __name__)


def _transform_sentiment_result(result: dict) -> dict:
    """Transform ML service result to frontend-compatible format."""
    return {
        'label': result.get('predicted_sentiment', 'neutral'),
        'positive_score': result.get('positive_score', 0.0),
        'negative_score': result.get('negative_score', 0.0),
        'neutral_score': result.get('neutral_score', 0.0),
        'aspects': result.get('aspects', {}),
        'confidence_score': result.get('confidence_score', 0.0),
        'inference_time_ms': result.get('inference_time_ms', 0),
        'model_loaded': result.get('model_loaded', False)
    }


@sentiment_bp.route('/analyze', methods=['POST'])
def analyze_sentiment_alias():
    """Alias for /sentiment/analyze to maintain frontend compatibility."""
    return analyze_sentiment()


@sentiment_bp.route('/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of a single text.
    
    Request Body:
        {
            "text": "Review text to analyze"
        }
    
    Returns:
        JSON response with sentiment prediction and confidence scores
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    try:
        model_service = current_app.extensions.get('model_service')
        if not model_service or not model_service.is_loaded:
            logger.warning("Model service not available, using fallback analysis")
            result = _fallback_sentiment_analysis(text)
        else:
            result = model_service.analyze(text)
        
        logger.info(f"Sentiment analysis completed: {result['predicted_sentiment']}")
        return jsonify(_transform_sentiment_result(result)), 200
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return jsonify({'error': 'Analysis failed', 'message': str(e)}), 500


@sentiment_bp.route('/sentiment/reviews/<int:review_id>', methods=['GET'])
def get_review_sentiment(review_id):
    """
    Get sentiment analysis for a specific review.
    
    Args:
        review_id: ID of the review
        
    Returns:
        JSON response with review and sentiment data
    """
    try:
        review = Review.query.get(review_id)
        if not review:
            return jsonify({'error': 'Review not found'}), 404
        
        result = {
            'review': review.to_dict(),
            'sentiment': review.sentiment_result.to_dict() if review.sentiment_result else None
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Failed to fetch review sentiment: {e}")
        return jsonify({'error': str(e)}), 500


@sentiment_bp.route('/sentiment/products/<product_id>', methods=['GET'])
def get_product_sentiment(product_id):
    """
    Get aggregated sentiment analysis for a product.
    
    Args:
        product_id: Product identifier
        
    Returns:
        JSON response with product sentiment summary
    """
    try:
        product = Product.query.get(product_id)
        if not product:
            return jsonify({'error': 'Product not found'}), 404
        
        sentiment_dist = SentimentResult.get_sentiment_distribution(product_id)
        avg_scores = SentimentResult.get_average_scores(product_id)
        
        result = {
            'product': product.to_dict(),
            'sentiment_distribution': sentiment_dist,
            'average_scores': avg_scores
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Failed to fetch product sentiment: {e}")
        return jsonify({'error': str(e)}), 500


@sentiment_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get overall dashboard statistics.
    
    Query Parameters:
        product_id (optional): Filter by product
    
    Returns:
        JSON response with total reviews and sentiment distribution
    """
    product_id = request.args.get('product_id')
    
    query = Review.query
    if product_id:
        query = query.filter_by(product_id=product_id)
    
    total_reviews = query.count()
    
    if total_reviews == 0:
        return jsonify({
            'total_reviews': 0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'avg_rating': 0
        })
    
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    reviews_with_sentiment = query.join(SentimentResult).all()
    for review in reviews_with_sentiment:
        if review.sentiment_result:
            label = review.sentiment_result.predicted_sentiment
            if label in sentiment_counts:
                sentiment_counts[label] += 1
    
    ratings = [r.raw_rating for r in query.all() if r.raw_rating]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    
    return jsonify({
        'total_reviews': total_reviews,
        'sentiment_distribution': sentiment_counts,
        'avg_rating': round(avg_rating, 2),
        'product_id': product_id
    })


@sentiment_bp.route('/reviews', methods=['GET'])
def get_reviews():
    """
    Get recent reviews with sentiment data.
    
    Query Parameters:
        product_id (optional): Filter by product
        page (int): Page number (default: 1)
        per_page (int): Items per page (default: 20)
        sentiment (optional): Filter by sentiment label
    
    Returns:
        JSON response with paginated reviews
    """
    product_id = request.args.get('product_id')
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    sentiment_filter = request.args.get('sentiment')
    
    query = Review.query
    if product_id:
        query = query.filter_by(product_id=product_id)
    if sentiment_filter:
        query = query.join(SentimentResult).filter(
            SentimentResult.predicted_sentiment == sentiment_filter
        )
    
    query = query.order_by(Review.ingested_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'reviews': [r.to_dict() for r in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    })


@sentiment_bp.route('/ingest', methods=['POST'])
def ingest_review():
    """
    Ingest a new review and analyze its sentiment.
    
    Request Body:
        {
            "text": "Review text",
            "product_id": "optional_product_id",
            "product_name": "Product Name",
            "category": "category",
            "user_id": "user123",
            "score": 5,
            "summary": "Great product",
            "source": "api"
        }
    
    Returns:
        JSON response with review ID and sentiment result
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    product_id = data.get('product_id', 'unknown')
    
    try:
        # Create or get product
        product = Product.query.get(product_id)
        if not product:
            product = Product(
                product_id=product_id,
                product_name=data.get('product_name', product_id),
                category=data.get('category', 'general')
            )
            db.session.add(product)
            db.session.flush()
        
        # Create review
        review = Review(
            product_id=product_id,
            reviewer_name=data.get('user_id'),
            review_text=text,
            cleaned_text=text,
            summary=data.get('summary'),
            raw_rating=data.get('score'),
            source=data.get('source', 'api'),
            review_date=datetime.now(timezone.utc)
        )
        db.session.add(review)
        db.session.flush()
        
        # Analyze sentiment
        model_service = current_app.extensions.get('model_service')
        if model_service and model_service.is_loaded:
            sentiment_result = model_service.analyze(text)
        else:
            sentiment_result = _fallback_sentiment_analysis(text)
        
        # Create sentiment result
        sentiment_db = SentimentResult(
            review_id=review.review_id,
            predicted_sentiment=sentiment_result.get('predicted_sentiment', 'neutral'),
            positive_score=sentiment_result.get('positive_score', 0.0),
            negative_score=sentiment_result.get('negative_score', 0.0),
            neutral_score=sentiment_result.get('neutral_score', 0.0),
            confidence_score=sentiment_result.get('confidence_score', 0.0),
            aspects=sentiment_result.get('aspects', {}),
            model_version='latest',
            model_name='distilbert-sentiment',
            inference_time_ms=sentiment_result.get('inference_time_ms', 0)
        )
        db.session.add(sentiment_db)
        db.session.commit()
        
        # Emit socket event
        try:
            from models import socketio
            socketio.emit('new_review', {
                'review': review.to_dict(),
                'sentiment': _transform_sentiment_result(sentiment_result)
            })
        except Exception as socket_err:
            logger.warning(f"Socket emit failed: {socket_err}")
        
        return jsonify({
            'review_id': review.review_id,
            'sentiment': _transform_sentiment_result(sentiment_result)
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Ingest failed: {e}")
        return jsonify({'error': 'Ingest failed', 'message': str(e)}), 500


@sentiment_bp.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """
    Analyze sentiment for multiple texts.
    
    Request Body:
        {
            "reviews": [{"text": "Review 1"}, {"text": "Review 2"}]
        }
    """
    data = request.get_json()
    if not data or 'reviews' not in data:
        return jsonify({'error': 'No reviews provided'}), 400
    
    reviews = data['reviews']
    if not isinstance(reviews, list):
        return jsonify({'error': 'Reviews must be a list'}), 400
    
    try:
        model_service = current_app.extensions.get('model_service')
        results = []
        for review in reviews:
            text = review.get('text', '') if isinstance(review, dict) else str(review)
            text = text.strip()
            if not text:
                results.append(_transform_sentiment_result(_fallback_sentiment_analysis('')))
                continue
            
            if model_service and model_service.is_loaded:
                result = model_service.analyze(text)
            else:
                result = _fallback_sentiment_analysis(text)
            results.append(_transform_sentiment_result(result))
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return jsonify({'error': 'Batch analysis failed', 'message': str(e)}), 500


def _fallback_sentiment_analysis(text: str) -> dict:
    """
    Fallback sentiment analysis using rule-based approach.
    Used when the ML model is not available.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment prediction
    """
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                      'love', 'perfect', 'best', 'awesome', 'happy', 'satisfied'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor',
                      'hate', 'disappointed', 'useless', 'waste', 'regret'}
    
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
            'aspects': {}
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
        'aspects': {}
    }
