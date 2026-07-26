from flask import Blueprint, request, jsonify
from collections import Counter
from datetime import datetime, timezone, timedelta
from app.models import Review, SentimentResult, Product
from app.database import db
from app.utils.logger import get_logger

logger = get_logger(__name__)

trends_bp = Blueprint('trends', __name__)


@trends_bp.route('/trends', methods=['GET'])
def trends_index():
    """Trends API index."""
    return jsonify({
        'message': 'Trends API',
        'endpoints': [
            '/api/trends',
            '/api/trends/sentiment-over-time',
            '/api/trends/keywords',
            '/api/trends/rating-distribution',
            '/api/trending-topics',
            '/api/trends-over-time'
        ]
    }), 200


@trends_bp.route('/trending-topics', methods=['GET'])
def get_trending_topics():
    """
    Get trending topics from reviews.
    
    Query Parameters:
        product_id (optional): Filter by product
        hours (int): Time window in hours (default: 168)
        top_n (int): Number of topics to return (default: 15)
    """
    product_id = request.args.get('product_id')
    hours = request.args.get('hours', 168, type=int)
    top_n = request.args.get('top_n', 15, type=int)
    
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = Review.query.filter(Review.ingested_at >= since)
        if product_id:
            query = query.filter_by(product_id=product_id)
        
        reviews = query.all()
        
        all_words = []
        for review in reviews:
            if review.cleaned_text:
                words = review.cleaned_text.lower().split()
                all_words.extend([w for w in words if len(w) > 3])
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(top_n)
        
        topics_with_sentiment = []
        for word, freq in top_words:
            relevant_reviews = [
                r for r in reviews
                if r.cleaned_text and word in r.cleaned_text.lower()
            ]
            dominant_sentiment = 'neutral'
            if relevant_reviews:
                sentiments = []
                for r in relevant_reviews[:20]:
                    if r.sentiment_result:
                        sentiments.append(r.sentiment_result.predicted_sentiment)
                if sentiments:
                    dominant_sentiment = Counter(sentiments).most_common(1)[0][0]
            
            topics_with_sentiment.append({
                'keyword': word,
                'frequency': freq,
                'dominant_sentiment': dominant_sentiment
            })
        
        return jsonify({
            'topics': topics_with_sentiment,
            'period_hours': hours,
            'total_reviews_analyzed': len(reviews)
        })
    except Exception as e:
        logger.error(f"Failed to get trending topics: {e}")
        return jsonify({'error': str(e)}), 500


@trends_bp.route('/trends-over-time', methods=['GET'])
def get_trends_over_time():
    """
    Get sentiment trends over time.
    
    Query Parameters:
        product_id (optional): Filter by product
        days (int): Number of days to look back (default: 30)
        aspect (optional): Filter by aspect
    """
    product_id = request.args.get('product_id')
    days = request.args.get('days', 30, type=int)
    aspect = request.args.get('aspect')
    
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = Review.query.filter(Review.ingested_at >= since)\
            .join(SentimentResult)\
            .order_by(Review.ingested_at.asc())
        
        if product_id:
            query = query.filter_by(product_id=product_id)
        
        reviews = query.all()
        
        daily_stats = {}
        for review in reviews:
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
                
                if aspect and review.sentiment_result.aspects:
                    aspect_data = review.sentiment_result.aspects.get(aspect, {})
                    if aspect_data:
                        if 'aspect_sentiment' not in daily_stats[day_key]:
                            daily_stats[day_key]['aspect_sentiment'] = {}
                        daily_stats[day_key]['aspect_sentiment'][aspect] = \
                            aspect_data.get('sentiment', 'neutral')
        
        trend_data = sorted(daily_stats.values(), key=lambda x: x['date'])
        
        return jsonify({
            'trends': trend_data,
            'aspect': aspect,
            'days': days
        })
    except Exception as e:
        logger.error(f"Failed to get trends over time: {e}")
        return jsonify({'error': str(e)}), 500


@trends_bp.route('/trends/sentiment-over-time', methods=['GET'])
def sentiment_over_time():
    """
    Get sentiment trends over time.
    
    Query Parameters:
        product_id (optional): Filter by product
        days (optional): Number of days to look back (default: 30)
        interval (optional): 'day', 'week', 'month' (default: 'day')
    
    Returns:
        JSON response with time-series sentiment data
    """
    product_id = request.args.get('product_id')
    days = request.args.get('days', 30, type=int)
    interval = request.args.get('interval', 'day')
    
    try:
        return get_trends_over_time_with_interval(product_id, days, interval)
    except Exception as e:
        logger.error(f"Failed to fetch sentiment trends: {e}")
        return jsonify({'error': str(e)}), 500


def get_trends_over_time_with_interval(product_id, days, interval):
    """Helper to get trends with interval grouping."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = Review.query.filter(Review.ingested_at >= since)\
        .join(SentimentResult)\
        .order_by(Review.ingested_at.asc())
    
    if product_id:
        query = query.filter_by(product_id=product_id)
    
    reviews = query.all()
    
    daily_stats = {}
    for review in reviews:
        if interval == 'day':
            period = review.ingested_at.strftime('%Y-%m-%d')
        elif interval == 'week':
            period = review.ingested_at.strftime('%Y-W%V')
        else:
            period = review.ingested_at.strftime('%Y-%m')
        
        if period not in daily_stats:
            daily_stats[period] = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        if review.sentiment_result:
            label = review.sentiment_result.predicted_sentiment
            if label in daily_stats[period]:
                daily_stats[period][label] += 1
    
    trends = []
    for period, counts in sorted(daily_stats.items()):
        trends.append({
            'period': period,
            **counts,
            'total': sum(counts.values())
        })
    
    return jsonify({
        'trends': trends,
        'interval': interval,
        'days': days,
        'product_id': product_id
    }), 200


@trends_bp.route('/trends/keywords', methods=['GET'])
def trending_keywords():
    """
    Get trending keywords from reviews.
    
    Query Parameters:
        product_id (optional): Filter by product
        top_n (optional): Number of keywords to return (default: 20)
        sentiment (optional): Filter by sentiment
    
    Returns:
        JSON response with trending keywords
    """
    product_id = request.args.get('product_id')
    top_n = request.args.get('top_n', 20, type=int)
    sentiment_filter = request.args.get('sentiment')
    
    try:
        query = Review.query
        
        if product_id:
            query = query.filter_by(product_id=product_id)
        if sentiment_filter:
            query = query.join(SentimentResult).filter(
                SentimentResult.predicted_sentiment == sentiment_filter
            )
        
        reviews = query.limit(1000).all()
        
        keywords = {}
        for review in reviews:
            if review.cleaned_text:
                words = review.cleaned_text.lower().split()
                for word in words:
                    if len(word) > 3:
                        keywords[word] = keywords.get(word, 0) + 1
        
        sorted_keywords = sorted(
            keywords.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return jsonify({
            'keywords': [{'word': k, 'frequency': v} for k, v in sorted_keywords],
            'total_reviews_analyzed': len(reviews)
        })
    except Exception as e:
        logger.error(f"Failed to fetch trending keywords: {e}")
        return jsonify({'error': str(e)}), 500


@trends_bp.route('/trends/rating-distribution', methods=['GET'])
def rating_distribution():
    """
    Get distribution of ratings over time.
    
    Query Parameters:
        product_id (optional): Filter by product
    
    Returns:
        JSON response with rating distribution
    """
    product_id = request.args.get('product_id')
    
    try:
        distribution = Review.get_rating_distribution(product_id)
        return jsonify(distribution), 200
    except Exception as e:
        logger.error(f"Failed to fetch rating distribution: {e}")
        return jsonify({'error': str(e)}), 500
