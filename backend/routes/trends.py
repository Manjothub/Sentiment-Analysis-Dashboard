from flask import Blueprint, request, jsonify
from collections import Counter
from datetime import datetime, timezone, timedelta
from ..models import db
from ..models.database import Review, Sentiment, Trend

trends_bp = Blueprint('trends', __name__)


def get_analyzer():
    from ..app import analyzer
    return analyzer


@trends_bp.route('/trending-topics', methods=['GET'])
def get_trending_topics():
    product_id = request.args.get('product_id')
    hours = request.args.get('hours', 24, type=int)
    top_n = request.args.get('top_n', 15, type=int)

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = Review.query.filter(Review.timestamp >= since)
    if product_id:
        query = query.filter_by(product_id=product_id)

    # Existing sentiment counts from Database to avoid N+1 inference
    recent_reviews = query.join(Sentiment).all()

    topics_with_sentiment = []
    for word, freq in topics:
        # Filter local objects
        relevant_reviews = [r for r in recent_reviews if word in r.text.lower()]

        if relevant_reviews:
            sentiments = [r.sentiment.label for r in relevant_reviews[:20]]
            avg_sentiment = Counter(sentiments).most_common(1)[0][0]
        else:
            avg_sentiment = 'neutral'

        topics_with_sentiment.append({
            'keyword': word,
            'frequency': freq,
            'dominant_sentiment': avg_sentiment
        })

    return jsonify({
        'topics': topics_with_sentiment,
        'period_hours': hours,
        'total_reviews_analyzed': len(texts)
    })


@trends_bp.route('/trends-over-time', methods=['GET'])
def get_trends_over_time():
    product_id = request.args.get('product_id')
    days = request.args.get('days', 30, type=int)
    aspect = request.args.get('aspect')

    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = Review.query.filter(Review.timestamp >= since)\
        .join(Sentiment)\
        .order_by(Review.timestamp.asc())

    if product_id:
        query = query.filter_by(product_id=product_id)

    reviews = query.all()

    daily_stats = {}
    for review in reviews:
        day_key = review.timestamp.strftime('%Y-%m-%d')
        if day_key not in daily_stats:
            daily_stats[day_key] = {
                'date': day_key,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'total': 0
            }

        if review.sentiment:
            daily_stats[day_key][review.sentiment.label] += 1
            daily_stats[day_key]['total'] += 1

            if aspect and review.sentiment.aspects:
                aspect_data = review.sentiment.aspects.get(aspect, {})
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


@trends_bp.route('/aspects', methods=['GET'])
def get_aspect_analysis():
    product_id = request.args.get('product_id')

    query = Review.query.join(Sentiment)
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
        if review.sentiment and review.sentiment.aspects:
            for aspect, data in review.sentiment.aspects.items():
                if aspect in aspects_summary:
                    aspects_summary[aspect]['mentions'] += 1
                    sent = data.get('sentiment', 'neutral')
                    aspects_summary[aspect][sent] += 1

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

    return jsonify(result)
