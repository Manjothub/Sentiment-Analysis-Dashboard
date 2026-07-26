from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
from ..models import db
from ..models.database import Review, Sentiment, Product
from ..services.nlp_service import SentimentAnalyzer

sentiment_bp = Blueprint('sentiment', __name__)


def get_analyzer():
    from ..app import analyzer
    return analyzer


@sentiment_bp.route('/analyze', methods=['POST'])
def analyze_review():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    analyzer = get_analyzer()
    result = analyzer.analyze_sentiment(data['text'])
    return jsonify(result)


@sentiment_bp.route('/reviews', methods=['GET'])
def get_reviews():
    product_id = request.args.get('product_id')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    sentiment_filter = request.args.get('sentiment')

    query = Review.query
    if product_id:
        query = query.filter_by(product_id=product_id)
    if sentiment_filter:
        query = query.join(Sentiment).filter(Sentiment.label == sentiment_filter)

    query = query.order_by(Review.timestamp.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'reviews': [r.to_dict() for r in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    })


@sentiment_bp.route('/stats', methods=['GET'])
def get_stats():
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

    reviews_with_sentiment = query.join(Sentiment).all()
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for r in reviews_with_sentiment:
        if r.sentiment:
            sentiment_counts[r.sentiment.label] = sentiment_counts.get(r.sentiment.label, 0) + 1

    ratings = [r.score for r in query.all() if r.score]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    return jsonify({
        'total_reviews': total_reviews,
        'sentiment_distribution': sentiment_counts,
        'avg_rating': round(avg_rating, 2),
        'product_id': product_id
    })


@sentiment_bp.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    data = request.get_json()
    if not data or 'reviews' not in data:
        return jsonify({'error': 'No reviews provided'}), 400

    analyzer = get_analyzer()
    texts = [r.get('text', '') for r in data['reviews']]
    results = [analyzer.analyze_sentiment(t) for t in texts]

    return jsonify({'results': results})


@sentiment_bp.route('/ingest', methods=['POST'])
def ingest_review():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    product_id = data.get('product_id', 'unknown')
    product = Product.query.get(product_id)
    if not product:
        product = Product(
            id=product_id,
            name=data.get('product_name', product_id),
            category=data.get('category', 'general')
        )
        db.session.add(product)
        db.session.commit()

    review = Review(
        product_id=product_id,
        user_id=data.get('user_id'),
        score=data.get('score'),
        summary=data.get('summary'),
        text=data['text'],
        source=data.get('source', 'api'),
        timestamp=datetime.fromtimestamp(
            data.get('timestamp', datetime.now(timezone.utc).timestamp()),
            tz=timezone.utc
        )
    )
    db.session.add(review)
    db.session.flush()

    analyzer = get_analyzer()
    sentiment_result = analyzer.analyze_sentiment(data['text'])

    sentiment = Sentiment(
        review_id=review.id,
        label=sentiment_result['label'],
        positive_score=sentiment_result['positive_score'],
        negative_score=sentiment_result['negative_score'],
        neutral_score=sentiment_result['neutral_score'],
        aspects=sentiment_result['aspects']
    )
    db.session.add(sentiment)
    db.session.commit()

    from ..models import socketio
    socketio.emit('new_review', {
        'review': review.to_dict(),
        'sentiment': sentiment_result
    })

    return jsonify({
        'review_id': review.id,
        'sentiment': sentiment_result
    }), 201
