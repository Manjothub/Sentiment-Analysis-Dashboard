from flask import Blueprint, request, jsonify
from ..models.database import Review, Product, CompetitorProduct

comparative_bp = Blueprint('comparative', __name__)


def get_analyzer():
    from ..app import analyzer
    return analyzer


@comparative_bp.route('/compare', methods=['GET'])
def compare_products():
    product_id = request.args.get('product_id')
    competitor_ids = request.args.getlist('competitor_ids')

    if not product_id:
        return jsonify({'error': 'product_id required'}), 400

    analyzer = get_analyzer()

    product_reviews = Review.query.filter_by(product_id=product_id).all()
    product_texts = [r.text for r in product_reviews if r.text]

    competitor_results = {}
    for comp_id in competitor_ids:
        comp_reviews = Review.query.filter_by(product_id=comp_id).all()
        comp_texts = [r.text for r in comp_reviews if r.text]
        if comp_texts:
            competitor_results[comp_id] = analyzer.comparative_analysis(
                product_texts[:100], comp_texts[:100]
            )

    own_analysis = analyzer.comparative_analysis(
        product_texts[:100], product_texts[:100]
    )

    return jsonify({
        'product_id': product_id,
        'product_sentiment': own_analysis['product'],
        'competitors': competitor_results
    })


@comparative_bp.route('/competitors', methods=['GET', 'POST'])
def manage_competitors():
    if request.method == 'GET':
        product_id = request.args.get('product_id')
        query = CompetitorProduct.query
        if product_id:
            query = query.filter_by(source_product_id=product_id)
        return jsonify({
            'competitors': [c.to_dict() for c in query.all()]
        })

    elif request.method == 'POST':
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'name required'}), 400

        competitor = CompetitorProduct(
            name=data['name'],
            source_product_id=data.get('source_product_id')
        )
        from ..models import db
        db.session.add(competitor)
        db.session.commit()

        return jsonify(competitor.to_dict()), 201


@comparative_bp.route('/compare/aspects', methods=['GET'])
def compare_aspects():
    product_id = request.args.get('product_id')
    competitor_id = request.args.get('competitor_id')

    if not product_id or not competitor_id:
        return jsonify({'error': 'product_id and competitor_id required'}), 400

    product_reviews = Review.query.filter_by(product_id=product_id)\
        .order_by(Review.timestamp.desc()).limit(100).all()
    competitor_reviews = Review.query.filter_by(product_id=competitor_id)\
        .order_by(Review.timestamp.desc()).limit(100).all()

    aspects = ['quality', 'shipping', 'customer_service', 'value', 'usability']
    comparison = {}

    for aspect in aspects:
        product_aspect_count = sum(
            1 for r in product_reviews
            if r.sentiment and r.sentiment.aspects
            and aspect in r.sentiment.aspects
            and r.sentiment.aspects[aspect].get('sentiment') == 'positive'
        )
        competitor_aspect_count = sum(
            1 for r in competitor_reviews
            if r.sentiment and r.sentiment.aspects
            and aspect in r.sentiment.aspects
            and r.sentiment.aspects[aspect].get('sentiment') == 'positive'
        )

        product_total = sum(
            1 for r in product_reviews
            if r.sentiment and r.sentiment.aspects
            and aspect in r.sentiment.aspects
        )
        competitor_total = sum(
            1 for r in competitor_reviews
            if r.sentiment and r.sentiment.aspects
            and aspect in r.sentiment.aspects
        )

        comparison[aspect] = {
            'product': {
                'positive_count': product_aspect_count,
                'total_mentions': product_total,
                'positive_pct': round(product_aspect_count / max(product_total, 1) * 100, 1)
            },
            'competitor': {
                'positive_count': competitor_aspect_count,
                'total_mentions': competitor_total,
                'positive_pct': round(competitor_aspect_count / max(competitor_total, 1) * 100, 1)
            }
        }

    return jsonify(comparison)
