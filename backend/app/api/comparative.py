"""
Comparative Analysis API Blueprint
===================================
Provides endpoints for comparing sentiment across products and competitors.
"""

from flask import Blueprint, jsonify, request
from sqlalchemy import func
from app.models import Review, SentimentResult, Product
from app.database import db
from app.utils.logger import get_logger

logger = get_logger(__name__)

comparative_bp = Blueprint('comparative', __name__)


@comparative_bp.route('/comparative/products', methods=['GET'])
def compare_products():
    """
    Compare sentiment across multiple products.
    
    Query Parameters:
        product_ids (required): Comma-separated list of product IDs
    
    Returns:
        JSON response with comparative sentiment analysis
    """
    product_ids_param = request.args.get('product_ids')
    
    if not product_ids_param:
        return jsonify({'error': 'product_ids parameter is required'}), 400
    
    product_ids = [pid.strip() for pid in product_ids_param.split(',')]
    
    try:
        comparison = []
        
        for product_id in product_ids:
            product = Product.query.get(product_id)
            if not product:
                continue
            
            sentiment_dist = SentimentResult.get_sentiment_distribution(product_id)
            avg_scores = SentimentResult.get_average_scores(product_id)
            
            comparison.append({
                'product_id': product_id,
                'product_name': product.product_name if product else product_id,
                'category': product.category,
                'sentiment_distribution': sentiment_dist,
                'average_scores': avg_scores
            })
        
        return jsonify({
            'comparison': comparison,
            'total_products': len(comparison)
        }), 200
        
    except Exception as e:
        logger.error(f"Product comparison failed: {e}")
        return jsonify({'error': str(e)}), 500


@comparative_bp.route('/comparative/categories', methods=['GET'])
def compare_categories():
    """
    Compare sentiment across product categories.
    
    Returns:
        JSON response with category-level sentiment comparison
    """
    try:
        categories = db.session.query(
            Product.category,
            func.count(Review.review_id).label('total_reviews'),
            func.avg(Review.raw_rating).label('avg_rating'),
            func.avg(SentimentResult.positive_score).label('avg_positive_score'),
            func.avg(SentimentResult.negative_score).label('avg_negative_score'),
            func.avg(SentimentResult.neutral_score).label('avg_neutral_score')
        ).join(
            Review, Product.product_id == Review.product_id
        ).join(
            SentimentResult, Review.review_id == SentimentResult.review_id
        ).group_by(
            Product.category
        ).all()
        
        result = []
        for row in categories:
            if row.category:
                result.append({
                    'category': row.category,
                    'total_reviews': row.total_reviews,
                    'avg_rating': round(float(row.avg_rating), 2) if row.avg_rating else None,
                    'avg_positive_score': round(float(row.avg_positive_score), 4) if row.avg_positive_score else 0,
                    'avg_negative_score': round(float(row.avg_negative_score), 4) if row.avg_negative_score else 0,
                    'avg_neutral_score': round(float(row.avg_neutral_score), 4) if row.avg_neutral_score else 0
                })
        
        return jsonify({
            'categories': result,
            'total_categories': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Category comparison failed: {e}")
        return jsonify({'error': str(e)}), 500


@comparative_bp.route('/comparative/competitors', methods=['GET'])
def compare_with_competitors():
    """
    Compare product sentiment with competitors.
    
    Query Parameters:
        product_id (required): Product ID to compare
    
    Returns:
        JSON response with competitor comparison
    """
    product_id = request.args.get('product_id')
    
    if not product_id:
        return jsonify({'error': 'product_id parameter is required'}), 400
    
    try:
        product = Product.query.get(product_id)
        if not product:
            return jsonify({'error': 'Product not found'}), 404
        
        # Get product sentiment
        product_sentiment = SentimentResult.get_sentiment_distribution(product_id)
        product_scores = SentimentResult.get_average_scores(product_id)
        
        # Get competitor products (same category)
        competitors = Product.query.filter(
            Product.category == product.category,
            Product.product_id != product_id,
            Product.is_competitor == True
        ).all()
        
        competitor_data = []
        for competitor in competitors:
            comp_sentiment = SentimentResult.get_sentiment_distribution(competitor.product_id)
            comp_scores = SentimentResult.get_average_scores(competitor.product_id)
            
            competitor_data.append({
                'product_id': competitor.product_id,
                'product_name': competitor.product_name,
                'sentiment_distribution': comp_sentiment,
                'average_scores': comp_scores
            })
        
        return jsonify({
            'product': {
                'product_id': product.product_id,
                'product_name': product.product_name,
                'category': product.category,
                'sentiment_distribution': product_sentiment,
                'average_scores': product_scores
            },
            'competitors': competitor_data,
            'total_competitors': len(competitor_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Competitor comparison failed: {e}")
        return jsonify({'error': str(e)}), 500
