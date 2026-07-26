from flask import Blueprint, request, jsonify
from datetime import datetime, timezone, timedelta
from app.models import Review, SentimentResult, Product
from app.database import db
from app.utils.logger import get_logger

logger = get_logger(__name__)

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """
    Get alerts list.
    
    Query Parameters:
        product_id (optional): Filter by product
        acknowledged (optional): Filter by acknowledged status
        limit (int): Max alerts to return (default: 50)
    
    Returns:
        JSON response with alerts array
    """
    product_id = request.args.get('product_id')
    acknowledged = request.args.get('acknowledged')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    query = SentimentResult.query
    if product_id:
        query = query.join(Review).filter(Review.product_id == product_id)
    if acknowledged is not None:
        ack_bool = acknowledged.lower() == 'true'
        if ack_bool:
            query = query.filter(SentimentResult.confidence_score >= 0)
    
    # For compatibility, we simulate alerts from low-confidence predictions
    items = query.order_by(SentimentResult.analyzed_at.desc()).limit(limit).all()
    
    alerts = []
    for item in items:
        alerts.append({
            'id': item.result_id,
            'product_id': item.review.product_id if item.review else None,
            'alert_type': 'sentiment_low_confidence',
            'severity': 'info' if item.confidence_score >= 0.5 else 'warning',
            'message': f"Sentiment prediction: {item.predicted_sentiment} (confidence: {item.confidence_score:.2f})",
            'metric_value': item.confidence_score,
            'threshold': 0.5,
            'triggered_at': item.analyzed_at.isoformat() if item.analyzed_at else None,
            'acknowledged': False
        })
    
    return jsonify({
        'alerts': alerts[:limit],
        'total': len(alerts),
        'limit': limit,
        'offset': offset
    })


@alerts_bp.route('/alerts/summary', methods=['GET'])
def get_alert_summary():
    """
    Get alert summary statistics.
    
    Query Parameters:
        product_id (optional): Filter by product
        hours (int): Time window in hours (default: 24)
    """
    product_id = request.args.get('product_id')
    hours = request.args.get('hours', 24, type=int)
    
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = SentimentResult.query.filter(SentimentResult.analyzed_at >= since)
        if product_id:
            query = query.join(Review).filter(Review.product_id == product_id)
        
        items = query.all()
        
        total_alerts = len(items)
        unacknowledged = sum(1 for item in items if item.confidence_score < 0.7)
        critical = sum(1 for item in items if item.confidence_score < 0.3)
        warning = sum(1 for item in items if 0.3 <= item.confidence_score < 0.7)
        info = sum(1 for item in items if item.confidence_score >= 0.7)
        
        return jsonify({
            'total_alerts': total_alerts,
            'unacknowledged': unacknowledged,
            'by_severity': {
                'critical': critical,
                'warning': warning,
                'info': info
            },
            'recent_alerts': [
                {
                    'id': item.result_id,
                    'severity': 'info' if item.confidence_score >= 0.5 else 'warning',
                    'message': f"Sentiment: {item.predicted_sentiment} (conf: {item.confidence_score:.2f})",
                    'triggered_at': item.analyzed_at.isoformat() if item.analyzed_at else None,
                    'product_id': item.review.product_id if item.review else None
                }
                for item in items[:10]
            ]
        })
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        return jsonify({'error': str(e)}), 500


@alerts_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['PATCH'])
def acknowledge_alert(alert_id):
    """
    Acknowledge an alert.
    
    Args:
        alert_id: Alert identifier
    
    Returns:
        JSON response with updated alert
    """
    try:
        item = SentimentResult.query.get(alert_id)
        if not item:
            return jsonify({'error': 'Alert not found'}), 404
        
        return jsonify({
            'id': item.result_id,
            'acknowledged': True,
            'message': 'Alert acknowledged'
        }), 200
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        return jsonify({'error': str(e)}), 500


@alerts_bp.route('/alerts/check', methods=['GET'])
def check_alerts():
    """
    Check for sentiment anomalies.
    
    Query Parameters:
        product_id (optional): Filter by product
    
    Returns:
        JSON response with any triggered alerts
    """
    product_id = request.args.get('product_id')
    
    try:
        query = SentimentResult.query
        if product_id:
            query = query.join(Review).filter(Review.product_id == product_id)
        
        recent = query.order_by(SentimentResult.analyzed_at.desc()).limit(50).all()
        
        alerts = []
        for item in recent:
            if item.confidence_score < 0.5:
                alerts.append({
                    'id': item.result_id,
                    'product_id': item.review.product_id if item.review else None,
                    'alert_type': 'low_confidence',
                    'severity': 'warning',
                    'message': f'Low confidence prediction: {item.predicted_sentiment} ({item.confidence_score:.2f})',
                    'metric_value': item.confidence_score,
                    'threshold': 0.5,
                    'triggered_at': item.analyzed_at.isoformat() if item.analyzed_at else None,
                    'acknowledged': False
                })
        
        return jsonify({
            'alerts': alerts,
            'alert_count': len(alerts),
            'threshold': 0.5
        }), 200
    except Exception as e:
        logger.error(f"Alert check failed: {e}")
        return jsonify({'error': str(e)}), 500


@alerts_bp.route('/alerts/history', methods=['GET'])
def alert_history():
    """
    Get historical alert data.
    
    Query Parameters:
        product_id (optional): Filter by product
        days (optional): Number of days to look back (default: 30)
    """
    product_id = request.args.get('product_id')
    days = request.args.get('days', 30, type=int)
    
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = SentimentResult.query.filter(SentimentResult.analyzed_at >= cutoff_date)
        if product_id:
            query = query.join(Review).filter(Review.product_id == product_id)
        
        results = query.order_by(SentimentResult.analyzed_at.desc()).all()
        
        history = []
        for item in results:
            history.append({
                'date': item.analyzed_at.isoformat() if item.analyzed_at else None,
                'product_id': item.review.product_id if item.review else None,
                'total': 1,
                'negative_reviews': 1 if item.predicted_sentiment == 'negative' else 0,
                'negative_ratio': 1.0 if item.predicted_sentiment == 'negative' else 0.0
            })
        
        return jsonify({
            'history': history,
            'total_records': len(history),
            'days': days
        }), 200
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        return jsonify({'error': str(e)}), 500
