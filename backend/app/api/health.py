"""
Health Check API Blueprint
==========================
Provides health check endpoints for monitoring application status,
database connectivity, and model availability.
"""

from flask import Blueprint, jsonify, current_app
from sqlalchemy import text
from app.database import db
from app.utils.logger import get_logger

logger = get_logger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the application, database, and model.
    
    Returns:
        JSON response with health status information
    """
    health_status = {
        'status': 'healthy',
        'application': 'running',
        'database': 'unknown',
        'model': 'not_loaded',
        'version': '1.0.0'
    }
    
    # Check database connectivity
    try:
        with db.engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        health_status['database'] = 'connected'
        logger.debug("Database health check passed")
    except Exception as e:
        health_status['database'] = f'error: {str(e)}'
        health_status['status'] = 'degraded'
        logger.error(f"Database health check failed: {e}")
    
    # Check model availability
    model_service = current_app.extensions.get('model_service')
    if model_service and model_service.is_loaded:
        health_status['model'] = 'loaded'
    else:
        health_status['model'] = 'not_loaded'
    
    # Determine overall status
    if health_status['database'] != 'connected':
        health_status['status'] = 'unhealthy'
    elif health_status['model'] == 'not_loaded':
        health_status['status'] = 'degraded'
    
    status_code = 200 if health_status['status'] in ['healthy', 'degraded'] else 503
    return jsonify(health_status), status_code


@health_bp.route('/health/database', methods=['GET'])
def database_health():
    """
    Detailed database health check.
    
    Returns:
        JSON response with database connection pool statistics
    """
    try:
        with db.engine.connect() as conn:
            # Get connection pool stats
            pool = db.engine.pool
            pool_status = {
                'size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
            }
            
            # Get database version
            result = conn.execute(text('SELECT version()'))
            db_version = result.fetchone()[0]
            
            # Get table counts
            from app.models import Product, Review, SentimentResult
            
            return jsonify({
                'status': 'connected',
                'database_version': db_version,
                'connection_pool': pool_status,
                'table_counts': {
                    'products': Product.query.count(),
                    'reviews': Review.query.count(),
                    'sentiment_results': SentimentResult.query.count()
                }
            }), 200
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return jsonify({
            'status': 'disconnected',
            'error': str(e)
        }), 503
