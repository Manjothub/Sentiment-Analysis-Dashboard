"""
Database Package
================
Exports database initialization and connection utilities.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Initialize SQLAlchemy instance
db = SQLAlchemy()


def init_db(app):
    """
    Initialize database with Flask application.
    
    Args:
        app: Flask application instance
    """
    db.init_app(app)
    
    with app.app_context():
        # Import all models to ensure they are registered with SQLAlchemy
        from app.models import Product, Review, SentimentResult
        
        # Create all tables
        db.create_all()
        
        app.logger.info("Database initialized successfully.")


__all__ = ['db', 'init_db']
