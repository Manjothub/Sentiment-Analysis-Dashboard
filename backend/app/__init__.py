"""
Flask Application Factory
=========================
Creates and configures the Flask application instance.
Follows the Application Factory pattern for scalability and testability.
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate

from app.config import Config
from app.database import db, init_db
from app.utils.logger import setup_logging

# Initialize extensions
migrate = Migrate()


def create_app(config_name: str = None) -> Flask:
    """
    Application factory function.

    Args:
        config_name: Configuration environment name (development, production, testing).
                     If None, defaults to FLASK_ENV or 'development'.

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    if config_name == 'testing':
        app.config.update({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED': False,
            'SECRET_KEY': 'test-secret-key',
        })
    else:
        app.config.from_object(Config)

    # Setup logging
    setup_logging(app)
    app.logger.info("Starting Sentiment Analysis Dashboard application...")

    # Initialize extensions
    CORS(app)
    db.init_app(app)
    migrate.init_app(app, db)

    # Initialize Socket.IO
    try:
        from models import socketio
        socketio.init_app(app, cors_allowed_origins='*')
        app.logger.info("Socket.IO initialized successfully")
    except Exception as e:
        app.logger.warning(f"Socket.IO initialization failed: {e}")

    # Register blueprints
    _register_blueprints(app)

    # Register ML services
    _register_services(app)

    # Register error handlers
    _register_error_handlers(app)

    # Create database tables if they don't exist
    with app.app_context():
        from app.models import Product, Review, SentimentResult, ModelVersion
        db.create_all()
        app.logger.info("Database tables created/verified successfully.")

    return app


def _register_blueprints(app: Flask) -> None:
    """Register all API blueprints with the Flask application."""
    from app.api.sentiment import sentiment_bp
    from app.api.trends import trends_bp
    from app.api.alerts import alerts_bp
    from app.api.comparative import comparative_bp
    from app.api.health import health_bp
    from app.api.model import model_bp

    app.register_blueprint(sentiment_bp, url_prefix='/api')
    app.register_blueprint(trends_bp, url_prefix='/api')
    app.register_blueprint(alerts_bp, url_prefix='/api')
    app.register_blueprint(comparative_bp, url_prefix='/api')
    app.register_blueprint(health_bp, url_prefix='/api')
    app.register_blueprint(model_bp, url_prefix='/api')

    app.logger.debug("API blueprints registered successfully.")


def _register_services(app: Flask) -> None:
    """Register ML services with the Flask application extensions."""
    try:
        from app.services.ml_pipeline.sentiment_service import SentimentService
        from app.services.ml_pipeline.aspect_service import AspectService
        from app.services.ml_pipeline.topic_model import TopicModeler

        # Determine model path
        model_path = app.config.get('MODEL_PATH', 'distilbert-base-uncased-finetuned-sst-2-english')
        if model_path and not os.path.exists(model_path) and '/' not in model_path and '\\' in model_path:
            model_path = 'distilbert-base-uncased-finetuned-sst-2-english'

        # Initialize services
        sentiment_service = SentimentService(model_path)
        aspect_service = AspectService()
        topic_modeler = TopicModeler()

        # Register with Flask extensions
        app.extensions['model_service'] = sentiment_service
        app.extensions['aspect_service'] = aspect_service
        app.extensions['topic_modeler'] = topic_modeler

        if sentiment_service.is_loaded:
            app.logger.info("Sentiment model loaded and registered successfully")
        else:
            app.logger.warning("Sentiment model not loaded (will use fallback)")

        if aspect_service.is_loaded:
            app.logger.info("Aspect extraction model loaded successfully")

    except Exception as e:
        app.logger.error(f"Failed to register ML services: {e}")
        app.logger.warning("Application will run without ML models")


def _register_error_handlers(app: Flask) -> None:
    """Register global error handlers for the application."""
    @app.errorhandler(400)
    def bad_request(error):
        app.logger.warning(f"Bad request: {error}")
        return {'error': 'Bad request', 'message': str(error)}, 400

    @app.errorhandler(404)
    def not_found(error):
        app.logger.warning(f"Resource not found: {error}")
        return {'error': 'Not found', 'message': 'The requested resource was not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error', 'message': 'An unexpected error occurred'}, 500
