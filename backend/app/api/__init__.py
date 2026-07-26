"""
API Package
===========
Exports all API blueprints for the application.
"""

from app.api.health import health_bp
from app.api.sentiment import sentiment_bp
from app.api.trends import trends_bp
from app.api.alerts import alerts_bp
from app.api.comparative import comparative_bp

__all__ = ['health_bp', 'sentiment_bp', 'trends_bp', 'alerts_bp', 'comparative_bp']
