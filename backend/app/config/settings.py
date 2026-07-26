"""
Configuration Settings
======================
Environment-aware configuration classes for the Sentiment Analysis Dashboard.
Uses python-dotenv to load environment variables from .env file.

Configuration Classes:
    - Config: Base configuration with common settings
    - DevelopmentConfig: Development environment settings
    - ProductionConfig: Production environment settings  
    - TestingConfig: Testing environment settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Base configuration class.
    Contains settings common to all environments.
    """
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = False
    TESTING = False

    # Database - PostgreSQL
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'sentiment_dashboard')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

    # Construct SQLAlchemy database URI
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
    }

    # Model paths
    MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ml_models', 'saved_models'))
    MODEL_PATH = os.path.join(MODEL_DIR, 'distilbert-sentiment')

    # Dataset paths
    DATASET_DIR = os.getenv('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'dataset'))
    PROCESSED_DIR = os.getenv('PROCESSED_DIR', os.path.join(DATASET_DIR, 'processed'))

    # Logging
    LOG_DIR = os.getenv('LOG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE_MAX_BYTES = 10485760  # 10MB
    LOG_FILE_BACKUP_COUNT = 5

    # Preprocessing
    MIN_REVIEW_LENGTH = 10
    MAX_REVIEW_LENGTH = 2000
    BATCH_SIZE = 64
    RANDOM_SEED = 42

    # API
    API_RATE_LIMIT = 100  # requests per minute
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 1800,
        'pool_pre_ping': True,
        'max_overflow': 10,
    }


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # Use in-memory SQLite for tests
