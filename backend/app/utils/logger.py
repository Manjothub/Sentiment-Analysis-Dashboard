"""
Logging Configuration Module
============================
Configures application-wide logging with both file and console handlers.
Supports log rotation, different log levels per environment, and structured logging.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from app.config import Config


def setup_logging(app=None, log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Sets up:
    - Console handler for stdout output
    - Rotating file handler for persistent logs
    - Different log levels based on environment
    
    Args:
        app: Flask application instance (optional)
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured root logger instance
    """
    # Determine log level
    if log_level is None:
        log_level = getattr(Config, 'LOG_LEVEL', 'INFO')
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = getattr(Config, 'LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = os.path.join(log_dir, 'sentiment_dashboard.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=getattr(Config, 'LOG_FILE_MAX_BYTES', 10485760),  # 10MB
        backupCount=getattr(Config, 'LOG_FILE_BACKUP_COUNT', 5)
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    error_log_file = os.path.join(log_dir, 'error.log')
    error_file_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=getattr(Config, 'LOG_FILE_MAX_BYTES', 10485760),
        backupCount=3
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_file_handler)
    
    # Log startup message
    root_logger.info(f"Logging initialized. Level: {log_level}, File: {log_file}")
    
    # If Flask app is provided, configure its logger
    if app:
        app.logger.handlers = root_logger.handlers
        app.logger.setLevel(numeric_level)
        app.logger.info(f"Flask application logger configured with level: {log_level}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance configured with the application's logging settings
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.logger.info("MyClass initialized")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
