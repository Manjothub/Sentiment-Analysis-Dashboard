"""
Database Connection Module
==========================
Manages database connections, sessions, and provides utility functions
for database operations. Handles connection pooling and error recovery.
"""

import os
import time
from typing import Optional
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from app.config import Config


def get_database_uri() -> str:
    """
    Construct database URI from environment variables.
    
    Returns:
        PostgreSQL connection URI string
    """
    return f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"


def create_database_engine(uri: Optional[str] = None, pool_size: int = 10, max_retries: int = 3):
    """
    Create a SQLAlchemy database engine with connection pooling.
    
    Args:
        uri: Database URI. If None, uses Config defaults.
        pool_size: Connection pool size
        max_retries: Number of times to retry connection on failure
        
    Returns:
        SQLAlchemy engine instance
        
    Raises:
        OperationalError: If unable to connect after max_retries
    """
    if uri is None:
        uri = get_database_uri()
    
    engine = create_engine(
        uri,
        pool_size=pool_size,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=Config.DEBUG
    )
    
    # Test connection with retries
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return engine
        except OperationalError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Database connection failed (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise OperationalError(f"Unable to connect to database after {max_retries} attempts: {e}")
    
    return engine


def create_session_factory(engine):
    """
    Create a thread-local scoped session factory.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        scoped_session factory
    """
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    return Session


def test_connection(engine) -> bool:
    """
    Test database connection and return status.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT version()")
            version = result.fetchone()[0]
            print(f"Connected to PostgreSQL: {version}")
            
            # Test if database exists
            db_name = Config.DB_NAME
            result = conn.execute(
                f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"
            )
            if result.fetchone():
                print(f"Database '{db_name}' exists.")
            else:
                print(f"Database '{db_name}' does not exist yet. It will be created on first use.")
            
            return True
    except SQLAlchemyError as e:
        print(f"Database connection test failed: {e}")
        return False


def get_table_names(engine) -> list:
    """
    Get list of table names in the database.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        List of table names
    """
    inspector = inspect(engine)
    return inspector.get_table_names()


def table_exists(engine, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        engine: SQLAlchemy engine instance
        table_name: Name of the table to check
        
    Returns:
        True if table exists, False otherwise
    """
    return table_name in get_table_names(engine)
