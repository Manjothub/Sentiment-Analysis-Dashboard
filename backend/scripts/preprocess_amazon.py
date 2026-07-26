"""
Amazon Reviews Preprocessing Script
====================================
Preprocesses the Amazon Reviews dataset for sentiment analysis.
Includes text cleaning, label mapping, and data validation.

Dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews
Location: dataset/amazon-reviews/Reviews.csv
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def load_amazon_reviews(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load Amazon Reviews CSV file.
    
    Args:
        filepath: Path to Reviews.csv
        nrows: Number of rows to load (None for all)
        
    Returns:
        DataFrame with Amazon reviews
    """
    logger.info(f"Loading Amazon reviews from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath, nrows=nrows)
    logger.info(f"Loaded {len(df):,} reviews with {len(df.columns)} columns")
    
    return df


def clean_text(text: str) -> str:
    """
    Clean and normalize review text.
    
    Steps:
    1. Remove HTML tags
    2. Remove URLs
    3. Remove email addresses
    4. Remove emojis and special characters
    5. Remove punctuation
    6. Normalize whitespace
    7. Convert to lowercase
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|https\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove emojis and special Unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove numbers (optional - keep for now)
    # text = re.sub(r'\d+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    return text


def map_rating_to_sentiment(rating: int) -> str:
    """
    Map numeric rating to sentiment label.
    
    Args:
        rating: Numeric rating (1-5)
        
    Returns:
        Sentiment label: 'positive', 'negative', or 'neutral'
    """
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'


def preprocess_amazon(
    input_path: str,
    output_path: str,
    nrows: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 2000,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Main preprocessing function for Amazon Reviews.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV
        nrows: Number of rows to process (None for all)
        min_length: Minimum review length to keep
        max_length: Maximum review length to keep
        remove_duplicates: Whether to remove duplicate reviews
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("=" * 60)
    logger.info("Starting Amazon Reviews preprocessing")
    logger.info("=" * 60)
    
    # Step 1: Load data
    df = load_amazon_reviews(input_path, nrows)
    initial_count = len(df)
    
    # Step 2: Filter required columns
    required_cols = ['Id', 'ProductId', 'UserId', 'ProfileName', 
                     'HelpfulnessNumerator', 'HelpfulnessDenominator',
                     'Score', 'Time', 'Summary', 'Text']
    
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]
    logger.info(f"Using {len(available_cols)} columns: {available_cols}")
    
    # Step 3: Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna(subset=['Text', 'Score'])
    missing_after = df.isnull().sum().sum()
    logger.info(f"Removed {missing_before - missing_after} rows with missing values")
    
    # Step 4: Validate ratings
    df = df[df['Score'].between(1, 5)]
    logger.info(f"Rows with valid ratings (1-5): {len(df):,}")
    
    # Step 5: Remove duplicates
    if remove_duplicates:
        duplicate_count = df.duplicated(subset=['Text']).sum()
        df = df.drop_duplicates(subset=['Text'])
        logger.info(f"Removed {duplicate_count} duplicate reviews")
    
    # Step 6: Clean text
    logger.info("Cleaning review text...")
    df['cleaned_text'] = df['Text'].apply(clean_text)
    
    # Step 7: Filter by review length
    df['review_length'] = df['cleaned_text'].str.len()
    df = df[df['review_length'].between(min_length, max_length)]
    logger.info(f"Rows after length filter ({min_length}-{max_length} chars): {len(df):,}")
    
    # Step 8: Map ratings to sentiment labels
    df['sentiment_label'] = df['Score'].apply(map_rating_to_sentiment)
    
    # Step 9: Convert timestamp
    if 'Time' in df.columns:
        df['review_date'] = pd.to_datetime(df['Time'], unit='s')
        df = df.drop(columns=['Time'])
    
    # Step 10: Add helpfulness ratio
    if 'HelpfulnessNumerator' in df.columns and 'HelpfulnessDenominator' in df.columns:
        df['helpfulness_ratio'] = np.where(
            df['HelpfulnessDenominator'] > 0,
            df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'],
            0
        )
    
    # Step 11: Add source column
    df['source'] = 'amazon'
    
    # Step 12: Sort by review date
    if 'review_date' in df.columns:
        df = df.sort_values('review_date', ascending=False)
    
    # Step 13: Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_path}")
    
    # Step 14: Log summary statistics
    sentiment_dist = df['sentiment_label'].value_counts()
    rating_dist = df['Score'].value_counts().sort_index()
    
    logger.info(f"\nProcessing Summary:")
    logger.info(f"{'=' * 40}")
    logger.info(f"Initial rows: {initial_count:,}")
    logger.info(f"Final rows: {len(df):,}")
    logger.info(f"Rows removed: {initial_count - len(df):,} ({(initial_count - len(df)) / initial_count * 100:.1f}%)")
    logger.info(f"\nSentiment Distribution:")
    for label, count in sentiment_dist.items():
        logger.info(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    logger.info(f"\nRating Distribution:")
    for rating, count in rating_dist.items():
        logger.info(f"  {rating}: {count:,} ({count/len(df)*100:.1f}%)")
    logger.info(f"\nAverage review length: {df['review_length'].mean():.1f} chars")
    logger.info(f"Median review length: {df['review_length'].median():.1f} chars")
    logger.info(f"Total unique products: {df['ProductId'].nunique():,}")
    logger.info(f"Total unique users: {df['UserId'].nunique():,}")
    logger.info(f"{'=' * 40}")
    
    return df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Preprocess Amazon Reviews dataset for sentiment analysis'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input Reviews.csv file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save cleaned CSV file'
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=None,
        help='Number of rows to process (default: all)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum review length in characters (default: 10)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2000,
        help='Maximum review length in characters (default: 2000)'
    )
    
    args = parser.parse_args()
    
    # Determine default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # backend/scripts/
    backend_dir = os.path.dirname(script_dir)                  # backend/
    project_dir = os.path.dirname(backend_dir)                 # project root
    
    if args.input is None:
        args.input = os.path.join(project_dir, 'dataset', 'amazon-reviews', 'Reviews.csv')
    
    if args.output is None:
        args.output = os.path.join(project_dir, 'dataset', 'processed', 'amazon_cleaned.csv')
    
    # Setup logging
    setup_logging()
    
    # Run preprocessing
    df = preprocess_amazon(
        input_path=args.input,
        output_path=args.output,
        nrows=args.nrows,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    logger.info(f"Preprocessing complete. Dataset saved to: {args.output}")
    return df


if __name__ == '__main__':
    main()
