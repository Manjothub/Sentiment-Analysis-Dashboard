"""
Twitter Sentiment Preprocessing Script
=======================================
Preprocesses the Twitter Sentiment dataset for sentiment analysis.
Uses the Sentiment140 dataset with 1.6 million tweets.

Dataset: https://www.kaggle.com/kazanova/sentiment140
Location: dataset/twitter-sentiment/training.1600000.processed.noemoticon.csv
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# Sentiment140 column names (CSV has no header)
TWITTER_COLUMNS = [
    'sentiment',       # 0 = negative, 2 = neutral, 4 = positive
    'id',              # Tweet ID
    'date',            # Date of tweet
    'query',           # Query (NO_QUERY if no query)
    'user',            # Username
    'text'             # Tweet text
]

# Sentiment mapping for Sentiment140
SENTIMENT140_MAP = {
    0: 'negative',
    2: 'neutral',
    4: 'positive'
}


def load_twitter_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load Twitter Sentiment140 CSV file.
    
    Args:
        filepath: Path to CSV file
        nrows: Number of rows to load (None for all)
        
    Returns:
        DataFrame with Twitter data
    """
    logger.info(f"Loading Twitter data from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(
        filepath,
        encoding='latin-1',
        header=None,
        names=TWITTER_COLUMNS,
        nrows=nrows
    )
    logger.info(f"Loaded {len(df):,} tweets with {len(df.columns)} columns")
    
    return df


def clean_tweet_text(text: str) -> str:
    """
    Clean and normalize tweet text.
    
    Steps:
    1. Remove HTML tags
    2. Remove URLs
    3. Remove usernames (@mentions)
    4. Remove hashtags
    5. Remove email addresses
    6. Remove emojis
    7. Remove punctuation
    8. Normalize whitespace
    9. Convert to lowercase
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|https\S+|www\.\S+', ' ', text)
    
    # Remove @usernames
    text = re.sub(r'@\w+', ' ', text)
    
    # Remove hashtags (keep the text after #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove emojis and special Unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    return text


def preprocess_twitter(
    input_path: str,
    output_path: str,
    nrows: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 280,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Main preprocessing function for Twitter Sentiment data.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV
        nrows: Number of rows to process (None for all)
        min_length: Minimum tweet length to keep
        max_length: Maximum tweet length to keep
        remove_duplicates: Whether to remove duplicate tweets
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("=" * 60)
    logger.info("Starting Twitter Sentiment preprocessing")
    logger.info("=" * 60)
    
    # Step 1: Load data
    df = load_twitter_data(input_path, nrows)
    initial_count = len(df)
    
    # Step 2: Map sentiment labels
    df['sentiment_label'] = df['sentiment'].map(SENTIMENT140_MAP)
    logger.info(f"Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
    
    # Step 3: Remove rows with unmapped sentiment
    df = df.dropna(subset=['sentiment_label'])
    logger.info(f"Rows after sentiment mapping: {len(df):,}")
    
    # Step 4: Remove duplicates
    if remove_duplicates:
        duplicate_count = df.duplicated(subset=['text']).sum()
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Removed {duplicate_count} duplicate tweets")
    
    # Step 5: Clean text
    logger.info("Cleaning tweet text...")
    df['cleaned_text'] = df['text'].apply(clean_tweet_text)
    
    # Step 6: Filter by text length
    df['review_length'] = df['cleaned_text'].str.len()
    df = df[df['review_length'].between(min_length, max_length)]
    logger.info(f"Rows after length filter ({min_length}-{max_length} chars): {len(df):,}")
    
    # Step 7: Parse dates
    df['review_date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Step 8: Drop empty text
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Step 9: Add source column
    df['source'] = 'twitter'
    
    # Step 10: Sort by date
    df = df.sort_values('review_date', ascending=False)
    
    # Step 11: Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_path}")
    
    # Step 12: Log summary statistics
    sentiment_dist = df['sentiment_label'].value_counts()
    
    logger.info(f"\nProcessing Summary:")
    logger.info(f"{'=' * 40}")
    logger.info(f"Initial rows: {initial_count:,}")
    logger.info(f"Final rows: {len(df):,}")
    logger.info(f"Rows removed: {initial_count - len(df):,} ({(initial_count - len(df)) / initial_count * 100:.1f}%)")
    logger.info(f"\nSentiment Distribution:")
    for label, count in sentiment_dist.items():
        logger.info(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    logger.info(f"\nAverage tweet length: {df['review_length'].mean():.1f} chars")
    logger.info(f"Median tweet length: {df['review_length'].median():.1f} chars")
    logger.info(f"Total unique users: {df['user'].nunique():,}")
    logger.info(f"{'=' * 40}")
    
    return df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Preprocess Twitter Sentiment140 dataset'
    )
    parser.add_argument('--input', type=str, default=None, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Path to save cleaned CSV')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to process')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # backend/scripts/
    backend_dir = os.path.dirname(script_dir)                  # backend/
    project_dir = os.path.dirname(backend_dir)                 # project root
    
    if args.input is None:
        args.input = os.path.join(
            project_dir, 'dataset', 'twitter-sentiment',
            'training.1600000.processed.noemoticon.csv'
        )
    
    if args.output is None:
        args.output = os.path.join(project_dir, 'dataset', 'processed', 'twitter_cleaned.csv')
    
    setup_logging()
    
    df = preprocess_twitter(input_path=args.input, output_path=args.output, nrows=args.nrows)
    logger.info(f"Preprocessing complete. Dataset saved to: {args.output}")
    return df


if __name__ == '__main__':
    main()
