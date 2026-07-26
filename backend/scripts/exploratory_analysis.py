"""
Exploratory Data Analysis (EDA) Script
========================================
Performs comprehensive EDA on the preprocessed datasets.
Generates statistics, visualizations, and insights.

Usage:
    python scripts/exploratory_analysis.py
    python scripts/exploratory_analysis.py --dataset ../dataset/processed/final_dataset.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load preprocessed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset: {filepath}")
    logger.info(f"Shape: {df.shape}")
    return df


def analyze_dataset_size(df: pd.DataFrame) -> dict:
    """Analyze dataset size and basic info."""
    logger.info("\n" + "=" * 50)
    logger.info("DATASET SIZE ANALYSIS")
    logger.info("=" * 50)
    
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'column_names': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    logger.info(f"Total rows: {info['total_rows']:,}")
    logger.info(f"Total columns: {info['total_columns']}")
    logger.info(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    logger.info(f"Columns: {info['column_names']}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    return info


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values in the dataset."""
    logger.info("\n" + "=" * 50)
    logger.info("MISSING VALUES ANALYSIS")
    logger.info("=" * 50)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'column': missing.index,
        'missing_count': missing.values,
        'missing_percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    logger.info(f"Total missing values: {df.isnull().sum().sum():,}")
    logger.info(f"Columns with missing values: {len(missing_df)}")
    
    if len(missing_df) > 0:
        logger.info("\nMissing value details:")
        for _, row in missing_df.iterrows():
            logger.info(f"  {row['column']}: {int(row['missing_count']):,} ({row['missing_percentage']:.2f}%)")
    
    return {
        'total_missing': int(df.isnull().sum().sum()),
        'columns_with_missing': len(missing_df),
        'missing_details': missing_df.to_dict('records')
    }


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """Analyze duplicate rows."""
    logger.info("\n" + "=" * 50)
    logger.info("DUPLICATE ANALYSIS")
    logger.info("=" * 50)
    
    total_duplicates = df.duplicated().sum()
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    text_duplicates = df.duplicated(subset=[text_col]).sum() if text_col in df.columns else 0
    
    logger.info(f"Total duplicate rows: {total_duplicates:,} ({total_duplicates/len(df)*100:.2f}%)")
    logger.info(f"Duplicate {text_col}: {text_duplicates:,} ({text_duplicates/len(df)*100:.2f}%)")
    
    return {
        'total_duplicates': int(total_duplicates),
        'text_duplicates': int(text_duplicates)
    }


def analyze_review_length(df: pd.DataFrame) -> dict:
    """Analyze review/tweet length statistics."""
    logger.info("\n" + "=" * 50)
    logger.info("REVIEW LENGTH ANALYSIS")
    logger.info("=" * 50)
    
    if 'review_length' in df.columns:
        lengths = df['review_length']
    elif 'cleaned_text' in df.columns:
        lengths = df['cleaned_text'].str.len()
    else:
        lengths = df['text'].str.len() if 'text' in df.columns else pd.Series()
    
    stats = {
        'average_length': float(lengths.mean()),
        'median_length': float(lengths.median()),
        'std_length': float(lengths.std()),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'q25': float(lengths.quantile(0.25)),
        'q75': float(lengths.quantile(0.75)),
        'q90': float(lengths.quantile(0.90)),
        'q95': float(lengths.quantile(0.95)),
        'q99': float(lengths.quantile(0.99))
    }
    
    logger.info(f"Average length: {stats['average_length']:.1f} characters")
    logger.info(f"Median length: {stats['median_length']:.1f} characters")
    logger.info(f"Std deviation: {stats['std_length']:.1f} characters")
    logger.info(f"Shortest: {stats['min_length']} characters")
    logger.info(f"Longest: {stats['max_length']} characters")
    logger.info(f"25th percentile: {stats['q25']:.1f}")
    logger.info(f"75th percentile: {stats['q75']:.1f}")
    logger.info(f"95th percentile: {stats['q95']:.1f}")
    logger.info(f"99th percentile: {stats['q99']:.1f}")
    
    return stats


def analyze_sentiment_distribution(df: pd.DataFrame) -> dict:
    """Analyze sentiment label distribution."""
    logger.info("\n" + "=" * 50)
    logger.info("SENTIMENT DISTRIBUTION ANALYSIS")
    logger.info("=" * 50)
    
    sentiment_col = 'sentiment_label' if 'sentiment_label' in df.columns else 'sentiment'
    if sentiment_col not in df.columns:
        logger.warning("No sentiment column found")
        return {}
    
    distribution = df[sentiment_col].value_counts()
    percentages = (distribution / len(df) * 100).round(2)
    
    logger.info("Distribution:")
    for label, count in distribution.items():
        logger.info(f"  {label}: {count:,} ({percentages[label]:.1f}%)")
    
    # Check for imbalance
    max_pct = percentages.max()
    min_pct = percentages.min()
    imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
    
    logger.info(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        logger.warning("⚠️ Significant class imbalance detected!")
    
    return {
        'distribution': distribution.to_dict(),
        'percentages': percentages.to_dict(),
        'imbalance_ratio': float(imbalance_ratio)
    }


def analyze_rating_distribution(df: pd.DataFrame) -> dict:
    """Analyze rating distribution (for Amazon data)."""
    logger.info("\n" + "=" * 50)
    logger.info("RATING DISTRIBUTION ANALYSIS")
    logger.info("=" * 50)
    
    if 'Score' not in df.columns and 'raw_rating' not in df.columns:
        logger.info("No rating column found in dataset")
        return {}
    
    rating_col = 'Score' if 'Score' in df.columns else 'raw_rating'
    distribution = df[rating_col].value_counts().sort_index()
    percentages = (distribution / len(df) * 100).round(2)
    
    logger.info("Rating distribution:")
    for rating, count in distribution.items():
        logger.info(f"  Rating {rating}: {count:,} ({percentages[rating]:.1f}%)")
    
    return {
        'distribution': distribution.to_dict(),
        'percentages': percentages.to_dict(),
        'mean_rating': float(df[rating_col].mean()),
        'median_rating': float(df[rating_col].median())
    }


def analyze_frequent_words(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Analyze most and least frequent words."""
    logger.info("\n" + "=" * 50)
    logger.info("WORD FREQUENCY ANALYSIS")
    logger.info("=" * 50)
    
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    if text_col not in df.columns:
        logger.warning("No text column found")
        return {}
    
    # Sample if dataset is large
    sample_size = min(100000, len(df))
    sampled_texts = df[text_col].sample(sample_size, random_state=42)
    
    # Count words
    all_words = []
    for text in sampled_texts:
        if isinstance(text, str):
            words = text.lower().split()
            all_words.extend([w for w in words if len(w) > 2])  # Filter short words
    
    word_counts = Counter(all_words)
    
    most_common = word_counts.most_common(top_n)
    least_common = word_counts.most_common()[-top_n:] if len(word_counts) >= top_n else []
    
    logger.info(f"\nTop {top_n} most frequent words:")
    for i, (word, count) in enumerate(most_common, 1):
        logger.info(f"  {i:2d}. '{word}': {count:,}")
    
    logger.info(f"\nTop {top_n} least frequent words:")
    for word, count in least_common:
        logger.info(f"  '{word}': {count:,}")
    
    return {
        'most_frequent': dict(most_common),
        'least_frequent': dict(least_common),
        'total_unique_words': len(word_counts),
        'total_words_sampled': len(all_words)
    }


def analyze_source_distribution(df: pd.DataFrame) -> dict:
    """Analyze distribution by data source."""
    logger.info("\n" + "=" * 50)
    logger.info("SOURCE DISTRIBUTION ANALYSIS")
    logger.info("=" * 50)
    
    if 'source' not in df.columns:
        logger.info("No source column found")
        return {}
    
    distribution = df['source'].value_counts()
    percentages = (distribution / len(df) * 100).round(2)
    
    logger.info("Source distribution:")
    for source, count in distribution.items():
        logger.info(f"  {source}: {count:,} ({percentages[source]:.1f}%)")
    
    return {
        'distribution': distribution.to_dict(),
        'percentages': percentages.to_dict()
    }


def run_full_eda(dataset_path: str) -> dict:
    """Run complete EDA pipeline."""
    logger.info("=" * 60)
    logger.info(f"STARTING EXPLORATORY DATA ANALYSIS")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    df = load_dataset(dataset_path)
    
    results = {
        'dataset_info': analyze_dataset_size(df),
        'missing_values': analyze_missing_values(df),
        'duplicates': analyze_duplicates(df),
        'review_length': analyze_review_length(df),
        'sentiment_distribution': analyze_sentiment_distribution(df),
        'rating_distribution': analyze_rating_distribution(df),
        'word_frequency': analyze_frequent_words(df),
        'source_distribution': analyze_source_distribution(df)
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("EDA COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run EDA on preprocessed dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Path to preprocessed dataset')
    
    args = parser.parse_args()
    
    setup_logging()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # backend/scripts/
    backend_dir = os.path.dirname(script_dir)                  # backend/
    project_dir = os.path.dirname(backend_dir)                 # project root
    
    if args.dataset is None:
        args.dataset = os.path.join(project_dir, 'dataset', 'processed', 'final_dataset.csv')
    
    if not os.path.exists(args.dataset):
        # Try amazon_cleaned.csv as fallback
        args.dataset = os.path.join(project_dir, 'dataset', 'processed', 'amazon_cleaned.csv')
        
        if not os.path.exists(args.dataset):
            logger.error(f"No preprocessed dataset found. Run preprocessing scripts first.")
            sys.exit(1)
    
    results = run_full_eda(args.dataset)
    logger.info(f"EDA results saved with {sum(len(v) if isinstance(v, dict) else 0 for v in results.values())} metrics")


if __name__ == '__main__':
    main()
