"""
Dataset Preparation Module
==========================
Loads preprocessed Amazon Reviews, performs stratified train/val/test split,
converts to Hugging Face DatasetDict, tokenizes with DistilBERT tokenizer,
and saves the processed dataset and tokenizer.

Usage:
    python -c "from app.services.ml_pipeline.prepare_dataset import prepare_dataset; prepare_dataset()"
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 256
LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
RANDOM_SEED = 42

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'amazon_cleaned.csv')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'hf_dataset')
DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'ml_models', 'saved_models', 'tokenizer')


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load the preprocessed Amazon Reviews CSV and validate its contents.

    Args:
        filepath: Path to the cleaned CSV file

    Returns:
        DataFrame with validated data

    Raises:
        FileNotFoundError: If the dataset file does not exist
        ValueError: If required columns are missing or data is invalid
    """
    logger.info(f"Loading dataset from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

    # Validate required columns
    required_cols = ['cleaned_text', 'sentiment_label']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna(subset=['cleaned_text', 'sentiment_label'])
    missing_after = df.isnull().sum().sum()
    if missing_before > 0:
        logger.warning(f"Removed {missing_before - missing_after} rows with missing values")

    # Validate sentiment labels
    valid_labels = set(LABEL_MAP.keys())
    actual_labels = set(df['sentiment_label'].unique())
    invalid_labels = actual_labels - valid_labels
    if invalid_labels:
        logger.warning(f"Found invalid labels: {invalid_labels}. Filtering them out.")
        df = df[df['sentiment_label'].isin(valid_labels)]

    # Remove duplicates
    dup_before = df.duplicated(subset=['cleaned_text']).sum()
    df = df.drop_duplicates(subset=['cleaned_text'])
    if dup_before > 0:
        logger.info(f"Removed {dup_before} duplicate texts")

    # Filter empty or too-short texts
    df = df[df['cleaned_text'].str.strip().str.len() >= 3]

    logger.info(f"Final dataset: {len(df):,} rows")
    return df


def analyze_class_balance(df: pd.DataFrame) -> dict:
    """
    Analyze class distribution in the dataset.

    Args:
        df: DataFrame with 'sentiment_label' column

    Returns:
        Dictionary with class counts and percentages
    """
    counts = df['sentiment_label'].value_counts()
    percentages = (counts / len(df) * 100).round(2)

    logger.info("Class distribution:")
    for label in ['positive', 'neutral', 'negative']:
        count = counts.get(label, 0)
        pct = percentages.get(label, 0)
        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")

    return {
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict(),
        'total': len(df)
    }


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/validation/test split.

    Args:
        df: DataFrame with 'sentiment_label' column
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)
        test_ratio: Proportion for testing (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # First split: train vs temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['sentiment_label'],
        random_state=random_state
    )

    # Second split: val vs test from temp
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['sentiment_label'],
        random_state=random_state
    )

    logger.info(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Verify class balance in each split
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = split_df['sentiment_label'].value_counts(normalize=True).mul(100).round(1)
        logger.info(f"  {name} distribution: {dist.to_dict()}")

    return train_df, val_df, test_df


def tokenize_function(examples, tokenizer):
    """
    Tokenize text examples for DistilBERT.

    Args:
        examples: Dictionary with 'cleaned_text' key containing list of texts
        tokenizer: Hugging Face tokenizer instance

    Returns:
        Dictionary with input_ids, attention_mask, and labels
    """
    return tokenizer(
        examples['cleaned_text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None  # Return lists for Dataset compatibility
    )


def create_dataset_dict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> DatasetDict:
    """
    Create a Hugging Face DatasetDict from pandas DataFrames.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Hugging Face tokenizer
        output_dir: Directory to save the dataset

    Returns:
        DatasetDict with train/val/test splits
    """
    logger.info("Creating Hugging Face DatasetDict...")

    # Convert labels to numeric
    for df in [train_df, val_df, test_df]:
        df['label'] = df['sentiment_label'].map(LABEL_MAP)

    # Create Dataset objects
    train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'label']])

    # Tokenize all splits
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['cleaned_text']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['cleaned_text']
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['cleaned_text']
    )

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Dataset saved to: {output_dir}")

    # Log dataset statistics
    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")
    logger.info(f"Test samples: {len(test_dataset):,}")

    return dataset_dict


def prepare_dataset(
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    force_recreate: bool = False
) -> DatasetDict:
    """
    Complete dataset preparation pipeline.

    Args:
        input_path: Path to cleaned CSV
        output_dir: Directory to save Hugging Face dataset
        tokenizer_path: Path to save tokenizer
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        force_recreate: If True, recreate even if output exists

    Returns:
        DatasetDict with train/val/test splits
    """
    logger.info("=" * 60)
    logger.info("STARTING DATASET PREPARATION")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Check if already prepared
    if os.path.exists(output_dir) and not force_recreate:
        logger.info(f"Dataset already exists at: {output_dir}")
        logger.info("Set force_recreate=True to regenerate")
        from datasets import load_from_disk
        return load_from_disk(output_dir)

    # Step 1: Load and validate data
    df = load_and_validate_data(input_path)

    # Step 2: Analyze class balance
    class_info = analyze_class_balance(df)

    # Step 3: Stratified split
    train_df, val_df, test_df = stratified_split(
        df, train_ratio, val_ratio, test_ratio
    )

    # Step 4: Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Save tokenizer
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to: {tokenizer_path}")

    # Step 5: Create DatasetDict
    dataset_dict = create_dataset_dict(
        train_df, val_df, test_df, tokenizer, output_dir
    )

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION COMPLETED")
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    logger.info("=" * 60)

    return dataset_dict


if __name__ == '__main__':
    """Run dataset preparation when executed directly."""
    from app.utils.logger import setup_logging
    setup_logging()
    prepare_dataset()
