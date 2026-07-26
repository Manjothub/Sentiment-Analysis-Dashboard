#!/usr/bin/env python3
"""
Data analysis script for Sentiment Analysis project.
Automatically detects datasets in dataset/amazon-reviews/ and dataset/twitter-sentiment/,
loads them with pandas, and prints exploratory statistics.
"""

import os
import pandas as pd
import numpy as np

def find_dataset_files():
    """Find CSV files in dataset folders."""
    # Navigate: scripts/ -> backend/ -> project_root/ -> dataset/
    base_dir = os.path.dirname(os.path.abspath(__file__))  # backend/scripts/
    backend_dir = os.path.dirname(base_dir)                 # backend/
    project_dir = os.path.dirname(backend_dir)               # project root
    dataset_dir = os.path.join(project_dir, 'dataset')

    files = []
    # Amazon reviews
    amazon_dir = os.path.join(dataset_dir, 'amazon-reviews')
    if os.path.isdir(amazon_dir):
        for f in os.listdir(amazon_dir):
            if f.lower().endswith('.csv'):
                files.append(('amazon', os.path.join(amazon_dir, f)))
    # Twitter sentiment
    twitter_dir = os.path.join(dataset_dir, 'twitter-sentiment')
    if os.path.isdir(twitter_dir):
        for f in os.listdir(twitter_dir):
            if f.lower().endswith('.csv'):
                files.append(('twitter', os.path.join(twitter_dir, f)))
    return files

def load_dataset(filepath):
    """Load CSV with pandas, handling encoding if needed."""
    try:
        # Try default encoding first
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        # Try latin-1 for Twitter data
        df = pd.read_csv(filepath, encoding='latin-1')
    return df

def analyze_dataset(name, df):
    """Print analysis of a single dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing dataset: {name}")
    print(f"{'='*60}")
    print(f"File shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
    print(missing_df[missing_df['missing_count'] > 0])
    if missing.sum() == 0:
        print("  No missing values found.")

    print("\nDuplicate rows:")
    duplicate_count = df.duplicated().sum()
    print(f"  Number of duplicate rows: {duplicate_count:,}")
    if duplicate_count > 0:
        print(f"  Percentage: {duplicate_count/len(df)*100:.2f}%")

    # Analyze target/sentiment column
    print("\nTarget/Label Analysis:")
    # Try to find sentiment/score column
    target_col = None
    possible_target_names = ['sentiment', 'Score', 'label', 'target', 'class', 'rating']
    for col in possible_target_names:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        # Look for any column that might contain sentiment/score
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['sentiment', 'score', 'rating', 'label']):
                target_col = col
                break

    if target_col:
        print(f"  Target column identified: '{target_col}'")
        print(f"  Data type: {df[target_col].dtype}")
        unique_vals = df[target_col].unique()
        print(f"  Unique values: {unique_vals}")
        if len(unique_vals) < 20:  # Show value counts for categorical/discrete
            print("  Value counts:")
            print(df[target_col].value_counts().head(20))
        else:
            print("  Basic statistics:")
            print(df[target_col].describe())
    else:
        print("  No obvious target/sentiment column found.")
        # Show columns that might be text
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            print(f"  Text columns found: {list(text_cols)}")
            # Show sample text length
            for col in text_cols[:3]:  # First 3 text columns
                avg_len = df[col].astype(str).str.len().mean()
                print(f"    Average length of '{col}': {avg_len:.1f} characters")

    # Additional info for Amazon dataset
    if name == 'amazon':
        if 'Score' in df.columns:
            print("\n  Rating distribution (Score 1-5):")
            rating_counts = df['Score'].value_counts().sort_index()
            for rating, count in rating_counts.items():
                print(f"    Rating {rating}: {count:,} ({count/len(df)*100:.1f}%)")

        # Check for helpfulness denominator/numerator if present
        helpful_cols = [c for c in df.columns if 'Helpfulness' in c]
        if helpful_cols:
            print(f"  Helpfulness columns: {helpful_cols}")

    # Additional info for Twitter dataset
    if name == 'twitter':
        if 'sentiment' in df.columns:
            print("\n  Original sentiment distribution:")
            # Sentiment140: 0=negative, 2=neutral, 4=positive
            sentiment_map = {0: 'Negative', 2: 'Neutral', 4: 'Positive'}
            sent_counts = df['sentiment'].value_counts().sort_index()
            for val, count in sent_counts.items():
                label = sentiment_map.get(val, f'Unknown ({val})')
                print(f"    Sentiment {val} ({label}): {count:,} ({count/len(df)*100:.1f}%)")

        # Check for date column
        if 'date' in df.columns:
            print(f"  Date column present: {df['date'].dtype}")
            try:
                df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                print(f"  Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")
            except:
                pass

    print("-" * 60)

def main():
    """Main function to run analysis."""
    print("Starting dataset analysis for Sentiment Analysis project...")

    dataset_files = find_dataset_files()

    if not dataset_files:
        print("No CSV datasets found in dataset/amazon-reviews/ or dataset/twitter-sentiment/")
        return

    print(f"Found {len(dataset_files)} dataset(s):")
    for name, path in dataset_files:
        print(f"  - {name}: {path}")

    for name, filepath in dataset_files:
        try:
            print(f"\nLoading {name} dataset from {filepath}...")
            df = load_dataset(filepath)
            analyze_dataset(name, df)
        except Exception as e:
            print(f"Error processing {name} dataset: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

    # Provide preprocessing recommendations
    print("\nPREPROCESSING RECOMMENDATIONS:")
    print("1. Handle missing values:")
    print("   - For text columns: fill empty strings or drop rows if critical")
    print("   - For numeric columns: impute with median/mean or drop")
    print("2. Text cleaning:")
    print("   - Convert to lowercase")
    print("   - Remove HTML tags, URLs, special characters, emojis (optional)")
    print("   - Remove extra whitespace")
    print("   - Consider expanding contractions")
    print("3. Tokenization:")
    print("   - Use tokenizer matching the model (e.g., DistilBERT tokenizer)")
    print("   - Set max length (typically 128-512 tokens)")
    print("4. Label encoding:")
    print("   - Amazon: Map ratings 1-2 -> negative, 3 -> neutral, 4-5 -> positive")
    print("   - Twitter: Map 0 -> negative, 2 -> neutral, 4 -> positive (or binary: 0/4 -> 0/1)")
    print("5. Handle class imbalance:")
    print("   - Consider stratified sampling, class weights, or resampling techniques")
    print("6. Train/validation/test split:")
    print("   - Use stratified split to maintain label distribution")
    print("   - Typical split: 80% train, 10% validation, 10% test")

if __name__ == "__main__":
    main()