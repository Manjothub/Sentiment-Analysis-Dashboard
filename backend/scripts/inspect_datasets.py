"""
Dataset Inspection Script
=========================
Quick inspection of Amazon Reviews and Twitter Sentiment datasets.
"""
import os
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(backend_dir)
    dataset_dir = os.path.join(project_dir, 'dataset')

    # Amazon Reviews
    amazon_path = os.path.join(dataset_dir, 'amazon-reviews', 'Reviews.csv')
    print("=" * 60)
    print("Amazon Reviews Dataset")
    print("=" * 60)
    df_amazon = pd.read_csv(amazon_path, nrows=1000)
    print(f"Shape: {df_amazon.shape}")
    print(f"Columns: {list(df_amazon.columns)}")
    print(f"\nMissing values:\n{df_amazon.isnull().sum()}")
    print(f"\nRating distribution (Score):")
    print(df_amazon['Score'].value_counts().sort_index())
    print(f"\nFirst 3 rows:")
    print(df_amazon.head(3))

    # Twitter Sentiment
    twitter_path = os.path.join(dataset_dir, 'twitter-sentiment', 'training.1600000.processed.noemoticon.csv')
    print("\n" + "=" * 60)
    print("Twitter Sentiment Dataset")
    print("=" * 60)
    df_twitter = pd.read_csv(twitter_path, nrows=1000, encoding='latin-1', header=None)
    print(f"Shape: {df_twitter.shape}")
    print(f"Columns: {list(df_twitter.columns)}")
    print(f"\nSentiment distribution (col 0: 0=neg, 2=neu, 4=pos):")
    print(df_twitter[0].value_counts().sort_index())
    print(f"\nFirst 3 rows:")
    print(df_twitter.head(3))

    print("\n" + "=" * 60)
    print("Dataset Inspection Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
