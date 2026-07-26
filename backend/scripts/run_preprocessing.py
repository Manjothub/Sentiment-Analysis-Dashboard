"""
Preprocessing Orchestrator
==========================
Orchestrates the preprocessing pipeline for both Amazon and Twitter datasets.
Runs both preprocessing scripts and creates the final merged dataset.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import setup_logging, get_logger
from scripts.preprocess_amazon import preprocess_amazon
from scripts.preprocess_twitter import preprocess_twitter

logger = get_logger(__name__)


def run_full_pipeline(
    amazon_path: str,
    twitter_path: str,
    output_dir: str,
    amazon_nrows: Optional[int] = None,
    twitter_nrows: Optional[int] = None
) -> dict:
    """
    Run the complete preprocessing pipeline for both datasets.
    
    Args:
        amazon_path: Path to Amazon Reviews CSV
        twitter_path: Path to Twitter Sentiment CSV
        output_dir: Directory to save processed files
        amazon_nrows: Number of Amazon rows to process
        twitter_nrows: Number of Twitter rows to process
        
    Returns:
        Dictionary with preprocessing results and statistics
    """
    logger.info("=" * 60)
    logger.info("STARTING FULL PREPROCESSING PIPELINE")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    results = {
        'amazon': {'status': 'pending'},
        'twitter': {'status': 'pending'},
        'merged': {'status': 'pending'},
        'pipeline_start': datetime.now().isoformat()
    }
    
    # Step 1: Preprocess Amazon Reviews
    try:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Preprocessing Amazon Reviews")
        logger.info("=" * 40)
        
        amazon_output = os.path.join(output_dir, 'amazon_cleaned.csv')
        amazon_df = preprocess_amazon(
            input_path=amazon_path,
            output_path=amazon_output,
            nrows=amazon_nrows
        )
        
        results['amazon'] = {
            'status': 'completed',
            'output_path': amazon_output,
            'total_rows': len(amazon_df),
            'columns': list(amazon_df.columns),
            'sentiment_distribution': amazon_df['sentiment_label'].value_counts().to_dict()
        }
        logger.info(f"Amazon preprocessing completed: {len(amazon_df):,} reviews")
        
    except Exception as e:
        logger.error(f"Amazon preprocessing failed: {e}")
        results['amazon'] = {'status': 'failed', 'error': str(e)}
    
    # Step 2: Preprocess Twitter Sentiment
    try:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Preprocessing Twitter Sentiment")
        logger.info("=" * 40)
        
        twitter_output = os.path.join(output_dir, 'twitter_cleaned.csv')
        twitter_df = preprocess_twitter(
            input_path=twitter_path,
            output_path=twitter_output,
            nrows=twitter_nrows
        )
        
        results['twitter'] = {
            'status': 'completed',
            'output_path': twitter_output,
            'total_rows': len(twitter_df),
            'columns': list(twitter_df.columns),
            'sentiment_distribution': twitter_df['sentiment_label'].value_counts().to_dict()
        }
        logger.info(f"Twitter preprocessing completed: {len(twitter_df):,} tweets")
        
    except Exception as e:
        logger.error(f"Twitter preprocessing failed: {e}")
        results['twitter'] = {'status': 'failed', 'error': str(e)}
    
    # Step 3: Create merged dataset if both succeeded
    if (results['amazon']['status'] == 'completed' and 
        results['twitter']['status'] == 'completed'):
        
        try:
            logger.info("\n" + "=" * 40)
            logger.info("STEP 3: Creating Merged Dataset")
            logger.info("=" * 40)
            
            # Normalize columns for merging
            amazon_subset = amazon_df[[
                'cleaned_text', 'sentiment_label', 'review_length', 'source',
                'review_date'
            ]].copy()
            
            twitter_subset = twitter_df[[
                'cleaned_text', 'sentiment_label', 'review_length', 'source',
                'review_date'
            ]].copy()
            
            # Merge datasets
            merged_df = pd.concat([amazon_subset, twitter_subset], ignore_index=True)
            
            # Shuffle merged dataset
            merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save merged dataset
            merged_output = os.path.join(output_dir, 'final_dataset.csv')
            merged_df.to_csv(merged_output, index=False)
            
            results['merged'] = {
                'status': 'completed',
                'output_path': merged_output,
                'total_rows': len(merged_df),
                'amazon_rows': len(amazon_subset),
                'twitter_rows': len(twitter_subset),
                'sentiment_distribution': merged_df['sentiment_label'].value_counts().to_dict()
            }
            
            logger.info(f"Merged dataset created: {len(merged_df):,} total rows")
            logger.info(f"  Amazon: {len(amazon_subset):,}")
            logger.info(f"  Twitter: {len(twitter_subset):,}")
            
        except Exception as e:
            logger.error(f"Dataset merging failed: {e}")
            results['merged'] = {'status': 'failed', 'error': str(e)}
    else:
        logger.warning("Skipping merge - one or both preprocessing steps failed")
    
    # Final summary
    results['pipeline_end'] = datetime.now().isoformat()
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Amazon: {results['amazon']['status']}")
    logger.info(f"Twitter: {results['twitter']['status']}")
    logger.info(f"Merged: {results['merged']['status']}")
    logger.info(f"Output directory: {output_dir}")
    
    return results


def main():
    """Main entry point for the preprocessing orchestrator."""
    parser = argparse.ArgumentParser(
        description='Run full preprocessing pipeline for both datasets'
    )
    parser.add_argument('--amazon-nrows', type=int, default=None, help='Amazon rows to process')
    parser.add_argument('--twitter-nrows', type=int, default=None, help='Twitter rows to process')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # backend/scripts/
    backend_dir = os.path.dirname(script_dir)                  # backend/
    project_dir = os.path.dirname(backend_dir)                 # project root
    
    amazon_path = os.path.join(project_dir, 'dataset', 'amazon-reviews', 'Reviews.csv')
    twitter_path = os.path.join(
        project_dir, 'dataset', 'twitter-sentiment',
        'training.1600000.processed.noemoticon.csv'
    )
    output_dir = args.output_dir or os.path.join(project_dir, 'dataset', 'processed')
    
    # Verify datasets exist
    if not os.path.exists(amazon_path):
        logger.error(f"Amazon dataset not found: {amazon_path}")
        sys.exit(1)
    
    if not os.path.exists(twitter_path):
        logger.error(f"Twitter dataset not found: {twitter_path}")
        sys.exit(1)
    
    # Run pipeline
    results = run_full_pipeline(
        amazon_path=amazon_path,
        twitter_path=twitter_path,
        output_dir=output_dir,
        amazon_nrows=args.amazon_nrows,
        twitter_nrows=args.twitter_nrows
    )
    
    # Print final summary
    if results['merged']['status'] == 'completed':
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Final dataset: {results['merged']['output_path']}")
        print(f"Total rows: {results['merged']['total_rows']:,}")
    else:
        print(f"\n❌ Pipeline completed with errors")
        print(f"Amazon: {results['amazon']['status']}")
        print(f"Twitter: {results['twitter']['status']}")
        print(f"Merged: {results['merged']['status']}")


if __name__ == '__main__':
    main()
