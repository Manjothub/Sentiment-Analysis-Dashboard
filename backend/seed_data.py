import os
import sys
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.database import db
from app.models import Product, Review, SentimentResult
from app.services.nlp_service import SentimentAnalyzer


def seed_from_csv(app, csv_path, sample_size=500):
    analyzer = SentimentAnalyzer()

    with app.app_context():
        if Product.query.first():
            print("Database already has data, skipping seed.")
            return

        print(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path, nrows=sample_size)
        df = df.dropna(subset=['Text'])

        product_ids = df['ProductId'].unique()[:20]
        for pid in product_ids:
            if not Product.query.get(pid):
                row = df[df['ProductId'] == pid].iloc[0]
                product = Product(
                    product_id=str(pid),
                    product_name=str(row.get('Summary', pid))[:100] or f"Product-{str(pid)[:8]}",
                    category='general'
                )
                db.session.add(product)
        db.session.commit()
        print(f"Created {len(product_ids)} products")

        count = 0
        for _, row in df.iterrows():
            if count >= sample_size:
                break
            try:
                text = str(row['Text'])[:2000]
                if not text.strip():
                    continue

                review = Review(
                    product_id=str(row['ProductId']),
                    reviewer_name=str(row.get('UserId', '')),
                    raw_rating=int(row['Score']) if pd.notna(row.get('Score')) else None,
                    summary=str(row.get('Summary', ''))[:200],
                    review_text=text,
                    cleaned_text=text,
                    source='amazon',
                    review_date=datetime.fromtimestamp(
                        float(row['Time']), tz=timezone.utc
                    ) if pd.notna(row.get('Time')) else datetime.now(timezone.utc)
                )
                db.session.add(review)
                db.session.flush()

                result = analyzer.analyze_sentiment(text)
                sentiment = SentimentResult(
                    review_id=review.review_id,
                    predicted_sentiment=result['label'],
                    positive_score=result['positive_score'],
                    negative_score=result['negative_score'],
                    neutral_score=result['neutral_score'],
                    aspects=result['aspects']
                )
                db.session.add(sentiment)
                count += 1

                if count % 100 == 0:
                    db.session.commit()
                    print(f"Seeded {count} reviews...")

            except Exception as e:
                print(f"Error on row: {e}")
                continue

        db.session.commit()
        print(f"Successfully seeded {count} reviews!")
        return count


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'dataset', 'amazon-reviews', 'Reviews.csv')

    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        print("Please ensure the dataset is at:", csv_path)
        sys.exit(1)

    app = create_app()
    seed_from_csv(app, csv_path, sample_size=500)
