"""
Sentiment Result Model
======================
SQLAlchemy ORM model for the sentiment_results table.
Stores sentiment analysis predictions, confidence scores, and aspect-based analysis.
"""

from datetime import datetime, timezone
from app.database import db


class SentimentResult(db.Model):
    """
    SentimentResult model representing sentiment analysis output for a review.
    
    Attributes:
        result_id: Primary key (auto-incrementing)
        review_id: Foreign key to reviews table (unique - one result per review)
        predicted_sentiment: Predicted sentiment label (positive/negative/neutral)
        positive_score: Confidence score for positive sentiment (0.0 to 1.0)
        negative_score: Confidence score for negative sentiment (0.0 to 1.0)
        neutral_score: Confidence score for neutral sentiment (0.0 to 1.0)
        confidence_score: Overall confidence score
        emotion: Detected emotion (joy, anger, sadness, etc.)
        aspects: JSONB field storing extracted aspects and their sentiments
        aspect_scores: JSONB field storing numerical scores for each aspect
        model_version: Version identifier of the model used
        model_name: Name of the model used
        inference_time_ms: Time taken for inference in milliseconds
        analyzed_at: Timestamp when analysis was performed
    """
    __tablename__ = 'sentiment_results'

    # Primary key
    result_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)

    # Foreign key to reviews (one-to-one relationship)
    review_id = db.Column(
        db.BigInteger,
        db.ForeignKey('reviews.review_id', ondelete='CASCADE'),
        nullable=False,
        unique=True,
        index=True
    )

    # Sentiment predictions
    predicted_sentiment = db.Column(db.String(20), nullable=False)
    positive_score = db.Column(db.Float, default=0.0)
    negative_score = db.Column(db.Float, default=0.0)
    neutral_score = db.Column(db.Float, default=0.0)
    confidence_score = db.Column(db.Float, default=0.0)

    # Detailed analysis
    emotion = db.Column(db.String(50), nullable=True)
    aspects = db.Column(db.JSON, default=dict)
    aspect_scores = db.Column(db.JSON, default=dict)

    # Model metadata
    model_version = db.Column(db.String(50), nullable=True)
    model_name = db.Column(db.String(100), default='distilbert-sentiment')
    inference_time_ms = db.Column(db.Integer, nullable=True)

    # Timestamps
    analyzed_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    __table_args__ = (
        db.CheckConstraint(
            "predicted_sentiment IN ('positive', 'negative', 'neutral')",
            name='check_predicted_sentiment'
        ),
        db.CheckConstraint(
            'positive_score >= 0 AND positive_score <= 1 '
            'AND negative_score >= 0 AND negative_score <= 1 '
            'AND neutral_score >= 0 AND neutral_score <= 1',
            name='check_score_range'
        ),
        db.Index('idx_sentiment_review_analysis', 'review_id', 'predicted_sentiment', 'analyzed_at'),
    )

    def __repr__(self) -> str:
        """String representation of the SentimentResult model."""
        return f"<SentimentResult {self.result_id}: Review {self.review_id} - {self.predicted_sentiment}>"

    def to_dict(self) -> dict:
        """
        Serialize sentiment result to dictionary.
        
        Returns:
            Dictionary with all sentiment result attributes
        """
        return {
            'result_id': self.result_id,
            'review_id': self.review_id,
            'predicted_sentiment': self.predicted_sentiment,
            'positive_score': self.positive_score,
            'negative_score': self.negative_score,
            'neutral_score': self.neutral_score,
            'confidence_score': self.confidence_score,
            'emotion': self.emotion,
            'aspects': self.aspects,
            'aspect_scores': self.aspect_scores,
            'model_version': self.model_version,
            'model_name': self.model_name,
            'inference_time_ms': self.inference_time_ms,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None
        }

    @classmethod
    def get_sentiment_distribution(cls, product_id: str = None) -> dict:
        """
        Get distribution of predicted sentiments.
        
        Args:
            product_id: Optional product filter
            
        Returns:
            Dictionary with sentiment counts and percentages
        """
        from sqlalchemy import func
        from app.models import Review
        
        query = cls.query.join(Review, cls.review_id == Review.review_id)
        
        if product_id:
            query = query.filter(Review.product_id == product_id)
        
        results = query.with_entities(
            cls.predicted_sentiment,
            func.count(cls.result_id)
        ).group_by(cls.predicted_sentiment).all()
        
        total = sum(count for _, count in results)
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for sentiment, count in results:
            if sentiment in distribution:
                distribution[sentiment] = count
        
        percentages = {
            k: round((v / total * 100), 2) if total > 0 else 0
            for k, v in distribution.items()
        }
        
        return {
            'counts': distribution,
            'percentages': percentages,
            'total': total
        }

    @classmethod
    def get_average_scores(cls, product_id: str = None) -> dict:
        """
        Get average confidence scores for each sentiment.
        
        Args:
            product_id: Optional product filter
            
        Returns:
            Dictionary with average scores
        """
        from sqlalchemy import func
        from app.models import Review
        
        query = cls.query.join(Review, cls.review_id == Review.review_id)
        
        if product_id:
            query = query.filter(Review.product_id == product_id)
        
        result = query.with_entities(
            func.avg(cls.positive_score),
            func.avg(cls.negative_score),
            func.avg(cls.neutral_score),
            func.avg(cls.confidence_score)
        ).first()
        
        return {
            'avg_positive_score': round(float(result[0] or 0), 4),
            'avg_negative_score': round(float(result[1] or 0), 4),
            'avg_neutral_score': round(float(result[2] or 0), 4),
            'avg_confidence_score': round(float(result[3] or 0), 4)
        }
