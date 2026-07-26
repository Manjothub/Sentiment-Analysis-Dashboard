"""
Review Model
============
SQLAlchemy ORM model for the reviews table.
Represents a product review with raw text and metadata.
"""

from datetime import datetime, timezone
from app.database import db


class Review(db.Model):
    """
    Review model representing a product review.
    
    Attributes:
        review_id: Primary key (auto-incrementing)
        product_id: Foreign key to products table
        reviewer_name: Reviewer username/ID
        reviewer_id: Internal reviewer identifier
        review_text: Original review text
        cleaned_text: Preprocessed/cleaned text
        summary: Review summary/title
        raw_rating: Original rating (1-5)
        review_date: Date of review
        verified_purchase: Verified purchase flag
        helpful_votes: Number of helpful votes
        total_votes: Total votes received
        source: Data source (amazon/twitter)
        ingested_at: When record was ingested
    """
    __tablename__ = 'reviews'

    # Primary key
    review_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)

    # Foreign key to products
    product_id = db.Column(
        db.String(50),
        db.ForeignKey('products.product_id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    # Reviewer information
    reviewer_name = db.Column(db.String(200), nullable=True)
    reviewer_id = db.Column(db.String(100), nullable=True)

    # Review content
    review_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text, nullable=True)
    summary = db.Column(db.String(500), nullable=True)

    # Ratings and metadata
    raw_rating = db.Column(db.SmallInteger, nullable=True)
    review_date = db.Column(db.DateTime(timezone=True), nullable=True)
    verified_purchase = db.Column(db.Boolean, default=False)
    helpful_votes = db.Column(db.Integer, default=0)
    total_votes = db.Column(db.Integer, default=0)

    # Processing metadata
    source = db.Column(db.String(20), default='amazon')
    ingested_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    sentiment_result = db.relationship(
        'SentimentResult',
        backref='review',
        uselist=False,
        cascade='all, delete-orphan',
        passive_deletes=True
    )

    __table_args__ = (
        db.CheckConstraint('raw_rating IS NULL OR (raw_rating >= 1 AND raw_rating <= 5)',
                           name='check_rating_range'),
        db.CheckConstraint('LENGTH(review_text) >= 1',
                           name='check_review_length'),
        db.Index('idx_reviews_product_rating', 'product_id', 'raw_rating'),
        db.Index('idx_reviews_product_date', 'product_id', review_date.desc()),
    )

    def __repr__(self) -> str:
        """String representation of the Review model."""
        return f"<Review {self.review_id}: Product {self.product_id}, Rating {self.raw_rating}>"

    def to_dict(self) -> dict:
        """
        Serialize review to dictionary.
        
        Returns:
            Dictionary with all review attributes
        """
        return {
            'review_id': self.review_id,
            'product_id': self.product_id,
            'reviewer_name': self.reviewer_name,
            'review_text': self.review_text[:200] if self.review_text else None,
            'cleaned_text': self.cleaned_text[:200] if self.cleaned_text else None,
            'summary': self.summary,
            'raw_rating': self.raw_rating,
            'review_date': self.review_date.isoformat() if self.review_date else None,
            'verified_purchase': self.verified_purchase,
            'helpful_votes': self.helpful_votes,
            'total_votes': self.total_votes,
            'source': self.source,
            'ingested_at': self.ingested_at.isoformat() if self.ingested_at else None,
            'sentiment': self.sentiment_result.to_dict() if self.sentiment_result else None
        }

    @classmethod
    def find_by_product(cls, product_id: str, limit: int = 100):
        """
        Find reviews for a given product.
        
        Args:
            product_id: Product identifier
            limit: Maximum number of reviews to return
            
        Returns:
            List of Review objects
        """
        return cls.query.filter_by(product_id=product_id)\
                       .order_by(cls.review_date.desc())\
                       .limit(limit)\
                       .all()

    @classmethod
    def find_by_rating(cls, rating: int, limit: int = 100):
        """
        Find reviews with a specific rating.
        
        Args:
            rating: Rating value (1-5)
            limit: Maximum number of reviews to return
            
        Returns:
            List of Review objects
        """
        return cls.query.filter_by(raw_rating=rating)\
                       .order_by(cls.review_date.desc())\
                       .limit(limit)\
                       .all()

    @classmethod
    def get_rating_distribution(cls, product_id: str = None) -> dict:
        """
        Get distribution of ratings.
        
        Args:
            product_id: Optional product filter
            
        Returns:
            Dictionary with rating counts
        """
        query = cls.query
        if product_id:
            query = query.filter_by(product_id=product_id)
        
        from sqlalchemy import func
        results = query.with_entities(
            cls.raw_rating,
            func.count(cls.review_id)
        ).group_by(cls.raw_rating).all()
        
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating, count in results:
            if rating in distribution:
                distribution[rating] = count
        
        return distribution
