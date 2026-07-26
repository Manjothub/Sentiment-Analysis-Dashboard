"""
Product Model
=============
SQLAlchemy ORM model for the products table.
Represents a product being reviewed, with metadata and relationships.
"""

from datetime import datetime, timezone
from app.database import db


class Product(db.Model):
    """
    Product model representing items being reviewed.
    
    Attributes:
        product_id: Primary key, Amazon product ID
        asin: Amazon Standard Identification Number (unique)
        product_name: Product title/name
        brand: Product brand name
        category: Product category
        price: Product price in decimal
        is_competitor: Flag for competitor products
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'products'

    # Primary key
    product_id = db.Column(db.String(50), primary_key=True)

    # Product identifiers
    asin = db.Column(db.String(20), unique=True, nullable=True)
    product_name = db.Column(db.String(500), nullable=False)
    brand = db.Column(db.String(200), nullable=True)
    category = db.Column(db.String(200), nullable=True)
    price = db.Column(db.Numeric(10, 2), nullable=True)

    # Metadata
    is_competitor = db.Column(db.Boolean, default=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    reviews = db.relationship(
        'Review',
        backref='product',
        lazy='dynamic',
        cascade='all, delete-orphan',
        passive_deletes=True
    )

    def __repr__(self) -> str:
        """String representation of the Product model."""
        return f"<Product {self.product_id}: {self.product_name[:50]}>"

    def to_dict(self) -> dict:
        """
        Serialize product to dictionary.
        
        Returns:
            Dictionary with all product attributes
        """
        return {
            'product_id': self.product_id,
            'asin': self.asin,
            'product_name': self.product_name,
            'brand': self.brand,
            'category': self.category,
            'price': float(self.price) if self.price else None,
            'is_competitor': self.is_competitor,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'total_reviews': self.reviews.count() if hasattr(self, 'reviews') else 0
        }

    @classmethod
    def find_by_category(cls, category: str):
        """
        Find all products in a given category.
        
        Args:
            category: Category name to search for
            
        Returns:
            List of Product objects
        """
        return cls.query.filter_by(category=category).all()

    @classmethod
    def find_competitors(cls):
        """Find all products marked as competitors."""
        return cls.query.filter_by(is_competitor=True).all()
