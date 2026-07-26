from datetime import datetime, timezone
from . import db


class Product(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(255))
    category = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    reviews = db.relationship('Review', backref='product', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Review(db.Model):
    __tablename__ = 'reviews'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.String(50), db.ForeignKey('products.id'), nullable=False)
    user_id = db.Column(db.String(100))
    score = db.Column(db.Integer)
    summary = db.Column(db.Text)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    source = db.Column(db.String(20), default='amazon')

    sentiment = db.relationship('Sentiment', backref='review', uselist=False)

    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'user_id': self.user_id,
            'score': self.score,
            'summary': self.summary,
            'text': self.text,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source': self.source,
            'sentiment': self.sentiment.to_dict() if self.sentiment else None
        }


class Sentiment(db.Model):
    __tablename__ = 'sentiments'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.id'), nullable=False)
    label = db.Column(db.String(20), nullable=False)
    positive_score = db.Column(db.Float, default=0.0)
    negative_score = db.Column(db.Float, default=0.0)
    neutral_score = db.Column(db.Float, default=0.0)
    aspects = db.Column(db.JSON, default=dict)
    analyzed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'review_id': self.review_id,
            'label': self.label,
            'positive_score': self.positive_score,
            'negative_score': self.negative_score,
            'neutral_score': self.neutral_score,
            'aspects': self.aspects,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None
        }


class Trend(db.Model):
    __tablename__ = 'trends'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    keyword = db.Column(db.String(100), nullable=False)
    frequency = db.Column(db.Integer, default=0)
    sentiment_label = db.Column(db.String(20))
    product_id = db.Column(db.String(50), db.ForeignKey('products.id'))
    recorded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'keyword': self.keyword,
            'frequency': self.frequency,
            'sentiment_label': self.sentiment_label,
            'product_id': self.product_id,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        }


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.String(50), db.ForeignKey('products.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(20), default='info')
    message = db.Column(db.Text, nullable=False)
    metric_value = db.Column(db.Float)
    threshold = db.Column(db.Float)
    triggered_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    acknowledged = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'acknowledged': self.acknowledged
        }


class CompetitorProduct(db.Model):
    __tablename__ = 'competitor_products'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    source_product_id = db.Column(db.String(50), db.ForeignKey('products.id'))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'source_product_id': self.source_product_id
        }
