from datetime import datetime, timezone, timedelta
from ..models.database import Alert, Sentiment, Review
from ..models import db


class AlertService:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.alert_configs = {
            'negativity_spike': {
                'severity': 'warning',
                'threshold': 0.4,
                'cooldown_minutes': 60
            },
            'aspect_quality_drop': {
                'severity': 'critical',
                'threshold': 0.5,
                'cooldown_minutes': 120
            },
            'volume_drop': {
                'severity': 'info',
                'threshold': 0.3,
                'cooldown_minutes': 240
            }
        }

    def check_and_create_alerts(self, product_id):
        alerts = []
        recent_reviews = Review.query.filter_by(product_id=product_id)\
            .order_by(Review.timestamp.desc()).limit(50).all()

        if len(recent_reviews) < 10:
            return alerts

        recent_sentiments = []
        for review in recent_reviews:
            if review.sentiment:
                recent_sentiments.append(review.sentiment.label)

        spike = self.analyzer.detect_sentiment_spike(recent_sentiments)
        if spike:
            existing = Alert.query.filter_by(
                product_id=product_id,
                alert_type='negativity_spike',
                acknowledged=False
            ).filter(
                Alert.triggered_at > datetime.now(timezone.utc) - timedelta(hours=1)
            ).first()
            if not existing:
                alert = Alert(
                    product_id=product_id,
                    alert_type='negativity_spike',
                    severity='warning',
                    message=f'Negative sentiment spike detected: '
                            f'{spike["negative_ratio"]:.0%} negative in last '
                            f'{spike["window_size"]} reviews',
                    metric_value=spike['negative_ratio'],
                    threshold=spike['threshold']
                )
                db.session.add(alert)
                alerts.append(alert)

        neg_reviews = [
            r for r in recent_reviews
            if r.sentiment and r.sentiment.label == 'negative'
        ]
        if neg_reviews:
            quality_neg = sum(
                1 for r in neg_reviews
                if r.sentiment and r.sentiment.aspects
                and r.sentiment.aspects.get('quality', {}).get('sentiment') == 'negative'
            )
            quality_ratio = quality_neg / max(len(neg_reviews), 1)
            if quality_ratio > self.alert_configs['aspect_quality_drop']['threshold']:
                existing = Alert.query.filter_by(
                    product_id=product_id,
                    alert_type='aspect_quality_drop',
                    acknowledged=False
                ).filter(
                    Alert.triggered_at > datetime.now(timezone.utc) - timedelta(hours=2)
                ).first()
                if not existing:
                    alert = Alert(
                        product_id=product_id,
                        alert_type='aspect_quality_drop',
                        severity='critical',
                        message=f'Critical drop in product quality sentiment: '
                                f'{quality_ratio:.0%} of negative reviews mention quality issues',
                        metric_value=quality_ratio,
                        threshold=self.alert_configs['aspect_quality_drop']['threshold']
                    )
                    db.session.add(alert)
                    alerts.append(alert)

        db.session.commit()
        return alerts
