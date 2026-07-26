from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
from ..models import db
from ..models.database import Alert, Review, Sentiment
from ..services.alert_service import AlertService

alerts_bp = Blueprint('alerts', __name__)


def get_alert_service():
    from ..app import alert_service
    return alert_service


@alerts_bp.route('/alerts', methods=['GET'])
def get_alerts():
    product_id = request.args.get('product_id')
    acknowledged = request.args.get('acknowledged')
    limit = request.args.get('limit', 50, type=int)

    query = Alert.query
    if product_id:
        query = query.filter_by(product_id=product_id)
    if acknowledged is not None:
        query = query.filter_by(acknowledged=acknowledged.lower() == 'true')

    alerts = query.order_by(Alert.triggered_at.desc()).limit(limit).all()
    return jsonify({'alerts': [a.to_dict() for a in alerts]})


@alerts_bp.route('/alerts/check', methods=['POST'])
def check_alerts():
    data = request.get_json()
    product_id = data.get('product_id')
    if not product_id:
        return jsonify({'error': 'product_id required'}), 400

    service = get_alert_service()
    new_alerts = service.check_and_create_alerts(product_id)

    from ..models import socketio
    for alert in new_alerts:
        socketio.emit('new_alert', alert.to_dict())

    return jsonify({
        'alerts_created': len(new_alerts),
        'alerts': [a.to_dict() for a in new_alerts]
    })


@alerts_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['PATCH'])
def acknowledge_alert(alert_id):
    alert = Alert.query.get(alert_id)
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404

    alert.acknowledged = True
    db.session.commit()

    return jsonify(alert.to_dict())


@alerts_bp.route('/alerts/summary', methods=['GET'])
def get_alert_summary():
    product_id = request.args.get('product_id')
    hours = request.args.get('hours', 24, type=int)

    since = datetime.now(timezone.utc).timestamp() - (hours * 3600)
    since_dt = datetime.fromtimestamp(since, tz=timezone.utc)

    query = Alert.query.filter(Alert.triggered_at >= since_dt)
    if product_id:
        query = query.filter_by(product_id=product_id)

    alerts = query.all()

    return jsonify({
        'total_alerts': len(alerts),
        'unacknowledged': sum(1 for a in alerts if not a.acknowledged),
        'by_severity': {
            'critical': sum(1 for a in alerts if a.severity == 'critical'),
            'warning': sum(1 for a in alerts if a.severity == 'warning'),
            'info': sum(1 for a in alerts if a.severity == 'info')
        },
        'recent_alerts': [a.to_dict() for a in alerts[:10]]
    })
