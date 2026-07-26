import { FiAlertTriangle, FiAlertCircle, FiInfo, FiCheckCircle } from 'react-icons/fi';

const severityConfig = {
  critical: { icon: FiAlertCircle, color: '#e74c3c', bg: 'rgba(231, 76, 60, 0.15)' },
  warning: { icon: FiAlertTriangle, color: '#f39c12', bg: 'rgba(243, 156, 18, 0.15)' },
  info: { icon: FiInfo, color: '#3498db', bg: 'rgba(52, 152, 219, 0.15)' },
};

const AlertPanel = ({ alerts = [], onAcknowledge }) => {
  const sortedAlerts = [...alerts].sort(
    (a, b) => new Date(b.triggered_at) - new Date(a.triggered_at)
  );

  const formatTime = (isoStr) => {
    const d = new Date(isoStr);
    return d.toLocaleString();
  };

  return (
    <div className="alert-panel">
      {sortedAlerts.length === 0 ? (
        <div className="text-center text-muted py-4">
          <FiCheckCircle size={24} />
          <p className="mt-2 mb-0">No alerts. Everything looks good!</p>
        </div>
      ) : (
        sortedAlerts.slice(0, 20).map((alert) => {
          const config = severityConfig[alert.severity] || severityConfig.info;
          const Icon = config.icon;
          return (
            <div
              key={alert.id}
              className="alert-item d-flex align-items-start p-3 mb-2 rounded"
              style={{
                backgroundColor: config.bg,
                borderLeft: `4px solid ${config.color}`,
                opacity: alert.acknowledged ? 0.6 : 1,
              }}
            >
              <Icon size={20} color={config.color} className="me-2 mt-1 flex-shrink-0" />
              <div className="flex-grow-1">
                <div className="d-flex justify-content-between align-items-start">
                  <span
                    className="badge mb-1"
                    style={{ backgroundColor: config.color, color: '#fff', fontSize: '0.7rem' }}
                  >
                    {alert.severity.toUpperCase()}
                  </span>
                  <small className="text-muted">{formatTime(alert.triggered_at)}</small>
                </div>
                <p className="mb-1" style={{ fontSize: '0.85rem' }}>
                  {alert.message}
                </p>
                {!alert.acknowledged && onAcknowledge && (
                  <button
                    className="btn btn-sm btn-outline-secondary mt-1"
                    style={{ fontSize: '0.75rem' }}
                    onClick={() => onAcknowledge(alert.id)}
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};

export default AlertPanel;
