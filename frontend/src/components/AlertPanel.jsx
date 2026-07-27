import { FiAlertTriangle, FiAlertCircle, FiInfo, FiCheckCircle } from 'react-icons/fi';

const severityConfig = {
  critical: { icon: FiAlertCircle, color: '#ef4444', bg: 'rgba(239, 68, 68, 0.12)' },
  warning: { icon: FiAlertTriangle, color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.12)' },
  info: { icon: FiInfo, color: '#6366f1', bg: 'rgba(99, 102, 241, 0.12)' },
};

const AlertPanel = ({ alerts = [], onAcknowledge }) => {
  const sortedAlerts = [...alerts].sort(
    (a, b) => {
      const aTime = a.triggered_at ? new Date(a.triggered_at).getTime() : 0;
      const bTime = b.triggered_at ? new Date(b.triggered_at).getTime() : 0;
      return bTime - aTime;
    }
  );

  const formatTime = (isoStr) => {
    const d = new Date(isoStr);
    return d.toLocaleString();
  };

  return (
    <div className="alert-panel">
      {sortedAlerts.length === 0 ? (
        <div className="text-center text-muted py-4">
          <FiCheckCircle size={28} className="text-success mb-2" />
          <p className="mt-2 mb-0" style={{ color: '#cbd5e1' }}>No active alerts. All sentiment metrics are healthy!</p>
        </div>
      ) : (
        sortedAlerts.slice(0, 20).map((alert) => {
          const config = severityConfig[alert.severity] || severityConfig.info;
          const Icon = config.icon;
          return (
            <div
              key={alert.id}
              className="alert-item d-flex align-items-start p-3 mb-2 rounded-3"
              style={{
                backgroundColor: config.bg,
                borderLeft: `4px solid ${config.color}`,
                border: '1px solid rgba(255, 255, 255, 0.05)',
                borderLeftWidth: '4px',
                opacity: alert.acknowledged ? 0.6 : 1,
              }}
            >
              <Icon size={20} color={config.color} className="me-2 mt-1 flex-shrink-0" />
              <div className="flex-grow-1">
                <div className="d-flex justify-content-between align-items-start">
                  <span
                    className="badge mb-1 px-2 py-1"
                    style={{ backgroundColor: config.color, color: '#fff', fontSize: '0.7rem' }}
                  >
                    {alert.severity.toUpperCase()}
                  </span>
                  <small style={{ color: '#cbd5e1', fontSize: '0.78rem' }}>{formatTime(alert.triggered_at)}</small>
                </div>
                <p className="mb-1 fw-medium" style={{ fontSize: '0.88rem', color: '#f8fafc' }}>
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
