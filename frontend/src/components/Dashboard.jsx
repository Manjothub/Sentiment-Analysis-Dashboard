import { useState, useEffect, useCallback, useRef } from 'react';
import {
  sentimentAPI, trendsAPI, alertsAPI, socket,
} from '../services/api';
import SentimentGauge from './SentimentGauge';
import SentimentChart from './SentimentChart';
import AspectAnalysis from './AspectAnalysis';
import TrendingTopics from './TrendingTopics';
import AlertPanel from './AlertPanel';
import ComparativeAnalysis from './ComparativeAnalysis';
import {
  FiBarChart2, FiTrendingUp, FiAlertTriangle,
  FiLayers, FiRefreshCw, FiSend, FiActivity,
  FiCpu, FiCheckCircle, FiXCircle, FiInfo, FiExternalLink,
} from 'react-icons/fi';

const Toast = ({ toasts, removeToast }) => (
  <div style={{ position: 'fixed', top: 20, right: 20, zIndex: 9999, display: 'flex', flexDirection: 'column', gap: 8 }}>
    {toasts.map((toast) => (
      <div
        key={toast.id}
        className="p-3 rounded shadow d-flex align-items-center gap-2"
        style={{
          minWidth: 280,
          backgroundColor: toast.type === 'error' ? 'rgba(231, 76, 60, 0.9)' : toast.type === 'success' ? 'rgba(46, 204, 113, 0.9)' : 'rgba(52, 152, 219, 0.9)',
          color: '#fff',
          fontSize: '0.85rem',
        }}
      >
        {toast.type === 'error' ? <FiXCircle /> : toast.type === 'success' ? <FiCheckCircle /> : <FiInfo />}
        <span className="flex-grow-1">{toast.message}</span>
        <button
          onClick={() => removeToast(toast.id)}
          style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '1rem' }}
        >
          ×
        </button>
      </div>
    ))}
  </div>
);

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [aspects, setAspects] = useState({});
  const [trendingTopics, setTrendingTopics] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [reviews, setReviews] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [aspectComparison, setAspectComparison] = useState({});
  const [analysisText, setAnalysisText] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [alertSummary, setAlertSummary] = useState({ total_alerts: 0, unacknowledged: 0 });
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [modelStatus, setModelStatus] = useState(null);
  const [toasts, setToasts] = useState([]);
  const toastIdRef = useRef(0);

  const addToast = useCallback((message, type = 'info') => {
    const id = ++toastIdRef.current;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const fetchAllData = useCallback(async (retryCount = 0) => {
    try {
      setError(null);
      const [statsRes, trendsRes, aspectsRes, topicsRes, alertsRes, reviewRes, modelRes] =
        await Promise.all([
          sentimentAPI.getStats().catch((err) => { console.error('stats failed', err); return null; }),
          trendsAPI.getTrendsOverTime({ days: 30 }).catch((err) => { console.error('trends failed', err); return null; }),
          trendsAPI.getAspectAnalysis().catch((err) => { console.error('aspects failed', err); return null; }),
          trendsAPI.getTrendingTopics({ hours: 168, top_n: 15 }).catch((err) => { console.error('topics failed', err); return null; }),
          alertsAPI.getAlerts({ limit: 20 }).catch((err) => { console.error('alerts failed', err); return null; }),
          sentimentAPI.getReviews({ per_page: 10 }).catch((err) => { console.error('reviews failed', err); return null; }),
          sentimentAPI.getModelStatus ? sentimentAPI.getModelStatus() : Promise.resolve(null),
        ]);
      
      if (statsRes) {
        setStats(statsRes.data);
        addToast('Dashboard data refreshed', 'success');
      }
      if (trendsRes) setTrendData(trendsRes.data.trends || []);
      if (aspectsRes) setAspects(aspectsRes.data || {});
      if (topicsRes) setTrendingTopics(topicsRes.data.topics || []);
      if (alertsRes) setAlerts(alertsRes.data.alerts || []);
      if (reviewRes) setReviews(reviewRes.data.reviews || []);
      if (modelRes) setModelStatus(modelRes.data);
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
      setError('Failed to load dashboard data. Please check your connection and try again.');
      if (retryCount < 3) {
        setTimeout(() => fetchAllData(retryCount + 1), 2000 * (retryCount + 1));
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  const fetchAlertSummary = useCallback(async () => {
    try {
      const res = await alertsAPI.getAlertSummary({ hours: 24 });
      setAlertSummary(res.data);
    } catch (err) {
      console.error('Failed to fetch alert summary:', err);
    }
  }, []);

  useEffect(() => {
    fetchAllData();
    fetchAlertSummary();
  }, [fetchAllData, fetchAlertSummary]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchAllData();
      fetchAlertSummary();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchAllData, fetchAlertSummary]);

  useEffect(() => {
    const onNewReview = (data) => {
      setReviews((prev) => {
        const next = data?.review ? [data.review, ...prev] : prev;
        return next.slice(0, 19);
      });
      setStats((prev) => {
        if (!prev) return prev;
        const label = data?.sentiment?.label || 'neutral';
        return {
          ...prev,
          total_reviews: prev.total_reviews + 1,
          sentiment_distribution: {
            ...prev.sentiment_distribution,
            [label]: (prev.sentiment_distribution?.[label] || 0) + 1,
          },
        };
      });
    };

    const onReviewProcessed = (data) => {
      onNewReview(data);
      addToast('New prediction processed', 'success');
    };

    const onNewAlert = (alert) => {
      setAlerts((prev) => [alert, ...prev.slice(0, 24)]);
      addToast(`New alert: ${alert.alert_type || 'sentiment alert'}`, 'error');
    };

    const onAlertSpike = (data) => {
      addToast(`Alert spike detected: ${data.message || 'Negative sentiment spike'}`, 'error');
    };

    socket.on('new_review', onNewReview);
    socket.on('review_processed', onReviewProcessed);
    socket.on('new_alert', onNewAlert);
    socket.on('alert_spike', onAlertSpike);

    return () => {
      socket.off('new_review', onNewReview);
      socket.off('review_processed', onReviewProcessed);
      socket.off('new_alert', onNewAlert);
      socket.off('alert_spike', onAlertSpike);
    };
  }, [addToast]);

  const handleAnalyzeText = async () => {
    if (!analysisText.trim()) return;
    try {
      setAnalysisLoading(true);
      setAnalysisResult(null);
      const res = await sentimentAPI.analyze(analysisText);
      setAnalysisResult(res.data);
      addToast('Analysis completed', 'success');
    } catch (err) {
      console.error('Analysis failed:', err);
      addToast('Analysis failed. Please try again.', 'error');
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleCheckAlerts = async () => {
    try {
      const res = await alertsAPI.checkAlerts('all');
      if (res.data.alerts_created > 0 || (res.data.alerts && res.data.alerts.length > 0)) {
        addToast(`Found ${res.data.alerts?.length || res.data.alerts_created || 0} alerts`, 'error');
      } else {
        addToast('No new alerts detected', 'info');
      }
      if (res.data.alerts_created > 0) fetchAllData();
    } catch (err) {
      console.error('Alert check failed:', err);
      addToast('Alert check failed. Please try again.', 'error');
    }
  };

  const handleAcknowledgeAlert = async (alertId) => {
    try {
      await alertsAPI.acknowledgeAlert(alertId);
      setAlerts((prev) =>
        prev.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a))
      );
      addToast('Alert acknowledged', 'success');
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
      addToast('Failed to acknowledge alert', 'error');
    }
  };

  const handleIngestReview = async () => {
    if (!analysisText.trim()) return;
    try {
      setAnalysisLoading(true);
      const res = await sentimentAPI.ingest({ text: analysisText, source: 'manual' });
      if (res.data.review_id) {
        addToast('Review submitted successfully', 'success');
        setAnalysisText('');
        fetchAllData();
      }
    } catch (err) {
      console.error('Ingest failed:', err);
      addToast('Failed to submit review', 'error');
    } finally {
      setAnalysisLoading(false);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: FiBarChart2 },
    { id: 'aspects', label: 'Aspect Analysis', icon: FiLayers },
    { id: 'trends', label: 'Trending Topics', icon: FiTrendingUp },
    { id: 'alerts', label: 'Alerts', icon: FiAlertTriangle },
    { id: 'compare', label: 'Compare', icon: FiActivity },
    { id: 'analyze', label: 'Analyze & Submit', icon: FiSend },
  ];

  const renderModelStatus = () => {
    if (!modelStatus) {
      return (
        <div className="card card-dark">
          <div className="card-body">
            <h5 className="card-title">Model Status</h5>
            <div className="text-center text-muted py-4">
              <FiCpu size={24} className="mb-2" />
              <p className="mb-0">Loading model information...</p>
            </div>
          </div>
        </div>
      );
    }

    const models = modelStatus.models || {};
    const sentiment = models.sentiment || {};
    const aspect = models.aspect_extraction || {};
    const topic = models.topic_modeling || {};

    return (
      <div className="card card-dark">
        <div className="card-body">
          <h5 className="card-title">Model Status</h5>
          <div className="row g-3">
            <div className="col-md-4">
              <div className="p-3 rounded" style={{ backgroundColor: sentiment.loaded ? 'rgba(46, 204, 113, 0.15)' : 'rgba(231, 76, 60, 0.15)' }}>
                <div className="d-flex align-items-center gap-2 mb-2">
                  <FiCpu style={{ color: sentiment.loaded ? '#2ecc71' : '#e74c3c' }} />
                  <strong>Sentiment</strong>
                </div>
                <small className="text-muted d-block">DistilBERT</small>
                <small className="text-muted d-block">Device: {sentiment.device || 'unknown'}</small>
                {sentiment.version && <small className="text-muted d-block">Version: {sentiment.version}</small>}
                <div className="mt-2">
                  <span className="badge" style={{ backgroundColor: sentiment.loaded ? '#2ecc71' : '#e74c3c' }}>
                    {sentiment.loaded ? 'Loaded' : 'Not Loaded'}
                  </span>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="p-3 rounded" style={{ backgroundColor: aspect.loaded ? 'rgba(46, 204, 113, 0.15)' : 'rgba(243, 156, 18, 0.15)' }}>
                <div className="d-flex align-items-center gap-2 mb-2">
                  <FiCpu style={{ color: aspect.loaded ? '#2ecc71' : '#f39c12' }} />
                  <strong>Aspect Extraction</strong>
                </div>
                <small className="text-muted d-block">facebook/bart-large-mnli</small>
                <div className="mt-2">
                  <span className="badge" style={{ backgroundColor: aspect.loaded ? '#2ecc71' : '#f39c12' }}>
                    {aspect.loaded ? 'Loaded' : 'Fallback Mode'}
                  </span>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="p-3 rounded" style={{ backgroundColor: topic.loaded ? 'rgba(46, 204, 113, 0.15)' : 'rgba(243, 156, 18, 0.15)' }}>
                <div className="d-flex align-items-center gap-2 mb-2">
                  <FiCpu style={{ color: topic.loaded ? '#2ecc71' : '#f39c12' }} />
                  <strong>Topic Modeling</strong>
                </div>
                <small className="text-muted d-block">BERTopic</small>
                {topic.num_topics > 0 && <small className="text-muted d-block">Topics: {topic.num_topics}</small>}
                <div className="mt-2">
                  <span className="badge" style={{ backgroundColor: topic.loaded ? '#2ecc71' : '#f39c12' }}>
                    {topic.loaded ? 'Fitted' : 'Not Fitted'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="d-flex flex-column justify-content-center align-items-center" style={{ minHeight: '60vh', gap: 16 }}>
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p className="text-muted">Loading dashboard data...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <Toast toasts={toasts} removeToast={removeToast} />

      {error && (
        <div className="alert alert-danger d-flex justify-content-between align-items-center mb-4" role="alert">
          <div>
            <FiXCircle className="me-2" />
            {error}
          </div>
          <button className="btn btn-sm btn-outline-danger" onClick={() => fetchAllData()}>
            <FiRefreshCw className="me-1" /> Retry
          </button>
        </div>
      )}

      <header className="dashboard-header d-flex justify-content-between align-items-center mb-4">
        <div>
          <h1 className="fw-bold mb-1">Sentiment Dashboard</h1>
          <p className="text-muted mb-0">
            {stats?.total_reviews || 0} reviews analyzed &middot;{' '}
            {alertSummary.unacknowledged} unacknowledged alerts
          </p>
        </div>
        <div className="d-flex align-items-center gap-2">
          <div className="form-check form-switch">
            <input
              className="form-check-input"
              type="checkbox"
              id="autoRefresh"
              checked={autoRefresh}
              onChange={() => setAutoRefresh(!autoRefresh)}
            />
            <label className="form-check-label text-light small" htmlFor="autoRefresh">
              Auto-refresh
            </label>
          </div>
          <button className="btn btn-outline-secondary btn-sm" onClick={() => fetchAllData()}>
            <FiRefreshCw className="me-1" /> Refresh
          </button>
          <button className="btn btn-outline-warning btn-sm" onClick={handleCheckAlerts}>
            <FiAlertTriangle className="me-1" /> Check Alerts
          </button>
        </div>
      </header>

      <nav className="nav nav-pills mb-4 gap-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={`nav-link ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon className="me-1" /> {tab.label}
            </button>
          );
        })}
      </nav>

      {activeTab === 'overview' && (
        <div className="overview-grid">
          <div className="card card-dark">
            <div className="card-body">
              <h5 className="card-title">Sentiment Overview</h5>
              <SentimentGauge
                positive={stats?.sentiment_distribution?.positive || 0}
                negative={stats?.sentiment_distribution?.negative || 0}
                neutral={stats?.sentiment_distribution?.neutral || 0}
              />
              <div className="row text-center mt-3 g-2">
                <div className="col-4">
                  <div className="p-2 rounded" style={{ backgroundColor: 'rgba(46, 204, 113, 0.15)' }}>
                    <small className="text-muted">Positive</small>
                    <h4 className="mb-0" style={{ color: '#2ecc71' }}>
                      {stats?.sentiment_distribution?.positive || 0}
                    </h4>
                  </div>
                </div>
                <div className="col-4">
                  <div className="p-2 rounded" style={{ backgroundColor: 'rgba(231, 76, 60, 0.15)' }}>
                    <small className="text-muted">Negative</small>
                    <h4 className="mb-0" style={{ color: '#e74c3c' }}>
                      {stats?.sentiment_distribution?.negative || 0}
                    </h4>
                  </div>
                </div>
                <div className="col-4">
                  <div className="p-2 rounded" style={{ backgroundColor: 'rgba(243, 156, 18, 0.15)' }}>
                    <small className="text-muted">Neutral</small>
                    <h4 className="mb-0" style={{ color: '#f39c12' }}>
                      {stats?.sentiment_distribution?.neutral || 0}
                    </h4>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="card card-dark">
            <div className="card-body">
              <h5 className="card-title">Sentiment Trend (30 Days)</h5>
              {trendData.length > 0 ? (
                <SentimentChart trendData={trendData} />
              ) : (
                <div className="text-center text-muted py-5">
                  <p>No trend data available yet</p>
                </div>
              )}
            </div>
          </div>

          <div className="card card-dark">
            <div className="card-body">
              <h5 className="card-title">Recent Reviews</h5>
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {reviews.length === 0 ? (
                  <div className="text-center text-muted py-3">
                    <FiInfo size={24} className="mb-2" />
                    <p className="mb-0">No reviews yet</p>
                  </div>
                ) : (
                  reviews.map((review, idx) => (
                    <div key={review.review_id || idx} className="review-item p-2 mb-2 rounded" style={{ backgroundColor: 'rgba(255,255,255,0.03)' }}>
                      <div className="d-flex justify-content-between">
                        <small className="text-muted">
                          {review.product_id?.slice(0, 8) || 'unknown'}... &middot; {review.source || 'api'}
                        </small>
                        <span
                          className="badge"
                          style={{
                            backgroundColor: review.sentiment?.predicted_sentiment === 'positive' ? '#2ecc71'
                              : review.sentiment?.predicted_sentiment === 'negative' ? '#e74c3c' : '#f39c12',
                          }}
                        >
                          {review.sentiment?.predicted_sentiment || 'pending'}
                        </span>
                      </div>
                      <p className="mb-0 small mt-1" style={{ color: '#a0a0b8' }}>
                        {review.review_text?.slice(0, 150) || review.cleaned_text?.slice(0, 150) || 'No text'}...
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {renderModelStatus()}
        </div>
      )}

      {activeTab === 'aspects' && (
        <div className="card card-dark">
          <div className="card-body">
            <h5 className="card-title">Aspect-Based Sentiment Analysis</h5>
            <p className="text-muted small">
              Shows how customers feel about specific aspects: product quality, shipping, customer service, value, and usability.
            </p>
            <AspectAnalysis aspects={aspects} />
          </div>
        </div>
      )}

      {activeTab === 'trends' && (
        <div className="card card-dark">
          <div className="card-body">
            <h5 className="card-title">Trending Topics</h5>
            <p className="text-muted small">
              Most frequently discussed keywords in recent reviews, colored by dominant sentiment.
            </p>
            <TrendingTopics topics={trendingTopics} />
          </div>
        </div>
      )}

      {activeTab === 'alerts' && (
        <div className="card card-dark">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h5 className="card-title mb-0">Alert System</h5>
                <small className="text-muted">
                  {alertSummary.total_alerts} alerts in last 24h ({alertSummary.by_severity?.critical || 0} critical,{' '}
                  {alertSummary.by_severity?.warning || 0} warning)
                </small>
              </div>
              <button className="btn btn-sm btn-outline-warning" onClick={handleCheckAlerts}>
                <FiAlertTriangle className="me-1" /> Check Now
              </button>
            </div>
            <AlertPanel alerts={alerts} onAcknowledge={handleAcknowledgeAlert} />
          </div>
        </div>
      )}

      {activeTab === 'compare' && (
        <div className="card card-dark">
          <div className="card-body">
            <h5 className="card-title">Comparative Analysis</h5>
            <p className="text-muted small">
              Compare sentiment across products and competitors.
            </p>
            <ComparativeAnalysis comparison={comparison} aspectComparison={aspectComparison} />
          </div>
        </div>
      )}

      {activeTab === 'analyze' && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="card card-dark h-100">
              <div className="card-body">
                <h5 className="card-title">Analyze &amp; Submit Review</h5>
                <p className="text-muted small">Enter a product review to analyze its sentiment or submit it to the dashboard.</p>
                <textarea
                  className="form-control mb-3"
                  rows="5"
                  placeholder="Paste a product review here..."
                  value={analysisText}
                  onChange={(e) => setAnalysisText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                      handleAnalyzeText();
                    }
                  }}
                />
                <div className="d-grid gap-2">
                  <button
                    className="btn btn-primary"
                    onClick={handleAnalyzeText}
                    disabled={analysisLoading || !analysisText.trim()}
                  >
                    {analysisLoading ? (
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true" />
                    ) : (
                      <FiSend className="me-1" />
                    )}
                    Analyze Sentiment
                  </button>
                  <button
                    className="btn btn-outline-success"
                    onClick={handleIngestReview}
                    disabled={analysisLoading || !analysisText.trim()}
                  >
                    {analysisLoading ? (
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true" />
                    ) : (
                      <FiExternalLink className="me-1" />
                    )}
                    Submit Review to Dashboard
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card card-dark h-100">
              <div className="card-body">
                <h5 className="card-title">Result</h5>
                {analysisResult ? (
                  <div>
                    <div className="text-center mb-4">
                      <span
                        className="badge fs-5 px-4 py-2"
                        style={{
                          backgroundColor:
                            analysisResult.label === 'positive' ? '#2ecc71'
                            : analysisResult.label === 'negative' ? '#e74c3c' : '#f39c12',
                        }}
                      >
                        {(analysisResult.label || 'neutral').toUpperCase()}
                      </span>
                    </div>
                    <div className="row g-3">
                      <div className="col-4">
                        <div className="p-3 rounded text-center" style={{ backgroundColor: 'rgba(46, 204, 113, 0.15)' }}>
                          <small className="text-muted">Positive</small>
                          <h5 className="mb-0" style={{ color: '#2ecc71' }}>
                            {(analysisResult.positive_score * 100).toFixed(1)}%
                          </h5>
                        </div>
                      </div>
                      <div className="col-4">
                        <div className="p-3 rounded text-center" style={{ backgroundColor: 'rgba(231, 76, 60, 0.15)' }}>
                          <small className="text-muted">Negative</small>
                          <h5 className="mb-0" style={{ color: '#e74c3c' }}>
                            {(analysisResult.negative_score * 100).toFixed(1)}%
                          </h5>
                        </div>
                      </div>
                      <div className="col-4">
                        <div className="p-3 rounded text-center" style={{ backgroundColor: 'rgba(243, 156, 18, 0.15)' }}>
                          <small className="text-muted">Neutral</small>
                          <h5 className="mb-0" style={{ color: '#f39c12' }}>
                            {(analysisResult.neutral_score * 100).toFixed(1)}%
                          </h5>
                        </div>
                      </div>
                    </div>
                    {analysisResult.aspects && Object.keys(analysisResult.aspects).length > 0 && (
                      <div className="mt-3">
                        <h6 className="text-muted">Detected Aspects:</h6>
                        <div className="d-flex flex-wrap gap-2">
                          {Object.entries(analysisResult.aspects).map(([key, val]) => (
                            <span
                              key={key}
                              className="badge"
                              style={{
                                backgroundColor:
                                  val.sentiment === 'positive' ? 'rgba(46, 204, 113, 0.2)'
                                  : val.sentiment === 'negative' ? 'rgba(231, 76, 60, 0.2)'
                                  : 'rgba(243, 156, 18, 0.2)',
                                color:
                                  val.sentiment === 'positive' ? '#2ecc71'
                                  : val.sentiment === 'negative' ? '#e74c3c' : '#f39c12',
                                border: '1px solid currentColor',
                              }}
                            >
                              {key}: {val.sentiment}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-muted py-5">
                    <FiSend size={24} className="mb-2" />
                    <p>Enter text and click analyze to see results</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
