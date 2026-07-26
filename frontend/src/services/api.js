import axios from 'axios';
import { io } from 'socket.io-client';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
});

export const socket = io(import.meta.env.VITE_API_URL || 'http://localhost:5000', {
  transports: ['websocket', 'polling'],
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 10,
});

export const sentimentAPI = {
  analyze: (text) => api.post('/analyze', { text }),
  analyzeBatch: (reviews) => api.post('/analyze-batch', { reviews }),
  ingest: (data) => api.post('/ingest', data),
  getStats: (productId) => api.get('/stats', { params: { product_id: productId } }),
  getReviews: (params) => api.get('/reviews', { params }),
  getModelStatus: () => api.get('/model/status'),
};

export const trendsAPI = {
  getTrendingTopics: (params) => api.get('/trending-topics', { params }),
  getTrendsOverTime: (params) => api.get('/trends-over-time', { params }),
  getAspectAnalysis: (params) => api.get('/aspects', { params }),
};

export const alertsAPI = {
  getAlerts: (params) => api.get('/alerts', { params }),
  checkAlerts: (productId) => api.post('/alerts/check', { product_id: productId }),
  acknowledgeAlert: (alertId) => api.patch(`/alerts/${alertId}/acknowledge`),
  getAlertSummary: (params) => api.get('/alerts/summary', { params }),
};

export const comparativeAPI = {
  compare: (params) => api.get('/compare', { params }),
  getCompetitors: (params) => api.get('/competitors', { params }),
  addCompetitor: (data) => api.post('/competitors', data),
  compareAspects: (params) => api.get('/compare/aspects', { params }),
};

export const dashboardAPI = {
  getOverview: (params) => api.get('/dashboard/overview', { params }),
};

export default api;
