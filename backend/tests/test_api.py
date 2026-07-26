"""
Integration Tests for API Endpoints
====================================
Tests for the Flask API endpoints including Phase 2 ML endpoints.
Uses Flask test client for HTTP-level testing.
"""

import os
import sys
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database import db


@pytest.fixture
def app():
    """Create a Flask application configured for testing."""
    app = create_app('testing')

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create a test client for the application."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_endpoint(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data

    def test_health_response_structure(self, client):
        """Test health response has correct structure."""
        response = client.get('/api/health')
        data = json.loads(response.data)
        assert data['status'] in ['healthy', 'degraded']


class TestModelStatusEndpoint:
    """Tests for model status endpoint."""

    def test_model_status_endpoint(self, client):
        """Test model status endpoint returns 200."""
        response = client.get('/api/model/status')
        # May return 503 if model not loaded, but should still have valid structure
        data = json.loads(response.data)
        assert 'models' in data
        assert 'sentiment' in data['models']
        assert 'aspect_extraction' in data['models']
        assert 'topic_modeling' in data['models']

    def test_model_status_structure(self, client):
        """Test model status response structure."""
        response = client.get('/api/model/status')
        data = json.loads(response.data)
        assert 'application' in data
        assert 'timestamp' in data
        assert 'overall_status' in data
        assert data['application'] == 'running'


class TestPredictEndpoint:
    """Tests for sentiment prediction endpoint."""

    def test_predict_no_text(self, client):
        """Test predict without text returns 400."""
        response = client.post('/api/predict', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_empty_text(self, client):
        """Test predict with empty text returns 400."""
        response = client.post('/api/predict', json={'text': ''})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_with_text(self, client):
        """Test predict with valid text returns prediction."""
        response = client.post('/api/predict', json={'text': 'Great product!'})
        # Should return 200 or 503 (if model not loaded)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'sentiment' in data
            assert 'metadata' in data

    def test_predict_with_aspects(self, client):
        """Test predict with aspect extraction enabled."""
        response = client.post('/api/predict', json={
            'text': 'Great product quality and fast shipping!',
            'include_aspects': True
        })
        assert response.status_code in [200, 503]

    def test_predict_with_product_id(self, client):
        """Test predict with product ID."""
        response = client.post('/api/predict', json={
            'text': 'Good product',
            'product_id': 'test_product_123'
        })
        assert response.status_code in [200, 503]

    def test_predict_invalid_json(self, client):
        """Test predict with invalid JSON."""
        response = client.post('/api/predict', data='not json', content_type='application/json')
        assert response.status_code == 400

    def test_predict_with_store(self, client):
        """Test predict with store_result enabled."""
        response = client.post('/api/predict', json={
            'text': 'Amazing product!',
            'store_result': True,
            'product_id': 'test_product'
        })
        assert response.status_code in [200, 503]


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_no_texts(self, client):
        """Test batch predict without texts array."""
        response = client.post('/api/batch_predict', json={})
        assert response.status_code == 400

    def test_batch_predict_empty_array(self, client):
        """Test batch predict with empty array."""
        response = client.post('/api/batch_predict', json={'texts': []})
        assert response.status_code == 400

    def test_batch_predict_valid(self, client):
        """Test batch predict with valid texts."""
        response = client.post('/api/batch_predict', json={
            'texts': ['Great product!', 'Terrible item.', 'Okay']
        })
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'results' in data
            assert 'statistics' in data
            assert data['total_texts'] == 3

    def test_batch_predict_too_many(self, client):
        """Test batch predict with too many texts."""
        texts = ['text'] * 101
        response = client.post('/api/batch_predict', json={'texts': texts})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_batch_predict_invalid_type(self, client):
        """Test batch predict with non-array texts."""
        response = client.post('/api/batch_predict', json={'texts': 'not an array'})
        assert response.status_code == 400

    def test_batch_predict_statistics(self, client):
        """Test batch predict returns statistics."""
        response = client.post('/api/batch_predict', json={
            'texts': ['Good', 'Bad', 'Great', 'Terrible', 'Fine']
        })
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = json.loads(response.data)
            stats = data['statistics']
            assert 'sentiment_distribution' in stats
            assert 'average_confidence' in stats

    def test_batch_predict_with_store(self, client):
        """Test batch predict with store_results enabled."""
        response = client.post('/api/batch_predict', json={
            'texts': ['Good product', 'Bad product'],
            'store_results': True,
            'product_id': 'test_product'
        })
        assert response.status_code in [200, 503]


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_endpoint(self, client):
        """Test model info endpoint returns valid response."""
        response = client.get('/api/model/info')
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'model_info' in data

    def test_model_info_structure(self, client):
        """Test model info response structure."""
        response = client.get('/api/model/info')
        data = json.loads(response.data)
        if response.status_code == 200:
            assert 'success' in data
            assert data['success'] == True
            assert 'model_info' in data
            assert 'timestamp' in data


class TestTopicsEndpoint:
    """Tests for topics endpoint."""

    def test_topics_endpoint(self, client):
        """Test topics endpoint returns valid response."""
        response = client.get('/api/topics')
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'topics' in data

    def test_topics_with_params(self, client):
        """Test topics endpoint with query parameters."""
        response = client.get('/api/topics?n_topics=5&include_docs=true')
        assert response.status_code in [200, 503]

    def test_topics_default_params(self, client):
        """Test topics endpoint with default parameters."""
        response = client.get('/api/topics')
        assert response.status_code in [200, 503]


class TestExistingEndpoints:
    """Tests that existing endpoints still work (regression tests)."""

    def test_sentiment_endpoint(self, client):
        """Test existing sentiment endpoint still works."""
        response = client.get('/api/sentiment/test_product')
        assert response.status_code in [200, 404, 503]

    def test_trends_endpoint(self, client):
        """Test existing trends endpoint still works."""
        response = client.get('/api/trends')
        assert response.status_code in [200, 503]

    def test_alerts_endpoint(self, client):
        """Test existing alerts endpoint still works."""
        response = client.get('/api/alerts')
        assert response.status_code in [200, 503]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
