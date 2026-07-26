"""
Unit Tests for ML Pipeline
===========================
Tests for dataset preparation, tokenization, training, and evaluation modules.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml_pipeline.prepare_dataset import (
    load_and_validate_data,
    analyze_class_balance,
    stratified_split,
    LABEL_MAP,
    ID2LABEL
)
from app.services.ml_pipeline.train_model import (
    TrainingConfig,
    compute_metrics,
    get_device
)
from app.services.ml_pipeline.evaluate_model import (
    calculate_metrics,
    find_misclassified
)


class TestPrepareDataset:
    """Tests for dataset preparation module."""

    def test_label_map_completeness(self):
        """Verify LABEL_MAP covers all expected labels."""
        assert 'positive' in LABEL_MAP
        assert 'neutral' in LABEL_MAP
        assert 'negative' in LABEL_MAP
        assert LABEL_MAP['positive'] == 2
        assert LABEL_MAP['neutral'] == 1
        assert LABEL_MAP['negative'] == 0

    def test_id2label_mapping(self):
        """Verify ID2LABEL reverse mapping is correct."""
        for label, idx in LABEL_MAP.items():
            assert ID2LABEL[idx] == label

    def test_load_and_validate_data_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_and_validate_data('/nonexistent/path.csv')

    def test_load_and_validate_data_missing_columns(self, tmp_path):
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({'wrong_col': ['text1', 'text2']})
        path = tmp_path / 'test.csv'
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match='Missing required columns'):
            load_and_validate_data(str(path))

    def test_load_and_validate_data_valid(self, tmp_path):
        """Test loading valid data works correctly."""
        df = pd.DataFrame({
            'cleaned_text': ['Great product', 'Terrible item', 'Okay'],
            'sentiment_label': ['positive', 'negative', 'neutral']
        })
        path = tmp_path / 'test.csv'
        df.to_csv(path, index=False)
        result = load_and_validate_data(str(path))
        assert len(result) == 3
        assert all(col in result.columns for col in ['cleaned_text', 'sentiment_label'])

    def test_load_and_validate_data_removes_invalid_labels(self, tmp_path):
        """Test that invalid sentiment labels are filtered out."""
        df = pd.DataFrame({
            'cleaned_text': ['Great', 'Terrible', 'Okay', 'Invalid'],
            'sentiment_label': ['positive', 'negative', 'neutral', 'unknown']
        })
        path = tmp_path / 'test.csv'
        df.to_csv(path, index=False)
        result = load_and_validate_data(str(path))
        assert len(result) == 3
        assert 'unknown' not in result['sentiment_label'].values

    def test_load_and_validate_data_removes_short_texts(self, tmp_path):
        """Test that very short texts are removed."""
        df = pd.DataFrame({
            'cleaned_text': ['Great product!', 'OK', 'A'],
            'sentiment_label': ['positive', 'neutral', 'negative']
        })
        path = tmp_path / 'test.csv'
        df.to_csv(path, index=False)
        result = load_and_validate_data(str(path))
        # 'OK' (len=2) and 'A' (len=1) both fail str.len() >= 3 filter
        assert len(result) == 1  # Only 'Great product!' remains

    def test_analyze_class_balance(self):
        """Test class balance analysis."""
        df = pd.DataFrame({
            'sentiment_label': ['positive', 'positive', 'negative', 'neutral', 'positive']
        })
        result = analyze_class_balance(df)
        assert result['total'] == 5
        assert result['counts']['positive'] == 3
        assert result['counts']['negative'] == 1
        assert result['counts']['neutral'] == 1

    def test_stratified_split_ratios(self):
        """Test that stratified split maintains correct ratios."""
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'cleaned_text': [f'Review {i}' for i in range(n_samples)],
            'sentiment_label': np.random.choice(
                ['positive', 'negative', 'neutral'],
                size=n_samples,
                p=[0.5, 0.3, 0.2]
            )
        })

        train_df, val_df, test_df = stratified_split(df, 0.8, 0.1, 0.1)

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == n_samples
        assert abs(len(train_df) / n_samples - 0.8) < 0.05
        assert abs(len(val_df) / n_samples - 0.1) < 0.05
        assert abs(len(test_df) / n_samples - 0.1) < 0.05

    def test_stratified_split_preserves_distribution(self):
        """Test that class distribution is preserved across splits."""
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'cleaned_text': [f'Review {i}' for i in range(n_samples)],
            'sentiment_label': np.random.choice(
                ['positive', 'negative', 'neutral'],
                size=n_samples,
                p=[0.5, 0.3, 0.2]
            )
        })

        original_dist = df['sentiment_label'].value_counts(normalize=True)
        train_df, val_df, test_df = stratified_split(df, 0.8, 0.1, 0.1)

        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_dist = split_df['sentiment_label'].value_counts(normalize=True)
            for label in ['positive', 'negative', 'neutral']:
                assert abs(split_dist.get(label, 0) - original_dist.get(label, 0)) < 0.1


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 2e-5
        assert config.batch_size == 16
        assert config.num_epochs == 10
        assert config.early_stopping_patience == 3
        assert config.max_length == 256

    def test_config_to_dict(self):
        """Test config serialization to dictionary."""
        config = TrainingConfig(learning_rate=1e-4, batch_size=32)
        config_dict = config.to_dict()
        assert config_dict['learning_rate'] == 1e-4
        assert config_dict['batch_size'] == 32
        assert config_dict['num_epochs'] == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=8,
            num_epochs=5,
            early_stopping_patience=5
        )
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8
        assert config.num_epochs == 5
        assert config.early_stopping_patience == 5


class TestComputeMetrics:
    """Tests for evaluation metrics computation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 2, 0, 1, 2])
        # Simulate logits (higher value = higher probability)
        logits = np.eye(3)[predictions] * 10

        metrics = compute_metrics((logits, labels))
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        labels = np.array([0, 0, 0])
        predictions = np.array([1, 1, 1])
        logits = np.eye(3)[predictions] * 10

        metrics = compute_metrics((logits, labels))
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0

    def test_mixed_predictions(self):
        """Test metrics with mixed correct/incorrect predictions."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 1, 0, 2, 2])
        logits = np.eye(3)[predictions] * 10

        metrics = compute_metrics((logits, labels))
        assert 0.5 <= metrics['accuracy'] <= 1.0
        assert 0.0 < metrics['f1'] <= 1.0

    def test_per_class_metrics(self):
        """Test per-class metric computation."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        predictions = np.array([0, 0, 1, 1, 2, 2])
        logits = np.eye(3)[predictions] * 10

        metrics = compute_metrics((logits, labels))
        assert metrics['f1_negative'] == 1.0
        assert metrics['f1_neutral'] == 1.0
        assert metrics['f1_positive'] == 1.0


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_basic_metrics(self):
        """Test basic metric calculation."""
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        predicted_labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.eye(3)[predicted_labels]

        metrics = calculate_metrics(true_labels, predicted_labels, probabilities)
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['total_samples'] == 6
        assert metrics['correct_predictions'] == 6
        assert metrics['incorrect_predictions'] == 0

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        predicted_labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.eye(3)[predicted_labels]

        metrics = calculate_metrics(true_labels, predicted_labels, probabilities)
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (3, 3)

    def test_per_class_support(self):
        """Test per-class support counts."""
        true_labels = np.array([0, 0, 0, 1, 1, 2])
        predicted_labels = np.array([0, 0, 0, 1, 1, 2])
        probabilities = np.eye(3)[predicted_labels]

        metrics = calculate_metrics(true_labels, predicted_labels, probabilities)
        assert metrics['per_class']['negative']['support'] == 3
        assert metrics['per_class']['neutral']['support'] == 2
        assert metrics['per_class']['positive']['support'] == 1


class TestFindMisclassified:
    """Tests for misclassified example detection."""

    def test_no_misclassified(self):
        """Test with no misclassified examples."""
        true_labels = np.array([0, 1, 2])
        predicted_labels = np.array([0, 1, 2])
        probabilities = np.array([[0.9, 0.05, 0.05],
                                  [0.05, 0.9, 0.05],
                                  [0.05, 0.05, 0.9]])
        texts = ['Great', 'Okay', 'Terrible']

        examples = find_misclassified(true_labels, predicted_labels, probabilities, texts)
        assert len(examples) == 0

    def test_all_misclassified(self):
        """Test with all misclassified examples."""
        true_labels = np.array([0, 1, 2])
        predicted_labels = np.array([1, 2, 0])
        probabilities = np.array([[0.1, 0.8, 0.1],
                                  [0.1, 0.1, 0.8],
                                  [0.8, 0.1, 0.1]])
        texts = ['Great', 'Okay', 'Terrible']

        examples = find_misclassified(true_labels, predicted_labels, probabilities, texts)
        assert len(examples) == 3

    def test_misclassified_details(self):
        """Test misclassified example details are correct."""
        true_labels = np.array([0, 1])
        predicted_labels = np.array([1, 0])
        probabilities = np.array([[0.2, 0.7, 0.1],
                                  [0.8, 0.1, 0.1]])
        texts = ['Should be negative', 'Should be neutral']

        examples = find_misclassified(true_labels, predicted_labels, probabilities, texts, top_n=2)
        assert len(examples) == 2
        # Sorted by confidence descending: index 1 (confidence 0.8) first, then index 0 (confidence 0.7)
        assert examples[0]['true_label'] == 'neutral'
        assert examples[0]['predicted_label'] == 'negative'
        assert examples[1]['true_label'] == 'negative'
        assert examples[1]['predicted_label'] == 'neutral'

    def test_misclassified_confidence(self):
        """Test confidence scores in misclassified examples."""
        true_labels = np.array([0, 0])
        predicted_labels = np.array([1, 1])
        probabilities = np.array([[0.1, 0.9, 0.0],
                                  [0.4, 0.6, 0.0]])
        texts = ['High confidence mistake', 'Low confidence mistake']

        examples = find_misclassified(true_labels, predicted_labels, probabilities, texts, top_n=2)
        # Should be sorted by confidence (highest first)
        assert examples[0]['confidence'] >= examples[1]['confidence']


class TestGetDevice:
    """Tests for device detection."""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_cuda_device(self, mock_get_device_properties, mock_get_device_name, mock_cuda):
        """Test CUDA device detection."""
        mock_cuda.return_value = True
        mock_get_device_name.return_value = 'NVIDIA Test GPU'
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8 GB
        mock_get_device_properties.return_value = mock_props
        device = get_device()
        assert device == 'cuda'

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_mps_device(self, mock_mps, mock_cuda):
        """Test MPS device detection."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        device = get_device()
        assert device == 'mps'

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_cpu_fallback(self, mock_mps, mock_cuda):
        """Test CPU fallback when no GPU available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        device = get_device()
        assert device == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
