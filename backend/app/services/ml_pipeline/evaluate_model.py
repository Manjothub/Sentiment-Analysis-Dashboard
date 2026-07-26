"""
Model Evaluation Module
=======================
Evaluates a trained DistilBERT model on the test dataset, generating:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (visualized)
- ROC Curve (one-vs-rest)
- Classification Report
- Misclassified examples analysis
- All plots saved to backend/logs/evaluation/

Usage:
    python -c "from app.services.ml_pipeline.evaluate_model import evaluate_model; evaluate_model()"
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
NUM_LABELS = 3
ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
CLASS_NAMES = [ID2LABEL[i] for i in range(NUM_LABELS)]
COLORS = {'negative': '#ff4444', 'neutral': '#ffbb33', 'positive': '#00C851'}

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'hf_dataset')
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'saved_models')
DEFAULT_EVAL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'logs', 'evaluation')


def load_model_and_tokenizer(model_dir: str):
    """
    Load a trained model and tokenizer from disk.

    Args:
        model_dir: Path to saved model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from: {model_dir}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Check what files exist
    files = os.listdir(model_dir)
    logger.info(f"Model directory contents: {files}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Set model to evaluation mode
    model.eval()

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer


def load_test_dataset(dataset_dir: str):
    """
    Load the test split from a Hugging Face dataset.

    Args:
        dataset_dir: Path to saved dataset

    Returns:
        Test dataset
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    dataset = load_from_disk(dataset_dir)
    test_dataset = dataset['test']

    logger.info(f"Test dataset loaded: {len(test_dataset):,} samples")
    return test_dataset


def get_predictions(
    model,
    tokenizer,
    test_dataset,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions for the test dataset.

    Args:
        model: Trained Hugging Face model
        tokenizer: Corresponding tokenizer
        test_dataset: Test dataset
        batch_size: Batch size for inference

    Returns:
        Tuple of (true_labels, predicted_labels, probabilities)
    """
    logger.info("Generating predictions...")

    # Create a temporary trainer for prediction
    training_args = TrainingArguments(
        output_dir=os.path.join(PROJECT_ROOT, 'backend', 'logs', '_tmp_eval'),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    # Get predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    true_labels = predictions.label_ids

    # Convert logits to probabilities and predictions
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predicted_labels = np.argmax(probabilities, axis=1)

    logger.info(f"Predictions generated: {len(predicted_labels):,} samples")

    return true_labels, predicted_labels, probabilities


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        probabilities: Prediction probabilities

    Returns:
        Dictionary with all metrics
    """
    logger.info("Calculating metrics...")

    # Overall metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted'
    )

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(true_labels, predicted_labels, labels=[0, 1, 2])

    # Generate classification report
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])

    # ROC AUC (one-vs-rest)
    try:
        true_binarized = label_binarize(true_labels, classes=[0, 1, 2])
        roc_auc_scores = {}
        for i, class_name in enumerate(CLASS_NAMES):
            if true_binarized.shape[1] > i:
                roc_auc_scores[class_name] = roc_auc_score(
                    true_binarized[:, i], probabilities[:, i]
                )
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {e}")
        roc_auc_scores = {name: 0.0 for name in CLASS_NAMES}

    metrics = {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': round(float(per_class_precision[i]), 4),
                'recall': round(float(per_class_recall[i]), 4),
                'f1_score': round(float(per_class_f1[i]), 4),
                'support': int(per_class_support[i])
            }
            for i in range(NUM_LABELS)
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc_scores,
        'total_samples': len(true_labels),
        'correct_predictions': int(np.sum(true_labels == predicted_labels)),
        'incorrect_predictions': int(np.sum(true_labels != predicted_labels)),
    }

    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")

    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_dir: str) -> str:
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        save_dir: Directory to save the plot

    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - DistilBERT Sentiment Analysis', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Add percentages
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / cm_sum * 100
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j + 0.5, i + 0.6, f'{cm_percent[i, j]:.1f}%',
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=10)

    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to: {path}")
    return path


def plot_roc_curves(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    save_dir: str
) -> str:
    """
    Plot and save ROC curves (one-vs-rest).

    Args:
        true_labels: Ground truth labels
        probabilities: Prediction probabilities
        save_dir: Directory to save the plot

    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 8))

    # Binarize labels for multi-class ROC
    true_binarized = label_binarize(true_labels, classes=[0, 1, 2])

    colors = ['#ff4444', '#ffbb33', '#00C851']

    for i, class_name in enumerate(CLASS_NAMES):
        if true_binarized.shape[1] <= i:
            continue

        fpr, tpr, _ = roc_curve(true_binarized[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            color=colors[i],
            lw=2,
            label=f'{class_name} (AUC = {roc_auc:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - DistilBERT Sentiment Analysis', fontsize=16, pad=20)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    path = os.path.join(save_dir, 'roc_curves.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved to: {path}")
    return path


def plot_class_metrics(metrics: Dict, save_dir: str) -> str:
    """
    Plot per-class precision, recall, F1 scores.

    Args:
        metrics: Metrics dictionary
        save_dir: Directory to save the plot

    Returns:
        Path to saved plot
    """
    classes = list(metrics['per_class'].keys())
    precision = [metrics['per_class'][c]['precision'] for c in classes]
    recall = [metrics['per_class'][c]['recall'] for c in classes]
    f1 = [metrics['per_class'][c]['f1_score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#FF9800')

    ax.set_xlabel('Sentiment Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    path = os.path.join(save_dir, 'per_class_metrics.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class metrics saved to: {path}")
    return path


def find_misclassified(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    texts: List[str],
    top_n: int = 20
) -> List[Dict]:
    """
    Find and analyze misclassified examples.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        probabilities: Prediction probabilities
        texts: Original text for each sample
        top_n: Number of misclassified examples to return

    Returns:
        List of misclassified example dictionaries
    """
    misclassified_indices = np.where(true_labels != predicted_labels)[0]

    # Sort by confidence (highest confidence mistakes first)
    confidences = np.max(probabilities, axis=1)
    mistake_confidences = confidences[misclassified_indices]
    sorted_indices = misclassified_indices[np.argsort(-mistake_confidences)]

    examples = []
    for idx in sorted_indices[:top_n]:
        examples.append({
            'index': int(idx),
            'text': str(texts[idx])[:200] if idx < len(texts) else '',
            'true_label': ID2LABEL[int(true_labels[idx])],
            'predicted_label': ID2LABEL[int(predicted_labels[idx])],
            'confidence': round(float(confidences[idx]), 4),
            'probabilities': {
                ID2LABEL[i]: round(float(probabilities[idx][i]), 4)
                for i in range(NUM_LABELS)
            }
        })

    logger.info(f"Found {len(misclassified_indices)} misclassified examples")
    logger.info(f"Misclassification rate: {len(misclassified_indices) / len(true_labels) * 100:.2f}%")

    return examples


def save_evaluation_report(metrics: Dict[str, Any], save_dir: str) -> str:
    """
    Save complete evaluation report as JSON.

    Args:
        metrics: All evaluation metrics
        save_dir: Directory to save the report

    Returns:
        Path to saved report
    """
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'model_type': 'DistilBERT',
        'num_classes': NUM_LABELS,
        'class_mapping': ID2LABEL,
        'metrics': metrics
    }

    path = os.path.join(save_dir, 'evaluation_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to: {path}")
    return path


def evaluate_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    eval_dir: str = DEFAULT_EVAL_DIR
) -> Dict[str, Any]:
    """
    Run complete model evaluation pipeline.

    Args:
        model_dir: Path to trained model directory
        dataset_dir: Path to Hugging Face dataset
        eval_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Create evaluation directory
    os.makedirs(eval_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # Load test dataset
    test_dataset = load_test_dataset(dataset_dir)

    # Get texts for misclassification analysis
    dataset_full = load_from_disk(dataset_dir)
    test_texts = dataset_full['test']['cleaned_text'] if 'cleaned_text' in dataset_full['test'].column_names else []

    # Generate predictions
    true_labels, predicted_labels, probabilities = get_predictions(model, tokenizer, test_dataset)

    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels, probabilities)

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    cm_path = plot_confusion_matrix(np.array(metrics['confusion_matrix']), eval_dir)
    roc_path = plot_roc_curves(true_labels, probabilities, eval_dir)
    metrics_path = plot_class_metrics(metrics, eval_dir)

    # Find misclassified examples
    misclassified = find_misclassified(
        true_labels, predicted_labels, probabilities,
        test_texts[:len(true_labels)]
    )

    # Save all results
    results = {
        **metrics,
        'visualizations': {
            'confusion_matrix': cm_path,
            'roc_curves': roc_path,
            'per_class_metrics': metrics_path
        },
        'misclassified_examples': misclassified[:10],  # Top 10 for display
        'all_misclassified_count': len(misclassified),
        'evaluation_dir': eval_dir
    }

    # Save comprehensive report
    save_evaluation_report(metrics, eval_dir)

    # Save misclassified examples separately
    misclassified_path = os.path.join(eval_dir, 'misclassified_examples.json')
    with open(misclassified_path, 'w') as f:
        json.dump(misclassified, f, indent=2)
    logger.info(f"Misclassified examples saved to: {misclassified_path}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Visualizations saved to: {eval_dir}")
    logger.info("=" * 60)

    return results


if __name__ == '__main__':
    """Run evaluation when executed directly."""
    from app.utils.logger import setup_logging
    setup_logging()
    evaluate_model()
