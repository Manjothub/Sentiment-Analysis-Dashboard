"""
DistilBERT Fine-Tuning Module
==============================
Fine-tunes DistilBERT for 3-class sentiment classification (positive/neutral/negative).
Supports early stopping, mixed precision, gradient accumulation, checkpointing,
resume training, TensorBoard logging, and automatic model registry updates.

Usage:
    python -c "from app.services.ml_pipeline.train_model import train_distilbert; train_distilbert()"
"""

import os
import sys
import json
import math
import shutil
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from datasets import load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 3
ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
RANDOM_SEED = 42

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed', 'hf_dataset')
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'checkpoints')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'saved_models')
DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'ml_models', 'saved_models', 'tokenizer')
DEFAULT_LOGS_DIR = os.path.join(PROJECT_ROOT, 'backend', 'logs')


@dataclass
class TrainingConfig:
    """Configuration for DistilBERT fine-tuning."""
    # Model parameters
    model_name: str = MODEL_NAME
    num_labels: int = NUM_LABELS

    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Mixed precision
    fp16: bool = torch.cuda.is_available()
    bf16: bool = False

    # Logging and saving
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3

    # Data
    max_length: int = 256
    dataset_dir: str = DEFAULT_DATASET_DIR
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    logs_dir: str = DEFAULT_LOGS_DIR

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for sentiment classification.

    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer

    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2]
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_negative': precision_per_class[0],
        'recall_negative': recall_per_class[0],
        'f1_negative': f1_per_class[0],
        'precision_neutral': precision_per_class[1],
        'recall_neutral': recall_per_class[1],
        'f1_neutral': f1_per_class[1],
        'precision_positive': precision_per_class[2],
        'recall_positive': recall_per_class[2],
        'f1_positive': f1_per_class[2],
    }


def load_datasets(dataset_dir: str) -> DatasetDict:
    """
    Load preprocessed Hugging Face dataset.

    Args:
        dataset_dir: Path to saved dataset

    Returns:
        DatasetDict with train/validation/test splits
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_dir}. "
            "Run prepare_dataset.py first."
        )

    logger.info(f"Loading dataset from: {dataset_dir}")
    dataset = load_from_disk(dataset_dir)

    logger.info(f"Train: {len(dataset['train']):,} samples")
    logger.info(f"Validation: {len(dataset['validation']):,} samples")
    logger.info(f"Test: {len(dataset['test']):,} samples")

    return dataset


def get_device() -> str:
    """
    Detect available device (CUDA, MPS, or CPU).

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected)")

    return device


def train_distilbert(
    config: Optional[TrainingConfig] = None,
    register_model: bool = True
) -> Dict[str, Any]:
    """
    Fine-tune DistilBERT for sentiment classification.

    Args:
        config: Training configuration (uses defaults if None)
        register_model: If True, register model in database after training

    Returns:
        Dictionary with training results and metrics
    """
    if config is None:
        config = TrainingConfig()

    logger.info("=" * 60)
    logger.info("STARTING DISTILBERT FINE-TUNING")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")

    # Detect device
    device = get_device()

    # Create output directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    # Load datasets
    dataset = load_datasets(config.dataset_dir)

    # Load tokenizer
    tokenizer_path = config.tokenizer_path
    if os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info(f"Loading tokenizer from Hugging Face: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    # Move model to device
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        evaluation_strategy='steps',
        eval_steps=config.eval_steps,
        save_strategy='steps',
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,

        # Training hyperparameters
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,

        # Mixed precision
        fp16=config.fp16 and device == 'cuda',
        bf16=config.bf16,

        # Logging
        logging_dir=os.path.join(config.logs_dir, 'tensorboard'),
        logging_steps=config.logging_steps,
        report_to=['tensorboard'],

        # Checkpointing
        resume_from_checkpoint=config.resume_from_checkpoint,

        # Reproducibility
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,

        # Other
        dataloader_num_workers=2,
        group_by_length=True,
        length_column_name='length',
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        ],
    )

    # Train the model
    logger.info("\n" + "=" * 40)
    logger.info("TRAINING PHASE")
    logger.info("=" * 40)

    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save the best model
    logger.info("\nSaving best model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Best model saved to: {config.output_dir}")

    # Save training metrics
    training_metrics = {
        'global_step': train_result.global_step,
        'training_loss': train_result.training_loss,
        'epoch': train_result.epoch,
    }
    metrics_path = os.path.join(config.output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")

    # Evaluate on test set
    logger.info("\n" + "=" * 40)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 40)

    test_results = trainer.evaluate(dataset['test'], metric_key_prefix='test')
    logger.info("Test set results:")
    for key, value in test_results.items():
        logger.info(f"  {key}: {value:.4f}")

    # Generate detailed classification report
    predictions = trainer.predict(dataset['test'])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        output_dict=True
    )

    report_path = os.path.join(config.logs_dir, 'evaluation', 'classification_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Classification report saved to: {report_path}")

    # Compile final results
    results = {
        'model_name': 'distilbert-sentiment',
        'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'training': training_metrics,
        'test_metrics': {
            k.replace('test_', ''): float(v)
            for k, v in test_results.items()
        },
        'classification_report': report,
        'config': config.to_dict(),
        'device': device,
        'training_date': datetime.now().isoformat(),
        'model_path': config.output_dir,
        'num_labels': NUM_LABELS,
        'id2label': ID2LABEL,
        'label2id': LABEL2ID,
    }

    # Register model in database
    if register_model:
        _register_model_in_db(results)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Best F1: {test_results.get('test_f1', 0):.4f}")
    logger.info(f"Best Accuracy: {test_results.get('test_accuracy', 0):.4f}")
    logger.info(f"Model saved at: {config.output_dir}")
    logger.info("=" * 60)

    return results


def _register_model_in_db(results: Dict[str, Any]) -> None:
    """
    Register trained model in the model_versions database table.

    Args:
        results: Training results dictionary with metrics
    """
    try:
        from app.database import db
        from app.models.model_version import ModelVersion
        from flask import Flask
        from app.config import Config

        # Create minimal Flask app for database access
        app = Flask(__name__)
        app.config.from_object(Config)
        db.init_app(app)

        with app.app_context():
            # Check if this version already exists
            existing = ModelVersion.query.filter_by(
                model_name=results['model_name'],
                version=results['version']
            ).first()

            if existing:
                logger.info(f"Model version {results['version']} already registered")
                return

            # Create new model version record
            model_version = ModelVersion(
                model_name=results['model_name'],
                version=results['version'],
                accuracy=results['test_metrics'].get('accuracy', 0),
                f1_score=results['test_metrics'].get('f1', 0),
                precision=results['test_metrics'].get('precision', 0),
                recall=results['test_metrics'].get('recall', 0),
                training_date=datetime.now(),
                parameters=results['config'],
                notes=f"Fine-tuned DistilBERT on Amazon Reviews. "
                      f"Device: {results['device']}. "
                      f"Epochs: {results['training'].get('epoch', 0):.1f}"
            )

            db.session.add(model_version)
            db.session.commit()
            logger.info(f"Model registered in database: {results['model_name']} v{results['version']}")

    except Exception as e:
        logger.warning(f"Failed to register model in database: {e}")
        logger.warning("Model was saved to disk but not registered in DB")


if __name__ == '__main__':
    """Run training when executed directly."""
    from app.utils.logger import setup_logging
    setup_logging()
    train_distilbert()
