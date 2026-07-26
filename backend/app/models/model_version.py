"""
Model Version Model
===================
SQLAlchemy ORM model for tracking model versions and their performance metrics.
"""

from datetime import datetime, timezone
from app.database import db


class ModelVersion(db.Model):
    """
    ModelVersion model for tracking trained model versions.
    
    Attributes:
        version_id: Primary key
        model_name: Name of the model
        version: Version string
        accuracy: Model accuracy score
        f1_score: Model F1 score
        precision: Model precision score
        recall: Model recall score
        training_date: When the model was trained
        parameters: JSONB field for training parameters
        notes: Additional notes about the model
        created_at: Record creation timestamp
    """
    __tablename__ = 'model_versions'

    version_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    training_date = db.Column(db.DateTime(timezone=True), nullable=True)
    parameters = db.Column(db.JSON, default=dict)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    __table_args__ = (
        db.UniqueConstraint('model_name', 'version', name='uq_model_version'),
    )

    def __repr__(self) -> str:
        return f"<ModelVersion {self.model_name}: v{self.version}>"

    def to_dict(self) -> dict:
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'version': self.version,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'parameters': self.parameters,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
