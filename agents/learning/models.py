"""
Data models for Learning Agent

Adaptive models and performance feedback models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class LearningMethod(str, Enum):
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ONLINE_LEARNING = "online_learning"
    ENSEMBLE_LEARNING = "ensemble_learning"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"


class ModelType(str, Enum):
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    LINEAR_MODEL = "linear_model"


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    model_type: ModelType
    timestamp: datetime
    
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Financial metrics
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float
    
    # Learning metrics
    training_loss: float
    validation_loss: float
    learning_rate: float
    convergence_epochs: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "learning_rate": self.learning_rate,
            "convergence_epochs": self.convergence_epochs
        }


@dataclass
class AdaptationResult:
    """Results from model adaptation"""
    adaptation_id: str
    timestamp: datetime
    
    # Performance change
    performance_before: ModelPerformance
    performance_after: ModelPerformance
    improvement: float
    
    # Adaptation details
    learning_method: LearningMethod
    features_added: List[str]
    features_removed: List[str]
    hyperparameters_changed: Dict[str, Any]
    
    # Validation
    cross_validation_score: float
    out_of_sample_performance: float
    statistical_significance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaptation_id": self.adaptation_id,
            "timestamp": self.timestamp.isoformat(),
            "performance_before": self.performance_before.to_dict(),
            "performance_after": self.performance_after.to_dict(),
            "improvement": self.improvement,
            "learning_method": self.learning_method.value,
            "features_added": self.features_added,
            "features_removed": self.features_removed,
            "hyperparameters_changed": self.hyperparameters_changed,
            "cross_validation_score": self.cross_validation_score,
            "out_of_sample_performance": self.out_of_sample_performance,
            "statistical_significance": self.statistical_significance
        }


@dataclass
class LearningAnalysis:
    """Complete learning system analysis"""
    system_id: str
    timestamp: datetime
    
    # Current model ensemble
    active_models: List[ModelPerformance]
    best_performing_model: str
    ensemble_performance: ModelPerformance
    
    # Learning progress
    recent_adaptations: List[AdaptationResult]
    learning_trajectory: List[float]  # Performance over time
    
    # System insights
    feature_importance: Dict[str, float]
    model_correlations: Dict[str, float]
    regime_detection: Dict[str, str]
    
    # Future projections
    expected_performance_improvement: float
    recommended_adaptations: List[str]
    next_learning_cycle: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "timestamp": self.timestamp.isoformat(),
            "active_models": [m.to_dict() for m in self.active_models],
            "best_performing_model": self.best_performing_model,
            "ensemble_performance": self.ensemble_performance.to_dict(),
            "recent_adaptations": [a.to_dict() for a in self.recent_adaptations],
            "learning_trajectory": self.learning_trajectory,
            "feature_importance": self.feature_importance,
            "model_correlations": self.model_correlations,
            "regime_detection": self.regime_detection,
            "expected_performance_improvement": self.expected_performance_improvement,
            "recommended_adaptations": self.recommended_adaptations,
            "next_learning_cycle": self.next_learning_cycle.isoformat()
        }
