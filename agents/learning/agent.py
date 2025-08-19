"""
Learning Agent

Adaptive models and performance optimization
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .models import (
    LearningAnalysis, ModelPerformance, AdaptationResult,
    LearningMethod, ModelType
)
from ..common.models import BaseAgent


class LearningAgent(BaseAgent):
    """Complete Adaptive Learning Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("learning", config)
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.analyze_learning_system(*args, **kwargs)
    
    async def analyze_learning_system(
        self,
        system_id: str = "trading_system_1"
    ) -> Dict[str, Any]:
        """Analyze learning system performance and adaptations"""
        
        # Create demo learning analysis
        analysis = self._create_demo_analysis(system_id)
        
        return {
            "learning_analysis": analysis.to_dict()
        }
    
    def _create_demo_analysis(self, system_id: str) -> LearningAnalysis:
        """Create demo learning analysis"""
        
        # Generate model performances
        models = []
        model_types = [ModelType.NEURAL_NETWORK, ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]
        
        for i, model_type in enumerate(model_types):
            performance = ModelPerformance(
                model_id=f"model_{i+1}",
                model_type=ModelType.NEURAL_NETWORK,
                timestamp=datetime.now(),
                accuracy=np.random.uniform(0.55, 0.75),
                precision=np.random.uniform(0.5, 0.8),
                recall=np.random.uniform(0.45, 0.75),
                f1_score=np.random.uniform(0.5, 0.75),
                sharpe_ratio=np.random.uniform(0.8, 2.2),
                max_drawdown=np.random.uniform(-0.15, -0.05),
                hit_rate=np.random.uniform(0.52, 0.68),
                profit_factor=np.random.uniform(1.1, 2.5),
                training_loss=np.random.uniform(0.1, 0.4),
                validation_loss=np.random.uniform(0.15, 0.5),
                learning_rate=np.random.uniform(0.001, 0.1),
                convergence_epochs=np.random.randint(50, 200)
            )
            models.append(performance)
        
        # Best performing model
        best_model = max(models, key=lambda m: m.sharpe_ratio)
        
        # Ensemble performance (typically better than individual models)
        ensemble = ModelPerformance(
            model_id="ensemble",
            model_type=ModelType.NEURAL_NETWORK,  # Placeholder
            timestamp=datetime.now(),
            accuracy=np.random.uniform(0.65, 0.80),
            precision=np.random.uniform(0.6, 0.85),
            recall=np.random.uniform(0.55, 0.80),
            f1_score=np.random.uniform(0.6, 0.8),
            sharpe_ratio=np.random.uniform(1.5, 2.8),
            max_drawdown=np.random.uniform(-0.12, -0.03),
            hit_rate=np.random.uniform(0.58, 0.72),
            profit_factor=np.random.uniform(1.5, 3.0),
            training_loss=np.random.uniform(0.05, 0.2),
            validation_loss=np.random.uniform(0.08, 0.25),
            learning_rate=0.01,
            convergence_epochs=100
        )
        
        # Generate adaptation results
        adaptations = []
        for i in range(3):
            before = ModelPerformance(
                model_id=f"adaptation_{i}_before",
                model_type=ModelType.GRADIENT_BOOSTING,
                timestamp=datetime.now() - timedelta(days=i*30),
                accuracy=np.random.uniform(0.5, 0.65),
                precision=np.random.uniform(0.45, 0.7),
                recall=np.random.uniform(0.4, 0.65),
                f1_score=np.random.uniform(0.45, 0.65),
                sharpe_ratio=np.random.uniform(0.5, 1.5),
                max_drawdown=np.random.uniform(-0.2, -0.08),
                hit_rate=np.random.uniform(0.48, 0.62),
                profit_factor=np.random.uniform(0.8, 1.8),
                training_loss=np.random.uniform(0.2, 0.5),
                validation_loss=np.random.uniform(0.25, 0.6),
                learning_rate=0.01,
                convergence_epochs=150
            )
            
            after = ModelPerformance(
                model_id=f"adaptation_{i}_after",
                model_type=before.model_type,
                timestamp=datetime.now() - timedelta(days=i*30-7),
                accuracy=before.accuracy + np.random.uniform(0.02, 0.08),
                precision=before.precision + np.random.uniform(0.01, 0.06),
                recall=before.recall + np.random.uniform(0.01, 0.06),
                f1_score=before.f1_score + np.random.uniform(0.01, 0.06),
                sharpe_ratio=before.sharpe_ratio + np.random.uniform(0.1, 0.5),
                max_drawdown=before.max_drawdown + np.random.uniform(0.01, 0.04),
                hit_rate=before.hit_rate + np.random.uniform(0.02, 0.06),
                profit_factor=before.profit_factor + np.random.uniform(0.1, 0.4),
                training_loss=before.training_loss - np.random.uniform(0.02, 0.08),
                validation_loss=before.validation_loss - np.random.uniform(0.02, 0.08),
                learning_rate=0.01,
                convergence_epochs=120
            )
            
            adaptation = AdaptationResult(
                adaptation_id=f"adaptation_{i+1}",
                timestamp=datetime.now() - timedelta(days=i*30),
                performance_before=before,
                performance_after=after,
                improvement=(after.sharpe_ratio - before.sharpe_ratio) / before.sharpe_ratio,
                learning_method=LearningMethod.REINFORCEMENT_LEARNING,
                features_added=[f"feature_{j}" for j in range(np.random.randint(1, 4))],
                features_removed=[f"old_feature_{j}" for j in range(np.random.randint(0, 2))],
                hyperparameters_changed={"learning_rate": 0.01, "batch_size": 64},
                cross_validation_score=np.random.uniform(0.6, 0.8),
                out_of_sample_performance=np.random.uniform(0.55, 0.75),
                statistical_significance=np.random.uniform(0.01, 0.05)
            )
            adaptations.append(adaptation)
        
        return LearningAnalysis(
            system_id=system_id,
            timestamp=datetime.now(),
            active_models=models,
            best_performing_model=best_model.model_id,
            ensemble_performance=ensemble,
            recent_adaptations=adaptations,
            learning_trajectory=[np.random.uniform(0.5, 2.0) for _ in range(10)],
            feature_importance={
                "technical_indicators": np.random.uniform(0.2, 0.4),
                "sentiment_scores": np.random.uniform(0.15, 0.3),
                "macro_factors": np.random.uniform(0.1, 0.25),
                "flow_metrics": np.random.uniform(0.1, 0.2),
                "volatility_measures": np.random.uniform(0.05, 0.15)
            },
            model_correlations={
                "model_1_vs_model_2": np.random.uniform(0.3, 0.7),
                "model_1_vs_model_3": np.random.uniform(0.2, 0.6),
                "model_2_vs_model_3": np.random.uniform(0.4, 0.8)
            },
            regime_detection={
                "current_regime": "bull_market",
                "regime_confidence": str(np.random.uniform(0.7, 0.95)),
                "expected_regime_duration": f"{np.random.randint(30, 180)} days"
            },
            expected_performance_improvement=np.random.uniform(0.05, 0.20),
            recommended_adaptations=[
                "Add new sentiment features",
                "Retrain with recent data",
                "Adjust position sizing algorithm"
            ],
            next_learning_cycle=datetime.now() + timedelta(days=np.random.randint(7, 30))
        )
