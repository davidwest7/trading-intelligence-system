#!/usr/bin/env python3
"""
Phase 1 Implementation: Uncertainty Quantification & Diversity-Aware Selection
Builds on top of our current agent system to add uncertainty and diversity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import cvxpy as cp
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UNCERTAINTY-AWARE AGENT BASE CLASS
# ============================================================================

@dataclass
class UncertaintyAwareOutput:
    """Enhanced agent output with uncertainty quantification"""
    mean_prediction: float
    uncertainty: float
    horizon: int
    confidence: float
    agent_id: str
    timestamp: datetime
    regime_conditional: Optional[Dict[str, float]] = None
    calibration_metrics: Optional[Dict[str, float]] = None

class UncertaintyAwareAgent:
    """Base class for agents that emit (μ, σ, horizon)"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.uncertainty_model = None
        self.horizon_model = None
        self.calibration_model = None
        self.prediction_history = []
        
    def predict_with_uncertainty(self, data: Dict[str, Any]) -> UncertaintyAwareOutput:
        """
        Returns prediction with uncertainty quantification
        """
        # Get base prediction
        mean_pred = self._get_base_prediction(data)
        
        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(data, mean_pred)
        
        # Predict horizon
        horizon = self._predict_horizon(data)
        
        # Calculate confidence
        confidence = self._calculate_confidence(mean_pred, uncertainty)
        
        # Create output
        output = UncertaintyAwareOutput(
            mean_prediction=mean_pred,
            uncertainty=uncertainty,
            horizon=horizon,
            confidence=confidence,
            agent_id=self.agent_id,
            timestamp=datetime.now()
        )
        
        # Store for calibration
        self.prediction_history.append(output)
        
        return output
    
    def _get_base_prediction(self, data: Dict[str, Any]) -> float:
        """Get base prediction from agent logic"""
        # Override in subclasses
        raise NotImplementedError
    
    def _estimate_uncertainty(self, data: Dict[str, Any], prediction: float) -> float:
        """Estimate prediction uncertainty"""
        if self.uncertainty_model is None:
            # Simple heuristic uncertainty based on data quality
            return self._heuristic_uncertainty(data, prediction)
        
        # Use trained uncertainty model
        features = self._extract_uncertainty_features(data)
        return self.uncertainty_model.predict(features)
    
    def _heuristic_uncertainty(self, data: Dict[str, Any], prediction: float) -> float:
        """Simple heuristic uncertainty estimation"""
        # Base uncertainty
        base_uncertainty = 0.1
        
        # Adjust based on data quality
        if 'data_quality' in data:
            base_uncertainty *= (2 - data['data_quality'])  # Higher quality = lower uncertainty
        
        # Adjust based on prediction magnitude
        if abs(prediction) > 0.5:
            base_uncertainty *= 1.5  # Higher predictions = higher uncertainty
        
        return min(base_uncertainty, 0.5)  # Cap at 50%
    
    def _predict_horizon(self, data: Dict[str, Any]) -> int:
        """Predict optimal holding horizon"""
        if self.horizon_model is None:
            # Default horizon based on agent type
            return self._default_horizon()
        
        features = self._extract_horizon_features(data)
        horizon = self.horizon_model.predict(features)
        return max(1, int(horizon))  # Minimum 1 day
    
    def _default_horizon(self) -> int:
        """Default horizon based on agent type"""
        horizon_map = {
            'technical': 5,      # 1 week
            'sentiment': 3,      # 3 days
            'moneyflows': 21,    # 3 weeks
            'undervalued': 252,  # 1 year
            'insider': 63,       # 3 months
            'macro': 126,        # 6 months
            'flow': 7,           # 1 week
            'top_performers': 126, # 6 months
            'causal': 21,        # 3 weeks
            'hedging': 63,       # 3 months
            'learning': 14,      # 2 weeks
            'value': 252         # 1 year
        }
        return horizon_map.get(self.agent_id.split('_')[0], 21)
    
    def _calculate_confidence(self, prediction: float, uncertainty: float) -> float:
        """Calculate confidence score"""
        # Higher confidence for lower uncertainty and higher signal strength
        signal_strength = abs(prediction)
        confidence = signal_strength / (signal_strength + uncertainty)
        return min(confidence, 1.0)
    
    def _extract_uncertainty_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features for uncertainty prediction"""
        # Simple feature extraction - override in subclasses
        features = []
        
        # Data quality features
        features.append(data.get('data_quality', 0.5))
        features.append(data.get('data_freshness', 1.0))
        
        # Market condition features
        features.append(data.get('volatility', 0.2))
        features.append(data.get('volume', 1000000))
        
        return np.array(features).reshape(1, -1)
    
    def _extract_horizon_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features for horizon prediction"""
        # Similar to uncertainty features
        return self._extract_uncertainty_features(data)
    
    def train_uncertainty_model(self, historical_data: List[Tuple[Dict, float, float]]):
        """Train uncertainty model on historical data"""
        X = []
        y = []
        
        for data, prediction, actual in historical_data:
            features = self._extract_uncertainty_features(data)
            error = abs(prediction - actual)
            X.append(features.flatten())
            y.append(error)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train quantile regressor for uncertainty
        self.uncertainty_model = QuantileRegressor(quantile=0.9, alpha=0.1)
        self.uncertainty_model.fit(X, y)
    
    def train_horizon_model(self, historical_data: List[Tuple[Dict, int, float]]):
        """Train horizon model on historical data"""
        X = []
        y = []
        
        for data, horizon, performance in historical_data:
            features = self._extract_horizon_features(data)
            X.append(features.flatten())
            y.append(horizon)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train gradient boosting for horizon prediction
        self.horizon_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.horizon_model.fit(X, y)

# ============================================================================
# DIVERSITY-AWARE SELECTOR
# ============================================================================

class DiversityAwareSelector:
    """Diversified slate bandits with correlation penalties"""
    
    def __init__(self, correlation_window: int = 252, diversity_weight: float = 0.1):
        self.correlation_window = correlation_window
        self.diversity_weight = diversity_weight
        self.correlation_matrix = None
        self.return_history = {}
        self.last_update = None
        
    def select_diversified_slate(self, opportunities: List[UncertaintyAwareOutput], 
                                k: int = 10, current_positions: Dict[str, float] = None) -> List[UncertaintyAwareOutput]:
        """
        Select K opportunities maximizing expected return minus correlation penalty
        """
        if len(opportunities) <= k:
            return opportunities
        
        # Update correlation matrix
        self._update_correlation_matrix(opportunities)
        
        # Calculate expected returns and risks
        expected_returns = []
        risks = []
        opportunity_ids = []
        
        for opp in opportunities:
            expected_returns.append(opp.mean_prediction)
            risks.append(opp.uncertainty)
            opportunity_ids.append(opp.agent_id)
        
        # Solve optimization problem
        selected_indices = self._solve_diversified_selection(
            expected_returns, risks, k, current_positions
        )
        
        return [opportunities[i] for i in selected_indices]
    
    def _update_correlation_matrix(self, opportunities: List[UncertaintyAwareOutput]):
        """Update correlation matrix based on recent returns"""
        # For now, use a simple correlation matrix
        # In practice, this would use actual historical returns
        
        n_agents = len(opportunities)
        if self.correlation_matrix is None:
            self.correlation_matrix = np.eye(n_agents) * 0.1  # Small base correlation
        
        # Update with new correlation estimates
        # This is a simplified version - in practice, you'd use actual return data
        for i, opp1 in enumerate(opportunities):
            for j, opp2 in enumerate(opportunities):
                if i != j:
                    # Simple correlation based on agent types
                    correlation = self._estimate_agent_correlation(opp1, opp2)
                    self.correlation_matrix[i, j] = correlation
                    self.correlation_matrix[j, i] = correlation
    
    def _estimate_agent_correlation(self, opp1: UncertaintyAwareOutput, opp2: UncertaintyAwareOutput) -> float:
        """Estimate correlation between two agents"""
        # Simple correlation estimation based on agent types
        agent1_type = opp1.agent_id.split('_')[0]
        agent2_type = opp2.agent_id.split('_')[0]
        
        # Define correlation matrix between agent types
        type_correlations = {
            ('technical', 'technical'): 0.8,
            ('sentiment', 'sentiment'): 0.7,
            ('moneyflows', 'moneyflows'): 0.6,
            ('technical', 'sentiment'): 0.3,
            ('technical', 'moneyflows'): 0.4,
            ('sentiment', 'moneyflows'): 0.2,
            ('undervalued', 'value'): 0.9,
            ('undervalued', 'technical'): 0.1,
            ('insider', 'sentiment'): 0.4,
            ('macro', 'sentiment'): 0.5,
            ('flow', 'moneyflows'): 0.7,
            ('top_performers', 'technical'): 0.6,
            ('causal', 'sentiment'): 0.5,
            ('hedging', 'macro'): 0.3,
            ('learning', 'technical'): 0.4,
        }
        
        # Get correlation for this pair
        pair = tuple(sorted([agent1_type, agent2_type]))
        return type_correlations.get(pair, 0.1)  # Default low correlation
    
    def _solve_diversified_selection(self, expected_returns: List[float], risks: List[float], 
                                   k: int, current_positions: Dict[str, float] = None) -> List[int]:
        """Solve diversified selection using quadratic programming"""
        
        n_opportunities = len(expected_returns)
        
        # Decision variables: x[i] = 1 if opportunity i is selected
        x = cp.Variable(n_opportunities, boolean=True)
        
        # Objective: maximize expected return - diversity penalty
        expected_return = cp.sum([expected_returns[i] * x[i] for i in range(n_opportunities)])
        
        # Diversity penalty: penalize high correlations
        diversity_penalty = 0
        for i in range(n_opportunities):
            for j in range(i+1, n_opportunities):
                if self.correlation_matrix is not None:
                    correlation = self.correlation_matrix[i, j]
                    diversity_penalty += self.diversity_weight * correlation * x[i] * x[j]
        
        objective = cp.Maximize(expected_return - diversity_penalty)
        
        # Constraints
        constraints = [
            cp.sum(x) == k,  # Select exactly k opportunities
        ]
        
        # Add position constraints if provided
        if current_positions:
            for i, opp_id in enumerate(range(n_opportunities)):
                if opp_id in current_positions:
                    # Don't select if already at max position
                    if current_positions[opp_id] >= 1.0:
                        constraints.append(x[i] == 0)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.GUROBI)  # or cp.CBC for open source
            if problem.status == cp.OPTIMAL:
                selected_indices = [i for i in range(n_opportunities) if x[i].value > 0.5]
                return selected_indices
            else:
                # Fallback to greedy selection
                return self._greedy_diversified_selection(expected_returns, risks, k)
        except:
            # Fallback to greedy selection
            return self._greedy_diversified_selection(expected_returns, risks, k)
    
    def _greedy_diversified_selection(self, expected_returns: List[float], risks: List[float], k: int) -> List[int]:
        """Greedy diversified selection as fallback"""
        n_opportunities = len(expected_returns)
        selected = []
        remaining = list(range(n_opportunities))
        
        for _ in range(k):
            if not remaining:
                break
            
            # Calculate scores for remaining opportunities
            scores = []
            for i in remaining:
                # Expected return
                score = expected_returns[i]
                
                # Diversity penalty
                diversity_penalty = 0
                for j in selected:
                    if self.correlation_matrix is not None:
                        correlation = self.correlation_matrix[i, j]
                        diversity_penalty += self.diversity_weight * correlation
                
                scores.append(score - diversity_penalty)
            
            # Select best remaining opportunity
            best_idx = remaining[np.argmax(scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected

# ============================================================================
# CALIBRATED BLENDER
# ============================================================================

class CalibratedBlender:
    """Calibrates agent outputs with uncertainty quantification"""
    
    def __init__(self):
        self.calibration_models = {}  # Per regime
        self.expected_calibration_error = {}
        self.calibration_history = []
        
    def calibrate_predictions(self, agent_outputs: List[UncertaintyAwareOutput], 
                            regime: str = 'default') -> List[UncertaintyAwareOutput]:
        """Apply calibration to agent predictions"""
        
        calibrated_outputs = []
        
        for output in agent_outputs:
            # Apply calibration if model exists
            if regime in self.calibration_models:
                calibrated_mu, calibrated_sigma = self._apply_calibration(
                    output.mean_prediction, output.uncertainty, regime
                )
            else:
                calibrated_mu = output.mean_prediction
                calibrated_sigma = output.uncertainty
            
            # Create calibrated output
            calibrated_output = UncertaintyAwareOutput(
                mean_prediction=calibrated_mu,
                uncertainty=calibrated_sigma,
                horizon=output.horizon,
                confidence=output.confidence,
                agent_id=output.agent_id,
                timestamp=output.timestamp,
                regime_conditional=output.regime_conditional,
                calibration_metrics={
                    'original_mean': output.mean_prediction,
                    'original_uncertainty': output.uncertainty,
                    'calibration_applied': regime in self.calibration_models
                }
            )
            
            calibrated_outputs.append(calibrated_output)
        
        return calibrated_outputs
    
    def _apply_calibration(self, mean: float, uncertainty: float, regime: str) -> Tuple[float, float]:
        """Apply calibration transformation"""
        # Simple calibration - in practice, use isotonic regression or Platt scaling
        calibration_params = self.calibration_models[regime]
        
        # Apply scaling and shifting
        calibrated_mean = calibration_params['scale'] * mean + calibration_params['shift']
        calibrated_uncertainty = calibration_params['uncertainty_scale'] * uncertainty
        
        return calibrated_mean, calibrated_uncertainty
    
    def update_calibration(self, predictions: List[UncertaintyAwareOutput], 
                          actual_returns: List[float], regime: str = 'default'):
        """Update calibration models with new data"""
        
        if len(predictions) != len(actual_returns):
            return
        
        # Calculate calibration parameters
        predicted_means = [p.mean_prediction for p in predictions]
        predicted_uncertainties = [p.uncertainty for p in predictions]
        
        # Simple linear calibration
        # In practice, use more sophisticated methods like isotonic regression
        
        # Mean calibration
        mean_scale = np.std(actual_returns) / np.std(predicted_means) if np.std(predicted_means) > 0 else 1.0
        mean_shift = np.mean(actual_returns) - mean_scale * np.mean(predicted_means)
        
        # Uncertainty calibration
        uncertainty_scale = np.std(actual_returns) / np.mean(predicted_uncertainties) if np.mean(predicted_uncertainties) > 0 else 1.0
        
        # Store calibration parameters
        self.calibration_models[regime] = {
            'scale': mean_scale,
            'shift': mean_shift,
            'uncertainty_scale': uncertainty_scale
        }
        
        # Calculate calibration error
        calibrated_means = [mean_scale * m + mean_shift for m in predicted_means]
        calibration_error = np.mean([(c - a) ** 2 for c, a in zip(calibrated_means, actual_returns)])
        self.expected_calibration_error[regime] = calibration_error

# ============================================================================
# ENHANCED AGENT IMPLEMENTATIONS
# ============================================================================

class EnhancedTechnicalAgent(UncertaintyAwareAgent):
    """Enhanced technical agent with uncertainty quantification"""
    
    def __init__(self):
        super().__init__("technical_agent")
        
    def _get_base_prediction(self, data: Dict[str, Any]) -> float:
        """Get technical analysis prediction"""
        close_prices = data.get('close_prices', [])
        
        if len(close_prices) < 20:
            return 0.0
        
        # Simple technical analysis
        sma_20 = np.mean(close_prices[-20:])
        current_price = close_prices[-1]
        momentum = (current_price - close_prices[-5]) / close_prices[-5]
        
        # Generate signal
        if current_price > sma_20 and momentum > 0.01:
            return 0.8  # Strong buy
        elif current_price < sma_20 and momentum < -0.01:
            return -0.7  # Strong sell
        else:
            return 0.1  # Weak signal
    
    def _extract_uncertainty_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features for technical uncertainty"""
        close_prices = data.get('close_prices', [])
        
        if len(close_prices) < 20:
            return np.array([0.5, 0.5, 0.2, 1000000]).reshape(1, -1)
        
        # Technical uncertainty features
        volatility = np.std(np.diff(close_prices[-20:])) / np.mean(close_prices[-20:])
        volume = data.get('volume', 1000000)
        data_quality = data.get('data_quality', 0.8)
        trend_strength = abs(np.corrcoef(range(len(close_prices[-20:])), close_prices[-20:])[0, 1])
        
        return np.array([data_quality, trend_strength, volatility, volume]).reshape(1, -1)

class EnhancedSentimentAgent(UncertaintyAwareAgent):
    """Enhanced sentiment agent with uncertainty quantification"""
    
    def __init__(self):
        super().__init__("sentiment_agent")
        
    def _get_base_prediction(self, data: Dict[str, Any]) -> float:
        """Get sentiment analysis prediction"""
        news_texts = data.get('news_texts', [])
        
        if not news_texts:
            return 0.0
        
        # Simple sentiment scoring
        positive_words = ['bullish', 'growth', 'profit', 'gain', 'positive', 'strong']
        negative_words = ['bearish', 'loss', 'decline', 'negative', 'risk', 'weak']
        
        total_sentiment = 0
        for text in news_texts:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            total_sentiment += positive_count - negative_count
        
        avg_sentiment = total_sentiment / len(news_texts)
        
        # Normalize to [-1, 1]
        return np.tanh(avg_sentiment / 2)
    
    def _extract_uncertainty_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features for sentiment uncertainty"""
        news_texts = data.get('news_texts', [])
        
        if not news_texts:
            return np.array([0.5, 0.5, 0.2, 0]).reshape(1, -1)
        
        # Sentiment uncertainty features
        data_quality = data.get('data_quality', 0.8)
        news_count = len(news_texts)
        avg_length = np.mean([len(text) for text in news_texts])
        sentiment_variance = np.var([len(text.split()) for text in news_texts])
        
        return np.array([data_quality, news_count / 10, avg_length / 100, sentiment_variance]).reshape(1, -1)

# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def run_enhanced_trading_system():
    """Example of running the enhanced trading system"""
    
    print("🚀 Enhanced Trading System - Phase 1 Implementation")
    print("=" * 60)
    
    # Initialize components
    technical_agent = EnhancedTechnicalAgent()
    sentiment_agent = EnhancedSentimentAgent()
    diversity_selector = DiversityAwareSelector()
    calibrated_blender = CalibratedBlender()
    
    # Generate test data
    test_data = {
        'close_prices': [100 + i * 0.1 + np.random.randn() * 0.5 for i in range(100)],
        'volume': 1500000,
        'data_quality': 0.9,
        'news_texts': [
            "Company reports strong quarterly growth and bullish outlook",
            "Market shows positive momentum with increasing volume",
            "Analysts predict continued growth in the sector"
        ]
    }
    
    # Get agent predictions with uncertainty
    print("\n📊 Agent Predictions with Uncertainty:")
    technical_output = technical_agent.predict_with_uncertainty(test_data)
    sentiment_output = sentiment_agent.predict_with_uncertainty(test_data)
    
    print(f"Technical Agent:")
    print(f"  Mean Prediction: {technical_output.mean_prediction:.3f}")
    print(f"  Uncertainty: {technical_output.uncertainty:.3f}")
    print(f"  Confidence: {technical_output.confidence:.3f}")
    print(f"  Horizon: {technical_output.horizon} days")
    
    print(f"\nSentiment Agent:")
    print(f"  Mean Prediction: {sentiment_output.mean_prediction:.3f}")
    print(f"  Uncertainty: {sentiment_output.uncertainty:.3f}")
    print(f"  Confidence: {sentiment_output.confidence:.3f}")
    print(f"  Horizon: {sentiment_output.horizon} days")
    
    # Calibrate predictions
    print("\n🔧 Calibrating Predictions:")
    agent_outputs = [technical_output, sentiment_output]
    calibrated_outputs = calibrated_blender.calibrate_predictions(agent_outputs)
    
    for output in calibrated_outputs:
        print(f"{output.agent_id}:")
        print(f"  Original: {output.calibration_metrics['original_mean']:.3f}")
        print(f"  Calibrated: {output.mean_prediction:.3f}")
    
    # Select diverse slate
    print("\n🎯 Diversity-Aware Selection:")
    selected_opportunities = diversity_selector.select_diversified_slate(
        calibrated_outputs, k=2
    )
    
    print(f"Selected {len(selected_opportunities)} opportunities:")
    for opp in selected_opportunities:
        print(f"  {opp.agent_id}: {opp.mean_prediction:.3f} ± {opp.uncertainty:.3f}")
    
    # Portfolio optimization (simplified)
    print("\n💼 Portfolio Optimization:")
    total_weight = 0
    for opp in selected_opportunities:
        # Simple Kelly-style weighting
        weight = opp.mean_prediction / (opp.uncertainty ** 2)
        weight = max(0, min(weight, 0.5))  # Cap at 50%
        total_weight += weight
        print(f"  {opp.agent_id}: {weight:.1%} allocation")
    
    print(f"Total allocation: {total_weight:.1%}")
    
    print("\n✅ Enhanced Trading System Demo Complete!")

if __name__ == "__main__":
    run_enhanced_trading_system()
