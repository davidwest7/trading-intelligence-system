"""
Unified Scoring System for Trading Opportunities

UnifiedScore = w1*Likelihood + w2*ExpectedReturn – w3*Risk + w4*Liquidity + w5*Conviction + w6*Recency + w7*RegimeFit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import yaml


@dataclass
class RawSignals:
    """Raw signals from agents"""
    likelihood: float  # 0-1
    expected_return: float  # Can be negative
    risk: float  # >= 0, higher is riskier
    liquidity: float = 1.0  # 0-1, higher is more liquid
    conviction: float = 0.5  # 0-1, agent's confidence
    recency: float = 1.0  # 0-1, how fresh the signal is
    regime_fit: float = 0.5  # 0-1, how well it fits current regime


@dataclass
class ScoringWeights:
    """Weights for unified scoring"""
    likelihood: float = 0.25
    expected_return: float = 0.20
    risk: float = 0.20  # Note: This is subtracted
    liquidity: float = 0.10
    conviction: float = 0.10
    recency: float = 0.10
    regime_fit: float = 0.05
    
    def normalize(self):
        """Ensure weights sum to 1.0 (excluding risk which is subtracted)"""
        total = (self.likelihood + self.expected_return + self.liquidity + 
                self.conviction + self.recency + self.regime_fit)
        if total > 0:
            scale = 1.0 / total
            self.likelihood *= scale
            self.expected_return *= scale
            self.liquidity *= scale
            self.conviction *= scale
            self.recency *= scale
            self.regime_fit *= scale


@dataclass
class ComponentScores:
    """Individual component scores after normalization"""
    likelihood_score: float
    return_score: float
    risk_score: float
    liquidity_score: float
    conviction_score: float
    recency_score: float
    regime_score: float


@dataclass
class ScoredOpportunity:
    """Opportunity with unified scoring"""
    id: str
    symbol: str
    strategy: str
    raw_signals: RawSignals
    component_scores: ComponentScores
    unified_score: float
    calibrated_probability: float
    rank: int
    percentile_rank: float
    confidence_interval: Tuple[float, float]


class UnifiedScorer:
    """
    Unified scoring system for trading opportunities
    
    Features:
    - Multi-component scoring with configurable weights
    - Isotonic/Platt calibration for probability estimates
    - Regime-aware scoring adjustments
    - Cross-sectional ranking
    - Diversification penalties
    
    TODO Items:
    1. Implement advanced calibration methods
    2. Add regime-dependent weight adjustments
    3. Implement online score recalibration
    4. Add factor exposure penalties
    5. Implement correlation-based diversification
    6. Add dynamic weight learning
    7. Implement score explanation/attribution
    8. Add A/B testing framework for scoring
    9. Implement score decay functions
    10. Add cross-validation for calibration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.weights_by_asset_class = self._load_default_weights()
        self.calibrator = None
        self.score_history = []
        
        if config_path:
            self._load_config(config_path)
        
    def _load_default_weights(self) -> Dict[str, ScoringWeights]:
        """Load default weights by asset class"""
        return {
            "equities": ScoringWeights(
                likelihood=0.25, expected_return=0.20, risk=0.20,
                liquidity=0.10, conviction=0.10, recency=0.10, regime_fit=0.05
            ),
            "fx": ScoringWeights(
                likelihood=0.30, expected_return=0.15, risk=0.25,
                liquidity=0.15, conviction=0.10, recency=0.05, regime_fit=0.0
            ),
            "crypto": ScoringWeights(
                likelihood=0.20, expected_return=0.25, risk=0.30,
                liquidity=0.05, conviction=0.15, recency=0.05, regime_fit=0.0
            ),
            "bonds": ScoringWeights(
                likelihood=0.35, expected_return=0.15, risk=0.15,
                liquidity=0.15, conviction=0.10, recency=0.05, regime_fit=0.05
            ),
            "commodities": ScoringWeights(
                likelihood=0.25, expected_return=0.20, risk=0.25,
                liquidity=0.10, conviction=0.10, recency=0.05, regime_fit=0.05
            )
        }
    
    def score_opportunities(self, opportunities: List[Dict[str, Any]], 
                          scoring_config: Dict[str, Any] = None) -> List[ScoredOpportunity]:
        """
        Score a batch of opportunities
        
        Args:
            opportunities: List of opportunity dictionaries
            scoring_config: Configuration for scoring
            
        Returns:
            List of scored opportunities ranked by score
        """
        if not opportunities:
            return []
            
        config = scoring_config or {}
        calibration_method = config.get('calibration_method', 'isotonic')
        regime_aware = config.get('regime_aware', True)
        diversification_penalty = config.get('diversification_penalty', 0.1)
        
        scored_opportunities = []
        
        for opp in opportunities:
            # Extract raw signals
            raw_signals = self._extract_raw_signals(opp)
            
            # Determine asset class and get weights
            asset_class = opp.get('metadata', {}).get('asset_class', 'equities')
            weights = self._get_weights_for_asset_class(asset_class, config)
            
            # Calculate component scores
            component_scores = self._calculate_component_scores(raw_signals, weights)
            
            # Calculate unified score
            unified_score = self._calculate_unified_score(component_scores, weights)
            
            # Apply regime adjustments if enabled
            if regime_aware:
                unified_score = self._apply_regime_adjustment(unified_score, raw_signals.regime_fit)
            
            # Create scored opportunity
            scored_opp = ScoredOpportunity(
                id=opp['id'],
                symbol=opp['symbol'],
                strategy=opp['strategy'],
                raw_signals=raw_signals,
                component_scores=component_scores,
                unified_score=unified_score,
                calibrated_probability=0.0,  # Will be set later
                rank=0,  # Will be set after ranking
                percentile_rank=0.0,
                confidence_interval=(0.0, 0.0)
            )
            
            scored_opportunities.append(scored_opp)
        
        # Rank opportunities
        scored_opportunities.sort(key=lambda x: x.unified_score, reverse=True)
        for i, opp in enumerate(scored_opportunities):
            opp.rank = i + 1
            opp.percentile_rank = (len(scored_opportunities) - i) / len(scored_opportunities)
        
        # Apply calibration
        if calibration_method != 'none':
            self._apply_calibration(scored_opportunities, calibration_method)
        
        # Apply diversification penalty
        if diversification_penalty > 0:
            self._apply_diversification_penalty(scored_opportunities, diversification_penalty)
        
        return scored_opportunities
    
    def _extract_raw_signals(self, opportunity: Dict[str, Any]) -> RawSignals:
        """Extract raw signals from opportunity dictionary"""
        signals_data = opportunity.get('raw_signals', {})
        
        return RawSignals(
            likelihood=signals_data.get('likelihood', 0.5),
            expected_return=signals_data.get('expected_return', 0.0),
            risk=signals_data.get('risk', 0.1),
            liquidity=signals_data.get('liquidity', 1.0),
            conviction=signals_data.get('conviction', 0.5),
            recency=signals_data.get('recency', 1.0),
            regime_fit=signals_data.get('regime_fit', 0.5)
        )
    
    def _get_weights_for_asset_class(self, asset_class: str, 
                                   config: Dict[str, Any]) -> ScoringWeights:
        """Get scoring weights for asset class"""
        # Check for config override
        if 'weights' in config:
            custom_weights = ScoringWeights(**config['weights'])
            custom_weights.normalize()
            return custom_weights
        
        # Use default weights for asset class
        default_weights = self.weights_by_asset_class.get(asset_class, 
                                                         self.weights_by_asset_class['equities'])
        return default_weights
    
    def _calculate_component_scores(self, raw_signals: RawSignals, 
                                  weights: ScoringWeights) -> ComponentScores:
        """Calculate normalized component scores"""
        
        # Normalize expected return to 0-1 scale (assuming returns are in -50% to +50% range)
        return_score = np.clip((raw_signals.expected_return + 0.5) / 1.0, 0, 1)
        
        # Risk score (higher risk = lower score)
        risk_score = np.clip(1.0 - raw_signals.risk, 0, 1)
        
        return ComponentScores(
            likelihood_score=raw_signals.likelihood,
            return_score=return_score,
            risk_score=risk_score,
            liquidity_score=raw_signals.liquidity,
            conviction_score=raw_signals.conviction,
            recency_score=raw_signals.recency,
            regime_score=raw_signals.regime_fit
        )
    
    def _calculate_unified_score(self, component_scores: ComponentScores, 
                               weights: ScoringWeights) -> float:
        """Calculate unified score using weighted components"""
        
        score = (
            weights.likelihood * component_scores.likelihood_score +
            weights.expected_return * component_scores.return_score +
            weights.liquidity * component_scores.liquidity_score +
            weights.conviction * component_scores.conviction_score +
            weights.recency * component_scores.recency_score +
            weights.regime_fit * component_scores.regime_score -
            weights.risk * (1.0 - component_scores.risk_score)  # Subtract risk
        )
        
        return np.clip(score, 0, 1)
    
    def _apply_regime_adjustment(self, base_score: float, regime_fit: float) -> float:
        """Apply regime-aware adjustments to score"""
        # Boost scores for opportunities that fit current regime
        regime_boost = 0.1 * (regime_fit - 0.5)  # ±5% adjustment
        adjusted_score = base_score + regime_boost
        return np.clip(adjusted_score, 0, 1)
    
    def _apply_calibration(self, opportunities: List[ScoredOpportunity], 
                         method: str):
        """Apply score calibration to convert scores to probabilities"""
        
        scores = [opp.unified_score for opp in opportunities]
        
        if method == 'isotonic':
            # Simple isotonic calibration (would need historical outcomes for training)
            calibrated_probs = self._isotonic_calibration(scores)
        elif method == 'platt':
            # Platt scaling (would need historical outcomes for training)
            calibrated_probs = self._platt_calibration(scores)
        else:
            # No calibration, just use raw scores
            calibrated_probs = scores
        
        for i, opp in enumerate(opportunities):
            opp.calibrated_probability = calibrated_probs[i]
            # Simple confidence intervals (would be better with proper uncertainty estimation)
            opp.confidence_interval = (
                max(0, calibrated_probs[i] - 0.1),
                min(1, calibrated_probs[i] + 0.1)
            )
    
    def _isotonic_calibration(self, scores: List[float]) -> List[float]:
        """Apply isotonic calibration (simplified)"""
        # TODO: Implement proper calibration with historical outcomes
        # For now, just apply a simple transformation
        scores_array = np.array(scores)
        # Apply sigmoid-like transformation
        calibrated = 1 / (1 + np.exp(-5 * (scores_array - 0.5)))
        return calibrated.tolist()
    
    def _platt_calibration(self, scores: List[float]) -> List[float]:
        """Apply Platt scaling (simplified)"""
        # TODO: Implement proper Platt scaling with historical outcomes
        # For now, just return normalized scores
        scores_array = np.array(scores)
        if scores_array.std() > 0:
            normalized = (scores_array - scores_array.mean()) / scores_array.std()
            calibrated = 1 / (1 + np.exp(-normalized))
        else:
            calibrated = scores_array
        return calibrated.tolist()
    
    def _apply_diversification_penalty(self, opportunities: List[ScoredOpportunity],
                                     penalty_weight: float):
        """Apply diversification penalty for correlated opportunities"""
        # TODO: Implement proper correlation-based diversification
        # For now, apply simple sector/symbol diversification
        
        symbol_counts = {}
        strategy_counts = {}
        
        for opp in opportunities:
            symbol_counts[opp.symbol] = symbol_counts.get(opp.symbol, 0) + 1
            strategy_counts[opp.strategy] = strategy_counts.get(opp.strategy, 0) + 1
        
        for opp in opportunities:
            # Penalize if multiple opportunities on same symbol or strategy
            symbol_penalty = (symbol_counts[opp.symbol] - 1) * penalty_weight * 0.1
            strategy_penalty = (strategy_counts[opp.strategy] - 1) * penalty_weight * 0.05
            
            total_penalty = symbol_penalty + strategy_penalty
            opp.unified_score = max(0, opp.unified_score - total_penalty)
    
    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Update weights if provided
            if 'weights_by_asset_class' in config:
                for asset_class, weights_dict in config['weights_by_asset_class'].items():
                    weights = ScoringWeights(**weights_dict)
                    weights.normalize()
                    self.weights_by_asset_class[asset_class] = weights
                    
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_default_config(self, config_path: str):
        """Save default configuration to YAML file"""
        config = {
            'weights_by_asset_class': {}
        }
        
        for asset_class, weights in self.weights_by_asset_class.items():
            config['weights_by_asset_class'][asset_class] = {
                'likelihood': weights.likelihood,
                'expected_return': weights.expected_return,
                'risk': weights.risk,
                'liquidity': weights.liquidity,
                'conviction': weights.conviction,
                'recency': weights.recency,
                'regime_fit': weights.regime_fit
            }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


# Example usage
if __name__ == "__main__":
    # Create sample opportunities
    opportunities = [
        {
            'id': 'opp_1',
            'symbol': 'AAPL',
            'strategy': 'technical',
            'raw_signals': {
                'likelihood': 0.8,
                'expected_return': 0.05,
                'risk': 0.02,
                'liquidity': 0.95,
                'conviction': 0.7,
                'recency': 0.9,
                'regime_fit': 0.6
            },
            'metadata': {'asset_class': 'equities'}
        },
        {
            'id': 'opp_2',
            'symbol': 'EURUSD',
            'strategy': 'sentiment',
            'raw_signals': {
                'likelihood': 0.6,
                'expected_return': 0.02,
                'risk': 0.015,
                'liquidity': 1.0,
                'conviction': 0.5,
                'recency': 1.0,
                'regime_fit': 0.8
            },
            'metadata': {'asset_class': 'fx'}
        }
    ]
    
    # Score opportunities
    scorer = UnifiedScorer()
    scored_opportunities = scorer.score_opportunities(opportunities)
    
    # Print results
    for opp in scored_opportunities:
        print(f"Rank {opp.rank}: {opp.symbol} - Score: {opp.unified_score:.3f}, "
              f"Prob: {opp.calibrated_probability:.3f}")
        
    # Save default config
    scorer.save_default_config('/Users/davidwestera/trading-intelligence-system/config/scoring_weights.yaml')
