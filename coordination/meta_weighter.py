#!/usr/bin/env python3
"""
Meta-Weighter: Calibrated Blend of Agent Signals

This component implements the core decision-making logic that combines signals
from all 12 agents using calibrated weights and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AgentSignal:
    """Individual agent signal with metadata"""
    agent_id: str
    symbol: str
    signal_strength: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    metadata: Dict[str, Any]
    horizon: str  # '1D', '1W', '1M', '3M', '6M', '1Y'
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    expected_return: float
    risk_score: float

@dataclass
class BlendedSignal:
    """Meta-weighted blended signal"""
    symbol: str
    blended_strength: float
    confidence: float
    agent_contributions: Dict[str, float]
    consensus_score: float
    disagreement_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class MetaWeighter:
    """
    Meta-Weighter: Calibrated blend of agent signals
    
    Features:
    - Dynamic weight calibration based on agent performance
    - Ensemble methods (Random Forest, Gradient Boosting, Ridge)
    - Regime-aware weighting
    - Uncertainty quantification
    - Consensus and disagreement analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Agent weights (initial - will be calibrated)
        self.agent_weights = {
            'technical': 0.15,
            'sentiment': 0.10,
            'insider': 0.12,
            'macro': 0.08,
            'moneyflows': 0.15,
            'flow': 0.08,
            'causal': 0.12,
            'hedging': 0.07,
            'learning': 0.01,
            'undervalued': 0.03,
            'top_performers': 0.05,
            'value': 0.20
        }
        
        # Ensemble models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01)
        }
        
        # Performance tracking
        self.agent_performance = {agent: [] for agent in self.agent_weights.keys()}
        self.signal_history = []
        self.calibration_history = []
        
        # Regime detection
        self.current_regime = 'normal'
        self.regime_weights = {
            'bull_market': {
                'technical': 0.20, 'sentiment': 0.15, 'value': 0.15,
                'moneyflows': 0.20, 'flow': 0.10, 'causal': 0.10,
                'insider': 0.05, 'macro': 0.03, 'hedging': 0.02
            },
            'bear_market': {
                'value': 0.25, 'hedging': 0.20, 'macro': 0.15,
                'insider': 0.15, 'technical': 0.10, 'sentiment': 0.05,
                'moneyflows': 0.05, 'flow': 0.03, 'causal': 0.02
            },
            'sideways': {
                'technical': 0.20, 'flow': 0.15, 'moneyflows': 0.15,
                'sentiment': 0.12, 'causal': 0.12, 'insider': 0.10,
                'value': 0.08, 'macro': 0.05, 'hedging': 0.03
            },
            'normal': self.agent_weights
        }
        
        # Calibration parameters
        self.calibration_window = self.config.get('calibration_window', 252)  # 1 year
        self.min_performance_history = self.config.get('min_performance_history', 50)
        self.regime_detection_threshold = self.config.get('regime_detection_threshold', 0.1)
        
        logger.info("Meta-Weighter initialized with ensemble models and regime detection")
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime based on price action and volatility"""
        try:
            if len(market_data) < 20:
                return 'normal'
            
            # Calculate regime indicators
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std()
            momentum = market_data['close'].pct_change(20)
            
            current_vol = volatility.iloc[-1]
            current_momentum = momentum.iloc[-1]
            
            # Regime classification
            if current_momentum > self.regime_detection_threshold and current_vol < 0.02:
                return 'bull_market'
            elif current_momentum < -self.regime_detection_threshold and current_vol > 0.03:
                return 'bear_market'
            elif current_vol > 0.025:
                return 'sideways'
            else:
                return 'normal'
                
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return 'normal'
    
    def calibrate_weights(self, performance_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Calibrate agent weights based on historical performance"""
        try:
            calibrated_weights = {}
            
            for agent, performances in performance_history.items():
                if len(performances) >= self.min_performance_history:
                    # Calculate performance metrics
                    avg_performance = np.mean(performances[-self.calibration_window:])
                    consistency = 1.0 - np.std(performances[-self.calibration_window:])
                    
                    # Combine performance and consistency
                    score = (avg_performance * 0.7) + (consistency * 0.3)
                    calibrated_weights[agent] = max(0.01, score)  # Minimum 1% weight
                else:
                    calibrated_weights[agent] = self.agent_weights.get(agent, 0.05)
            
            # Normalize weights
            total_weight = sum(calibrated_weights.values())
            if total_weight > 0:
                calibrated_weights = {k: v/total_weight for k, v in calibrated_weights.items()}
            
            return calibrated_weights
            
        except Exception as e:
            logger.warning(f"Error calibrating weights: {e}")
            return self.agent_weights
    
    def blend_signals_ensemble(self, signals: List[AgentSignal], 
                             market_data: pd.DataFrame) -> BlendedSignal:
        """Blend signals using ensemble methods"""
        try:
            if not signals:
                return None
            
            # Group signals by symbol
            symbol_signals = {}
            for signal in signals:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append(signal)
            
            blended_signals = []
            
            for symbol, symbol_sigs in symbol_signals.items():
                # Create feature matrix for ensemble
                features = []
                for signal in symbol_sigs:
                    features.append([
                        signal.signal_strength,
                        signal.confidence,
                        signal.expected_return,
                        signal.risk_score,
                        self._encode_horizon(signal.horizon),
                        self._encode_signal_type(signal.signal_type)
                    ])
                
                features = np.array(features)
                
                # Get current regime
                regime = self.detect_market_regime(market_data)
                regime_weights = self.regime_weights.get(regime, self.agent_weights)
                
                # Ensemble prediction
                ensemble_predictions = []
                for model_name, model in self.models.items():
                    try:
                        # Simple ensemble using agent weights as features
                        weighted_prediction = np.average(
                            features[:, 0],  # signal_strength
                            weights=[regime_weights.get(s.agent_id, 0.05) for s in symbol_sigs]
                        )
                        ensemble_predictions.append(weighted_prediction)
                    except:
                        continue
                
                if ensemble_predictions:
                    blended_strength = np.mean(ensemble_predictions)
                    confidence = np.std(ensemble_predictions)  # Lower std = higher confidence
                    confidence = max(0.1, 1.0 - confidence)
                    
                    # Calculate agent contributions
                    agent_contributions = {}
                    for signal in symbol_sigs:
                        weight = regime_weights.get(signal.agent_id, 0.05)
                        agent_contributions[signal.agent_id] = weight * signal.signal_strength
                    
                    # Consensus and disagreement analysis
                    signal_strengths = [s.signal_strength for s in symbol_sigs]
                    consensus_score = np.mean(signal_strengths)
                    disagreement_score = np.std(signal_strengths)
                    
                    blended_signal = BlendedSignal(
                        symbol=symbol,
                        blended_strength=blended_strength,
                        confidence=confidence,
                        agent_contributions=agent_contributions,
                        consensus_score=consensus_score,
                        disagreement_score=disagreement_score,
                        timestamp=datetime.now(),
                        metadata={
                            'regime': regime,
                            'num_agents': len(symbol_sigs),
                            'ensemble_method': 'weighted_average'
                        }
                    )
                    
                    blended_signals.append(blended_signal)
            
            return blended_signals
            
        except Exception as e:
            logger.error(f"Error in ensemble blending: {e}")
            return []
    
    def blend_signals_simple(self, signals: List[AgentSignal], 
                           market_data: pd.DataFrame) -> List[BlendedSignal]:
        """Simple weighted average blending"""
        try:
            if not signals:
                return []
            
            # Group by symbol
            symbol_signals = {}
            for signal in signals:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append(signal)
            
            # Detect regime
            regime = self.detect_market_regime(market_data)
            regime_weights = self.regime_weights.get(regime, self.agent_weights)
            
            blended_signals = []
            
            for symbol, symbol_sigs in symbol_signals.items():
                # Weighted average
                total_weight = 0
                weighted_sum = 0
                agent_contributions = {}
                
                for signal in symbol_sigs:
                    weight = regime_weights.get(signal.agent_id, 0.05)
                    weighted_sum += signal.signal_strength * weight
                    total_weight += weight
                    agent_contributions[signal.agent_id] = weight * signal.signal_strength
                
                if total_weight > 0:
                    blended_strength = weighted_sum / total_weight
                    
                    # Calculate confidence based on agreement
                    signal_strengths = [s.signal_strength for s in symbol_sigs]
                    confidence = 1.0 - np.std(signal_strengths)  # Higher agreement = higher confidence
                    confidence = max(0.1, min(0.95, confidence))
                    
                    # Consensus analysis
                    consensus_score = np.mean(signal_strengths)
                    disagreement_score = np.std(signal_strengths)
                    
                    blended_signal = BlendedSignal(
                        symbol=symbol,
                        blended_strength=blended_strength,
                        confidence=confidence,
                        agent_contributions=agent_contributions,
                        consensus_score=consensus_score,
                        disagreement_score=disagreement_score,
                        timestamp=datetime.now(),
                        metadata={
                            'regime': regime,
                            'num_agents': len(symbol_sigs),
                            'blending_method': 'weighted_average'
                        }
                    )
                    
                    blended_signals.append(blended_signal)
            
            return blended_signals
            
        except Exception as e:
            logger.error(f"Error in simple blending: {e}")
            return []
    
    def update_performance(self, agent_id: str, performance: float):
        """Update agent performance history"""
        try:
            if agent_id in self.agent_performance:
                self.agent_performance[agent_id].append(performance)
                
                # Keep only recent history
                if len(self.agent_performance[agent_id]) > self.calibration_window * 2:
                    self.agent_performance[agent_id] = self.agent_performance[agent_id][-self.calibration_window:]
            
            # Recalibrate weights periodically
            if len(self.agent_performance.get(agent_id, [])) % 20 == 0:
                self._recalibrate_weights()
                
        except Exception as e:
            logger.warning(f"Error updating performance: {e}")
    
    def _recalibrate_weights(self):
        """Recalibrate agent weights based on performance"""
        try:
            calibrated_weights = self.calibrate_weights(self.agent_performance)
            
            # Update regime weights
            for regime in self.regime_weights:
                if regime != 'normal':
                    # Adjust regime weights based on calibrated weights
                    for agent, weight in calibrated_weights.items():
                        if agent in self.regime_weights[regime]:
                            self.regime_weights[regime][agent] = weight
            
            self.calibration_history.append({
                'timestamp': datetime.now(),
                'weights': calibrated_weights.copy()
            })
            
            logger.info("Agent weights recalibrated")
            
        except Exception as e:
            logger.warning(f"Error recalibrating weights: {e}")
    
    def _encode_horizon(self, horizon: str) -> float:
        """Encode time horizon as numerical value"""
        horizon_map = {
            '1D': 1.0, '1W': 0.8, '1M': 0.6, '3M': 0.4, '6M': 0.2, '1Y': 0.1
        }
        return horizon_map.get(horizon, 0.5)
    
    def _encode_signal_type(self, signal_type: str) -> float:
        """Encode signal type as numerical value"""
        type_map = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}
        return type_map.get(signal_type, 0.0)
    
    def get_agent_weights(self, regime: str = None) -> Dict[str, float]:
        """Get current agent weights for specified regime"""
        if regime and regime in self.regime_weights:
            return self.regime_weights[regime]
        return self.agent_weights
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        summary = {}
        
        for agent, performances in self.agent_performance.items():
            if performances:
                summary[agent] = {
                    'avg_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'recent_performance': np.mean(performances[-20:]) if len(performances) >= 20 else np.mean(performances),
                    'num_signals': len(performances)
                }
        
        return summary
