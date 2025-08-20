#!/usr/bin/env python3
"""
Cost Model Learning for Execution Optimization

Implements Almgren-Chriss market impact model with residual slippage learning
using GBDT/QR models for cost prediction and execution optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import asyncio

from common.observability.telemetry import log_event, trace_operation


logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order types for execution"""
    MARKET = "market"
    LIMIT = "limit"
    POV = "pov"  # Percentage of Volume
    PEGGED = "pegged"
    ICEBERG = "iceberg"


class VenueType(str, Enum):
    """Trading venues"""
    PRIMARY = "primary"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    ATS = "ats"


@dataclass
class ExecutionState:
    """Execution state features"""
    symbol: str
    current_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    mid_price: float
    volume_24h: float
    volatility: float
    time_of_day: float  # 0-1 normalized
    day_of_week: int   # 0-6
    timestamp: datetime
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array"""
        return np.array([
            self.current_price,
            self.bid_price,
            self.ask_price,
            self.bid_size,
            self.ask_size,
            self.spread,
            self.mid_price,
            self.volume_24h,
            self.volatility,
            self.time_of_day,
            self.day_of_week
        ])


@dataclass
class ExecutionAction:
    """Execution action"""
    order_type: OrderType
    venue: VenueType
    price: float
    size: float
    urgency: float  # 0-1, higher = more urgent
    timestamp: datetime
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array"""
        return np.array([
            list(OrderType).index(self.order_type),
            list(VenueType).index(self.venue),
            self.price,
            self.size,
            self.urgency
        ])


@dataclass
class ExecutionResult:
    """Execution result with realized costs"""
    action: ExecutionAction
    realized_price: float
    filled_size: float
    slippage: float  # Realized slippage
    market_impact: float
    timing_cost: float
    total_cost: float
    fill_rate: float
    execution_time: float  # seconds
    timestamp: datetime


class AlmgrenChrissModel:
    """
    Almgren-Chriss Market Impact Model
    
    Implements the theoretical market impact model as baseline
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eta = config.get('eta', 0.1)  # Linear impact parameter
        self.gamma = config.get('gamma', 0.1)  # Temporary impact parameter
        self.sigma = config.get('sigma', 0.02)  # Volatility
        self.risk_aversion = config.get('risk_aversion', 1.0)
        
        logger.info("Almgren-Chriss model initialized")
    
    def calculate_optimal_schedule(self, order_size: float, 
                                 time_horizon: float,
                                 current_price: float,
                                 volatility: float) -> Tuple[List[float], float]:
        """
        Calculate optimal execution schedule
        
        Args:
            order_size: Total order size
            time_horizon: Execution time horizon
            current_price: Current market price
            volatility: Asset volatility
            
        Returns:
            (schedule, expected_cost)
        """
        try:
            # Almgren-Chriss parameters
            T = time_horizon  # Time horizon
            X = order_size    # Total order size
            S0 = current_price  # Initial price
            
            # Model parameters
            eta = self.eta * volatility  # Linear impact
            gamma = self.gamma * volatility  # Temporary impact
            sigma = volatility
            
            # Optimal trading rate (constant)
            optimal_rate = X / T
            
            # Expected cost components
            linear_impact = 0.5 * eta * X * optimal_rate
            temporary_impact = gamma * X * optimal_rate
            timing_cost = 0.5 * self.risk_aversion * sigma**2 * X**2 * T / 3
            
            total_cost = linear_impact + temporary_impact + timing_cost
            
            # Generate schedule (constant rate)
            n_steps = max(1, int(T * 60))  # Assume 1-minute steps
            schedule = [optimal_rate] * n_steps
            
            return schedule, total_cost
            
        except Exception as e:
            logger.error(f"Almgren-Chriss calculation failed: {e}")
            return [order_size], order_size * 0.001  # Fallback
    
    def estimate_market_impact(self, order_size: float, 
                             current_price: float,
                             volume_24h: float) -> float:
        """Estimate market impact for order size"""
        try:
            # Normalize order size by daily volume
            volume_ratio = order_size / max(volume_24h, 1e-6)
            
            # Linear impact model
            impact = self.eta * volume_ratio * current_price
            
            return impact
            
        except Exception as e:
            logger.error(f"Market impact estimation failed: {e}")
            return 0.0


class ResidualSlippageModel:
    """
    Residual Slippage Learning Model
    
    Learns residual slippage beyond Almgren-Chriss predictions
    using GBDT/QR models with rich feature engineering
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get('model_type', 'gbdt')  # 'gbdt' or 'qr'
        self.learning_rate = config.get('learning_rate', 0.1)
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.min_samples_leaf = config.get('min_samples_leaf', 10)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Training data
        self.training_data: List[Tuple[np.ndarray, float]] = []
        self.min_training_samples = config.get('min_training_samples', 100)
        self.max_training_samples = config.get('max_training_samples', 10000)
        
        # Performance tracking
        self.prediction_count = 0
        self.avg_error = 0.0
        self.last_retrain = None
        
        logger.info("Residual Slippage Model initialized")
    
    def _create_model(self) -> lgb.LGBMRegressor:
        """Create LightGBM model"""
        if self.model_type == 'qr':
            # Quantile regression for uncertainty
            return lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5,  # Median
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_samples=self.min_samples_leaf,
                random_state=42,
                verbose=-1
            )
        else:
            # Standard GBDT
            return lgb.LGBMRegressor(
                objective='regression',
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_samples=self.min_samples_leaf,
                random_state=42,
                verbose=-1
            )
    
    def _extract_features(self, state: ExecutionState, 
                         action: ExecutionAction,
                         ac_cost: float) -> np.ndarray:
        """Extract features for residual slippage prediction"""
        try:
            # Market state features
            market_features = state.to_array()
            
            # Action features
            action_features = action.to_array()
            
            # Derived features
            derived_features = np.array([
                action.size / max(state.volume_24h, 1e-6),  # Size ratio
                action.size * action.urgency,  # Urgency-weighted size
                state.spread / state.mid_price,  # Normalized spread
                state.volatility * action.urgency,  # Volatility-urgency interaction
                ac_cost / action.size if action.size > 0 else 0,  # AC cost per unit
                state.time_of_day * action.urgency,  # Time-urgency interaction
                state.bid_size / max(state.ask_size, 1e-6),  # Bid-ask imbalance
                action.price / state.mid_price - 1,  # Price deviation
                state.volume_24h * state.volatility,  # Volume-volatility interaction
                action.urgency * state.spread  # Urgency-spread interaction
            ])
            
            # Combine all features
            features = np.concatenate([
                market_features,
                action_features,
                derived_features
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(27)  # Default feature vector
    
    async def add_training_sample(self, state: ExecutionState,
                          action: ExecutionAction,
                          ac_cost: float,
                          realized_slippage: float):
        """Add training sample for online learning"""
        try:
            features = self._extract_features(state, action, ac_cost)
            self.training_data.append((features, realized_slippage))
            
            # Maintain training data size
            if len(self.training_data) > self.max_training_samples:
                self.training_data.pop(0)
            
            # Retrain if enough samples
            if len(self.training_data) >= self.min_training_samples:
                await self._retrain_model()
                
        except Exception as e:
            logger.error(f"Training sample addition failed: {e}")
    
    async def _retrain_model(self):
        """Retrain the model with accumulated data"""
        try:
            if len(self.training_data) < self.min_training_samples:
                return
            
            # Prepare training data
            X = np.array([sample[0] for sample in self.training_data])
            y = np.array([sample[1] for sample in self.training_data])
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Create and train model
            self.model = self._create_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_val_scaled)
            mae = np.mean(np.abs(y_val - y_pred))
            self.avg_error = mae
            
            # Update feature names
            self.feature_names = [
                f"market_{i}" for i in range(11)
            ] + [
                f"action_{i}" for i in range(5)
            ] + [
                "size_ratio", "urgency_size", "norm_spread", "vol_urgency",
                "ac_cost_unit", "time_urgency", "bid_ask_imbalance",
                "price_deviation", "vol_vol_interaction", "urgency_spread"
            ]
            
            self.last_retrain = datetime.utcnow()
            
            logger.info(f"Model retrained with {len(self.training_data)} samples, MAE: {mae:.6f}")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def predict_residual_slippage(self, state: ExecutionState,
                                action: ExecutionAction,
                                ac_cost: float) -> Tuple[float, float]:
        """
        Predict residual slippage
        
        Returns:
            (predicted_residual, confidence)
        """
        try:
            if self.model is None:
                return 0.0, 0.0
            
            # Extract features
            features = self._extract_features(state, action, ac_cost)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict residual
            residual = self.model.predict(features_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = max(0.1, 1.0 - self.avg_error / 0.001)  # Normalize by typical slippage
            
            self.prediction_count += 1
            
            return residual, confidence
            
        except Exception as e:
            logger.error(f"Residual slippage prediction failed: {e}")
            return 0.0, 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        return importance_dict


class CostModelLearner:
    """
    Cost Model Learning System
    
    Combines Almgren-Chriss baseline with residual slippage learning
    for comprehensive cost prediction and execution optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Model components
        self.ac_model = AlmgrenChrissModel(config.get('ac_config', {}))
        self.residual_model = ResidualSlippageModel(config.get('residual_config', {}))
        
        # Cost tracking
        self.cost_history: List[ExecutionResult] = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Performance metrics
        self.total_predictions = 0
        self.avg_prediction_error = 0.0
        self.cost_reduction = 0.0
        self.last_retrain = None
        
        logger.info("Cost Model Learning System initialized")
    
    async def predict_execution_cost(self, state: ExecutionState,
                                   action: ExecutionAction,
                                   order_size: float,
                                   trace_id: str) -> Dict[str, float]:
        """
        Predict execution cost for given state and action
        
        Returns:
            Dictionary with cost components
        """
        async with trace_operation("cost_prediction", trace_id=trace_id):
            try:
                # Almgren-Chriss baseline cost
                ac_schedule, ac_cost = self.ac_model.calculate_optimal_schedule(
                    order_size, 1.0, state.current_price, state.volatility
                )
                
                # Market impact
                market_impact = self.ac_model.estimate_market_impact(
                    order_size, state.current_price, state.volume_24h
                )
                
                # Residual slippage prediction
                residual_slippage, confidence = self.residual_model.predict_residual_slippage(
                    state, action, ac_cost
                )
                
                # Total predicted cost
                total_cost = ac_cost + residual_slippage
                
                # Cost breakdown
                cost_breakdown = {
                    'ac_baseline': ac_cost,
                    'market_impact': market_impact,
                    'residual_slippage': residual_slippage,
                    'total_cost': total_cost,
                    'confidence': confidence,
                    'cost_per_unit': total_cost / max(order_size, 1e-6)
                }
                
                self.total_predictions += 1
                
                return cost_breakdown
                
            except Exception as e:
                logger.error(f"Cost prediction failed: {e}", extra={'trace_id': trace_id})
                return {
                    'ac_baseline': 0.0,
                    'market_impact': 0.0,
                    'residual_slippage': 0.0,
                    'total_cost': 0.0,
                    'confidence': 0.0,
                    'cost_per_unit': 0.0
                }
    
    async def record_execution_result(self, result: ExecutionResult, trace_id: str):
        """Record execution result for learning"""
        try:
            # Add to history
            self.cost_history.append(result)
            
            # Maintain history size
            if len(self.cost_history) > self.max_history_size:
                self.cost_history.pop(0)
            
            # Calculate prediction error
            predicted_cost = result.action.price * result.action.size * 0.001  # Simplified
            actual_cost = result.total_cost
            error = abs(predicted_cost - actual_cost)
            
            # Update average error
            self.avg_prediction_error = (
                (self.avg_prediction_error * (self.total_predictions - 1) + error) / 
                self.total_predictions
            )
            
            # Add training sample to residual model
            # Note: This would need the original state and AC cost
            # For now, use simplified approach
            state = ExecutionState(
                symbol=result.action.order_type,  # Placeholder
                current_price=result.realized_price,
                bid_price=result.realized_price * 0.999,
                ask_price=result.realized_price * 1.001,
                bid_size=1000,
                ask_size=1000,
                spread=result.realized_price * 0.002,
                mid_price=result.realized_price,
                volume_24h=1000000,
                volatility=0.02,
                time_of_day=0.5,
                day_of_week=1,
                timestamp=result.timestamp
            )
            
            ac_cost = result.market_impact  # Simplified
            realized_residual = result.slippage - ac_cost
            
            await self.residual_model.add_training_sample(
                state, result.action, ac_cost, realized_residual
            )
            
            # Log metrics
            await self._log_execution_metrics(result, error, trace_id)
            
        except Exception as e:
            logger.error(f"Execution result recording failed: {e}")
    
    async def optimize_execution(self, state: ExecutionState,
                               order_size: float,
                               urgency: float,
                               trace_id: str) -> ExecutionAction:
        """
        Optimize execution strategy
        
        Returns:
            Optimal execution action
        """
        async with trace_operation("execution_optimization", trace_id=trace_id):
            try:
                # Define action space
                order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.POV]
                venues = [VenueType.PRIMARY, VenueType.DARK_POOL, VenueType.ECN]
                
                best_action = None
                best_cost = float('inf')
                
                # Grid search over action space
                for order_type in order_types:
                    for venue in venues:
                        # Generate action
                        action = self._generate_action(
                            state, order_type, venue, order_size, urgency
                        )
                        
                        # Predict cost
                        cost_breakdown = await self.predict_execution_cost(
                            state, action, order_size, trace_id
                        )
                        
                        # Update best action
                        if cost_breakdown['total_cost'] < best_cost:
                            best_cost = cost_breakdown['total_cost']
                            best_action = action
                
                return best_action or self._generate_default_action(
                    state, order_size, urgency
                )
                
            except Exception as e:
                logger.error(f"Execution optimization failed: {e}", extra={'trace_id': trace_id})
                return self._generate_default_action(state, order_size, urgency)
    
    def _generate_action(self, state: ExecutionState,
                        order_type: OrderType,
                        venue: VenueType,
                        order_size: float,
                        urgency: float) -> ExecutionAction:
        """Generate execution action"""
        # Price calculation
        if order_type == OrderType.MARKET:
            price = state.ask_price if order_size > 0 else state.bid_price
        elif order_type == OrderType.LIMIT:
            price = state.mid_price * (1 + 0.001 * urgency)  # Slight premium
        else:  # POV
            price = state.mid_price
        
        return ExecutionAction(
            order_type=order_type,
            venue=venue,
            price=price,
            size=order_size,
            urgency=urgency,
            timestamp=datetime.utcnow()
        )
    
    def _generate_default_action(self, state: ExecutionState,
                               order_size: float,
                               urgency: float) -> ExecutionAction:
        """Generate default execution action"""
        return ExecutionAction(
            order_type=OrderType.MARKET,
            venue=VenueType.PRIMARY,
            price=state.ask_price if order_size > 0 else state.bid_price,
            size=order_size,
            urgency=urgency,
            timestamp=datetime.utcnow()
        )
    
    async def _log_execution_metrics(self, result: ExecutionResult, 
                                   error: float, trace_id: str):
        """Log execution metrics"""
        try:
            await log_event("execution_result_recorded", {
                "trace_id": trace_id,
                "symbol": result.action.order_type,  # Placeholder
                "order_type": result.action.order_type.value,
                "venue": result.action.venue.value,
                "size": result.action.size,
                "urgency": result.action.urgency,
                "realized_price": result.realized_price,
                "slippage": result.slippage,
                "total_cost": result.total_cost,
                "fill_rate": result.fill_rate,
                "execution_time": result.execution_time,
                "prediction_error": error,
                "avg_prediction_error": self.avg_prediction_error
            })
            
        except Exception as e:
            logger.error(f"Metrics logging failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_predictions": self.total_predictions,
            "avg_prediction_error": self.avg_prediction_error,
            "cost_reduction": self.cost_reduction,
            "training_samples": len(self.residual_model.training_data),
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "feature_importance": self.residual_model.get_feature_importance()
        }
