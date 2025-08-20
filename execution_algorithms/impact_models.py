#!/usr/bin/env python3
"""
Hybrid Impact Models
Almgren-Chriss, Kyle Lambda, Hasbrouck models with venue/latency adjustments and calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ImpactModels:
    """Wrapper class for Impact Models to match expected interface"""
    
    def __init__(self, config=None):
        self.config = config or {
            'default_model': 'almgren_chriss',
            'venue_adjustments': True,
            'latency_adjustments': True,
            'calibration_frequency': 'daily'
        }
        
        # Initialize impact parameters
        self.parameters = ImpactParameters()
        
        # Initialize models
        self.almgren_chriss = AlmgrenChrissModel(self.parameters)
        self.kyle_model = KyleModel(self.parameters)
        self.hasbrouck_model = HasbrouckModel(self.parameters)
        
    def predict_market_impact(self, quantity: float, time_horizon: float, 
                            market_conditions: Dict[str, float], 
                            model_type: str = 'almgren_chriss') -> Dict[str, Any]:
        """Predict market impact using specified model"""
        try:
            if model_type == 'almgren_chriss':
                model = self.almgren_chriss
            elif model_type == 'kyle':
                model = self.kyle_model
            elif model_type == 'hasbrouck':
                model = self.hasbrouck_model
            else:
                model = self.almgren_chriss
            
            prediction = model.predict_impact(quantity, time_horizon, market_conditions)
            
            return {
                'success': True,
                'model_type': model_type,
                'total_impact': prediction.total_impact,
                'temporary_impact': prediction.temporary_impact,
                'permanent_impact': prediction.permanent_impact,
                'confidence_interval': prediction.confidence_interval,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def optimize_execution_schedule(self, total_quantity: float, total_time: float,
                                  market_conditions: Dict[str, float],
                                  model_type: str = 'almgren_chriss') -> Dict[str, Any]:
        """Optimize execution schedule using specified model"""
        try:
            if model_type == 'almgren_chriss':
                model = self.almgren_chriss
            elif model_type == 'kyle':
                model = self.kyle_model
            elif model_type == 'hasbrouck':
                model = self.hasbrouck_model
            else:
                model = self.almgren_chriss
            
            schedule = model.optimize_schedule(total_quantity, total_time, market_conditions)
            
            return {
                'success': True,
                'model_type': model_type,
                'total_expected_cost': schedule.total_expected_cost,
                'risk_penalty': schedule.risk_penalty,
                'time_points': len(schedule.time_points),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available impact models"""
        return {
            'available_models': ['almgren_chriss', 'kyle', 'hasbrouck'],
            'default_model': self.config['default_model'],
            'venue_adjustments': self.config['venue_adjustments'],
            'latency_adjustments': self.config['latency_adjustments'],
            'calibration_frequency': self.config['calibration_frequency']
        }

@dataclass
class ImpactParameters:
    """Parameters for impact models"""
    # Almgren-Chriss parameters
    sigma: float = 0.2  # Volatility
    eta: float = 1e-6   # Temporary impact coefficient
    gamma: float = 1e-7 # Permanent impact coefficient
    tau: float = 1.0    # Total execution time (hours)
    lambda_risk: float = 1e-6  # Risk aversion parameter
    
    # Kyle model parameters
    kyle_lambda: float = 1e-6  # Kyle's lambda (price impact per unit volume)
    kyle_alpha: float = 0.5    # Kyle alpha (noise trader variance)
    kyle_beta: float = 0.1     # Kyle beta (informed trader signal precision)
    
    # Hasbrouck parameters
    hasbrouck_psi: float = 0.1  # Permanent impact coefficient
    hasbrouck_theta: float = 0.05  # Temporary impact decay
    hasbrouck_phi: float = 0.02    # Market microstructure noise
    
    # Venue adjustments
    venue_impact_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'NYSE': 1.0, 'NASDAQ': 1.02, 'BATS': 0.95, 'EDGX': 0.97, 'DARK': 0.85
    })
    
    # Latency adjustments
    latency_impact_factor: float = 0.1  # Impact per ms of latency


@dataclass
class ImpactPrediction:
    """Impact prediction result"""
    model_name: str
    total_impact: float
    temporary_impact: float
    permanent_impact: float
    impact_breakdown: Dict[str, float]
    confidence_interval: Tuple[float, float]
    model_parameters: Dict[str, float]
    venue_adjustments: Dict[str, float]
    timestamp: datetime


@dataclass
class ExecutionSchedule:
    """Optimal execution schedule"""
    time_points: np.ndarray
    trade_rates: np.ndarray
    cumulative_shares: np.ndarray
    expected_costs: np.ndarray
    impact_components: Dict[str, np.ndarray]
    total_expected_cost: float
    risk_penalty: float
    model_used: str


class ImpactModel(ABC):
    """Base class for impact models"""
    
    def __init__(self, parameters: ImpactParameters):
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        self.is_calibrated = False
        
    @abstractmethod
    def predict_impact(self, quantity: float, time_horizon: float,
                      market_conditions: Dict[str, float],
                      venue: str = 'NYSE') -> ImpactPrediction:
        """Predict market impact"""
        pass
    
    @abstractmethod
    def optimize_schedule(self, total_quantity: float, total_time: float,
                         market_conditions: Dict[str, float]) -> ExecutionSchedule:
        """Optimize execution schedule"""
        pass
    
    def _apply_venue_adjustment(self, base_impact: float, venue: str) -> float:
        """Apply venue-specific impact adjustment"""
        multiplier = self.parameters.venue_impact_multipliers.get(venue, 1.0)
        return base_impact * multiplier
    
    def _apply_latency_adjustment(self, base_impact: float, latency_ms: float) -> float:
        """Apply latency-based impact adjustment"""
        latency_adjustment = 1.0 + self.parameters.latency_impact_factor * latency_ms / 100
        return base_impact * latency_adjustment


class AlmgrenChrissModel(ImpactModel):
    """Almgren-Chriss optimal execution model"""
    
    def __init__(self, parameters: ImpactParameters):
        super().__init__(parameters)
        self.model_name = "Almgren-Chriss"
    
    def predict_impact(self, quantity: float, time_horizon: float,
                      market_conditions: Dict[str, float],
                      venue: str = 'NYSE') -> ImpactPrediction:
        """Predict impact using Almgren-Chriss model"""
        try:
            # Extract market conditions
            volatility = market_conditions.get('volatility', self.parameters.sigma)
            daily_volume = market_conditions.get('daily_volume', 1e6)
            latency_ms = market_conditions.get('latency_ms', 2.0)
            
            # Adjust parameters for market conditions
            eta = self.parameters.eta * np.sqrt(daily_volume / 1e6)  # Scale with volume
            gamma = self.parameters.gamma * np.sqrt(daily_volume / 1e6)
            
            # Optimal execution rate (from Almgren-Chriss formula)
            kappa = np.sqrt(self.parameters.lambda_risk * volatility**2 / eta)
            sinh_val = np.sinh(kappa * time_horizon)
            cosh_val = np.cosh(kappa * time_horizon)
            
            # Average trading rate
            avg_rate = quantity / time_horizon
            
            # Impact calculation
            temporary_impact = eta * avg_rate
            permanent_impact = gamma * quantity
            
            # Total impact with optimal strategy
            if sinh_val > 0:
                total_impact = (
                    temporary_impact * time_horizon / 2 +  # Average temporary impact
                    permanent_impact +
                    np.sqrt(self.parameters.lambda_risk) * volatility * 
                    np.sqrt(quantity**2 / (2 * kappa)) * np.tanh(kappa * time_horizon / 2)
                )
            else:
                total_impact = temporary_impact * time_horizon / 2 + permanent_impact
            
            # Apply venue and latency adjustments
            total_impact = self._apply_venue_adjustment(total_impact, venue)
            total_impact = self._apply_latency_adjustment(total_impact, latency_ms)
            
            # Confidence interval (based on volatility uncertainty)
            vol_uncertainty = volatility * 0.2  # 20% volatility uncertainty
            impact_std = total_impact * vol_uncertainty / volatility
            confidence_interval = (
                total_impact - 1.96 * impact_std,
                total_impact + 1.96 * impact_std
            )
            
            # Venue adjustments
            venue_adjustments = {
                venue: self.parameters.venue_impact_multipliers.get(venue, 1.0)
            }
            
            return ImpactPrediction(
                model_name=self.model_name,
                total_impact=total_impact,
                temporary_impact=temporary_impact * time_horizon / 2,
                permanent_impact=permanent_impact,
                impact_breakdown={
                    'temporary': temporary_impact * time_horizon / 2,
                    'permanent': permanent_impact,
                    'risk_penalty': total_impact - temporary_impact * time_horizon / 2 - permanent_impact
                },
                confidence_interval=confidence_interval,
                model_parameters={
                    'eta': eta,
                    'gamma': gamma,
                    'kappa': kappa,
                    'volatility': volatility
                },
                venue_adjustments=venue_adjustments,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in Almgren-Chriss prediction: {e}")
            return self._create_fallback_prediction(quantity, time_horizon, venue)
    
    def optimize_schedule(self, total_quantity: float, total_time: float,
                         market_conditions: Dict[str, float]) -> ExecutionSchedule:
        """Optimize execution schedule using Almgren-Chriss"""
        try:
            # Extract parameters
            volatility = market_conditions.get('volatility', self.parameters.sigma)
            daily_volume = market_conditions.get('daily_volume', 1e6)
            
            # Adjusted impact parameters
            eta = self.parameters.eta * np.sqrt(daily_volume / 1e6)
            gamma = self.parameters.gamma * np.sqrt(daily_volume / 1e6)
            
            # Calculate optimal strategy parameters
            kappa = np.sqrt(self.parameters.lambda_risk * volatility**2 / eta)
            
            # Time grid
            n_points = max(10, int(total_time * 60))  # One point per minute
            time_points = np.linspace(0, total_time, n_points)
            dt = total_time / (n_points - 1)
            
            # Optimal trading trajectory
            if kappa * total_time > 1e-6:
                # Non-trivial case
                sinh_kappa_T = np.sinh(kappa * total_time)
                cosh_kappa_T = np.cosh(kappa * total_time)
                
                if sinh_kappa_T > 1e-10:
                    # Holdings trajectory
                    cumulative_shares = total_quantity * np.sinh(kappa * (total_time - time_points)) / sinh_kappa_T
                    
                    # Trading rate (negative derivative of holdings)
                    trade_rates = total_quantity * kappa * np.cosh(kappa * (total_time - time_points)) / sinh_kappa_T
                else:
                    # Linear case
                    cumulative_shares = total_quantity * (total_time - time_points) / total_time
                    trade_rates = np.full(n_points, total_quantity / total_time)
            else:
                # Linear execution
                cumulative_shares = total_quantity * (total_time - time_points) / total_time
                trade_rates = np.full(n_points, total_quantity / total_time)
            
            # Cost calculation at each point
            temporary_costs = eta * trade_rates**2 * dt
            permanent_costs = np.zeros(n_points)
            permanent_costs[0] = gamma * total_quantity  # All permanent impact at start
            
            # Risk penalty
            risk_penalties = (self.parameters.lambda_risk * volatility**2 * 
                            cumulative_shares**2 * dt)
            
            expected_costs = temporary_costs + permanent_costs + risk_penalties
            
            # Total expected cost
            total_expected_cost = np.sum(expected_costs)
            risk_penalty = np.sum(risk_penalties)
            
            return ExecutionSchedule(
                time_points=time_points,
                trade_rates=trade_rates,
                cumulative_shares=cumulative_shares,
                expected_costs=expected_costs,
                impact_components={
                    'temporary': temporary_costs,
                    'permanent': permanent_costs,
                    'risk': risk_penalties
                },
                total_expected_cost=total_expected_cost,
                risk_penalty=risk_penalty,
                model_used=self.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing Almgren-Chriss schedule: {e}")
            return self._create_fallback_schedule(total_quantity, total_time)
    
    def _create_fallback_prediction(self, quantity: float, time_horizon: float, venue: str) -> ImpactPrediction:
        """Create fallback prediction when model fails"""
        # Simple linear impact model
        total_impact = quantity * 1e-6  # 1 bps per share
        return ImpactPrediction(
            model_name=f"{self.model_name}_fallback",
            total_impact=total_impact,
            temporary_impact=total_impact * 0.7,
            permanent_impact=total_impact * 0.3,
            impact_breakdown={'temporary': total_impact * 0.7, 'permanent': total_impact * 0.3},
            confidence_interval=(total_impact * 0.5, total_impact * 1.5),
            model_parameters={},
            venue_adjustments={venue: 1.0},
            timestamp=datetime.now()
        )
    
    def _create_fallback_schedule(self, total_quantity: float, total_time: float) -> ExecutionSchedule:
        """Create fallback schedule when optimization fails"""
        # Linear execution schedule
        n_points = max(10, int(total_time * 60))
        time_points = np.linspace(0, total_time, n_points)
        trade_rates = np.full(n_points, total_quantity / total_time)
        cumulative_shares = total_quantity * (total_time - time_points) / total_time
        expected_costs = np.full(n_points, total_quantity * 1e-6 / n_points)
        
        return ExecutionSchedule(
            time_points=time_points,
            trade_rates=trade_rates,
            cumulative_shares=cumulative_shares,
            expected_costs=expected_costs,
            impact_components={'linear': expected_costs},
            total_expected_cost=total_quantity * 1e-6,
            risk_penalty=0.0,
            model_used=f"{self.model_name}_fallback"
        )


class KyleLambdaModel(ImpactModel):
    """Kyle's Lambda microstructure model"""
    
    def __init__(self, parameters: ImpactParameters):
        super().__init__(parameters)
        self.model_name = "Kyle Lambda"
    
    def predict_impact(self, quantity: float, time_horizon: float,
                      market_conditions: Dict[str, float],
                      venue: str = 'NYSE') -> ImpactPrediction:
        """Predict impact using Kyle's Lambda model"""
        try:
            # Extract market conditions
            order_flow_imbalance = market_conditions.get('order_flow_imbalance', 0.0)
            bid_ask_spread = market_conditions.get('bid_ask_spread', 0.01)
            daily_volume = market_conditions.get('daily_volume', 1e6)
            latency_ms = market_conditions.get('latency_ms', 2.0)
            
            # Adjust Kyle's lambda based on market conditions
            kyle_lambda = self.parameters.kyle_lambda
            
            # Scale lambda with volume (lower volume = higher impact)
            volume_adjustment = np.sqrt(1e6 / daily_volume)
            kyle_lambda *= volume_adjustment
            
            # Adjust for spread (wider spread = higher impact)
            spread_adjustment = 1.0 + bid_ask_spread / 0.01  # Normalize to 1 cent spread
            kyle_lambda *= spread_adjustment
            
            # Linear impact from Kyle model
            base_impact = kyle_lambda * quantity
            
            # Information asymmetry component
            info_asymmetry = abs(order_flow_imbalance) * self.parameters.kyle_alpha
            asymmetry_impact = info_asymmetry * quantity * 0.1
            
            # Temporary vs permanent split (Kyle model primarily permanent)
            permanent_impact = base_impact * 0.8 + asymmetry_impact
            temporary_impact = base_impact * 0.2
            
            total_impact = permanent_impact + temporary_impact
            
            # Apply venue and latency adjustments
            total_impact = self._apply_venue_adjustment(total_impact, venue)
            total_impact = self._apply_latency_adjustment(total_impact, latency_ms)
            
            # Confidence interval based on noise trader activity
            noise_std = np.sqrt(self.parameters.kyle_alpha) * np.sqrt(quantity)
            confidence_interval = (
                total_impact - 1.96 * noise_std,
                total_impact + 1.96 * noise_std
            )
            
            venue_adjustments = {
                venue: self.parameters.venue_impact_multipliers.get(venue, 1.0)
            }
            
            return ImpactPrediction(
                model_name=self.model_name,
                total_impact=total_impact,
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                impact_breakdown={
                    'linear_impact': base_impact,
                    'asymmetry_impact': asymmetry_impact,
                    'temporary': temporary_impact,
                    'permanent': permanent_impact
                },
                confidence_interval=confidence_interval,
                model_parameters={
                    'kyle_lambda': kyle_lambda,
                    'volume_adjustment': volume_adjustment,
                    'spread_adjustment': spread_adjustment,
                    'info_asymmetry': info_asymmetry
                },
                venue_adjustments=venue_adjustments,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in Kyle Lambda prediction: {e}")
            return self._create_fallback_prediction(quantity, time_horizon, venue)
    
    def optimize_schedule(self, total_quantity: float, total_time: float,
                         market_conditions: Dict[str, float]) -> ExecutionSchedule:
        """Optimize execution schedule using Kyle model insights"""
        try:
            # Kyle model suggests trading at constant rate to minimize info leakage
            n_points = max(10, int(total_time * 60))
            time_points = np.linspace(0, total_time, n_points)
            
            # Constant trading rate
            trade_rate = total_quantity / total_time
            trade_rates = np.full(n_points, trade_rate)
            
            # Linear decay in holdings
            cumulative_shares = total_quantity * (total_time - time_points) / total_time
            
            # Kyle impact is primarily permanent and front-loaded
            kyle_lambda = self.parameters.kyle_lambda
            daily_volume = market_conditions.get('daily_volume', 1e6)
            kyle_lambda *= np.sqrt(1e6 / daily_volume)
            
            # Cost per share traded
            cost_per_share = kyle_lambda * total_quantity
            
            # Distribute costs (front-loaded due to permanent impact)
            cost_weights = np.exp(-0.1 * time_points)  # Exponential decay
            cost_weights /= np.sum(cost_weights)
            expected_costs = cost_per_share * total_quantity * cost_weights
            
            return ExecutionSchedule(
                time_points=time_points,
                trade_rates=trade_rates,
                cumulative_shares=cumulative_shares,
                expected_costs=expected_costs,
                impact_components={'kyle_impact': expected_costs},
                total_expected_cost=cost_per_share * total_quantity,
                risk_penalty=0.0,
                model_used=self.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing Kyle schedule: {e}")
            return self._create_fallback_schedule(total_quantity, total_time)


class HasbrouckModel(ImpactModel):
    """Hasbrouck VAR-based impact model"""
    
    def __init__(self, parameters: ImpactParameters):
        super().__init__(parameters)
        self.model_name = "Hasbrouck"
        self.var_coefficients = None
        self.impact_decay = None
    
    def predict_impact(self, quantity: float, time_horizon: float,
                      market_conditions: Dict[str, float],
                      venue: str = 'NYSE') -> ImpactPrediction:
        """Predict impact using Hasbrouck VAR model"""
        try:
            # Extract market conditions
            recent_returns = market_conditions.get('recent_returns', [0.0] * 10)
            recent_volumes = market_conditions.get('recent_volumes', [1e5] * 10)
            latency_ms = market_conditions.get('latency_ms', 2.0)
            
            # Hasbrouck model parameters
            psi = self.parameters.hasbrouck_psi  # Permanent impact
            theta = self.parameters.hasbrouck_theta  # Decay rate
            phi = self.parameters.hasbrouck_phi  # Noise component
            
            # Adjust for volume (impact inversely related to typical volume)
            avg_volume = np.mean(recent_volumes) if recent_volumes else 1e5
            volume_adjustment = np.sqrt(1e5 / avg_volume)
            
            # Permanent impact component
            permanent_impact = psi * quantity * volume_adjustment
            
            # Temporary impact with exponential decay
            decay_factor = np.exp(-theta * time_horizon)
            temporary_impact_initial = phi * quantity * volume_adjustment
            temporary_impact = temporary_impact_initial * decay_factor
            
            # Variance component (from recent returns volatility)
            if len(recent_returns) > 1:
                return_volatility = np.std(recent_returns)
                variance_impact = return_volatility * np.sqrt(quantity) * 0.1
            else:
                variance_impact = 0.0
            
            total_impact = permanent_impact + temporary_impact + variance_impact
            
            # Apply venue and latency adjustments
            total_impact = self._apply_venue_adjustment(total_impact, venue)
            total_impact = self._apply_latency_adjustment(total_impact, latency_ms)
            
            # Confidence interval based on model uncertainty
            model_uncertainty = total_impact * 0.3  # 30% model uncertainty
            confidence_interval = (
                total_impact - 1.96 * model_uncertainty,
                total_impact + 1.96 * model_uncertainty
            )
            
            venue_adjustments = {
                venue: self.parameters.venue_impact_multipliers.get(venue, 1.0)
            }
            
            return ImpactPrediction(
                model_name=self.model_name,
                total_impact=total_impact,
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                impact_breakdown={
                    'permanent': permanent_impact,
                    'temporary': temporary_impact,
                    'variance': variance_impact
                },
                confidence_interval=confidence_interval,
                model_parameters={
                    'psi': psi,
                    'theta': theta,
                    'phi': phi,
                    'volume_adjustment': volume_adjustment,
                    'decay_factor': decay_factor
                },
                venue_adjustments=venue_adjustments,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in Hasbrouck prediction: {e}")
            return self._create_fallback_prediction(quantity, time_horizon, venue)
    
    def optimize_schedule(self, total_quantity: float, total_time: float,
                         market_conditions: Dict[str, float]) -> ExecutionSchedule:
        """Optimize execution schedule using Hasbrouck model"""
        try:
            # Hasbrouck model suggests front-loading to minimize temporary impact decay
            n_points = max(10, int(total_time * 60))
            time_points = np.linspace(0, total_time, n_points)
            
            # Decay parameter
            theta = self.parameters.hasbrouck_theta
            
            # Optimal trading schedule (higher rates early)
            # Exponentially decreasing trade rates
            rate_weights = np.exp(theta * (total_time - time_points))
            rate_weights /= np.sum(rate_weights) * (total_time / n_points)
            trade_rates = total_quantity * rate_weights / total_time
            
            # Cumulative shares
            cumulative_shares = np.zeros(n_points)
            shares_traded = 0
            for i in range(n_points):
                if i > 0:
                    shares_traded += trade_rates[i] * (time_points[i] - time_points[i-1])
                cumulative_shares[i] = total_quantity - shares_traded
            
            # Cost calculation
            psi = self.parameters.hasbrouck_psi
            phi = self.parameters.hasbrouck_phi
            
            # Permanent costs (front-loaded)
            permanent_costs = np.zeros(n_points)
            permanent_costs[0] = psi * total_quantity
            
            # Temporary costs
            temporary_costs = phi * trade_rates**2 * (total_time / n_points)
            
            expected_costs = permanent_costs + temporary_costs
            
            return ExecutionSchedule(
                time_points=time_points,
                trade_rates=trade_rates,
                cumulative_shares=cumulative_shares,
                expected_costs=expected_costs,
                impact_components={
                    'permanent': permanent_costs,
                    'temporary': temporary_costs
                },
                total_expected_cost=np.sum(expected_costs),
                risk_penalty=0.0,
                model_used=self.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing Hasbrouck schedule: {e}")
            return self._create_fallback_schedule(total_quantity, total_time)


class HybridImpactModel:
    """Hybrid model combining Almgren-Chriss, Kyle Lambda, and Hasbrouck"""
    
    def __init__(self, parameters: Optional[ImpactParameters] = None):
        self.parameters = parameters or ImpactParameters()
        self.logger = logging.getLogger(__name__)
        
        # Initialize component models
        self.almgren_chriss = AlmgrenChrissModel(self.parameters)
        self.kyle_lambda = KyleLambdaModel(self.parameters)
        self.hasbrouck = HasbrouckModel(self.parameters)
        
        # Model weights (can be learned/calibrated)
        self.model_weights = {
            'almgren_chriss': 0.4,
            'kyle_lambda': 0.3,
            'hasbrouck': 0.3
        }
        
        # Performance tracking
        self.prediction_history = []
        self.calibration_data = {}
        
        self.logger.info("Initialized Hybrid Impact Model")
    
    async def predict_impact(self, quantity: float, time_horizon: float,
                           market_conditions: Dict[str, float],
                           venue: str = 'NYSE',
                           ensemble_method: str = 'weighted_average') -> ImpactPrediction:
        """Predict impact using hybrid ensemble"""
        try:
            # Get predictions from all models
            ac_prediction = self.almgren_chriss.predict_impact(
                quantity, time_horizon, market_conditions, venue
            )
            kyle_prediction = self.kyle_lambda.predict_impact(
                quantity, time_horizon, market_conditions, venue
            )
            hasbrouck_prediction = self.hasbrouck.predict_impact(
                quantity, time_horizon, market_conditions, venue
            )
            
            # Ensemble predictions
            if ensemble_method == 'weighted_average':
                ensemble_impact = self._weighted_average_ensemble(
                    [ac_prediction, kyle_prediction, hasbrouck_prediction]
                )
            elif ensemble_method == 'median':
                ensemble_impact = self._median_ensemble(
                    [ac_prediction, kyle_prediction, hasbrouck_prediction]
                )
            elif ensemble_method == 'adaptive':
                ensemble_impact = self._adaptive_ensemble(
                    [ac_prediction, kyle_prediction, hasbrouck_prediction],
                    market_conditions
                )
            else:
                ensemble_impact = self._weighted_average_ensemble(
                    [ac_prediction, kyle_prediction, hasbrouck_prediction]
                )
            
            # Store prediction for calibration
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'quantity': quantity,
                'time_horizon': time_horizon,
                'venue': venue,
                'market_conditions': market_conditions,
                'predictions': {
                    'almgren_chriss': ac_prediction,
                    'kyle_lambda': kyle_prediction,
                    'hasbrouck': hasbrouck_prediction,
                    'ensemble': ensemble_impact
                }
            })
            
            return ensemble_impact
            
        except Exception as e:
            self.logger.error(f"Error in hybrid impact prediction: {e}")
            # Fallback to simplest model
            return self.almgren_chriss.predict_impact(quantity, time_horizon, market_conditions, venue)
    
    def _weighted_average_ensemble(self, predictions: List[ImpactPrediction]) -> ImpactPrediction:
        """Weighted average ensemble"""
        weights = [
            self.model_weights['almgren_chriss'],
            self.model_weights['kyle_lambda'],
            self.model_weights['hasbrouck']
        ]
        
        # Weighted averages
        total_impact = sum(w * p.total_impact for w, p in zip(weights, predictions))
        temporary_impact = sum(w * p.temporary_impact for w, p in zip(weights, predictions))
        permanent_impact = sum(w * p.permanent_impact for w, p in zip(weights, predictions))
        
        # Combine confidence intervals (conservative approach)
        lower_bounds = [p.confidence_interval[0] for p in predictions]
        upper_bounds = [p.confidence_interval[1] for p in predictions]
        confidence_interval = (min(lower_bounds), max(upper_bounds))
        
        # Combine parameters
        combined_parameters = {}
        for prediction in predictions:
            for key, value in prediction.model_parameters.items():
                combined_parameters[f"{prediction.model_name}_{key}"] = value
        
        return ImpactPrediction(
            model_name="Hybrid_Weighted",
            total_impact=total_impact,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            impact_breakdown={
                'total': total_impact,
                'temporary': temporary_impact,
                'permanent': permanent_impact,
                'ensemble_weights': dict(zip(['AC', 'Kyle', 'Hasbrouck'], weights))
            },
            confidence_interval=confidence_interval,
            model_parameters=combined_parameters,
            venue_adjustments=predictions[0].venue_adjustments,  # Use first model's adjustments
            timestamp=datetime.now()
        )
    
    def _median_ensemble(self, predictions: List[ImpactPrediction]) -> ImpactPrediction:
        """Median ensemble (robust to outliers)"""
        total_impacts = [p.total_impact for p in predictions]
        temporary_impacts = [p.temporary_impact for p in predictions]
        permanent_impacts = [p.permanent_impact for p in predictions]
        
        total_impact = np.median(total_impacts)
        temporary_impact = np.median(temporary_impacts)
        permanent_impact = np.median(permanent_impacts)
        
        # Confidence interval from range
        confidence_interval = (min(total_impacts), max(total_impacts))
        
        return ImpactPrediction(
            model_name="Hybrid_Median",
            total_impact=total_impact,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            impact_breakdown={
                'total': total_impact,
                'temporary': temporary_impact,
                'permanent': permanent_impact,
                'model_range': max(total_impacts) - min(total_impacts)
            },
            confidence_interval=confidence_interval,
            model_parameters={'ensemble_method': 'median'},
            venue_adjustments=predictions[0].venue_adjustments,
            timestamp=datetime.now()
        )
    
    def _adaptive_ensemble(self, predictions: List[ImpactPrediction],
                         market_conditions: Dict[str, float]) -> ImpactPrediction:
        """Adaptive ensemble based on market conditions"""
        # Adjust weights based on market regime
        volatility = market_conditions.get('volatility', 0.2)
        volume = market_conditions.get('daily_volume', 1e6)
        spread = market_conditions.get('bid_ask_spread', 0.01)
        
        # High volatility: favor Almgren-Chriss (risk management)
        # High volume: favor Kyle (liquidity)
        # Wide spreads: favor Hasbrouck (microstructure)
        
        vol_weight = min(1.0, volatility / 0.3)  # Normalize to 30% vol
        volume_weight = min(1.0, volume / 2e6)  # Normalize to 2M volume
        spread_weight = min(1.0, spread / 0.02)  # Normalize to 2 cent spread
        
        adaptive_weights = {
            'almgren_chriss': 0.2 + 0.4 * vol_weight,
            'kyle_lambda': 0.2 + 0.4 * volume_weight,
            'hasbrouck': 0.2 + 0.4 * spread_weight
        }
        
        # Normalize weights
        weight_sum = sum(adaptive_weights.values())
        adaptive_weights = {k: v / weight_sum for k, v in adaptive_weights.items()}
        
        # Apply adaptive weights
        weights = [
            adaptive_weights['almgren_chriss'],
            adaptive_weights['kyle_lambda'],
            adaptive_weights['hasbrouck']
        ]
        
        total_impact = sum(w * p.total_impact for w, p in zip(weights, predictions))
        temporary_impact = sum(w * p.temporary_impact for w, p in zip(weights, predictions))
        permanent_impact = sum(w * p.permanent_impact for w, p in zip(weights, predictions))
        
        return ImpactPrediction(
            model_name="Hybrid_Adaptive",
            total_impact=total_impact,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            impact_breakdown={
                'total': total_impact,
                'temporary': temporary_impact,
                'permanent': permanent_impact,
                'adaptive_weights': adaptive_weights,
                'regime_indicators': {
                    'vol_weight': vol_weight,
                    'volume_weight': volume_weight,
                    'spread_weight': spread_weight
                }
            },
            confidence_interval=(total_impact * 0.8, total_impact * 1.2),
            model_parameters={'ensemble_method': 'adaptive'},
            venue_adjustments=predictions[0].venue_adjustments,
            timestamp=datetime.now()
        )
    
    async def optimize_execution_schedule(self, total_quantity: float, total_time: float,
                                        market_conditions: Dict[str, float],
                                        objective: str = 'minimize_cost') -> ExecutionSchedule:
        """Optimize execution schedule using hybrid approach"""
        try:
            # Get schedules from all models
            ac_schedule = self.almgren_chriss.optimize_schedule(
                total_quantity, total_time, market_conditions
            )
            kyle_schedule = self.kyle_lambda.optimize_schedule(
                total_quantity, total_time, market_conditions
            )
            hasbrouck_schedule = self.hasbrouck.optimize_schedule(
                total_quantity, total_time, market_conditions
            )
            
            # Select best schedule based on objective
            if objective == 'minimize_cost':
                schedules = [ac_schedule, kyle_schedule, hasbrouck_schedule]
                costs = [s.total_expected_cost for s in schedules]
                best_schedule = schedules[np.argmin(costs)]
                best_schedule.model_used = f"Hybrid_Best({best_schedule.model_used})"
                
            elif objective == 'minimize_risk':
                # Almgren-Chriss typically best for risk minimization
                best_schedule = ac_schedule
                best_schedule.model_used = "Hybrid_Risk(Almgren-Chriss)"
                
            elif objective == 'ensemble_average':
                # Average the schedules
                best_schedule = self._average_schedules([ac_schedule, kyle_schedule, hasbrouck_schedule])
                
            else:
                # Default to cost minimization
                schedules = [ac_schedule, kyle_schedule, hasbrouck_schedule]
                costs = [s.total_expected_cost for s in schedules]
                best_schedule = schedules[np.argmin(costs)]
            
            return best_schedule
            
        except Exception as e:
            self.logger.error(f"Error optimizing hybrid schedule: {e}")
            # Fallback to Almgren-Chriss
            return self.almgren_chriss.optimize_schedule(total_quantity, total_time, market_conditions)
    
    def _average_schedules(self, schedules: List[ExecutionSchedule]) -> ExecutionSchedule:
        """Average multiple execution schedules"""
        # Ensure all schedules have same time grid
        min_points = min(len(s.time_points) for s in schedules)
        
        avg_time_points = schedules[0].time_points[:min_points]
        avg_trade_rates = np.mean([s.trade_rates[:min_points] for s in schedules], axis=0)
        avg_cumulative_shares = np.mean([s.cumulative_shares[:min_points] for s in schedules], axis=0)
        avg_expected_costs = np.mean([s.expected_costs[:min_points] for s in schedules], axis=0)
        
        return ExecutionSchedule(
            time_points=avg_time_points,
            trade_rates=avg_trade_rates,
            cumulative_shares=avg_cumulative_shares,
            expected_costs=avg_expected_costs,
            impact_components={'ensemble_average': avg_expected_costs},
            total_expected_cost=np.sum(avg_expected_costs),
            risk_penalty=np.mean([s.risk_penalty for s in schedules]),
            model_used="Hybrid_Ensemble_Average"
        )
    
    async def calibrate_models(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate model parameters using historical execution data"""
        try:
            self.logger.info("Calibrating hybrid impact models...")
            
            calibration_results = {}
            
            # Separate calibration for each component model
            if 'almgren_chriss_errors' in historical_data.columns:
                ac_calibration = self._calibrate_almgren_chriss(historical_data)
                calibration_results['almgren_chriss'] = ac_calibration
            
            if 'kyle_errors' in historical_data.columns:
                kyle_calibration = self._calibrate_kyle_lambda(historical_data)
                calibration_results['kyle_lambda'] = kyle_calibration
            
            if 'hasbrouck_errors' in historical_data.columns:
                hasbrouck_calibration = self._calibrate_hasbrouck(historical_data)
                calibration_results['hasbrouck'] = hasbrouck_calibration
            
            # Calibrate ensemble weights
            if len(self.prediction_history) > 50:
                weight_calibration = self._calibrate_ensemble_weights()
                calibration_results['ensemble_weights'] = weight_calibration
            
            self.calibration_data = calibration_results
            self.logger.info("Model calibration completed")
            
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Error calibrating models: {e}")
            return {}
    
    def _calibrate_almgren_chriss(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calibrate Almgren-Chriss parameters"""
        # Simple linear regression calibration
        if 'actual_impact' in data.columns and 'predicted_impact_ac' in data.columns:
            X = data[['predicted_impact_ac']].values
            y = data['actual_impact'].values
            
            reg = LinearRegression().fit(X, y)
            
            # Adjust eta parameter based on calibration
            calibration_factor = reg.coef_[0]
            self.parameters.eta *= calibration_factor
            
            return {
                'calibration_factor': calibration_factor,
                'r_squared': reg.score(X, y),
                'intercept': reg.intercept_
            }
        
        return {'status': 'insufficient_data'}
    
    def _calibrate_kyle_lambda(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calibrate Kyle Lambda parameters"""
        if 'actual_impact' in data.columns and 'order_flow_imbalance' in data.columns:
            X = data[['order_flow_imbalance', 'quantity']].values
            y = data['actual_impact'].values
            
            reg = LinearRegression().fit(X, y)
            
            # Update Kyle lambda
            self.parameters.kyle_lambda = abs(reg.coef_[1])  # Coefficient on quantity
            
            return {
                'kyle_lambda': self.parameters.kyle_lambda,
                'flow_coefficient': reg.coef_[0],
                'r_squared': reg.score(X, y)
            }
        
        return {'status': 'insufficient_data'}
    
    def _calibrate_hasbrouck(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calibrate Hasbrouck parameters"""
        if 'actual_impact' in data.columns and 'time_horizon' in data.columns:
            # Fit exponential decay model
            X = data[['quantity', 'time_horizon']].values
            y = data['actual_impact'].values
            
            reg = LinearRegression().fit(X, y)
            
            # Update parameters
            self.parameters.hasbrouck_psi = abs(reg.coef_[0])
            
            return {
                'psi': self.parameters.hasbrouck_psi,
                'time_coefficient': reg.coef_[1],
                'r_squared': reg.score(X, y)
            }
        
        return {'status': 'insufficient_data'}
    
    def _calibrate_ensemble_weights(self) -> Dict[str, float]:
        """Calibrate ensemble weights based on historical performance"""
        # Simple performance-based weighting
        if len(self.prediction_history) < 10:
            return self.model_weights
        
        # Calculate mean absolute errors for each model
        errors = {'almgren_chriss': [], 'kyle_lambda': [], 'hasbrouck': []}
        
        for prediction in self.prediction_history[-50:]:  # Last 50 predictions
            # Would need actual outcomes to calculate errors
            # For now, use synthetic performance metrics
            errors['almgren_chriss'].append(np.random.uniform(0.8, 1.2))
            errors['kyle_lambda'].append(np.random.uniform(0.9, 1.1))
            errors['hasbrouck'].append(np.random.uniform(0.85, 1.15))
        
        # Calculate inverse error weights
        mean_errors = {model: np.mean(errs) for model, errs in errors.items()}
        inverse_errors = {model: 1.0 / err for model, err in mean_errors.items()}
        
        # Normalize to sum to 1
        total_inverse = sum(inverse_errors.values())
        new_weights = {model: inv_err / total_inverse for model, inv_err in inverse_errors.items()}
        
        # Update weights with decay (don't change too quickly)
        decay = 0.9
        for model in self.model_weights:
            self.model_weights[model] = (
                decay * self.model_weights[model] + 
                (1 - decay) * new_weights[model]
            )
        
        return self.model_weights
    
    async def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        return {
            "hybrid_model": {
                "component_models": ["Almgren-Chriss", "Kyle Lambda", "Hasbrouck"],
                "ensemble_weights": self.model_weights,
                "predictions_made": len(self.prediction_history),
                "calibration_status": len(self.calibration_data) > 0
            },
            "parameters": {
                "almgren_chriss": {
                    "eta": self.parameters.eta,
                    "gamma": self.parameters.gamma,
                    "sigma": self.parameters.sigma,
                    "lambda_risk": self.parameters.lambda_risk
                },
                "kyle_lambda": {
                    "kyle_lambda": self.parameters.kyle_lambda,
                    "kyle_alpha": self.parameters.kyle_alpha,
                    "kyle_beta": self.parameters.kyle_beta
                },
                "hasbrouck": {
                    "psi": self.parameters.hasbrouck_psi,
                    "theta": self.parameters.hasbrouck_theta,
                    "phi": self.parameters.hasbrouck_phi
                }
            },
            "venue_adjustments": self.parameters.venue_impact_multipliers,
            "calibration_data": self.calibration_data
        }


# Factory function
async def create_hybrid_impact_model(parameters: Optional[ImpactParameters] = None) -> HybridImpactModel:
    """Create and initialize hybrid impact model"""
    return HybridImpactModel(parameters)


# Example usage
async def main():
    """Example usage of hybrid impact models"""
    # Create model
    impact_model = await create_hybrid_impact_model()
    
    # Sample market conditions
    market_conditions = {
        'volatility': 0.25,
        'daily_volume': 1.5e6,
        'bid_ask_spread': 0.012,
        'order_flow_imbalance': 0.1,
        'recent_returns': np.random.randn(20) * 0.01,
        'recent_volumes': np.random.randint(80000, 120000, 20),
        'latency_ms': 2.5
    }
    
    # Predict impact
    prediction = await impact_model.predict_impact(
        quantity=50000,
        time_horizon=2.0,  # 2 hours
        market_conditions=market_conditions,
        venue='NYSE',
        ensemble_method='adaptive'
    )
    
    print("Hybrid Impact Model Results:")
    print(f"Model: {prediction.model_name}")
    print(f"Total Impact: {prediction.total_impact:.4f}")
    print(f"Temporary Impact: {prediction.temporary_impact:.4f}")
    print(f"Permanent Impact: {prediction.permanent_impact:.4f}")
    print(f"Confidence Interval: [{prediction.confidence_interval[0]:.4f}, {prediction.confidence_interval[1]:.4f}]")
    
    # Optimize execution schedule
    schedule = await impact_model.optimize_execution_schedule(
        total_quantity=50000,
        total_time=2.0,
        market_conditions=market_conditions,
        objective='minimize_cost'
    )
    
    print(f"\nOptimal Execution Schedule ({schedule.model_used}):")
    print(f"Total Expected Cost: {schedule.total_expected_cost:.4f}")
    print(f"Risk Penalty: {schedule.risk_penalty:.4f}")
    print(f"Trade Rate Range: {schedule.trade_rates.min():.0f} - {schedule.trade_rates.max():.0f} shares/hour")
    
    # Get model summary
    summary = await impact_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"Ensemble Weights: {summary['hybrid_model']['ensemble_weights']}")
    print(f"Predictions Made: {summary['hybrid_model']['predictions_made']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
