#!/usr/bin/env python3
"""
Real-Time Risk Management System

Implements live risk monitoring with automatic throttling, drawdown governance,
and Kelly criterion with volatility caps for dynamic risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import asyncio
from collections import deque

from common.observability.telemetry import log_event, trace_operation


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for monitoring"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThrottleAction(str, Enum):
    """Throttle actions"""
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    PAUSE_TRADING = "pause_trading"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_95: float
    cvar_95: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    correlation: float
    sector_concentration: float
    leverage: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'volatility': self.volatility,
            'beta': self.beta,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'correlation': self.correlation,
            'sector_concentration': self.sector_concentration,
            'leverage': self.leverage,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskThreshold:
    """Risk threshold configuration"""
    var_limit: float
    cvar_limit: float
    volatility_limit: float
    drawdown_limit: float
    correlation_limit: float
    sector_limit: float
    leverage_limit: float
    
    def check_breach(self, metrics: RiskMetrics) -> List[str]:
        """Check which thresholds are breached"""
        breaches = []
        
        if metrics.var_95 > self.var_limit:
            breaches.append('var_limit')
        if metrics.cvar_95 > self.cvar_limit:
            breaches.append('cvar_limit')
        if metrics.volatility > self.volatility_limit:
            breaches.append('volatility_limit')
        if metrics.current_drawdown > self.drawdown_limit:
            breaches.append('drawdown_limit')
        if metrics.correlation > self.correlation_limit:
            breaches.append('correlation_limit')
        if metrics.sector_concentration > self.sector_limit:
            breaches.append('sector_limit')
        if metrics.leverage > self.leverage_limit:
            breaches.append('leverage_limit')
        
        return breaches


@dataclass
class ThrottleDecision:
    """Throttle decision"""
    action: ThrottleAction
    reduction_factor: float  # 0-1, how much to reduce
    reason: str
    breached_thresholds: List[str]
    risk_level: RiskLevel
    timestamp: datetime
    duration: timedelta  # How long to apply throttle


class KellyCriterion:
    """
    Kelly Criterion Implementation
    
    Calculates optimal position sizes with volatility caps and drawdown governance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)
        self.volatility_cap = config.get('volatility_cap', 0.20)
        self.drawdown_threshold = config.get('drawdown_threshold', 0.10)
        self.drawdown_reduction = config.get('drawdown_reduction', 0.5)
        
        # Performance tracking
        self.returns_history: deque = deque(maxlen=252)  # 1 year
        self.volatility_history: deque = deque(maxlen=60)  # 60 days
        
        logger.info("Kelly Criterion initialized")
    
    def calculate_kelly_fraction(self, expected_return: float, 
                               volatility: float,
                               current_drawdown: float = 0.0) -> float:
        """
        Calculate Kelly fraction with volatility cap and drawdown governance
        
        Args:
            expected_return: Expected return of the strategy
            volatility: Strategy volatility
            current_drawdown: Current portfolio drawdown
            
        Returns:
            Kelly fraction (0-1)
        """
        try:
            # Basic Kelly formula: f = μ / σ²
            if volatility <= 0:
                return 0.0
            
            kelly_fraction = expected_return / (volatility ** 2)
            
            # Apply volatility cap
            if volatility > self.volatility_cap:
                kelly_fraction *= (self.volatility_cap / volatility)
            
            # Apply drawdown governance
            if current_drawdown > self.drawdown_threshold:
                kelly_fraction *= self.drawdown_reduction
            
            # Apply maximum fraction limit
            kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
            
            # Ensure non-negative
            kelly_fraction = max(0.0, kelly_fraction)
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly fraction calculation failed: {e}")
            return 0.0
    
    def update_performance(self, returns: List[float], volatilities: List[float]):
        """Update performance history"""
        try:
            self.returns_history.extend(returns)
            self.volatility_history.extend(volatilities)
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
    
    def get_historical_metrics(self) -> Tuple[float, float]:
        """Get historical expected return and volatility"""
        if len(self.returns_history) < 30:
            return 0.0, 0.02  # Default values
        
        returns_array = np.array(list(self.returns_history))
        volatilities_array = np.array(list(self.volatility_history))
        
        expected_return = np.mean(returns_array)
        avg_volatility = np.mean(volatilities_array)
        
        return expected_return, avg_volatility


class RealTimeRiskMonitor:
    """
    Real-Time Risk Monitoring System
    
    Features:
    - Live risk metric calculation
    - Automatic threshold monitoring
    - Dynamic throttling decisions
    - Kelly criterion integration
    - Emergency stop mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk thresholds
        self.thresholds = RiskThreshold(
            var_limit=config.get('var_limit', 0.02),
            cvar_limit=config.get('cvar_limit', 0.03),
            volatility_limit=config.get('volatility_limit', 0.20),
            drawdown_limit=config.get('drawdown_limit', 0.10),
            correlation_limit=config.get('correlation_limit', 0.30),
            sector_limit=config.get('sector_limit', 0.25),
            leverage_limit=config.get('leverage_limit', 2.0)
        )
        
        # Kelly criterion
        self.kelly_criterion = KellyCriterion(config.get('kelly_config', {}))
        
        # State tracking
        self.current_metrics: Optional[RiskMetrics] = None
        self.metrics_history: deque = deque(maxlen=1000)
        self.throttle_history: List[ThrottleDecision] = []
        
        # Performance tracking
        self.breach_count = 0
        self.throttle_count = 0
        self.emergency_stop_count = 0
        
        # Callbacks
        self.breach_callbacks: List[Callable] = []
        self.throttle_callbacks: List[Callable] = []
        
        logger.info("Real-Time Risk Monitor initialized")
    
    async def update_risk_metrics(self, portfolio_data: Dict[str, Any], 
                                trace_id: str) -> RiskMetrics:
        """
        Update risk metrics with latest portfolio data
        
        Args:
            portfolio_data: Portfolio positions, prices, returns
            trace_id: Trace ID for observability
            
        Returns:
            Updated risk metrics
        """
        async with trace_operation("risk_metrics_update", trace_id=trace_id):
            try:
                # Extract portfolio data
                positions = portfolio_data.get('positions', {})
                prices = portfolio_data.get('prices', {})
                returns = portfolio_data.get('returns', [])
                portfolio_value = portfolio_data.get('portfolio_value', 1e6)
                
                # Calculate risk metrics
                var_95 = self._calculate_var(returns, 0.95)
                cvar_95 = self._calculate_cvar(returns, 0.95)
                volatility = self._calculate_volatility(returns)
                beta = self._calculate_beta(returns, portfolio_data.get('market_returns', []))
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                max_drawdown = self._calculate_max_drawdown(returns)
                current_drawdown = self._calculate_current_drawdown(returns)
                correlation = self._calculate_correlation(positions, prices)
                sector_concentration = self._calculate_sector_concentration(positions)
                leverage = self._calculate_leverage(positions, portfolio_value)
                
                # Create metrics object
                metrics = RiskMetrics(
                    var_95=var_95,
                    cvar_95=cvar_95,
                    volatility=volatility,
                    beta=beta,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    current_drawdown=current_drawdown,
                    correlation=correlation,
                    sector_concentration=sector_concentration,
                    leverage=leverage,
                    timestamp=datetime.utcnow()
                )
                
                # Update state
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Update Kelly criterion
                if returns:
                    self.kelly_criterion.update_performance(returns, [volatility])
                
                # Check for breaches
                await self._check_risk_breaches(metrics, trace_id)
                
                return metrics
                
            except Exception as e:
                logger.error(f"Risk metrics update failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_metrics()
    
    async def get_throttle_decision(self, intended_size: float,
                                  opportunity_risk: float,
                                  trace_id: str) -> ThrottleDecision:
        """
        Get throttle decision for intended trade
        
        Args:
            intended_size: Intended position size
            opportunity_risk: Risk of the opportunity
            trace_id: Trace ID for observability
            
        Returns:
            Throttle decision
        """
        async with trace_operation("throttle_decision", trace_id=trace_id):
            try:
                if not self.current_metrics:
                    return self._get_default_throttle()
                
                # Check for breaches
                breaches = self.thresholds.check_breach(self.current_metrics)
                
                if not breaches:
                    return ThrottleDecision(
                        action=ThrottleAction.NONE,
                        reduction_factor=1.0,
                        reason="No risk breaches",
                        breached_thresholds=[],
                        risk_level=RiskLevel.LOW,
                        timestamp=datetime.utcnow(),
                        duration=timedelta(minutes=0)
                    )
                
                # Determine risk level
                risk_level = self._determine_risk_level(breaches)
                
                # Calculate Kelly fraction
                expected_return, historical_vol = self.kelly_criterion.get_historical_metrics()
                kelly_fraction = self.kelly_criterion.calculate_kelly_fraction(
                    expected_return, historical_vol, self.current_metrics.current_drawdown
                )
                
                # Determine throttle action
                action, reduction_factor, reason = self._determine_throttle_action(
                    breaches, risk_level, kelly_fraction, intended_size, opportunity_risk
                )
                
                # Create throttle decision
                decision = ThrottleDecision(
                    action=action,
                    reduction_factor=reduction_factor,
                    reason=reason,
                    breached_thresholds=breaches,
                    risk_level=risk_level,
                    timestamp=datetime.utcnow(),
                    duration=self._calculate_throttle_duration(risk_level)
                )
                
                # Record decision
                self.throttle_history.append(decision)
                self.throttle_count += 1
                
                # Log decision
                await self._log_throttle_decision(decision, trace_id)
                
                return decision
                
            except Exception as e:
                logger.error(f"Throttle decision failed: {e}", extra={'trace_id': trace_id})
                return self._get_default_throttle()
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 10:
            return 0.02  # Default VaR
        
        returns_array = np.array(returns)
        var_percentile = (1 - confidence) * 100
        return np.percentile(returns_array, var_percentile)
    
    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) < 10:
            return 0.03  # Default CVaR
        
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence) * 100)
        tail_returns = returns_array[returns_array <= var_threshold]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate portfolio volatility"""
        if len(returns) < 10:
            return 0.02  # Default volatility
        
        returns_array = np.array(returns)
        return np.std(returns_array) * np.sqrt(252)  # Annualized
    
    def _calculate_beta(self, returns: List[float], market_returns: List[float]) -> float:
        """Calculate portfolio beta"""
        if len(returns) < 10 or len(market_returns) < 10:
            return 1.0  # Default beta
        
        returns_array = np.array(returns)
        market_array = np.array(market_returns)
        
        # Ensure same length
        min_length = min(len(returns_array), len(market_array))
        returns_array = returns_array[:min_length]
        market_array = market_array[:min_length]
        
        # Calculate beta
        covariance = np.cov(returns_array, market_array)[0, 1]
        market_variance = np.var(market_array)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 10:
            return 0.0  # Default Sharpe
        
        returns_array = np.array(returns)
        excess_return = np.mean(returns_array) - 0.02 / 252  # Assume 2% risk-free rate
        volatility = np.std(returns_array)
        
        return excess_return / volatility if volatility > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(returns) < 10:
            return 0.0  # Default drawdown
        
        returns_array = np.array(returns)
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def _calculate_current_drawdown(self, returns: List[float]) -> float:
        """Calculate current drawdown"""
        if len(returns) < 10:
            return 0.0  # Default drawdown
        
        returns_array = np.array(returns)
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown[-1] if len(drawdown) > 0 else 0.0
    
    def _calculate_correlation(self, positions: Dict[str, float], 
                            prices: Dict[str, List[float]]) -> float:
        """Calculate portfolio correlation"""
        if len(positions) < 2:
            return 0.0  # Default correlation
        
        # Simplified correlation calculation
        # In production, would calculate pairwise correlations
        return 0.3  # Placeholder
    
    def _calculate_sector_concentration(self, positions: Dict[str, float]) -> float:
        """Calculate sector concentration"""
        if not positions:
            return 0.0
        
        # Simplified sector mapping
        sector_mapping = {
            'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'TSLA': 'Automotive', 'AMZN': 'Consumer', 'NVDA': 'Technology',
            'META': 'Technology', 'JPM': 'Financial', 'XOM': 'Energy', 'JNJ': 'Healthcare'
        }
        
        sector_exposures = {}
        total_exposure = sum(abs(pos) for pos in positions.values())
        
        for symbol, position in positions.items():
            sector = sector_mapping.get(symbol, 'Other')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(position)
        
        # Calculate concentration (max sector exposure)
        max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0
        return max_sector_exposure / total_exposure if total_exposure > 0 else 0.0
    
    def _calculate_leverage(self, positions: Dict[str, float], 
                          portfolio_value: float) -> float:
        """Calculate portfolio leverage"""
        if portfolio_value <= 0:
            return 1.0  # Default leverage
        
        gross_exposure = sum(abs(pos) for pos in positions.values())
        return gross_exposure / portfolio_value
    
    def _determine_risk_level(self, breaches: List[str]) -> RiskLevel:
        """Determine risk level based on breaches"""
        if 'emergency_stop' in breaches or len(breaches) >= 4:
            return RiskLevel.CRITICAL
        elif len(breaches) >= 3:
            return RiskLevel.HIGH
        elif len(breaches) >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_throttle_action(self, breaches: List[str],
                                 risk_level: RiskLevel,
                                 kelly_fraction: float,
                                 intended_size: float,
                                 opportunity_risk: float) -> Tuple[ThrottleAction, float, str]:
        """Determine throttle action and reduction factor"""
        
        if risk_level == RiskLevel.CRITICAL:
            return ThrottleAction.EMERGENCY_STOP, 0.0, "Critical risk level - emergency stop"
        
        elif risk_level == RiskLevel.HIGH:
            # Significant reduction
            reduction_factor = min(0.3, kelly_fraction)
            return ThrottleAction.REDUCE_SIZE, reduction_factor, "High risk level - significant reduction"
        
        elif risk_level == RiskLevel.MEDIUM:
            # Moderate reduction
            reduction_factor = min(0.6, kelly_fraction)
            return ThrottleAction.REDUCE_SIZE, reduction_factor, "Medium risk level - moderate reduction"
        
        else:
            # Low risk - apply Kelly fraction
            reduction_factor = kelly_fraction
            return ThrottleAction.NONE, reduction_factor, "Low risk level - Kelly fraction applied"
    
    def _calculate_throttle_duration(self, risk_level: RiskLevel) -> timedelta:
        """Calculate throttle duration"""
        durations = {
            RiskLevel.LOW: timedelta(minutes=0),
            RiskLevel.MEDIUM: timedelta(minutes=30),
            RiskLevel.HIGH: timedelta(hours=2),
            RiskLevel.CRITICAL: timedelta(hours=24)
        }
        return durations.get(risk_level, timedelta(minutes=0))
    
    async def _check_risk_breaches(self, metrics: RiskMetrics, trace_id: str):
        """Check for risk breaches and trigger callbacks"""
        try:
            breaches = self.thresholds.check_breach(metrics)
            
            if breaches:
                self.breach_count += 1
                
                # Trigger breach callbacks
                for callback in self.breach_callbacks:
                    try:
                        await callback(breaches, metrics, trace_id)
                    except Exception as e:
                        logger.error(f"Breach callback failed: {e}")
                
                # Log breach
                await self._log_risk_breach(breaches, metrics, trace_id)
                
        except Exception as e:
            logger.error(f"Risk breach check failed: {e}")
    
    async def _log_risk_breach(self, breaches: List[str], 
                             metrics: RiskMetrics, trace_id: str):
        """Log risk breach event"""
        try:
            await log_event("risk_breach_detected", {
                "trace_id": trace_id,
                "breaches": breaches,
                "metrics": metrics.to_dict(),
                "breach_count": self.breach_count,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Risk breach logging failed: {e}")
    
    async def _log_throttle_decision(self, decision: ThrottleDecision, trace_id: str):
        """Log throttle decision"""
        try:
            await log_event("throttle_decision_made", {
                "trace_id": trace_id,
                "action": decision.action.value,
                "reduction_factor": decision.reduction_factor,
                "reason": decision.reason,
                "breached_thresholds": decision.breached_thresholds,
                "risk_level": decision.risk_level.value,
                "duration_minutes": decision.duration.total_seconds() / 60,
                "throttle_count": self.throttle_count,
                "timestamp": decision.timestamp.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Throttle decision logging failed: {e}")
    
    def _get_default_metrics(self) -> RiskMetrics:
        """Get default risk metrics"""
        return RiskMetrics(
            var_95=-0.02,
            cvar_95=-0.025,
            volatility=0.15,
            beta=1.0,
            sharpe_ratio=0.5,
            max_drawdown=-0.05,
            current_drawdown=0.0,
            correlation=0.3,
            sector_concentration=0.2,
            leverage=1.2,
            timestamp=datetime.utcnow()
        )
    
    def _get_default_throttle(self) -> ThrottleDecision:
        """Get default throttle decision"""
        return ThrottleDecision(
            action=ThrottleAction.NONE,
            reduction_factor=1.0,
            reason="Default throttle",
            breached_thresholds=[],
            risk_level=RiskLevel.LOW,
            timestamp=datetime.utcnow(),
            duration=timedelta(minutes=0)
        )
    
    def add_breach_callback(self, callback: Callable):
        """Add breach callback function"""
        self.breach_callbacks.append(callback)
    
    def add_throttle_callback(self, callback: Callable):
        """Add throttle callback function"""
        self.throttle_callbacks.append(callback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "breach_count": self.breach_count,
            "throttle_count": self.throttle_count,
            "emergency_stop_count": self.emergency_stop_count,
            "current_risk_level": self.current_metrics.volatility if self.current_metrics else 0,
            "current_drawdown": self.current_metrics.current_drawdown if self.current_metrics else 0,
            "metrics_history_length": len(self.metrics_history),
            "throttle_history_length": len(self.throttle_history)
        }
