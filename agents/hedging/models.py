"""
Data models for Hedging Strategy Agent

Portfolio risk analysis and hedge optimization models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class HedgeType(str, Enum):
    MARKET_HEDGE = "market_hedge"
    SECTOR_HEDGE = "sector_hedge"
    CURRENCY_HEDGE = "currency_hedge"
    VOLATILITY_HEDGE = "volatility_hedge"
    CREDIT_HEDGE = "credit_hedge"
    COMMODITY_HEDGE = "commodity_hedge"


class HedgeInstrument(str, Enum):
    OPTIONS = "options"
    FUTURES = "futures"
    ETF = "etf"
    INVERSE_ETF = "inverse_etf"
    SWAPS = "swaps"
    BONDS = "bonds"


@dataclass
class RiskMetrics:
    """Portfolio risk assessment metrics"""
    portfolio_id: str
    timestamp: datetime
    
    # Value at Risk
    var_1d_95: float
    var_1d_99: float
    var_10d_95: float
    expected_shortfall: float
    
    # Risk factors
    market_beta: float
    sector_exposures: Dict[str, float]
    style_exposures: Dict[str, float]  # Growth, value, momentum
    
    # Correlation risks
    avg_correlation: float
    max_correlation: float
    concentration_risk: float
    
    # Volatility metrics
    portfolio_volatility: float
    tracking_error: float
    information_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "timestamp": self.timestamp.isoformat(),
            "var_1d_95": self.var_1d_95,
            "var_1d_99": self.var_1d_99,
            "var_10d_95": self.var_10d_95,
            "expected_shortfall": self.expected_shortfall,
            "market_beta": self.market_beta,
            "sector_exposures": self.sector_exposures,
            "style_exposures": self.style_exposures,
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "concentration_risk": self.concentration_risk,
            "portfolio_volatility": self.portfolio_volatility,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio
        }


@dataclass
class HedgeRecommendation:
    """Individual hedge recommendation"""
    hedge_id: str
    hedge_type: HedgeType
    instrument: HedgeInstrument
    underlying_symbol: str
    
    # Position details
    recommended_size: float
    hedge_ratio: float
    target_exposure: float
    
    # Cost analysis
    estimated_cost: float
    cost_percentage: float  # As % of portfolio
    break_even_move: float
    
    # Effectiveness
    expected_correlation: float
    hedge_effectiveness: float  # 0-1
    time_decay_risk: float
    
    # Risk reduction
    var_reduction: float
    volatility_reduction: float
    max_drawdown_reduction: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hedge_id": self.hedge_id,
            "hedge_type": self.hedge_type.value,
            "instrument": self.instrument.value,
            "underlying_symbol": self.underlying_symbol,
            "recommended_size": self.recommended_size,
            "hedge_ratio": self.hedge_ratio,
            "target_exposure": self.target_exposure,
            "estimated_cost": self.estimated_cost,
            "cost_percentage": self.cost_percentage,
            "break_even_move": self.break_even_move,
            "expected_correlation": self.expected_correlation,
            "hedge_effectiveness": self.hedge_effectiveness,
            "time_decay_risk": self.time_decay_risk,
            "var_reduction": self.var_reduction,
            "volatility_reduction": self.volatility_reduction,
            "max_drawdown_reduction": self.max_drawdown_reduction
        }


@dataclass
class HedgingAnalysis:
    """Complete hedging strategy analysis"""
    portfolio_id: str
    timestamp: datetime
    
    # Current risk assessment
    current_risk_metrics: RiskMetrics
    
    # Hedge recommendations
    recommended_hedges: List[HedgeRecommendation]
    optimal_hedge_combination: List[str]  # Hedge IDs
    
    # Portfolio optimization
    optimized_weights: Dict[str, float]
    risk_budget_allocation: Dict[str, float]
    
    # Scenario analysis
    stress_test_results: Dict[str, float]
    tail_risk_analysis: Dict[str, float]
    
    # Performance projections
    expected_return_unhedged: float
    expected_return_hedged: float
    risk_adjusted_return_improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "timestamp": self.timestamp.isoformat(),
            "current_risk_metrics": self.current_risk_metrics.to_dict(),
            "recommended_hedges": [h.to_dict() for h in self.recommended_hedges],
            "optimal_hedge_combination": self.optimal_hedge_combination,
            "optimized_weights": self.optimized_weights,
            "risk_budget_allocation": self.risk_budget_allocation,
            "stress_test_results": self.stress_test_results,
            "tail_risk_analysis": self.tail_risk_analysis,
            "expected_return_unhedged": self.expected_return_unhedged,
            "expected_return_hedged": self.expected_return_hedged,
            "risk_adjusted_return_improvement": self.risk_adjusted_return_improvement
        }
