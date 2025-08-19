"""
Data models for Top Performers Agent

Momentum analysis and performance attribution models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class PerformanceMetric(str, Enum):
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    SHARPE_RATIO = "sharpe_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"


class MomentumType(str, Enum):
    PRICE_MOMENTUM = "price_momentum"
    EARNINGS_MOMENTUM = "earnings_momentum"
    REVISION_MOMENTUM = "revision_momentum"
    TECHNICAL_MOMENTUM = "technical_momentum"


class SectorRotationType(str, Enum):
    GROWTH_TO_VALUE = "growth_to_value"
    VALUE_TO_GROWTH = "value_to_growth"
    CYCLICAL_TO_DEFENSIVE = "cyclical_to_defensive"
    DEFENSIVE_TO_CYCLICAL = "defensive_to_cyclical"
    LARGE_TO_SMALL = "large_to_small"
    SMALL_TO_LARGE = "small_to_large"


@dataclass
class PerformanceData:
    """Individual performance data point"""
    ticker: str
    timestamp: datetime
    period: str  # 1d, 1w, 1m, 3m, 6m, 1y
    total_return: float
    risk_adjusted_return: float
    sharpe_ratio: float
    alpha: float
    beta: float
    volatility: float
    max_drawdown: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "period": self.period,
            "total_return": self.total_return,
            "risk_adjusted_return": self.risk_adjusted_return,
            "sharpe_ratio": self.sharpe_ratio,
            "alpha": self.alpha,
            "beta": self.beta,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown
        }


@dataclass
class MomentumScore:
    """Momentum scoring across different factors"""
    ticker: str
    timestamp: datetime
    
    # Different momentum types
    price_momentum_1m: float
    price_momentum_3m: float
    price_momentum_6m: float
    price_momentum_12m: float
    
    earnings_momentum: float
    revision_momentum: float
    technical_momentum: float
    
    # Combined momentum score
    composite_momentum: float
    momentum_persistence: float
    momentum_acceleration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "price_momentum_1m": self.price_momentum_1m,
            "price_momentum_3m": self.price_momentum_3m,
            "price_momentum_6m": self.price_momentum_6m,
            "price_momentum_12m": self.price_momentum_12m,
            "earnings_momentum": self.earnings_momentum,
            "revision_momentum": self.revision_momentum,
            "technical_momentum": self.technical_momentum,
            "composite_momentum": self.composite_momentum,
            "momentum_persistence": self.momentum_persistence,
            "momentum_acceleration": self.momentum_acceleration
        }


@dataclass
class SectorPerformance:
    """Sector performance analysis"""
    sector: str
    timestamp: datetime
    
    # Performance metrics
    sector_return_1w: float
    sector_return_1m: float
    sector_return_3m: float
    sector_return_ytd: float
    
    # Relative performance
    relative_to_market_1w: float
    relative_to_market_1m: float
    relative_to_market_3m: float
    
    # Sector characteristics
    avg_pe_ratio: float
    avg_dividend_yield: float
    avg_market_cap: float
    momentum_score: float
    
    # Top performers in sector
    top_performers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector": self.sector,
            "timestamp": self.timestamp.isoformat(),
            "sector_return_1w": self.sector_return_1w,
            "sector_return_1m": self.sector_return_1m,
            "sector_return_3m": self.sector_return_3m,
            "sector_return_ytd": self.sector_return_ytd,
            "relative_to_market_1w": self.relative_to_market_1w,
            "relative_to_market_1m": self.relative_to_market_1m,
            "relative_to_market_3m": self.relative_to_market_3m,
            "avg_pe_ratio": self.avg_pe_ratio,
            "avg_dividend_yield": self.avg_dividend_yield,
            "avg_market_cap": self.avg_market_cap,
            "momentum_score": self.momentum_score,
            "top_performers": self.top_performers
        }


@dataclass
class PerformanceAttribution:
    """Performance attribution analysis"""
    ticker: str
    timestamp: datetime
    period: str
    
    # Factor contributions
    market_factor: float
    sector_factor: float
    style_factor: float
    size_factor: float
    momentum_factor: float
    quality_factor: float
    value_factor: float
    
    # Alpha breakdown
    stock_specific_alpha: float
    explained_alpha: float
    unexplained_alpha: float
    
    # Risk attribution
    systematic_risk: float
    idiosyncratic_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "period": self.period,
            "market_factor": self.market_factor,
            "sector_factor": self.sector_factor,
            "style_factor": self.style_factor,
            "size_factor": self.size_factor,
            "momentum_factor": self.momentum_factor,
            "quality_factor": self.quality_factor,
            "value_factor": self.value_factor,
            "stock_specific_alpha": self.stock_specific_alpha,
            "explained_alpha": self.explained_alpha,
            "unexplained_alpha": self.unexplained_alpha,
            "systematic_risk": self.systematic_risk,
            "idiosyncratic_risk": self.idiosyncratic_risk
        }


@dataclass
class TopPerformersAnalysis:
    """Complete top performers analysis result"""
    timestamp: datetime
    analysis_period: str
    universe_size: int
    
    # Top performers by different metrics
    top_by_return: List[PerformanceData]
    top_by_risk_adjusted: List[PerformanceData]
    top_by_momentum: List[MomentumScore]
    top_by_alpha: List[PerformanceData]
    
    # Sector analysis
    sector_performance: List[SectorPerformance]
    sector_rotation_detected: Optional[SectorRotationType]
    sector_rotation_strength: float
    
    # Performance attribution
    performance_attributions: List[PerformanceAttribution]
    
    # Market insights
    market_momentum: float
    dispersion: float  # Cross-sectional dispersion
    concentration: float  # Performance concentration
    
    # Style analysis
    growth_vs_value: float  # Positive = growth outperforming
    large_vs_small: float   # Positive = large cap outperforming
    momentum_factor_performance: float
    
    # Predictions
    momentum_continuation_probability: float
    sector_rotation_probability: float
    expected_persistence_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "analysis_period": self.analysis_period,
            "universe_size": self.universe_size,
            "top_by_return": [p.to_dict() for p in self.top_by_return],
            "top_by_risk_adjusted": [p.to_dict() for p in self.top_by_risk_adjusted],
            "top_by_momentum": [m.to_dict() for m in self.top_by_momentum],
            "top_by_alpha": [p.to_dict() for p in self.top_by_alpha],
            "sector_performance": [s.to_dict() for s in self.sector_performance],
            "sector_rotation_detected": self.sector_rotation_detected.value if self.sector_rotation_detected else None,
            "sector_rotation_strength": self.sector_rotation_strength,
            "performance_attributions": [a.to_dict() for a in self.performance_attributions],
            "market_momentum": self.market_momentum,
            "dispersion": self.dispersion,
            "concentration": self.concentration,
            "growth_vs_value": self.growth_vs_value,
            "large_vs_small": self.large_vs_small,
            "momentum_factor_performance": self.momentum_factor_performance,
            "momentum_continuation_probability": self.momentum_continuation_probability,
            "sector_rotation_probability": self.sector_rotation_probability,
            "expected_persistence_days": self.expected_persistence_days
        }


@dataclass
class TopPerformersRequest:
    """Request for top performers analysis"""
    universe: List[str] = field(default_factory=list)  # Specific tickers or empty for broad universe
    analysis_period: str = "1m"  # 1w, 1m, 3m, 6m, 1y
    num_top_performers: int = 20
    include_sector_analysis: bool = True
    include_attribution: bool = True
    min_market_cap: float = 1e9  # $1B minimum
    exclude_recent_ipos: bool = True
    
    def validate(self) -> bool:
        """Validate request parameters"""
        valid_periods = ["1w", "1m", "3m", "6m", "1y"]
        
        return (
            self.analysis_period in valid_periods and
            self.num_top_performers > 0 and
            self.min_market_cap >= 0
        )
