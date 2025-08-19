"""
Data models for Technical Strategy Agent
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class MarketRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class ImbalanceZone:
    """Imbalance/FVG zone definition"""
    start_price: float
    end_price: float
    timeframe: str
    strength: float  # 0-1
    volume_imbalance: float
    timestamp: datetime
    filled: bool = False


@dataclass
class LiquidityLevel:
    """Support/Resistance liquidity level"""
    price: float
    strength: float  # 0-1
    volume: float
    touches: int
    timeframe: str
    level_type: Literal["support", "resistance", "order_block"]
    last_test: datetime


@dataclass
class TechnicalFeatures:
    """Technical analysis features for an asset"""
    imbalance_zones: List[ImbalanceZone] = field(default_factory=list)
    fair_value_gaps: List[ImbalanceZone] = field(default_factory=list)
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)
    order_blocks: List[LiquidityLevel] = field(default_factory=list)
    trend_strength: float = 0.0
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "imbalance_zones": [
                {
                    "start_price": zone.start_price,
                    "end_price": zone.end_price,
                    "timeframe": zone.timeframe,
                    "strength": zone.strength,
                    "volume_imbalance": zone.volume_imbalance,
                    "filled": zone.filled
                }
                for zone in self.imbalance_zones
            ],
            "fair_value_gaps": [
                {
                    "start_price": gap.start_price,
                    "end_price": gap.end_price,
                    "timeframe": gap.timeframe,
                    "strength": gap.strength
                }
                for gap in self.fair_value_gaps
            ],
            "liquidity_levels": [
                {
                    "price": level.price,
                    "strength": level.strength,
                    "volume": level.volume,
                    "touches": level.touches,
                    "level_type": level.level_type
                }
                for level in self.liquidity_levels
            ],
            "order_blocks": [
                {
                    "price": block.price,
                    "strength": block.strength,
                    "level_type": block.level_type
                }
                for block in self.order_blocks
            ],
            "trend_strength": self.trend_strength,
            "volatility_regime": self.volatility_regime.value
        }


@dataclass
class RiskMetrics:
    """Risk metrics for a trading opportunity"""
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float
    win_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate
        }


@dataclass
class TimeframeAlignment:
    """Multi-timeframe alignment analysis"""
    primary: str
    confirmation: List[str]
    alignment_score: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary,
            "confirmation": self.confirmation,
            "alignment_score": self.alignment_score
        }


@dataclass
class TechnicalOpportunity:
    """A trading opportunity identified by technical analysis"""
    symbol: str
    strategy: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    risk_reward_ratio: float
    confidence_score: float  # 0-1
    timeframe_alignment: TimeframeAlignment
    technical_features: TechnicalFeatures
    risk_metrics: RiskMetrics
    timestamp: datetime
    expiry: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": f"{self.symbol}_{self.strategy}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "confidence_score": self.confidence_score,
            "timeframe_alignment": self.timeframe_alignment.to_dict(),
            "technical_features": self.technical_features.to_dict(),
            "risk_metrics": self.risk_metrics.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None
        }


@dataclass
class AnalysisPayload:
    """Input payload for technical analysis"""
    symbols: List[str]
    timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    strategies: List[str] = field(default_factory=lambda: ["imbalance", "fvg", "liquidity_sweep", "trend"])
    min_score: float = 0.6
    max_risk: float = 0.02
    lookback_periods: int = 200


@dataclass
class AnalysisMetadata:
    """Metadata about the analysis performed"""
    analysis_time_ms: int
    symbols_analyzed: int
    opportunities_found: int
    market_regime: MarketRegime
    overall_bias: Literal["bullish", "bearish", "neutral"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_time_ms": self.analysis_time_ms,
            "symbols_analyzed": self.symbols_analyzed,
            "opportunities_found": self.opportunities_found,
            "market_regime": self.market_regime.value,
            "overall_bias": self.overall_bias
        }
