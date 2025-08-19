"""
Data models for Direction-of-Flow (DoF) Agent

Market microstructure and order flow analysis models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class FlowDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class RegimeType(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class VolumeProfile(str, Enum):
    BULLISH_VOLUME = "bullish_volume"      # Higher volume on up moves
    BEARISH_VOLUME = "bearish_volume"      # Higher volume on down moves
    NEUTRAL_VOLUME = "neutral_volume"      # Balanced volume
    CLIMAX_VOLUME = "climax_volume"        # Exhaustion volume
    LOW_VOLUME = "low_volume"              # Below average volume


@dataclass
class OrderFlowMetrics:
    """Order flow microstructure metrics"""
    bid_ask_spread: float
    bid_size: float
    ask_size: float
    bid_ask_ratio: float
    market_impact: float           # Price impact per unit volume
    kyle_lambda: float             # Market depth parameter
    amihud_illiquidity: float      # Amihud illiquidity measure
    volume_weighted_spread: float
    effective_spread: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_ask_spread": self.bid_ask_spread,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "bid_ask_ratio": self.bid_ask_ratio,
            "market_impact": self.market_impact,
            "kyle_lambda": self.kyle_lambda,
            "amihud_illiquidity": self.amihud_illiquidity,
            "volume_weighted_spread": self.volume_weighted_spread,
            "effective_spread": self.effective_spread
        }


@dataclass
class VolumeProfileData:
    """Volume profile analysis data"""
    price_levels: List[float]
    volume_at_price: List[float]
    poc: float                     # Point of Control (highest volume)
    value_area_high: float         # 70% volume area high
    value_area_low: float          # 70% volume area low
    profile_type: VolumeProfile
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "price_levels": self.price_levels,
            "volume_at_price": self.volume_at_price,
            "poc": self.poc,
            "value_area_high": self.value_area_high,
            "value_area_low": self.value_area_low,
            "profile_type": self.profile_type.value
        }


@dataclass
class MoneyFlowData:
    """Money flow analysis data"""
    money_flow_index: float        # MFI oscillator
    accumulation_distribution: float  # A/D line value
    on_balance_volume: float       # OBV value
    volume_price_trend: float      # VPT indicator
    ease_of_movement: float        # EMV indicator
    chaikin_money_flow: float      # CMF indicator
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "money_flow_index": self.money_flow_index,
            "accumulation_distribution": self.accumulation_distribution,
            "on_balance_volume": self.on_balance_volume,
            "volume_price_trend": self.volume_price_trend,
            "ease_of_movement": self.ease_of_movement,
            "chaikin_money_flow": self.chaikin_money_flow
        }


@dataclass
class RegimeState:
    """Market regime state from HMM analysis"""
    regime_type: RegimeType
    probability: float             # Probability of being in this regime
    persistence: float             # Expected regime duration
    volatility: float              # Regime volatility
    mean_return: float             # Expected return in this regime
    transition_probabilities: Dict[str, float]  # Prob of transitioning to other regimes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_type": self.regime_type.value,
            "probability": self.probability,
            "persistence": self.persistence,
            "volatility": self.volatility,
            "mean_return": self.mean_return,
            "transition_probabilities": self.transition_probabilities
        }


@dataclass
class FlowSignal:
    """Individual flow signal"""
    signal_type: str               # Type of flow signal
    strength: float                # Signal strength (0-1)
    direction: FlowDirection
    timeframe: str
    timestamp: datetime
    confidence: float
    supporting_evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "strength": self.strength,
            "direction": self.direction.value,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence
        }


@dataclass
class FlowAnalysis:
    """Complete flow analysis result"""
    ticker: str
    timestamp: datetime
    overall_direction: FlowDirection
    flow_strength: float           # Overall flow strength (0-1)
    confidence: float              # Analysis confidence (0-1)
    
    # Regime analysis
    current_regime: RegimeState
    regime_stability: float        # How stable the current regime is
    
    # Order flow
    order_flow_metrics: OrderFlowMetrics
    net_flow: float                # Net buying/selling pressure
    flow_persistence: float        # How persistent the flow is
    
    # Volume analysis
    volume_profile: VolumeProfileData
    money_flow: MoneyFlowData
    volume_trend: str              # "increasing", "decreasing", "stable"
    
    # Signals
    flow_signals: List[FlowSignal]
    signal_consensus: float        # Agreement between signals (-1 to 1)
    
    # Predictions
    short_term_direction: FlowDirection  # Next 1-4 hours
    medium_term_direction: FlowDirection # Next 1-3 days
    flow_divergence: bool          # Price vs flow divergence detected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "overall_direction": self.overall_direction.value,
            "flow_strength": self.flow_strength,
            "confidence": self.confidence,
            "current_regime": self.current_regime.to_dict(),
            "regime_stability": self.regime_stability,
            "order_flow_metrics": self.order_flow_metrics.to_dict(),
            "net_flow": self.net_flow,
            "flow_persistence": self.flow_persistence,
            "volume_profile": self.volume_profile.to_dict(),
            "money_flow": self.money_flow.to_dict(),
            "volume_trend": self.volume_trend,
            "flow_signals": [signal.to_dict() for signal in self.flow_signals],
            "signal_consensus": self.signal_consensus,
            "short_term_direction": self.short_term_direction.value,
            "medium_term_direction": self.medium_term_direction.value,
            "flow_divergence": self.flow_divergence
        }


@dataclass
class FlowRequest:
    """Request for flow analysis"""
    tickers: List[str]
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    lookback_periods: int = 100
    include_microstructure: bool = True
    include_regime_analysis: bool = True
    min_confidence: float = 0.6
    
    def validate(self) -> bool:
        """Validate request parameters"""
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        return (
            len(self.tickers) > 0 and
            all(tf in valid_timeframes for tf in self.timeframes) and
            self.lookback_periods > 10 and
            0.0 <= self.min_confidence <= 1.0
        )


@dataclass
class MarketTick:
    """Individual market tick data"""
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    
    @property
    def is_uptick(self) -> Optional[bool]:
        """Determine if this is an uptick (requires comparison with previous tick)"""
        # This would be set by the processing logic
        return getattr(self, '_is_uptick', None)
    
    @is_uptick.setter  
    def is_uptick(self, value: bool):
        self._is_uptick = value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "is_uptick": self.is_uptick
        }


@dataclass
class FlowMetrics:
    """Aggregated flow metrics for a time period"""
    period_start: datetime
    period_end: datetime
    
    # Volume metrics
    total_volume: float
    buy_volume: float
    sell_volume: float
    neutral_volume: float
    
    # Flow metrics
    net_buying_pressure: float     # (buy_volume - sell_volume) / total_volume
    volume_weighted_price: float   # VWAP for the period
    price_volume_correlation: float
    
    # Tick metrics
    uptick_ratio: float            # Proportion of upticks
    downtick_ratio: float          # Proportion of downticks
    tick_imbalance: float          # Uptick ratio - downtick ratio
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_volume": self.total_volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "neutral_volume": self.neutral_volume,
            "net_buying_pressure": self.net_buying_pressure,
            "volume_weighted_price": self.volume_weighted_price,
            "price_volume_correlation": self.price_volume_correlation,
            "uptick_ratio": self.uptick_ratio,
            "downtick_ratio": self.downtick_ratio,
            "tick_imbalance": self.tick_imbalance
        }
