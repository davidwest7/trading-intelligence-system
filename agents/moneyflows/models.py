"""
Data models for Money Flows Agent

Institutional flow tracking and dark pool analysis models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class FlowType(str, Enum):
    INSTITUTIONAL = "institutional"
    RETAIL = "retail"
    DARK_POOL = "dark_pool"
    ALGORITHMIC = "algorithmic"
    MARKET_MAKER = "market_maker"


class FlowDirection(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NEUTRAL = "neutral"


class InstitutionType(str, Enum):
    PENSION_FUND = "pension_fund"
    MUTUAL_FUND = "mutual_fund"
    HEDGE_FUND = "hedge_fund"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    INSURANCE = "insurance"
    ENDOWMENT = "endowment"
    FOREIGN_INVESTOR = "foreign_investor"


@dataclass
class InstitutionalFlow:
    """Individual institutional flow data point"""
    timestamp: datetime
    ticker: str
    flow_type: FlowType
    direction: FlowDirection
    volume: float
    notional_value: float
    price_impact: float
    confidence: float
    estimated_institution_type: Optional[InstitutionType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "flow_type": self.flow_type.value,
            "direction": self.direction.value,
            "volume": self.volume,
            "notional_value": self.notional_value,
            "price_impact": self.price_impact,
            "confidence": self.confidence,
            "estimated_institution_type": self.estimated_institution_type.value if self.estimated_institution_type else None
        }


@dataclass
class DarkPoolActivity:
    """Dark pool trading activity analysis"""
    ticker: str
    timestamp: datetime
    dark_pool_volume: float
    lit_market_volume: float
    dark_pool_ratio: float  # Dark pool volume / total volume
    estimated_block_trades: int
    avg_trade_size: float
    volume_weighted_price: float
    participation_rate: float  # Dark pool volume / ADV
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "dark_pool_volume": self.dark_pool_volume,
            "lit_market_volume": self.lit_market_volume,
            "dark_pool_ratio": self.dark_pool_ratio,
            "estimated_block_trades": self.estimated_block_trades,
            "avg_trade_size": self.avg_trade_size,
            "volume_weighted_price": self.volume_weighted_price,
            "participation_rate": self.participation_rate
        }


@dataclass
class VolumeConcentration:
    """Volume concentration analysis"""
    ticker: str
    timestamp: datetime
    herfindahl_index: float  # Concentration measure
    top_5_venues_share: float  # Share of top 5 trading venues
    fragmentation_score: float  # Market fragmentation measure
    venue_breakdown: Dict[str, float]  # Venue -> volume share
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "herfindahl_index": self.herfindahl_index,
            "top_5_venues_share": self.top_5_venues_share,
            "fragmentation_score": self.fragmentation_score,
            "venue_breakdown": self.venue_breakdown
        }


@dataclass
class FlowPattern:
    """Detected flow pattern"""
    pattern_id: str
    pattern_type: str  # accumulation, distribution, rotation, etc.
    ticker: str
    start_time: datetime
    duration: timedelta
    strength: float  # 0-1 pattern strength
    confidence: float  # 0-1 confidence in detection
    associated_flows: List[InstitutionalFlow]
    key_characteristics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "ticker": self.ticker,
            "start_time": self.start_time.isoformat(),
            "duration": self.duration.total_seconds(),
            "strength": self.strength,
            "confidence": self.confidence,
            "flow_count": len(self.associated_flows),
            "key_characteristics": self.key_characteristics
        }


@dataclass
class MoneyFlowAnalysis:
    """Complete money flow analysis result"""
    ticker: str
    timestamp: datetime
    analysis_period: timedelta
    
    # Flow summary
    total_institutional_inflow: float
    total_institutional_outflow: float
    net_institutional_flow: float
    retail_flow_estimate: float
    
    # Dark pool analysis
    dark_pool_activity: DarkPoolActivity
    
    # Volume analysis
    volume_concentration: VolumeConcentration
    unusual_volume_detected: bool
    volume_anomaly_score: float
    
    # Pattern detection
    detected_patterns: List[FlowPattern]
    dominant_flow_type: FlowType
    
    # Institution analysis
    institution_breakdown: Dict[str, float]  # Institution type -> flow
    foreign_flow_estimate: float
    
    # Signals
    accumulation_score: float  # -1 to 1, 1 = strong accumulation
    distribution_score: float  # -1 to 1, 1 = strong distribution
    rotation_signal: float  # -1 to 1, sector/style rotation
    
    # Predictions
    short_term_flow_direction: FlowDirection
    flow_persistence_probability: float
    estimated_completion_time: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "analysis_period": self.analysis_period.total_seconds(),
            "total_institutional_inflow": self.total_institutional_inflow,
            "total_institutional_outflow": self.total_institutional_outflow,
            "net_institutional_flow": self.net_institutional_flow,
            "retail_flow_estimate": self.retail_flow_estimate,
            "dark_pool_activity": self.dark_pool_activity.to_dict(),
            "volume_concentration": self.volume_concentration.to_dict(),
            "unusual_volume_detected": self.unusual_volume_detected,
            "volume_anomaly_score": self.volume_anomaly_score,
            "detected_patterns": [p.to_dict() for p in self.detected_patterns],
            "dominant_flow_type": self.dominant_flow_type.value,
            "institution_breakdown": self.institution_breakdown,
            "foreign_flow_estimate": self.foreign_flow_estimate,
            "accumulation_score": self.accumulation_score,
            "distribution_score": self.distribution_score,
            "rotation_signal": self.rotation_signal,
            "short_term_flow_direction": self.short_term_flow_direction.value,
            "flow_persistence_probability": self.flow_persistence_probability,
            "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None
        }


@dataclass
class MoneyFlowRequest:
    """Request for money flow analysis"""
    tickers: List[str]
    analysis_period: str = "1d"  # 1h, 4h, 1d, 1w
    include_dark_pools: bool = True
    include_institution_breakdown: bool = True
    min_confidence: float = 0.6
    detect_patterns: bool = True
    
    def validate(self) -> bool:
        """Validate request parameters"""
        valid_periods = ["1h", "4h", "1d", "1w"]
        
        return (
            len(self.tickers) > 0 and
            self.analysis_period in valid_periods and
            0.0 <= self.min_confidence <= 1.0
        )
