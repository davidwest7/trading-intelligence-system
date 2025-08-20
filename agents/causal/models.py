"""
Data models for Causal Impact Agent

Event studies and statistical inference models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class EventType(str, Enum):
    EARNINGS_ANNOUNCEMENT = "earnings_announcement"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY_APPROVAL = "regulatory_approval"
    MANAGEMENT_CHANGE = "management_change"
    DIVIDEND_ANNOUNCEMENT = "dividend_announcement"
    GUIDANCE_UPDATE = "guidance_update"
    LEGAL_SETTLEMENT = "legal_settlement"
    PARTNERSHIP_DEAL = "partnership_deal"
    SPLIT_SPINOFF = "split_spinoff"
    OTHER = "other"


class CausalMethod(str, Enum):
    EVENT_STUDY = "event_study"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    SYNTHETIC_CONTROL = "synthetic_control"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"


@dataclass
class CausalEvent:
    """Event for causal analysis"""
    event_id: str
    ticker: str
    event_date: datetime
    event_type: EventType
    event_description: str
    
    # Event characteristics
    expected_impact: Optional[float]  # Expected price impact
    magnitude: float  # Event magnitude (0-1 scale)
    surprise_factor: float  # How unexpected was the event
    
    # Market context
    market_cap_at_event: float
    sector: str
    trading_volume_ratio: float  # Volume vs normal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "event_date": self.event_date.isoformat(),
            "event_type": self.event_type.value,
            "event_description": self.event_description,
            "expected_impact": self.expected_impact,
            "magnitude": self.magnitude,
            "surprise_factor": self.surprise_factor,
            "market_cap_at_event": self.market_cap_at_event,
            "sector": self.sector,
            "trading_volume_ratio": self.trading_volume_ratio
        }


@dataclass
class EventStudyResult:
    """Results from event study analysis"""
    event: CausalEvent
    
    # Study parameters
    estimation_window: int  # Days for estimation
    event_window: List[int]  # Event window (e.g., [-1, 1])
    
    # Abnormal returns
    cumulative_abnormal_return: float
    abnormal_returns: List[float]  # Daily abnormal returns
    
    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: List[float]
    
    # Performance metrics
    r_squared: float
    prediction_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "estimation_window": self.estimation_window,
            "event_window": self.event_window,
            "cumulative_abnormal_return": self.cumulative_abnormal_return,
            "abnormal_returns": self.abnormal_returns,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval,
            "r_squared": self.r_squared,
            "prediction_error": self.prediction_error
        }


@dataclass
class ImpactMeasurement:
    """Impact measurement for causal analysis"""
    ticker: str
    event_id: str
    impact_value: float
    confidence_level: float
    measurement_date: datetime
    impact_type: str
    statistical_significance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "event_id": self.event_id,
            "impact_value": self.impact_value,
            "confidence_level": self.confidence_level,
            "measurement_date": self.measurement_date.isoformat(),
            "impact_type": self.impact_type,
            "statistical_significance": self.statistical_significance
        }

@dataclass
class CausalAnalysis:
    """Complete causal impact analysis"""
    ticker: str
    timestamp: datetime
    analysis_period: timedelta
    method: CausalMethod
    
    # Events analyzed
    analyzed_events: List[EventStudyResult]
    significant_events: List[EventStudyResult]
    
    # Aggregate results
    avg_event_impact: float
    impact_volatility: float
    success_rate: float  # % of events with expected direction
    
    # Pattern analysis
    seasonal_patterns: Dict[str, float]
    event_type_impacts: Dict[str, float]
    
    # Predictive insights
    predicted_next_event_impact: Optional[float]
    confidence_in_prediction: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "analysis_period": self.analysis_period.total_seconds(),
            "method": self.method.value,
            "analyzed_events": [e.to_dict() for e in self.analyzed_events],
            "significant_events": [e.to_dict() for e in self.significant_events],
            "avg_event_impact": self.avg_event_impact,
            "impact_volatility": self.impact_volatility,
            "success_rate": self.success_rate,
            "seasonal_patterns": self.seasonal_patterns,
            "event_type_impacts": self.event_type_impacts,
            "predicted_next_event_impact": self.predicted_next_event_impact,
            "confidence_in_prediction": self.confidence_in_prediction
        }
