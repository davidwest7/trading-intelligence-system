"""
Data models for Macro/Geopolitical Analysis Agent

Economic and geopolitical analysis models for market intelligence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class EconomicIndicatorType(str, Enum):
    GDP = "gdp"
    INFLATION = "inflation"
    UNEMPLOYMENT = "unemployment"
    INTEREST_RATES = "interest_rates"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    MANUFACTURING_PMI = "manufacturing_pmi"
    SERVICES_PMI = "services_pmi"
    RETAIL_SALES = "retail_sales"
    HOUSING_DATA = "housing_data"
    TRADE_BALANCE = "trade_balance"


class GeopoliticalEventType(str, Enum):
    MILITARY_CONFLICT = "military_conflict"
    TRADE_WAR = "trade_war"
    SANCTIONS = "sanctions"
    ELECTIONS = "elections"
    POLICY_CHANGE = "policy_change"
    CENTRAL_BANK_ACTION = "central_bank_action"
    COMMODITY_DISRUPTION = "commodity_disruption"
    CYBER_ATTACK = "cyber_attack"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"


class ImpactSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketImpact(str, Enum):
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class Region(str, Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    GLOBAL = "global"


@dataclass
class EconomicIndicator:
    """Economic indicator data point"""
    indicator_type: EconomicIndicatorType
    country: str
    value: float
    previous_value: Optional[float]
    forecast: Optional[float]
    release_date: datetime
    next_release: Optional[datetime]
    importance: ImpactSeverity
    currency_impact: MarketImpact
    equity_impact: MarketImpact
    bond_impact: MarketImpact
    
    @property
    def surprise_factor(self) -> float:
        """Calculate surprise factor vs forecast"""
        if self.forecast is None:
            return 0.0
        
        if abs(self.forecast) < 1e-6:  # Avoid division by zero
            return 0.0
            
        return (self.value - self.forecast) / abs(self.forecast)
    
    @property
    def momentum(self) -> float:
        """Calculate momentum vs previous value"""
        if self.previous_value is None:
            return 0.0
            
        if abs(self.previous_value) < 1e-6:
            return 0.0
            
        return (self.value - self.previous_value) / abs(self.previous_value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_type": self.indicator_type.value,
            "country": self.country,
            "value": self.value,
            "previous_value": self.previous_value,
            "forecast": self.forecast,
            "release_date": self.release_date.isoformat(),
            "next_release": self.next_release.isoformat() if self.next_release else None,
            "importance": self.importance.value,
            "currency_impact": self.currency_impact.value,
            "equity_impact": self.equity_impact.value,
            "bond_impact": self.bond_impact.value,
            "surprise_factor": self.surprise_factor,
            "momentum": self.momentum
        }


@dataclass
class GeopoliticalEvent:
    """Geopolitical event with market impact assessment"""
    event_id: str
    event_type: GeopoliticalEventType
    title: str
    description: str
    region: Region
    countries_involved: List[str]
    start_date: datetime
    estimated_duration: Optional[timedelta]
    severity: ImpactSeverity
    probability: float  # 0-1 probability of occurrence/escalation
    
    # Market impact assessments
    equity_impact: MarketImpact
    currency_impact: MarketImpact
    commodity_impact: MarketImpact
    bond_impact: MarketImpact
    
    # Affected sectors/assets
    affected_sectors: List[str]
    affected_currencies: List[str]
    affected_commodities: List[str]
    
    # Confidence and sources
    confidence: float  # 0-1 confidence in assessment
    sources: List[str]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "region": self.region.value,
            "countries_involved": self.countries_involved,
            "start_date": self.start_date.isoformat(),
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None,
            "severity": self.severity.value,
            "probability": self.probability,
            "equity_impact": self.equity_impact.value,
            "currency_impact": self.currency_impact.value,
            "commodity_impact": self.commodity_impact.value,
            "bond_impact": self.bond_impact.value,
            "affected_sectors": self.affected_sectors,
            "affected_currencies": self.affected_currencies,
            "affected_commodities": self.affected_commodities,
            "confidence": self.confidence,
            "sources": self.sources,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class CentralBankAction:
    """Central bank action/communication"""
    bank: str  # Federal Reserve, ECB, BOJ, etc.
    action_type: str  # rate_decision, qe, guidance, speech
    date: datetime
    description: str
    key_points: List[str]
    
    # Rate information
    current_rate: Optional[float]
    previous_rate: Optional[float]
    rate_change: Optional[float]
    
    # Market expectations
    market_expectation: Optional[float]
    surprise_factor: float
    
    # Impact assessment
    market_impact: MarketImpact
    currency_strength: float  # -1 to 1, -1 = very bearish for currency
    bond_yield_impact: float  # Expected basis points change
    
    hawkish_dovish_score: float  # -1 (very dovish) to 1 (very hawkish)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bank": self.bank,
            "action_type": self.action_type,
            "date": self.date.isoformat(),
            "description": self.description,
            "key_points": self.key_points,
            "current_rate": self.current_rate,
            "previous_rate": self.previous_rate,
            "rate_change": self.rate_change,
            "market_expectation": self.market_expectation,
            "surprise_factor": self.surprise_factor,
            "market_impact": self.market_impact.value,
            "currency_strength": self.currency_strength,
            "bond_yield_impact": self.bond_yield_impact,
            "hawkish_dovish_score": self.hawkish_dovish_score
        }


@dataclass
class MacroTheme:
    """Major macro theme affecting markets"""
    theme_id: str
    name: str
    description: str
    start_date: datetime
    current_phase: str  # emerging, developing, mature, declining
    
    # Theme strength and momentum
    strength: float  # 0-1 how strong the theme is
    momentum: float  # -1 to 1, momentum of the theme
    
    # Market relevance
    equity_relevance: float  # 0-1
    fx_relevance: float      # 0-1
    commodity_relevance: float  # 0-1
    bond_relevance: float    # 0-1
    
    # Related events and indicators
    related_events: List[str]  # Event IDs
    key_indicators: List[str]  # Indicator types to monitor
    
    # Time horizon
    expected_duration: Optional[timedelta]
    key_catalysts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "current_phase": self.current_phase,
            "strength": self.strength,
            "momentum": self.momentum,
            "equity_relevance": self.equity_relevance,
            "fx_relevance": self.fx_relevance,
            "commodity_relevance": self.commodity_relevance,
            "bond_relevance": self.bond_relevance,
            "related_events": self.related_events,
            "key_indicators": self.key_indicators,
            "expected_duration": self.expected_duration.total_seconds() if self.expected_duration else None,
            "key_catalysts": self.key_catalysts
        }


@dataclass
class MacroAnalysis:
    """Complete macro/geopolitical analysis result"""
    timestamp: datetime
    analysis_horizon: str  # short_term, medium_term, long_term
    
    # Economic environment
    global_growth_outlook: MarketImpact
    inflation_environment: str  # deflationary, low, moderate, high, hyperinflation
    interest_rate_cycle: str   # cutting, neutral, hiking
    
    # Key indicators
    recent_indicators: List[EconomicIndicator]
    upcoming_indicators: List[EconomicIndicator]
    
    # Geopolitical environment
    active_events: List[GeopoliticalEvent]
    emerging_risks: List[GeopoliticalEvent]
    
    # Central bank activity
    recent_cb_actions: List[CentralBankAction]
    upcoming_cb_meetings: List[Dict[str, Any]]
    
    # Macro themes
    dominant_themes: List[MacroTheme]
    
    # Overall assessment
    risk_environment: ImpactSeverity  # Current global risk level
    market_regime: str  # risk_on, risk_off, neutral, transitioning
    
    # Currency outlook
    usd_strength_outlook: MarketImpact
    safe_haven_demand: float  # 0-1
    
    # Sector/asset impacts
    sector_impacts: Dict[str, MarketImpact]  # sector -> impact
    commodity_impacts: Dict[str, MarketImpact]  # commodity -> impact
    regional_impacts: Dict[str, MarketImpact]  # region -> impact
    
    # Confidence and risks
    analysis_confidence: float  # 0-1
    key_risks: List[str]
    key_opportunities: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "analysis_horizon": self.analysis_horizon,
            "global_growth_outlook": self.global_growth_outlook.value,
            "inflation_environment": self.inflation_environment,
            "interest_rate_cycle": self.interest_rate_cycle,
            "recent_indicators": [ind.to_dict() for ind in self.recent_indicators],
            "upcoming_indicators": [ind.to_dict() for ind in self.upcoming_indicators],
            "active_events": [event.to_dict() for event in self.active_events],
            "emerging_risks": [event.to_dict() for event in self.emerging_risks],
            "recent_cb_actions": [action.to_dict() for action in self.recent_cb_actions],
            "upcoming_cb_meetings": self.upcoming_cb_meetings,
            "dominant_themes": [theme.to_dict() for theme in self.dominant_themes],
            "risk_environment": self.risk_environment.value,
            "market_regime": self.market_regime,
            "usd_strength_outlook": self.usd_strength_outlook.value,
            "safe_haven_demand": self.safe_haven_demand,
            "sector_impacts": {k: v.value for k, v in self.sector_impacts.items()},
            "commodity_impacts": {k: v.value for k, v in self.commodity_impacts.items()},
            "regional_impacts": {k: v.value for k, v in self.regional_impacts.items()},
            "analysis_confidence": self.analysis_confidence,
            "key_risks": self.key_risks,
            "key_opportunities": self.key_opportunities
        }


@dataclass
class MacroRequest:
    """Request for macro analysis"""
    analysis_horizon: str = "medium_term"  # short_term, medium_term, long_term
    regions: List[str] = field(default_factory=lambda: ["global"])
    include_geopolitical: bool = True
    include_central_banks: bool = True
    include_economic_calendar: bool = True
    lookback_days: int = 30
    lookahead_days: int = 60
    min_importance: str = "medium"  # low, medium, high, critical
    
    def validate(self) -> bool:
        """Validate request parameters"""
        valid_horizons = ["short_term", "medium_term", "long_term"]
        valid_importance = ["low", "medium", "high", "critical"]
        
        return (
            self.analysis_horizon in valid_horizons and
            self.min_importance in valid_importance and
            self.lookback_days > 0 and
            self.lookahead_days > 0
        )
