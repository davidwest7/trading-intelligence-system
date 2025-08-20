"""
Message Contracts for Optimized Trading System

Defines the core data structures for inter-service communication
with versioning, traceability, and uncertainty quantification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator
import numpy as np


class SignalType(str, Enum):
    """Types of trading signals"""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FLOW = "flow"
    MACRO = "macro"
    MONEY_FLOWS = "money_flows"
    UNDERVALUED = "undervalued"
    INSIDER = "insider"
    CAUSAL = "causal"
    HEDGING = "hedging"
    LEARNING = "learning"
    TOP_PERFORMERS = "top_performers"
    VALUE = "value"


class RegimeType(str, Enum):
    """Market regime types"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    LIQUID = "liquid"
    ILLIQUID = "illiquid"


class HorizonType(str, Enum):
    """Time horizons for signals"""
    INTRADAY = "intraday"  # Minutes to hours
    SHORT_TERM = "short_term"  # 1-5 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months


class DirectionType(str, Enum):
    """Signal directions"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Signal(BaseModel):
    """
    Standardized signal from any agent with uncertainty quantification
    """
    # Core identification
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(..., description="Trace ID for end-to-end tracking")
    agent_id: str = Field(..., description="Agent that generated this signal")
    agent_type: SignalType = Field(..., description="Type of agent")
    
    # Asset information
    symbol: str = Field(..., description="Trading symbol")
    exchange: Optional[str] = Field(None, description="Exchange")
    asset_class: Optional[str] = Field(None, description="Asset class")
    
    # Signal values with uncertainty
    mu: float = Field(..., description="Expected return (mean)")
    sigma: float = Field(..., description="Uncertainty (standard deviation)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent confidence [0,1]")
    
    # Time and regime information
    horizon: HorizonType = Field(..., description="Time horizon")
    regime: RegimeType = Field(..., description="Market regime")
    direction: DirectionType = Field(..., description="Signal direction")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(..., description="Model version hash")
    feature_version: str = Field(..., description="Feature version hash")
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Schema versioning
    schema_version: str = Field(default="1.0.0")
    
    @validator('mu', 'sigma')
    def validate_numeric_values(cls, v):
        """Validate numeric values are finite"""
        if not np.isfinite(v):
            raise ValueError(f"Value must be finite, got {v}")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is in [0,1]"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create from dictionary"""
        return cls(**data)


class Opportunity(BaseModel):
    """
    Trading opportunity with blended signals and risk metrics
    """
    # Core identification
    opportunity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(..., description="Trace ID for end-to-end tracking")
    
    # Asset information
    symbol: str = Field(..., description="Trading symbol")
    exchange: Optional[str] = Field(None, description="Exchange")
    asset_class: Optional[str] = Field(None, description="Asset class")
    
    # Blended signal values
    mu_blended: float = Field(..., description="Blended expected return")
    sigma_blended: float = Field(..., description="Blended uncertainty")
    confidence_blended: float = Field(..., ge=0.0, le=1.0, description="Blended confidence")
    
    # Risk metrics
    var_95: Optional[float] = Field(None, description="95% Value at Risk")
    cvar_95: Optional[float] = Field(None, description="95% Conditional Value at Risk")
    sharpe_ratio: Optional[float] = Field(None, description="Risk-adjusted return")
    
    # Agent contributions
    agent_signals: Dict[str, Signal] = Field(default_factory=dict)
    agent_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Time and regime
    horizon: HorizonType = Field(..., description="Time horizon")
    regime: RegimeType = Field(..., description="Market regime")
    direction: DirectionType = Field(..., description="Opportunity direction")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    blender_version: str = Field(..., description="Meta-weighter version")
    
    # Schema versioning
    schema_version: str = Field(default="1.0.0")
    
    @validator('mu_blended', 'sigma_blended')
    def validate_numeric_values(cls, v):
        """Validate numeric values are finite"""
        if not np.isfinite(v):
            raise ValueError(f"Value must be finite, got {v}")
        return v
    
    @validator('confidence_blended')
    def validate_confidence(cls, v):
        """Validate confidence is in [0,1]"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Opportunity':
        """Create from dictionary"""
        return cls(**data)


class Intent(BaseModel):
    """
    Trading intent with position sizing and risk constraints
    """
    # Core identification
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(..., description="Trace ID for end-to-end tracking")
    opportunity_id: str = Field(..., description="Source opportunity ID")
    
    # Asset information
    symbol: str = Field(..., description="Trading symbol")
    exchange: Optional[str] = Field(None, description="Exchange")
    
    # Position sizing
    direction: DirectionType = Field(..., description="Trade direction")
    size_eur: float = Field(..., gt=0, description="Position size in EUR")
    size_shares: Optional[int] = Field(None, description="Position size in shares")
    
    # Risk metrics
    risk_eur: float = Field(..., description="Risk amount in EUR")
    risk_pct: float = Field(..., ge=0.0, le=1.0, description="Risk as % of account")
    var_95: float = Field(..., description="95% Value at Risk")
    cvar_95: float = Field(..., description="95% Conditional Value at Risk")
    
    # Constraints
    max_position_size: float = Field(..., description="Maximum position size")
    max_risk_per_trade: float = Field(..., description="Maximum risk per trade")
    
    # Execution parameters
    urgency: str = Field(default="normal", description="Execution urgency")
    order_type: str = Field(default="market", description="Order type")
    venue: Optional[str] = Field(None, description="Preferred venue")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sizer_version: str = Field(..., description="RL sizer version")
    risk_version: str = Field(..., description="Risk manager version")
    
    # Schema versioning
    schema_version: str = Field(default="1.0.0")
    
    @validator('size_eur', 'risk_eur', 'var_95', 'cvar_95')
    def validate_positive_values(cls, v):
        """Validate positive values"""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v
    
    @validator('risk_pct')
    def validate_risk_percentage(cls, v):
        """Validate risk percentage is in [0,1]"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Risk percentage must be in [0,1], got {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """Create from dictionary"""
        return cls(**data)


class DecisionLog(BaseModel):
    """
    Complete decision log for auditability and learning
    """
    # Core identification
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(..., description="Trace ID for end-to-end tracking")
    
    # Decision components
    signals: List[Signal] = Field(default_factory=list)
    opportunities: List[Opportunity] = Field(default_factory=list)
    intents: List[Intent] = Field(default_factory=list)
    
    # Execution results
    fills: List[Dict[str, Any]] = Field(default_factory=list)
    slippage: Optional[float] = Field(None, description="Total slippage")
    execution_cost: Optional[float] = Field(None, description="Execution cost")
    
    # Performance metrics
    pnl: Optional[float] = Field(None, description="Realized P&L")
    pnl_pct: Optional[float] = Field(None, description="P&L as percentage")
    
    # Risk metrics
    portfolio_risk: Optional[float] = Field(None, description="Portfolio risk")
    drawdown: Optional[float] = Field(None, description="Current drawdown")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_latency_ms: Optional[float] = Field(None, description="Decision latency")
    
    # Schema versioning
    schema_version: str = Field(default="1.0.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionLog':
        """Create from dictionary"""
        return cls(**data)


# Utility functions for contract validation
def validate_signal_contract(signal_data: Dict[str, Any]) -> bool:
    """Validate signal contract"""
    try:
        Signal(**signal_data)
        return True
    except Exception as e:
        print(f"Signal validation failed: {e}")
        return False


def validate_opportunity_contract(opportunity_data: Dict[str, Any]) -> bool:
    """Validate opportunity contract"""
    try:
        Opportunity(**opportunity_data)
        return True
    except Exception as e:
        print(f"Opportunity validation failed: {e}")
        return False


def validate_intent_contract(intent_data: Dict[str, Any]) -> bool:
    """Validate intent contract"""
    try:
        Intent(**intent_data)
        return True
    except Exception as e:
        print(f"Intent validation failed: {e}")
        return False


def validate_decision_log_contract(decision_data: Dict[str, Any]) -> bool:
    """Validate decision log contract"""
    try:
        DecisionLog(**decision_data)
        return True
    except Exception as e:
        print(f"Decision log validation failed: {e}")
        return False


# Schema registry for versioning
SCHEMA_REGISTRY = {
    "Signal": {
        "1.0.0": Signal,
    },
    "Opportunity": {
        "1.0.0": Opportunity,
    },
    "Intent": {
        "1.0.0": Intent,
    },
    "DecisionLog": {
        "1.0.0": DecisionLog,
    }
}


def get_schema_class(schema_name: str, version: str = "1.0.0"):
    """Get schema class by name and version"""
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {schema_name}")
    
    if version not in SCHEMA_REGISTRY[schema_name]:
        raise ValueError(f"Unknown version {version} for schema {schema_name}")
    
    return SCHEMA_REGISTRY[schema_name][version]


def list_available_schemas() -> Dict[str, List[str]]:
    """List all available schemas and versions"""
    return {
        schema_name: list(versions.keys())
        for schema_name, versions in SCHEMA_REGISTRY.items()
    }
