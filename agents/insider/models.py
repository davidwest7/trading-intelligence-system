"""
Data models for Insider Activity Agent

SEC filing analysis and insider sentiment models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    OPTION_EXERCISE = "option_exercise"
    GIFT = "gift"
    INHERITANCE = "inheritance"
    OTHER = "other"


class InsiderRole(str, Enum):
    CEO = "ceo"
    CFO = "cfo"
    COO = "coo"
    DIRECTOR = "director"
    PRESIDENT = "president"
    VP = "vice_president"
    OFFICER = "officer"
    MAJOR_SHAREHOLDER = "major_shareholder"
    OTHER = "other"


class FilingType(str, Enum):
    FORM_4 = "form_4"
    FORM_3 = "form_3"
    FORM_5 = "form_5"
    FORM_144 = "form_144"
    SCHEDULE_13D = "schedule_13d"
    SCHEDULE_13G = "schedule_13g"


class SentimentSignal(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class InsiderTransaction:
    """Individual insider transaction"""
    ticker: str
    filing_date: datetime
    transaction_date: datetime
    
    # Insider information
    insider_name: str
    insider_role: InsiderRole
    insider_cik: str  # SEC Central Index Key
    
    # Transaction details
    transaction_type: TransactionType
    shares_traded: float
    price_per_share: float
    total_value: float
    
    # Holdings information
    shares_owned_after: float
    ownership_percentage: float
    
    # Filing metadata
    filing_type: FilingType
    is_direct_holding: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "filing_date": self.filing_date.isoformat(),
            "transaction_date": self.transaction_date.isoformat(),
            "insider_name": self.insider_name,
            "insider_role": self.insider_role.value,
            "insider_cik": self.insider_cik,
            "transaction_type": self.transaction_type.value,
            "shares_traded": self.shares_traded,
            "price_per_share": self.price_per_share,
            "total_value": self.total_value,
            "shares_owned_after": self.shares_owned_after,
            "ownership_percentage": self.ownership_percentage,
            "filing_type": self.filing_type.value,
            "is_direct_holding": self.is_direct_holding
        }


@dataclass
class InsiderProfile:
    """Insider profile and track record"""
    insider_name: str
    insider_cik: str
    current_roles: List[str]
    
    # Track record
    total_transactions: int
    total_buy_transactions: int
    total_sell_transactions: int
    
    # Performance metrics
    avg_return_after_buy: float
    avg_return_after_sell: float
    success_rate_buys: float
    success_rate_sells: float
    
    # Timing analysis
    avg_days_before_earnings: float
    tends_to_buy_before_positive_earnings: bool
    tends_to_sell_before_negative_earnings: bool
    
    # Transaction patterns
    avg_transaction_size: float
    typical_holding_period: timedelta
    seasonality_patterns: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "insider_name": self.insider_name,
            "insider_cik": self.insider_cik,
            "current_roles": self.current_roles,
            "total_transactions": self.total_transactions,
            "total_buy_transactions": self.total_buy_transactions,
            "total_sell_transactions": self.total_sell_transactions,
            "avg_return_after_buy": self.avg_return_after_buy,
            "avg_return_after_sell": self.avg_return_after_sell,
            "success_rate_buys": self.success_rate_buys,
            "success_rate_sells": self.success_rate_sells,
            "avg_days_before_earnings": self.avg_days_before_earnings,
            "tends_to_buy_before_positive_earnings": self.tends_to_buy_before_positive_earnings,
            "tends_to_sell_before_negative_earnings": self.tends_to_sell_before_negative_earnings,
            "avg_transaction_size": self.avg_transaction_size,
            "typical_holding_period": self.typical_holding_period.total_seconds(),
            "seasonality_patterns": self.seasonality_patterns
        }


@dataclass
class InsiderSentiment:
    """Aggregate insider sentiment for a stock"""
    ticker: str
    timestamp: datetime
    lookback_period: timedelta
    
    # Transaction summary
    total_buy_value: float
    total_sell_value: float
    net_insider_activity: float  # Buy - Sell
    
    # Participant analysis
    num_buyers: int
    num_sellers: int
    net_participants: int  # Buyers - Sellers
    
    # Role-based analysis
    ceo_sentiment: Optional[SentimentSignal]
    cfo_sentiment: Optional[SentimentSignal]
    director_sentiment: SentimentSignal
    officer_sentiment: SentimentSignal
    
    # Timing analysis
    activity_near_earnings: bool
    days_to_next_earnings: Optional[int]
    activity_vs_historical: float  # Current vs historical average
    
    # Sentiment scores
    raw_sentiment_score: float  # -1 to 1
    adjusted_sentiment_score: float  # Adjusted for historical performance
    confidence_level: float  # 0 to 1
    
    # Composite signals
    overall_sentiment: SentimentSignal
    strength: float  # 0 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "lookback_period": self.lookback_period.total_seconds(),
            "total_buy_value": self.total_buy_value,
            "total_sell_value": self.total_sell_value,
            "net_insider_activity": self.net_insider_activity,
            "num_buyers": self.num_buyers,
            "num_sellers": self.num_sellers,
            "net_participants": self.net_participants,
            "ceo_sentiment": self.ceo_sentiment.value if self.ceo_sentiment else None,
            "cfo_sentiment": self.cfo_sentiment.value if self.cfo_sentiment else None,
            "director_sentiment": self.director_sentiment.value,
            "officer_sentiment": self.officer_sentiment.value,
            "activity_near_earnings": self.activity_near_earnings,
            "days_to_next_earnings": self.days_to_next_earnings,
            "activity_vs_historical": self.activity_vs_historical,
            "raw_sentiment_score": self.raw_sentiment_score,
            "adjusted_sentiment_score": self.adjusted_sentiment_score,
            "confidence_level": self.confidence_level,
            "overall_sentiment": self.overall_sentiment.value,
            "strength": self.strength
        }


@dataclass
class TransactionPattern:
    """Detected insider transaction pattern"""
    pattern_id: str
    pattern_type: str  # cluster_buying, systematic_selling, etc.
    ticker: str
    start_date: datetime
    end_date: datetime
    
    # Pattern characteristics
    participants: List[str]
    total_value: float
    avg_transaction_size: float
    frequency: float  # Transactions per day
    
    # Statistical significance
    z_score: float  # Statistical significance vs historical
    p_value: float
    is_statistically_significant: bool
    
    # Performance context
    stock_performance_during_pattern: float
    stock_performance_after_pattern: float
    market_relative_performance: float
    
    # Pattern strength
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "participants": self.participants,
            "total_value": self.total_value,
            "avg_transaction_size": self.avg_transaction_size,
            "frequency": self.frequency,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "is_statistically_significant": self.is_statistically_significant,
            "stock_performance_during_pattern": self.stock_performance_during_pattern,
            "stock_performance_after_pattern": self.stock_performance_after_pattern,
            "market_relative_performance": self.market_relative_performance,
            "strength": self.strength,
            "confidence": self.confidence
        }


@dataclass
class InsiderAnalysis:
    """Complete insider activity analysis"""
    ticker: str
    timestamp: datetime
    analysis_period: timedelta
    
    # Recent transactions
    recent_transactions: List[InsiderTransaction]
    significant_transactions: List[InsiderTransaction]
    
    # Insider sentiment
    current_sentiment: InsiderSentiment
    sentiment_trend: str  # improving, deteriorating, stable
    
    # Pattern detection
    detected_patterns: List[TransactionPattern]
    unusual_activity_detected: bool
    
    # Key insights
    key_buyers: List[InsiderProfile]
    key_sellers: List[InsiderProfile]
    
    # Predictive analysis
    predicted_next_transaction_type: Optional[TransactionType]
    estimated_price_impact: float
    time_to_next_expected_activity: Optional[timedelta]
    
    # Risk assessment
    regulatory_risk_score: float
    information_asymmetry_score: float
    
    # Performance tracking
    historical_accuracy: float
    insider_performance_vs_market: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "analysis_period": self.analysis_period.total_seconds(),
            "recent_transactions": [t.to_dict() for t in self.recent_transactions],
            "significant_transactions": [t.to_dict() for t in self.significant_transactions],
            "current_sentiment": self.current_sentiment.to_dict(),
            "sentiment_trend": self.sentiment_trend,
            "detected_patterns": [p.to_dict() for p in self.detected_patterns],
            "unusual_activity_detected": self.unusual_activity_detected,
            "key_buyers": [b.to_dict() for b in self.key_buyers],
            "key_sellers": [s.to_dict() for s in self.key_sellers],
            "predicted_next_transaction_type": self.predicted_next_transaction_type.value if self.predicted_next_transaction_type else None,
            "estimated_price_impact": self.estimated_price_impact,
            "time_to_next_expected_activity": self.time_to_next_expected_activity.total_seconds() if self.time_to_next_expected_activity else None,
            "regulatory_risk_score": self.regulatory_risk_score,
            "information_asymmetry_score": self.information_asymmetry_score,
            "historical_accuracy": self.historical_accuracy,
            "insider_performance_vs_market": self.insider_performance_vs_market
        }


@dataclass
class InsiderRequest:
    """Request for insider activity analysis"""
    tickers: List[str]
    lookback_period: str = "90d"  # 30d, 90d, 180d, 1y
    include_patterns: bool = True
    include_profiles: bool = True
    min_transaction_value: float = 100000  # $100k minimum
    exclude_systematic_selling: bool = True  # Exclude routine option exercises
    min_confidence: float = 0.6
    
    def validate(self) -> bool:
        """Validate request parameters"""
        valid_periods = ["30d", "90d", "180d", "1y"]
        
        return (
            len(self.tickers) > 0 and
            self.lookback_period in valid_periods and
            self.min_transaction_value >= 0 and
            0 <= self.min_confidence <= 1
        )
