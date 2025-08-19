"""
Data models for Undervalued Agent

Fundamental analysis and value screening models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class ValuationMethod(str, Enum):
    DCF = "dcf"
    RELATIVE_VALUATION = "relative_valuation"
    ASSET_BASED = "asset_based"
    EARNINGS_POWER = "earnings_power"
    LIQUIDATION_VALUE = "liquidation_value"


class ScreeningCriteria(str, Enum):
    DEEP_VALUE = "deep_value"
    QUALITY_VALUE = "quality_value"
    GROWTH_AT_REASONABLE_PRICE = "garp"
    NET_NET = "net_net"
    ASSET_PLAYS = "asset_plays"
    SPECIAL_SITUATIONS = "special_situations"


class FinancialHealth(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DISTRESSED = "distressed"


@dataclass
class FinancialMetrics:
    """Core financial metrics"""
    ticker: str
    timestamp: datetime
    
    # Profitability
    revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    ebitda: float
    
    # Per share metrics
    earnings_per_share: float
    book_value_per_share: float
    sales_per_share: float
    cash_per_share: float
    
    # Margins
    gross_margin: float
    operating_margin: float
    net_margin: float
    ebitda_margin: float
    
    # Returns
    return_on_equity: float
    return_on_assets: float
    return_on_invested_capital: float
    
    # Growth rates
    revenue_growth_1y: float
    revenue_growth_3y: float
    earnings_growth_1y: float
    earnings_growth_3y: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "revenue": self.revenue,
            "gross_profit": self.gross_profit,
            "operating_income": self.operating_income,
            "net_income": self.net_income,
            "ebitda": self.ebitda,
            "earnings_per_share": self.earnings_per_share,
            "book_value_per_share": self.book_value_per_share,
            "sales_per_share": self.sales_per_share,
            "cash_per_share": self.cash_per_share,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "ebitda_margin": self.ebitda_margin,
            "return_on_equity": self.return_on_equity,
            "return_on_assets": self.return_on_assets,
            "return_on_invested_capital": self.return_on_invested_capital,
            "revenue_growth_1y": self.revenue_growth_1y,
            "revenue_growth_3y": self.revenue_growth_3y,
            "earnings_growth_1y": self.earnings_growth_1y,
            "earnings_growth_3y": self.earnings_growth_3y
        }


@dataclass
class BalanceSheetMetrics:
    """Balance sheet analysis metrics"""
    ticker: str
    timestamp: datetime
    
    # Assets
    total_assets: float
    current_assets: float
    cash_and_equivalents: float
    accounts_receivable: float
    inventory: float
    property_plant_equipment: float
    
    # Liabilities
    total_liabilities: float
    current_liabilities: float
    long_term_debt: float
    total_debt: float
    
    # Equity
    shareholders_equity: float
    retained_earnings: float
    
    # Ratios
    current_ratio: float
    quick_ratio: float
    debt_to_equity: float
    debt_to_assets: float
    asset_turnover: float
    working_capital: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "total_assets": self.total_assets,
            "current_assets": self.current_assets,
            "cash_and_equivalents": self.cash_and_equivalents,
            "accounts_receivable": self.accounts_receivable,
            "inventory": self.inventory,
            "property_plant_equipment": self.property_plant_equipment,
            "total_liabilities": self.total_liabilities,
            "current_liabilities": self.current_liabilities,
            "long_term_debt": self.long_term_debt,
            "total_debt": self.total_debt,
            "shareholders_equity": self.shareholders_equity,
            "retained_earnings": self.retained_earnings,
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "debt_to_equity": self.debt_to_equity,
            "debt_to_assets": self.debt_to_assets,
            "asset_turnover": self.asset_turnover,
            "working_capital": self.working_capital
        }


@dataclass
class ValuationMetrics:
    """Valuation ratios and metrics"""
    ticker: str
    timestamp: datetime
    current_price: float
    
    # Traditional ratios
    pe_ratio: float
    pb_ratio: float
    ps_ratio: float
    pcf_ratio: float  # Price to cash flow
    peg_ratio: float  # PE to growth
    
    # Enterprise value ratios
    enterprise_value: float
    ev_to_revenue: float
    ev_to_ebitda: float
    ev_to_ebit: float
    
    # Dividend metrics
    dividend_yield: float
    dividend_payout_ratio: float
    dividend_growth_rate: float
    
    # Advanced metrics
    price_to_free_cash_flow: float
    price_to_book_tangible: float
    price_to_sales_ttm: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "ps_ratio": self.ps_ratio,
            "pcf_ratio": self.pcf_ratio,
            "peg_ratio": self.peg_ratio,
            "enterprise_value": self.enterprise_value,
            "ev_to_revenue": self.ev_to_revenue,
            "ev_to_ebitda": self.ev_to_ebitda,
            "ev_to_ebit": self.ev_to_ebit,
            "dividend_yield": self.dividend_yield,
            "dividend_payout_ratio": self.dividend_payout_ratio,
            "dividend_growth_rate": self.dividend_growth_rate,
            "price_to_free_cash_flow": self.price_to_free_cash_flow,
            "price_to_book_tangible": self.price_to_book_tangible,
            "price_to_sales_ttm": self.price_to_sales_ttm
        }


@dataclass
class DCFValuation:
    """Discounted Cash Flow valuation model"""
    ticker: str
    timestamp: datetime
    
    # Assumptions
    discount_rate: float
    terminal_growth_rate: float
    projection_years: int
    
    # Cash flow projections
    projected_free_cash_flows: List[float]
    terminal_value: float
    
    # Valuation results
    enterprise_value: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float
    
    # Sensitivity analysis
    base_case_value: float
    bull_case_value: float
    bear_case_value: float
    
    # Margin of safety
    current_price: float
    margin_of_safety: float
    upside_potential: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "discount_rate": self.discount_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "projection_years": self.projection_years,
            "projected_free_cash_flows": self.projected_free_cash_flows,
            "terminal_value": self.terminal_value,
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "shares_outstanding": self.shares_outstanding,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "base_case_value": self.base_case_value,
            "bull_case_value": self.bull_case_value,
            "bear_case_value": self.bear_case_value,
            "current_price": self.current_price,
            "margin_of_safety": self.margin_of_safety,
            "upside_potential": self.upside_potential
        }


@dataclass
class QualityScore:
    """Business quality assessment"""
    ticker: str
    timestamp: datetime
    
    # Profitability quality
    profitability_score: float  # 0-100
    profitability_consistency: float
    profitability_trend: float
    
    # Financial strength
    balance_sheet_strength: float
    debt_management: float
    liquidity_score: float
    
    # Management quality
    capital_allocation_score: float
    shareholder_friendliness: float
    governance_score: float
    
    # Business moat
    competitive_position: float
    industry_attractiveness: float
    barriers_to_entry: float
    
    # Overall scores
    composite_quality_score: float
    financial_health: FinancialHealth
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "profitability_score": self.profitability_score,
            "profitability_consistency": self.profitability_consistency,
            "profitability_trend": self.profitability_trend,
            "balance_sheet_strength": self.balance_sheet_strength,
            "debt_management": self.debt_management,
            "liquidity_score": self.liquidity_score,
            "capital_allocation_score": self.capital_allocation_score,
            "shareholder_friendliness": self.shareholder_friendliness,
            "governance_score": self.governance_score,
            "competitive_position": self.competitive_position,
            "industry_attractiveness": self.industry_attractiveness,
            "barriers_to_entry": self.barriers_to_entry,
            "composite_quality_score": self.composite_quality_score,
            "financial_health": self.financial_health.value
        }


@dataclass
class ValueOpportunity:
    """Identified value investment opportunity"""
    ticker: str
    timestamp: datetime
    
    # Classification
    screening_criteria: ScreeningCriteria
    opportunity_type: str
    confidence_level: float
    
    # Valuation summary
    current_price: float
    intrinsic_value: float
    margin_of_safety: float
    upside_potential: float
    
    # Key metrics
    key_ratios: Dict[str, float]
    quality_score: float
    catalyst_probability: float
    
    # Risk assessment
    financial_risk: float
    business_risk: float
    market_risk: float
    overall_risk_rating: str
    
    # Investment thesis
    thesis_summary: str
    key_catalysts: List[str]
    potential_risks: List[str]
    time_horizon: str
    
    # Comparable analysis
    peer_comparison: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "screening_criteria": self.screening_criteria.value,
            "opportunity_type": self.opportunity_type,
            "confidence_level": self.confidence_level,
            "current_price": self.current_price,
            "intrinsic_value": self.intrinsic_value,
            "margin_of_safety": self.margin_of_safety,
            "upside_potential": self.upside_potential,
            "key_ratios": self.key_ratios,
            "quality_score": self.quality_score,
            "catalyst_probability": self.catalyst_probability,
            "financial_risk": self.financial_risk,
            "business_risk": self.business_risk,
            "market_risk": self.market_risk,
            "overall_risk_rating": self.overall_risk_rating,
            "thesis_summary": self.thesis_summary,
            "key_catalysts": self.key_catalysts,
            "potential_risks": self.potential_risks,
            "time_horizon": self.time_horizon,
            "peer_comparison": self.peer_comparison
        }


@dataclass
class UndervaluedAnalysis:
    """Complete undervalued stock analysis"""
    timestamp: datetime
    analysis_universe: List[str]
    screening_criteria: List[ScreeningCriteria]
    
    # Value opportunities
    identified_opportunities: List[ValueOpportunity]
    top_value_picks: List[str]
    
    # Market analysis
    market_valuation_level: float  # Expensive/cheap overall
    sector_valuations: Dict[str, float]
    style_performance: Dict[str, float]  # Value vs growth
    
    # Quality analysis
    quality_distribution: Dict[str, int]  # Quality score ranges
    high_quality_value_stocks: List[str]
    
    # Risk analysis
    portfolio_risk_metrics: Dict[str, float]
    correlation_analysis: Dict[str, float]
    
    # Performance metrics
    expected_returns: Dict[str, float]
    risk_adjusted_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "analysis_universe": self.analysis_universe,
            "screening_criteria": [c.value for c in self.screening_criteria],
            "identified_opportunities": [op.to_dict() for op in self.identified_opportunities],
            "top_value_picks": self.top_value_picks,
            "market_valuation_level": self.market_valuation_level,
            "sector_valuations": self.sector_valuations,
            "style_performance": self.style_performance,
            "quality_distribution": self.quality_distribution,
            "high_quality_value_stocks": self.high_quality_value_stocks,
            "portfolio_risk_metrics": self.portfolio_risk_metrics,
            "correlation_analysis": self.correlation_analysis,
            "expected_returns": self.expected_returns,
            "risk_adjusted_scores": self.risk_adjusted_scores
        }


@dataclass
class UndervaluedRequest:
    """Request for undervalued stock analysis"""
    universe: List[str] = field(default_factory=list)
    screening_criteria: List[ScreeningCriteria] = field(default_factory=lambda: [ScreeningCriteria.QUALITY_VALUE])
    min_market_cap: float = 1e9  # $1B minimum
    max_pe_ratio: float = 20.0
    min_roe: float = 0.10  # 10% ROE minimum
    min_margin_of_safety: float = 0.20  # 20% margin of safety
    max_debt_to_equity: float = 0.50
    include_dcf_analysis: bool = True
    include_quality_scoring: bool = True
    num_recommendations: int = 10
    
    def validate(self) -> bool:
        """Validate request parameters"""
        return (
            self.min_market_cap >= 0 and
            self.max_pe_ratio > 0 and
            0 <= self.min_roe <= 1 and
            0 <= self.min_margin_of_safety <= 1 and
            self.max_debt_to_equity >= 0 and
            self.num_recommendations > 0
        )
