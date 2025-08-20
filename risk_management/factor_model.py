#!/usr/bin/env python3
"""
Multi-Factor Risk Model with Crowding Constraints
Statistical & fundamental factors, crowding/borrow limits, exposure nets (style/sector/term), scenario stress
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from scipy import stats
from scipy.linalg import inv
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class FactorModel:
    """Wrapper class for Multi-Factor Risk Model to match expected interface"""
    
    def __init__(self, config=None):
        self.config = config or {
            'risk_factors': ['market', 'size', 'value', 'momentum', 'quality'],
            'max_position_size': 0.05,
            'max_sector_exposure': 0.25,
            'max_factor_exposure': 0.15,
            'confidence_level': 0.95
        }
        self.risk_model = MultiFactorRiskModel(config)
    
    async def calculate_risk_metrics(self, portfolio_data):
        """Calculate risk metrics for portfolio"""
        return await self.risk_model.calculate_portfolio_risk(portfolio_data)
    
    async def get_factor_exposures(self, positions):
        """Get factor exposures for positions"""
        return await self.risk_model.get_factor_exposures(positions)
    
    async def run_stress_tests(self, portfolio_data):
        """Run stress tests on portfolio"""
        return await self.risk_model.run_stress_tests(portfolio_data)
    
    def get_model_info(self):
        """Get information about the risk model"""
        return {
            'model_type': 'Multi-Factor Risk Model',
            'risk_factors': self.config['risk_factors'],
            'max_position_size': self.config['max_position_size'],
            'max_sector_exposure': self.config['max_sector_exposure'],
            'confidence_level': self.config['confidence_level']
        }

@dataclass
class RiskFactor:
    """Risk factor definition"""
    factor_id: str
    factor_name: str
    factor_type: str  # 'style', 'sector', 'country', 'currency', 'term_structure', 'statistical'
    description: str
    data_source: str
    frequency: str  # 'daily', 'weekly', 'monthly'
    is_tradeable: bool = False
    benchmark_weight: float = 0.0
    factor_loadings: Dict[str, float] = field(default_factory=dict)
    risk_premium: float = 0.0
    volatility: float = 0.0


@dataclass
class CrowdingIndicator:
    """Crowding indicator for positions"""
    indicator_id: str
    indicator_name: str
    measurement_type: str  # 'positioning', 'flow', 'sentiment', 'concentration'
    current_level: float
    percentile_rank: float  # 0-100 percentile
    z_score: float
    threshold_levels: Dict[str, float]  # 'low', 'medium', 'high', 'extreme'
    affected_securities: List[str]
    time_series: pd.Series = field(default_factory=pd.Series)


@dataclass
class ExposureLimit:
    """Exposure limit definition"""
    limit_id: str
    limit_type: str  # 'factor', 'sector', 'country', 'currency', 'term', 'crowding'
    limit_name: str
    max_long_exposure: float
    max_short_exposure: float
    max_net_exposure: float
    max_gross_exposure: float
    warning_threshold: float = 0.8  # Percentage of limit for warning
    hard_limit: bool = True
    affected_universe: List[str] = field(default_factory=list)


@dataclass
class RiskAttribution:
    """Risk attribution results"""
    attribution_date: datetime
    total_risk: float
    factor_contributions: Dict[str, float]
    specific_risk: float
    concentration_risk: float
    crowding_risk: float
    factor_exposures: Dict[str, float]
    risk_decomposition: Dict[str, float]
    largest_contributors: List[Tuple[str, float]]
    risk_warnings: List[str]


@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_id: str
    scenario_name: str
    scenario_description: str
    baseline_value: float
    stressed_value: float
    pnl_impact: float
    return_impact: float
    risk_factor_shocks: Dict[str, float]
    position_level_impacts: Dict[str, float]
    worst_contributors: List[Tuple[str, float]]
    var_impact: float
    timestamp: datetime


class StyleFactorModel:
    """Style factor model (momentum, value, quality, etc.)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.style_factors = self._define_style_factors()
        self.factor_loadings = {}
        self.factor_returns = pd.DataFrame()
        self.is_fitted = False
    
    def _define_style_factors(self) -> List[RiskFactor]:
        """Define standard style factors"""
        return [
            RiskFactor(
                factor_id="momentum",
                factor_name="Momentum",
                factor_type="style",
                description="12-1 month momentum factor",
                data_source="computed",
                frequency="daily"
            ),
            RiskFactor(
                factor_id="value",
                factor_name="Value",
                factor_type="style",
                description="Book-to-price and earnings yield",
                data_source="fundamentals",
                frequency="monthly"
            ),
            RiskFactor(
                factor_id="quality",
                factor_name="Quality",
                factor_type="style",
                description="ROE, debt-to-equity, earnings quality",
                data_source="fundamentals",
                frequency="quarterly"
            ),
            RiskFactor(
                factor_id="size",
                factor_name="Size",
                factor_type="style",
                description="Market capitalization factor",
                data_source="market_data",
                frequency="daily"
            ),
            RiskFactor(
                factor_id="volatility",
                factor_name="Low Volatility",
                factor_type="style",
                description="Inverse of realized volatility",
                data_source="computed",
                frequency="daily"
            ),
            RiskFactor(
                factor_id="profitability",
                factor_name="Profitability",
                factor_type="style",
                description="ROA, ROE, profit margins",
                data_source="fundamentals",
                frequency="quarterly"
            ),
            RiskFactor(
                factor_id="growth",
                factor_name="Growth",
                factor_type="style",
                description="Sales and earnings growth",
                data_source="fundamentals",
                frequency="quarterly"
            ),
            RiskFactor(
                factor_id="leverage",
                factor_name="Leverage",
                factor_type="style",
                description="Financial leverage factor",
                data_source="fundamentals",
                frequency="quarterly"
            )
        ]
    
    def fit(self, returns_data: pd.DataFrame, fundamental_data: pd.DataFrame) -> None:
        """Fit style factor model"""
        try:
            self.logger.info("Fitting style factor model...")
            
            # Calculate factor exposures for each security
            factor_exposures = {}
            
            for factor in self.style_factors:
                if factor.factor_id == "momentum":
                    # 12-1 month momentum
                    factor_exposures[factor.factor_id] = self._calculate_momentum(returns_data)
                elif factor.factor_id == "value":
                    factor_exposures[factor.factor_id] = self._calculate_value(fundamental_data)
                elif factor.factor_id == "quality":
                    factor_exposures[factor.factor_id] = self._calculate_quality(fundamental_data)
                elif factor.factor_id == "size":
                    factor_exposures[factor.factor_id] = self._calculate_size(fundamental_data)
                elif factor.factor_id == "volatility":
                    factor_exposures[factor.factor_id] = self._calculate_low_vol(returns_data)
                elif factor.factor_id == "profitability":
                    factor_exposures[factor.factor_id] = self._calculate_profitability(fundamental_data)
                elif factor.factor_id == "growth":
                    factor_exposures[factor.factor_id] = self._calculate_growth(fundamental_data)
                elif factor.factor_id == "leverage":
                    factor_exposures[factor.factor_id] = self._calculate_leverage(fundamental_data)
            
            # Convert to DataFrame
            self.factor_loadings = pd.DataFrame(factor_exposures)
            
            # Calculate factor returns using cross-sectional regression
            self.factor_returns = self._calculate_factor_returns(returns_data, self.factor_loadings)
            
            self.is_fitted = True
            self.logger.info(f"Fitted style factor model with {len(self.style_factors)} factors")
            
        except Exception as e:
            self.logger.error(f"Error fitting style factor model: {e}")
            self.is_fitted = False
    
    def _calculate_momentum(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate momentum factor exposures"""
        # 12-1 month momentum (skip most recent month)
        lookback_start = -252  # ~12 months
        lookback_end = -21     # Skip most recent month
        
        momentum_returns = returns_data.iloc[lookback_start:lookback_end].sum()
        return (momentum_returns - momentum_returns.mean()) / momentum_returns.std()
    
    def _calculate_value(self, fundamental_data: pd.DataFrame) -> pd.Series:
        """Calculate value factor exposures"""
        # Combine multiple value metrics
        value_metrics = []
        
        if 'book_to_price' in fundamental_data.columns:
            value_metrics.append(fundamental_data['book_to_price'])
        if 'earnings_yield' in fundamental_data.columns:
            value_metrics.append(fundamental_data['earnings_yield'])
        if 'sales_to_price' in fundamental_data.columns:
            value_metrics.append(fundamental_data['sales_to_price'])
        
        if value_metrics:
            # Average of standardized metrics
            value_score = pd.concat(value_metrics, axis=1).mean(axis=1)
            return (value_score - value_score.mean()) / value_score.std()
        else:
            # Fallback to random scores
            return pd.Series(np.random.randn(len(fundamental_data)), index=fundamental_data.index)
    
    def _calculate_quality(self, fundamental_data: pd.DataFrame) -> pd.Series:
        """Calculate quality factor exposures"""
        quality_metrics = []
        
        if 'roe' in fundamental_data.columns:
            quality_metrics.append(fundamental_data['roe'])
        if 'roa' in fundamental_data.columns:
            quality_metrics.append(fundamental_data['roa'])
        if 'debt_to_equity' in fundamental_data.columns:
            quality_metrics.append(-fundamental_data['debt_to_equity'])  # Lower debt is better
        
        if quality_metrics:
            quality_score = pd.concat(quality_metrics, axis=1).mean(axis=1)
            return (quality_score - quality_score.mean()) / quality_score.std()
        else:
            return pd.Series(np.random.randn(len(fundamental_data)), index=fundamental_data.index)
    
    def _calculate_size(self, fundamental_data: pd.DataFrame) -> pd.Series:
        """Calculate size factor exposures"""
        if 'market_cap' in fundamental_data.columns:
            log_mcap = np.log(fundamental_data['market_cap'])
            return (log_mcap - log_mcap.mean()) / log_mcap.std()
        else:
            return pd.Series(np.random.randn(len(fundamental_data)), index=fundamental_data.index)
    
    def _calculate_low_vol(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate low volatility factor exposures"""
        # Realized volatility over past 60 days
        realized_vol = returns_data.tail(60).std() * np.sqrt(252)
        # Negative because we want low volatility
        return -(realized_vol - realized_vol.mean()) / realized_vol.std()
    
    def _calculate_profitability(self, fundamental_data: pd.DataFrame) -> pd.Series:
        """Calculate profitability factor exposures"""
        profitability_metrics = []
        
        if 'gross_margin' in fundamental_data.columns:
            profitability_metrics.append(fundamental_data['gross_margin'])
        if 'operating_margin' in fundamental_data.columns:
            profitability_metrics.append(fundamental_data['operating_margin'])
        if 'net_margin' in fundamental_data.columns:
            profitability_metrics.append(fundamental_data['net_margin'])
        
        if profitability_metrics:
            profitability_score = pd.concat(profitability_metrics, axis=1).mean(axis=1)
            return (profitability_score - profitability_score.mean()) / profitability_score.std()
        else:
            return pd.Series(np.random.randn(len(fundamental_data)), index=fundamental_data.index)
    
    def _calculate_growth(self, fundamental_data: pd.DataFrame) -> pd.Series:
        """Calculate growth factor exposures"""
        growth_metrics = []
        
        if 'revenue_growth' in fundamental_data.columns:
            growth_metrics.append(fundamental_data['revenue_growth'])
        if 'earnings_growth' in fundamental_data.columns:
            growth_metrics.append(fundamental_data['earnings_growth'])
        if 'book_value_growth' in fundamental_data.columns:
            growth_metrics.append(fundamental_data['book_value_growth'])
        
        if growth_metrics:
            growth_score = pd.concat(growth_metrics, axis=1).mean(axis=1)
            return (growth_score - growth_score.mean()) / growth_score.std()
        else:
            return pd.Series(np.random.randn(len(fundamental_data)), index=fundamental_data.index)
    
    def _calculate_leverage(self, fundamental_data: pd.DataFrame) -> pd.Series:
        """Calculate leverage factor exposures"""
        if 'debt_to_assets' in fundamental_data.columns:
            leverage = fundamental_data['debt_to_assets']
            return (leverage - leverage.mean()) / leverage.std()
        else:
            return pd.Series(np.random.randn(len(fundamental_data)), index=fundamental_data.index)
    
    def _calculate_factor_returns(self, returns_data: pd.DataFrame, 
                                factor_loadings: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor returns using cross-sectional regression"""
        try:
            factor_returns = pd.DataFrame(index=returns_data.index, columns=factor_loadings.columns)
            
            for date in returns_data.index:
                if date in returns_data.index:
                    # Cross-sectional regression for this date
                    y = returns_data.loc[date].dropna()
                    X = factor_loadings.loc[y.index].dropna()
                    
                    # Align data
                    common_securities = y.index.intersection(X.index)
                    if len(common_securities) > len(X.columns):
                        y_aligned = y.loc[common_securities]
                        X_aligned = X.loc[common_securities]
                        
                        # Regression
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(X_aligned, y_aligned)
                        factor_returns.loc[date] = reg.coef_
            
            return factor_returns.astype(float)
            
        except Exception as e:
            self.logger.error(f"Error calculating factor returns: {e}")
            # Return zero factor returns as fallback
            return pd.DataFrame(0.0, index=returns_data.index, columns=factor_loadings.columns)


class CrowdingModel:
    """Model for detecting and measuring crowding in positions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.crowding_indicators = {}
        self.crowding_history = pd.DataFrame()
        self.thresholds = self._set_default_thresholds()
    
    def _set_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Set default crowding thresholds"""
        return {
            'positioning': {'low': 25, 'medium': 50, 'high': 75, 'extreme': 90},
            'flow': {'low': 20, 'medium': 40, 'high': 70, 'extreme': 85},
            'sentiment': {'low': 30, 'medium': 50, 'high': 80, 'extreme': 95},
            'concentration': {'low': 15, 'medium': 35, 'high': 65, 'extreme': 85}
        }
    
    def calculate_crowding_indicators(self, positions_data: pd.DataFrame,
                                    flow_data: pd.DataFrame,
                                    sentiment_data: pd.DataFrame) -> Dict[str, CrowdingIndicator]:
        """Calculate crowding indicators"""
        try:
            indicators = {}
            
            # 1. Position concentration indicator
            indicators['position_concentration'] = self._calculate_position_concentration(positions_data)
            
            # 2. Flow crowding indicator
            indicators['flow_crowding'] = self._calculate_flow_crowding(flow_data)
            
            # 3. Sentiment extremes indicator
            indicators['sentiment_extremes'] = self._calculate_sentiment_extremes(sentiment_data)
            
            # 4. Factor crowding indicator
            indicators['factor_crowding'] = self._calculate_factor_crowding(positions_data)
            
            # 5. Correlation crowding indicator
            indicators['correlation_crowding'] = self._calculate_correlation_crowding(positions_data)
            
            self.crowding_indicators = indicators
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating crowding indicators: {e}")
            return {}
    
    def _calculate_position_concentration(self, positions_data: pd.DataFrame) -> CrowdingIndicator:
        """Calculate position concentration indicator"""
        # Calculate Herfindahl index for concentration
        weights = positions_data.abs() / positions_data.abs().sum()
        herfindahl = (weights ** 2).sum()
        
        # Convert to percentile (historical comparison)
        historical_herfindahl = [herfindahl]  # Would use actual history in production
        percentile_rank = stats.percentileofscore(historical_herfindahl, herfindahl)
        z_score = (herfindahl - np.mean(historical_herfindahl)) / np.std(historical_herfindahl)
        
        return CrowdingIndicator(
            indicator_id="position_concentration",
            indicator_name="Position Concentration",
            measurement_type="concentration",
            current_level=float(herfindahl),
            percentile_rank=float(percentile_rank),
            z_score=float(z_score),
            threshold_levels=self.thresholds['concentration'],
            affected_securities=list(positions_data.index),
            time_series=pd.Series([herfindahl], index=[datetime.now()])
        )
    
    def _calculate_flow_crowding(self, flow_data: pd.DataFrame) -> CrowdingIndicator:
        """Calculate flow-based crowding indicator"""
        # Calculate flow concentration in recent period
        recent_flows = flow_data.tail(21).mean()  # 21-day average
        flow_concentration = (recent_flows.abs() / recent_flows.abs().sum()).max()
        
        # Historical percentile
        historical_flows = [flow_concentration]  # Would use actual history
        percentile_rank = stats.percentileofscore(historical_flows, flow_concentration)
        z_score = 0.0  # Would calculate from history
        
        return CrowdingIndicator(
            indicator_id="flow_crowding",
            indicator_name="Flow Crowding",
            measurement_type="flow",
            current_level=float(flow_concentration),
            percentile_rank=float(percentile_rank),
            z_score=float(z_score),
            threshold_levels=self.thresholds['flow'],
            affected_securities=list(flow_data.columns),
            time_series=pd.Series([flow_concentration], index=[datetime.now()])
        )
    
    def _calculate_sentiment_extremes(self, sentiment_data: pd.DataFrame) -> CrowdingIndicator:
        """Calculate sentiment-based crowding indicator"""
        # Look for extreme sentiment readings
        current_sentiment = sentiment_data.iloc[-1].mean()
        sentiment_extreme = abs(current_sentiment - 0.5) * 2  # Normalize to [0,1]
        
        percentile_rank = sentiment_extreme * 100  # Simplified
        z_score = (sentiment_extreme - 0.5) / 0.2  # Simplified
        
        return CrowdingIndicator(
            indicator_id="sentiment_extremes",
            indicator_name="Sentiment Extremes",
            measurement_type="sentiment",
            current_level=float(sentiment_extreme),
            percentile_rank=float(percentile_rank),
            z_score=float(z_score),
            threshold_levels=self.thresholds['sentiment'],
            affected_securities=list(sentiment_data.columns),
            time_series=pd.Series([sentiment_extreme], index=[datetime.now()])
        )
    
    def _calculate_factor_crowding(self, positions_data: pd.DataFrame) -> CrowdingIndicator:
        """Calculate factor-based crowding indicator"""
        # Would require factor loadings in production
        # For now, calculate simple factor exposure concentration
        factor_exposure = positions_data.std()  # Simplified proxy
        concentration = factor_exposure.max() / factor_exposure.mean() if factor_exposure.mean() > 0 else 1.0
        
        percentile_rank = min(concentration * 25, 100)  # Simplified
        z_score = (concentration - 1.0) / 0.5
        
        return CrowdingIndicator(
            indicator_id="factor_crowding",
            indicator_name="Factor Crowding",
            measurement_type="positioning",
            current_level=float(concentration),
            percentile_rank=float(percentile_rank),
            z_score=float(z_score),
            threshold_levels=self.thresholds['positioning'],
            affected_securities=list(positions_data.index),
            time_series=pd.Series([concentration], index=[datetime.now()])
        )
    
    def _calculate_correlation_crowding(self, positions_data: pd.DataFrame) -> CrowdingIndicator:
        """Calculate correlation-based crowding indicator"""
        # Calculate average correlation between positions
        if len(positions_data) > 1:
            correlation_matrix = positions_data.T.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        else:
            avg_correlation = 0.0
        
        percentile_rank = (avg_correlation + 1) * 50  # Convert [-1,1] to [0,100]
        z_score = avg_correlation / 0.3  # Rough standardization
        
        return CrowdingIndicator(
            indicator_id="correlation_crowding",
            indicator_name="Correlation Crowding",
            measurement_type="positioning",
            current_level=float(avg_correlation),
            percentile_rank=float(percentile_rank),
            z_score=float(z_score),
            threshold_levels=self.thresholds['positioning'],
            affected_securities=list(positions_data.index),
            time_series=pd.Series([avg_correlation], index=[datetime.now()])
        )


class MultiFactorRiskModel:
    """Main multi-factor risk model with crowding constraints"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.style_model = StyleFactorModel()
        self.crowding_model = CrowdingModel()
        
        # Risk model state
        self.factor_covariance_matrix = None
        self.specific_risk = None
        self.is_fitted = False
        
        # Exposure limits
        self.exposure_limits = self._set_default_limits()
        
        # Current portfolio state
        self.current_positions = pd.Series()
        self.current_exposures = {}
        self.current_risk_attribution = None
        
    def _set_default_limits(self) -> List[ExposureLimit]:
        """Set default exposure limits"""
        return [
            ExposureLimit(
                limit_id="factor_momentum",
                limit_type="factor",
                limit_name="Momentum Factor",
                max_long_exposure=0.15,
                max_short_exposure=-0.15,
                max_net_exposure=0.10,
                max_gross_exposure=0.20
            ),
            ExposureLimit(
                limit_id="factor_value",
                limit_type="factor", 
                limit_name="Value Factor",
                max_long_exposure=0.20,
                max_short_exposure=-0.20,
                max_net_exposure=0.15,
                max_gross_exposure=0.25
            ),
            ExposureLimit(
                limit_id="sector_tech",
                limit_type="sector",
                limit_name="Technology Sector",
                max_long_exposure=0.30,
                max_short_exposure=-0.10,
                max_net_exposure=0.25,
                max_gross_exposure=0.35
            ),
            ExposureLimit(
                limit_id="crowding_concentration",
                limit_type="crowding",
                limit_name="Position Concentration",
                max_long_exposure=0.25,
                max_short_exposure=-0.25,
                max_net_exposure=0.20,
                max_gross_exposure=0.30
            )
        ]
    
    async def fit(self, returns_data: pd.DataFrame, 
                 fundamental_data: pd.DataFrame,
                 positions_data: Optional[pd.DataFrame] = None,
                 flow_data: Optional[pd.DataFrame] = None,
                 sentiment_data: Optional[pd.DataFrame] = None) -> None:
        """Fit the multi-factor risk model"""
        try:
            self.logger.info("Fitting multi-factor risk model...")
            
            # 1. Fit style factor model
            self.style_model.fit(returns_data, fundamental_data)
            
            # 2. Calculate factor covariance matrix
            self.factor_covariance_matrix = self._estimate_factor_covariance(
                self.style_model.factor_returns
            )
            
            # 3. Calculate specific risk
            self.specific_risk = self._estimate_specific_risk(
                returns_data, self.style_model
            )
            
            # 4. Calculate crowding indicators if data available
            if positions_data is not None:
                if flow_data is None:
                    flow_data = pd.DataFrame(0, index=positions_data.index, columns=positions_data.columns)
                if sentiment_data is None:
                    sentiment_data = pd.DataFrame(0.5, index=positions_data.index, columns=positions_data.columns)
                
                self.crowding_model.calculate_crowding_indicators(
                    positions_data, flow_data, sentiment_data
                )
            
            self.is_fitted = True
            self.logger.info("Successfully fitted multi-factor risk model")
            
        except Exception as e:
            self.logger.error(f"Error fitting risk model: {e}")
    
    def calculate_factor_risk(self) -> float:
        """Calculate factor risk for the current portfolio"""
        try:
            if not self.is_fitted:
                return 0.0
            
            if self.factor_covariance_matrix is None:
                return 0.0
            
            # Calculate factor risk as the square root of factor variance
            factor_risk = np.sqrt(np.trace(self.factor_covariance_matrix))
            
            return factor_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating factor risk: {e}")
            return 0.0
            self.is_fitted = False
    
    def _estimate_factor_covariance(self, factor_returns: pd.DataFrame) -> np.ndarray:
        """Estimate factor covariance matrix"""
        try:
            # Use Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(factor_returns.dropna()).covariance_, lw.shrinkage_
            
            return cov_matrix
            
        except Exception as e:
            self.logger.error(f"Error estimating factor covariance: {e}")
            # Fallback to diagonal matrix
            n_factors = len(factor_returns.columns)
            return np.eye(n_factors) * 0.01
    
    def _estimate_specific_risk(self, returns_data: pd.DataFrame, 
                              style_model: StyleFactorModel) -> pd.Series:
        """Estimate specific risk for each security"""
        try:
            specific_risks = {}
            
            for security in returns_data.columns:
                if security in style_model.factor_loadings.index:
                    # Calculate residual risk from factor model
                    security_returns = returns_data[security].dropna()
                    factor_loadings = style_model.factor_loadings.loc[security]
                    factor_returns = style_model.factor_returns
                    
                    # Align data
                    common_dates = security_returns.index.intersection(factor_returns.index)
                    if len(common_dates) > 20:
                        y = security_returns.loc[common_dates]
                        X = factor_returns.loc[common_dates]
                        
                        # Calculate residuals
                        predicted_returns = X @ factor_loadings
                        residuals = y - predicted_returns
                        specific_risk = residuals.std()
                        
                        specific_risks[security] = specific_risk
                    else:
                        # Fallback
                        specific_risks[security] = security_returns.std()
                else:
                    # Fallback for securities without factor loadings
                    specific_risks[security] = returns_data[security].std()
            
            return pd.Series(specific_risks)
            
        except Exception as e:
            self.logger.error(f"Error estimating specific risk: {e}")
            # Fallback to historical volatility
            return returns_data.std()
    
    async def calculate_portfolio_risk(self, positions: pd.Series) -> RiskAttribution:
        """Calculate portfolio risk and attribution"""
        try:
            if not self.is_fitted:
                raise ValueError("Risk model must be fitted before risk calculation")
            
            self.current_positions = positions
            
            # 1. Calculate factor exposures
            factor_exposures = self._calculate_portfolio_factor_exposures(positions)
            
            # 2. Calculate factor risk contribution
            factor_risk_contributions = self._calculate_factor_risk_contributions(factor_exposures)
            
            # 3. Calculate specific risk
            specific_risk = self._calculate_portfolio_specific_risk(positions)
            
            # 4. Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(positions)
            
            # 5. Calculate crowding risk
            crowding_risk = self._calculate_crowding_risk(positions)
            
            # 6. Total risk
            total_factor_risk = np.sqrt(sum(factor_risk_contributions.values()))
            total_risk = np.sqrt(total_factor_risk**2 + specific_risk**2 + concentration_risk**2 + crowding_risk**2)
            
            # 7. Risk decomposition
            risk_decomposition = {
                'factor_risk': total_factor_risk / total_risk,
                'specific_risk': specific_risk / total_risk,
                'concentration_risk': concentration_risk / total_risk,
                'crowding_risk': crowding_risk / total_risk
            }
            
            # 8. Find largest risk contributors
            all_contributions = {**factor_risk_contributions, 'specific': specific_risk, 'concentration': concentration_risk}
            largest_contributors = sorted(all_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            # 9. Generate risk warnings
            risk_warnings = self._generate_risk_warnings(factor_exposures, positions)
            
            self.current_risk_attribution = RiskAttribution(
                attribution_date=datetime.now(),
                total_risk=total_risk,
                factor_contributions=factor_risk_contributions,
                specific_risk=specific_risk,
                concentration_risk=concentration_risk,
                crowding_risk=crowding_risk,
                factor_exposures=factor_exposures,
                risk_decomposition=risk_decomposition,
                largest_contributors=largest_contributors,
                risk_warnings=risk_warnings
            )
            
            return self.current_risk_attribution
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            # Return empty attribution as fallback
            return RiskAttribution(
                attribution_date=datetime.now(),
                total_risk=0.0,
                factor_contributions={},
                specific_risk=0.0,
                concentration_risk=0.0,
                crowding_risk=0.0,
                factor_exposures={},
                risk_decomposition={},
                largest_contributors=[],
                risk_warnings=["Error in risk calculation"]
            )
    
    def _calculate_portfolio_factor_exposures(self, positions: pd.Series) -> Dict[str, float]:
        """Calculate portfolio factor exposures"""
        factor_exposures = {}
        
        # Get factor loadings for portfolio securities
        portfolio_securities = positions.index.intersection(self.style_model.factor_loadings.index)
        
        if len(portfolio_securities) > 0:
            portfolio_loadings = self.style_model.factor_loadings.loc[portfolio_securities]
            portfolio_weights = positions.loc[portfolio_securities] / positions.loc[portfolio_securities].abs().sum()
            
            # Calculate weighted average exposures
            for factor in portfolio_loadings.columns:
                factor_exposures[factor] = (portfolio_weights * portfolio_loadings[factor]).sum()
        
        return factor_exposures
    
    def _calculate_factor_risk_contributions(self, factor_exposures: Dict[str, float]) -> Dict[str, float]:
        """Calculate factor risk contributions"""
        contributions = {}
        
        if self.factor_covariance_matrix is not None and len(factor_exposures) > 0:
            factor_names = list(factor_exposures.keys())
            exposures_vector = np.array([factor_exposures[f] for f in factor_names])
            
            # Risk contribution = exposure^T * Cov * exposure for each factor
            for i, factor in enumerate(factor_names):
                factor_vector = np.zeros(len(factor_names))
                factor_vector[i] = exposures_vector[i]
                
                risk_contribution = np.sqrt(factor_vector @ self.factor_covariance_matrix @ factor_vector)
                contributions[factor] = risk_contribution
        
        return contributions
    
    def _calculate_portfolio_specific_risk(self, positions: pd.Series) -> float:
        """Calculate portfolio specific risk"""
        if self.specific_risk is None:
            return 0.0
        
        portfolio_securities = positions.index.intersection(self.specific_risk.index)
        
        if len(portfolio_securities) > 0:
            weights = positions.loc[portfolio_securities] / positions.loc[portfolio_securities].abs().sum()
            specific_risks = self.specific_risk.loc[portfolio_securities]
            
            # Portfolio specific risk (assuming independence)
            portfolio_specific_risk = np.sqrt((weights**2 * specific_risks**2).sum())
            return portfolio_specific_risk
        
        return 0.0
    
    def _calculate_concentration_risk(self, positions: pd.Series) -> float:
        """Calculate concentration risk"""
        # Herfindahl index based concentration risk
        weights = positions.abs() / positions.abs().sum()
        herfindahl = (weights**2).sum()
        
        # Convert to risk metric (higher concentration = higher risk)
        concentration_risk = np.sqrt(herfindahl) * 0.1  # Scale factor
        
        return concentration_risk
    
    def _calculate_crowding_risk(self, positions: pd.Series) -> float:
        """Calculate crowding risk"""
        crowding_risk = 0.0
        
        # Aggregate crowding indicators
        for indicator in self.crowding_model.crowding_indicators.values():
            if indicator.percentile_rank > 75:  # High crowding
                risk_multiplier = (indicator.percentile_rank - 75) / 25 * 0.05  # Max 5% risk
                crowding_risk += risk_multiplier
        
        return crowding_risk
    
    def _generate_risk_warnings(self, factor_exposures: Dict[str, float], 
                              positions: pd.Series) -> List[str]:
        """Generate risk warnings based on limits and crowding"""
        warnings = []
        
        # Check factor exposure limits
        for limit in self.exposure_limits:
            if limit.limit_type == "factor" and limit.limit_name.lower().replace(" factor", "") in factor_exposures:
                factor_name = limit.limit_name.lower().replace(" factor", "")
                exposure = factor_exposures[factor_name]
                
                if exposure > limit.max_long_exposure * limit.warning_threshold:
                    warnings.append(f"High {limit.limit_name} exposure: {exposure:.2%}")
                elif exposure < limit.max_short_exposure * limit.warning_threshold:
                    warnings.append(f"High short {limit.limit_name} exposure: {exposure:.2%}")
        
        # Check crowding warnings
        for indicator in self.crowding_model.crowding_indicators.values():
            if indicator.percentile_rank > 80:
                warnings.append(f"High {indicator.indicator_name}: {indicator.percentile_rank:.0f}th percentile")
        
        # Check concentration
        weights = positions.abs() / positions.abs().sum()
        max_weight = weights.max()
        if max_weight > 0.2:
            warnings.append(f"High concentration: largest position {max_weight:.1%}")
        
        return warnings
    
    async def run_stress_tests(self, positions: pd.Series, 
                             stress_scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> List[StressTestResult]:
        """Run stress tests on portfolio"""
        try:
            if stress_scenarios is None:
                stress_scenarios = self._get_default_stress_scenarios()
            
            results = []
            baseline_value = positions.sum()  # Simplified portfolio value
            
            for scenario_id, factor_shocks in stress_scenarios.items():
                # Apply factor shocks to calculate stressed portfolio value
                stressed_value = self._apply_stress_scenario(positions, factor_shocks)
                
                pnl_impact = stressed_value - baseline_value
                return_impact = pnl_impact / baseline_value if baseline_value != 0 else 0.0
                
                # Calculate position-level impacts
                position_impacts = self._calculate_position_stress_impacts(positions, factor_shocks)
                
                # Find worst contributors
                worst_contributors = sorted(position_impacts.items(), key=lambda x: x[1])[:5]
                
                # Calculate VaR impact (simplified)
                var_impact = abs(pnl_impact) * 1.5  # Rough approximation
                
                result = StressTestResult(
                    scenario_id=scenario_id,
                    scenario_name=scenario_id.replace('_', ' ').title(),
                    scenario_description=f"Stress test scenario: {scenario_id}",
                    baseline_value=baseline_value,
                    stressed_value=stressed_value,
                    pnl_impact=pnl_impact,
                    return_impact=return_impact,
                    risk_factor_shocks=factor_shocks,
                    position_level_impacts=position_impacts,
                    worst_contributors=worst_contributors,
                    var_impact=var_impact,
                    timestamp=datetime.now()
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running stress tests: {e}")
            return []
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Get default stress test scenarios"""
        return {
            'market_crash': {
                'momentum': -0.15,
                'value': 0.10,
                'quality': 0.08,
                'volatility': -0.20,
                'size': -0.12
            },
            'rate_shock': {
                'growth': -0.10,
                'value': -0.08,
                'momentum': -0.05,
                'leverage': -0.15,
                'quality': 0.05
            },
            'growth_shock': {
                'growth': -0.20,
                'profitability': -0.15,
                'momentum': -0.10,
                'size': -0.08,
                'leverage': -0.12
            },
            'volatility_spike': {
                'volatility': -0.25,
                'momentum': -0.12,
                'quality': 0.10,
                'size': -0.15,
                'leverage': -0.10
            }
        }
    
    def _apply_stress_scenario(self, positions: pd.Series, factor_shocks: Dict[str, float]) -> float:
        """Apply stress scenario to calculate portfolio impact"""
        # Simplified stress test calculation
        total_impact = 0.0
        
        # Calculate factor exposures
        factor_exposures = self._calculate_portfolio_factor_exposures(positions)
        
        # Apply shocks
        for factor, shock in factor_shocks.items():
            if factor in factor_exposures:
                factor_impact = factor_exposures[factor] * shock * positions.abs().sum()
                total_impact += factor_impact
        
        return positions.sum() + total_impact
    
    def _calculate_position_stress_impacts(self, positions: pd.Series, 
                                         factor_shocks: Dict[str, float]) -> Dict[str, float]:
        """Calculate stress impacts at position level"""
        position_impacts = {}
        
        for security in positions.index:
            if security in self.style_model.factor_loadings.index:
                loadings = self.style_model.factor_loadings.loc[security]
                
                # Calculate impact for this security
                impact = 0.0
                for factor, shock in factor_shocks.items():
                    if factor in loadings.index:
                        impact += loadings[factor] * shock * positions[security]
                
                position_impacts[security] = impact
            else:
                position_impacts[security] = 0.0
        
        return position_impacts
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk model summary"""
        summary = {
            "model_status": {
                "is_fitted": self.is_fitted,
                "style_factors": len(self.style_model.style_factors) if self.style_model.is_fitted else 0,
                "crowding_indicators": len(self.crowding_model.crowding_indicators)
            },
            "exposure_limits": {
                "total_limits": len(self.exposure_limits),
                "factor_limits": len([l for l in self.exposure_limits if l.limit_type == "factor"]),
                "sector_limits": len([l for l in self.exposure_limits if l.limit_type == "sector"]),
                "crowding_limits": len([l for l in self.exposure_limits if l.limit_type == "crowding"])
            }
        }
        
        if self.current_risk_attribution:
            summary["current_risk"] = {
                "total_risk": self.current_risk_attribution.total_risk,
                "factor_risk_share": self.current_risk_attribution.risk_decomposition.get('factor_risk', 0),
                "specific_risk_share": self.current_risk_attribution.risk_decomposition.get('specific_risk', 0),
                "concentration_risk": self.current_risk_attribution.concentration_risk,
                "crowding_risk": self.current_risk_attribution.crowding_risk,
                "warnings_count": len(self.current_risk_attribution.risk_warnings)
            }
        
        if self.crowding_model.crowding_indicators:
            summary["crowding_status"] = {
                indicator.indicator_id: {
                    "current_level": indicator.current_level,
                    "percentile_rank": indicator.percentile_rank,
                    "risk_level": self._assess_crowding_risk_level(indicator.percentile_rank)
                }
                for indicator in self.crowding_model.crowding_indicators.values()
            }
        
        return summary
    
    def _assess_crowding_risk_level(self, percentile_rank: float) -> str:
        """Assess crowding risk level"""
        if percentile_rank >= 90:
            return "extreme"
        elif percentile_rank >= 75:
            return "high"
        elif percentile_rank >= 50:
            return "medium"
        else:
            return "low"


# Factory function
async def create_multi_factor_risk_model() -> MultiFactorRiskModel:
    """Create and initialize multi-factor risk model"""
    return MultiFactorRiskModel()


# Example usage
async def main():
    """Example usage of multi-factor risk model"""
    # Create sample data
    np.random.seed(42)
    n_periods = 252
    n_securities = 100
    
    # Sample returns data
    returns_data = pd.DataFrame(
        np.random.randn(n_periods, n_securities) * 0.02,
        columns=[f"STOCK_{i}" for i in range(n_securities)],
        index=pd.date_range(end=datetime.now(), periods=n_periods, freq='1D')
    )
    
    # Sample fundamental data
    fundamental_data = pd.DataFrame({
        'market_cap': np.random.lognormal(15, 1.5, n_securities),
        'book_to_price': np.random.uniform(0.5, 3.0, n_securities),
        'roe': np.random.uniform(0.05, 0.30, n_securities),
        'debt_to_equity': np.random.uniform(0.1, 2.0, n_securities),
        'revenue_growth': np.random.uniform(-0.1, 0.3, n_securities)
    }, index=returns_data.columns)
    
    # Sample positions
    positions = pd.Series(
        np.random.uniform(-0.05, 0.05, n_securities),
        index=returns_data.columns
    )
    
    # Create and fit risk model
    risk_model = await create_multi_factor_risk_model()
    await risk_model.fit(returns_data, fundamental_data)
    
    # Calculate portfolio risk
    risk_attribution = await risk_model.calculate_portfolio_risk(positions)
    
    print("Multi-Factor Risk Model Results:")
    print(f"Total Portfolio Risk: {risk_attribution.total_risk:.2%}")
    print(f"Factor Risk: {risk_attribution.risk_decomposition.get('factor_risk', 0):.1%}")
    print(f"Specific Risk: {risk_attribution.risk_decomposition.get('specific_risk', 0):.1%}")
    print(f"Concentration Risk: {risk_attribution.concentration_risk:.4f}")
    print(f"Crowding Risk: {risk_attribution.crowding_risk:.4f}")
    print(f"Risk Warnings: {len(risk_attribution.risk_warnings)}")
    
    # Run stress tests
    stress_results = await risk_model.run_stress_tests(positions)
    print(f"\nStress Test Results: {len(stress_results)} scenarios")
    for result in stress_results[:2]:  # Show first 2 scenarios
        print(f"  {result.scenario_name}: {result.return_impact:.2%} impact")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
