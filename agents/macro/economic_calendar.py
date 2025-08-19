"""
Economic Calendar and Data Provider

Fetches and analyzes economic indicators from various sources:
- FRED (Federal Reserve Economic Data)
- Trading Economics API
- Alpha Vantage
- Yahoo Finance
- Custom scrapers for central bank communications
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass

from .models import (
    EconomicIndicator, EconomicIndicatorType, ImpactSeverity,
    MarketImpact, CentralBankAction
)


class EconomicCalendarProvider:
    """
    Provider for economic calendar data and analysis
    
    Features:
    - Economic indicator fetching and parsing
    - Impact assessment and surprise calculation
    - Central bank communication analysis
    - Economic data forecasting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # API keys (would be loaded from config)
        self.alpha_vantage_key = self.config.get('alpha_vantage_key')
        self.trading_economics_key = self.config.get('trading_economics_key')
        self.fred_key = self.config.get('fred_key')
        
        # Indicator importance mapping
        self.indicator_importance = {
            EconomicIndicatorType.GDP: ImpactSeverity.CRITICAL,
            EconomicIndicatorType.INFLATION: ImpactSeverity.HIGH,
            EconomicIndicatorType.UNEMPLOYMENT: ImpactSeverity.HIGH,
            EconomicIndicatorType.INTEREST_RATES: ImpactSeverity.CRITICAL,
            EconomicIndicatorType.CONSUMER_CONFIDENCE: ImpactSeverity.MEDIUM,
            EconomicIndicatorType.MANUFACTURING_PMI: ImpactSeverity.HIGH,
            EconomicIndicatorType.SERVICES_PMI: ImpactSeverity.MEDIUM,
            EconomicIndicatorType.RETAIL_SALES: ImpactSeverity.MEDIUM,
            EconomicIndicatorType.HOUSING_DATA: ImpactSeverity.MEDIUM,
            EconomicIndicatorType.TRADE_BALANCE: ImpactSeverity.MEDIUM
        }
        
        # Central banks
        self.central_banks = {
            'fed': 'Federal Reserve',
            'ecb': 'European Central Bank',
            'boj': 'Bank of Japan',
            'boe': 'Bank of England',
            'pboc': 'People\'s Bank of China',
            'snb': 'Swiss National Bank',
            'boc': 'Bank of Canada',
            'rba': 'Reserve Bank of Australia'
        }
    
    async def get_recent_indicators(self, lookback_days: int = 30,
                                  countries: List[str] = None) -> List[EconomicIndicator]:
        """
        Get recent economic indicators
        
        Args:
            lookback_days: Days to look back
            countries: List of countries to include
            
        Returns:
            List of recent economic indicators
        """
        if countries is None:
            countries = ['US', 'EU', 'JP', 'UK', 'CN']
        
        indicators = []
        
        # In real implementation, would fetch from various APIs
        # For demo, generate realistic mock data
        indicators.extend(await self._generate_mock_indicators(countries, lookback_days))
        
        return indicators
    
    async def get_upcoming_indicators(self, lookahead_days: int = 30,
                                    countries: List[str] = None) -> List[EconomicIndicator]:
        """
        Get upcoming economic indicator releases
        
        Args:
            lookahead_days: Days to look ahead
            countries: List of countries to include
            
        Returns:
            List of upcoming economic indicators with forecasts
        """
        if countries is None:
            countries = ['US', 'EU', 'JP', 'UK', 'CN']
        
        # Generate mock upcoming indicators
        indicators = await self._generate_mock_upcoming_indicators(countries, lookahead_days)
        
        return indicators
    
    async def _generate_mock_indicators(self, countries: List[str], 
                                      lookback_days: int) -> List[EconomicIndicator]:
        """Generate realistic mock economic indicators"""
        indicators = []
        
        # Common indicator types and their typical ranges
        indicator_configs = {
            EconomicIndicatorType.GDP: {'range': (0.5, 4.0), 'frequency': 90},
            EconomicIndicatorType.INFLATION: {'range': (0.0, 5.0), 'frequency': 30},
            EconomicIndicatorType.UNEMPLOYMENT: {'range': (3.0, 10.0), 'frequency': 30},
            EconomicIndicatorType.MANUFACTURING_PMI: {'range': (40.0, 60.0), 'frequency': 30},
            EconomicIndicatorType.SERVICES_PMI: {'range': (40.0, 60.0), 'frequency': 30},
            EconomicIndicatorType.CONSUMER_CONFIDENCE: {'range': (80.0, 120.0), 'frequency': 30},
            EconomicIndicatorType.RETAIL_SALES: {'range': (-2.0, 5.0), 'frequency': 30}
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for country in countries:
            for indicator_type, config in indicator_configs.items():
                # Check if indicator should be released in this period
                frequency = config['frequency']
                days_since_start = (end_date - start_date).days
                
                if days_since_start >= frequency:
                    # Generate release date
                    release_date = start_date + timedelta(
                        days=np.random.randint(0, min(days_since_start, frequency))
                    )
                    
                    # Generate values
                    value_range = config['range']
                    current_value = np.random.uniform(*value_range)
                    previous_value = current_value + np.random.normal(0, 0.1)
                    forecast = current_value + np.random.normal(0, 0.05)
                    
                    # Determine impacts based on surprise
                    surprise = (current_value - forecast) / abs(forecast) if forecast != 0 else 0
                    
                    indicator = EconomicIndicator(
                        indicator_type=indicator_type,
                        country=country,
                        value=current_value,
                        previous_value=previous_value,
                        forecast=forecast,
                        release_date=release_date,
                        next_release=release_date + timedelta(days=frequency),
                        importance=self.indicator_importance[indicator_type],
                        currency_impact=self._assess_currency_impact(surprise, indicator_type),
                        equity_impact=self._assess_equity_impact(surprise, indicator_type),
                        bond_impact=self._assess_bond_impact(surprise, indicator_type)
                    )
                    
                    indicators.append(indicator)
        
        return indicators
    
    async def _generate_mock_upcoming_indicators(self, countries: List[str],
                                               lookahead_days: int) -> List[EconomicIndicator]:
        """Generate mock upcoming indicators with forecasts"""
        indicators = []
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=lookahead_days)
        
        # Key upcoming indicators
        upcoming_schedule = [
            (EconomicIndicatorType.GDP, 7),  # GDP in 7 days
            (EconomicIndicatorType.INFLATION, 14),  # CPI in 14 days
            (EconomicIndicatorType.UNEMPLOYMENT, 5),  # Jobs report in 5 days
            (EconomicIndicatorType.MANUFACTURING_PMI, 2),  # PMI in 2 days
            (EconomicIndicatorType.RETAIL_SALES, 10)  # Retail sales in 10 days
        ]
        
        for country in countries[:2]:  # Limit to 2 countries for demo
            for indicator_type, days_ahead in upcoming_schedule:
                release_date = start_date + timedelta(days=days_ahead)
                
                if release_date <= end_date:
                    # Generate forecast based on historical patterns
                    forecast = self._generate_realistic_forecast(indicator_type, country)
                    
                    indicator = EconomicIndicator(
                        indicator_type=indicator_type,
                        country=country,
                        value=forecast,  # Will be actual value when released
                        previous_value=forecast * (1 + np.random.normal(0, 0.02)),
                        forecast=forecast,
                        release_date=release_date,
                        next_release=None,  # Unknown for future
                        importance=self.indicator_importance[indicator_type],
                        currency_impact=MarketImpact.NEUTRAL,  # TBD
                        equity_impact=MarketImpact.NEUTRAL,   # TBD
                        bond_impact=MarketImpact.NEUTRAL      # TBD
                    )
                    
                    indicators.append(indicator)
        
        return indicators
    
    def _generate_realistic_forecast(self, indicator_type: EconomicIndicatorType, 
                                   country: str) -> float:
        """Generate realistic forecast values for indicators"""
        # Country-specific base values
        base_values = {
            'US': {
                EconomicIndicatorType.GDP: 2.5,
                EconomicIndicatorType.INFLATION: 3.2,
                EconomicIndicatorType.UNEMPLOYMENT: 4.1,
                EconomicIndicatorType.MANUFACTURING_PMI: 51.2,
                EconomicIndicatorType.RETAIL_SALES: 0.4
            },
            'EU': {
                EconomicIndicatorType.GDP: 1.8,
                EconomicIndicatorType.INFLATION: 2.8,
                EconomicIndicatorType.UNEMPLOYMENT: 6.2,
                EconomicIndicatorType.MANUFACTURING_PMI: 49.8,
                EconomicIndicatorType.RETAIL_SALES: 0.2
            }
        }
        
        country_defaults = base_values.get(country, base_values['US'])
        base_value = country_defaults.get(indicator_type, 50.0)
        
        # Add some random variation
        return base_value * (1 + np.random.normal(0, 0.05))
    
    def _assess_currency_impact(self, surprise: float, 
                              indicator_type: EconomicIndicatorType) -> MarketImpact:
        """Assess currency impact based on surprise factor"""
        # Positive surprise generally strengthens currency
        abs_surprise = abs(surprise)
        
        if abs_surprise < 0.01:  # Less than 1% surprise
            return MarketImpact.NEUTRAL
        elif abs_surprise < 0.05:  # 1-5% surprise
            return MarketImpact.BULLISH if surprise > 0 else MarketImpact.BEARISH
        else:  # > 5% surprise
            return MarketImpact.VERY_BULLISH if surprise > 0 else MarketImpact.VERY_BEARISH
    
    def _assess_equity_impact(self, surprise: float,
                            indicator_type: EconomicIndicatorType) -> MarketImpact:
        """Assess equity market impact"""
        # Growth indicators generally positive for equities
        growth_indicators = [
            EconomicIndicatorType.GDP,
            EconomicIndicatorType.RETAIL_SALES,
            EconomicIndicatorType.MANUFACTURING_PMI
        ]
        
        if indicator_type in growth_indicators:
            if surprise > 0.02:
                return MarketImpact.BULLISH
            elif surprise < -0.02:
                return MarketImpact.BEARISH
        
        # Inflation can be negative for equities if too high
        if indicator_type == EconomicIndicatorType.INFLATION:
            if surprise > 0.05:  # Much higher than expected
                return MarketImpact.BEARISH
            elif surprise < -0.05:  # Much lower than expected
                return MarketImpact.BULLISH
        
        return MarketImpact.NEUTRAL
    
    def _assess_bond_impact(self, surprise: float,
                          indicator_type: EconomicIndicatorType) -> MarketImpact:
        """Assess bond market impact"""
        # Growth surprises generally negative for bonds (higher yields)
        # Inflation surprises generally negative for bonds
        
        inflation_sensitive = [
            EconomicIndicatorType.INFLATION,
            EconomicIndicatorType.GDP,
            EconomicIndicatorType.RETAIL_SALES
        ]
        
        if indicator_type in inflation_sensitive:
            if surprise > 0.02:
                return MarketImpact.BEARISH  # Higher yields, lower bond prices
            elif surprise < -0.02:
                return MarketImpact.BULLISH  # Lower yields, higher bond prices
        
        return MarketImpact.NEUTRAL
    
    async def analyze_central_bank_communications(self, lookback_days: int = 30) -> List[CentralBankAction]:
        """
        Analyze recent central bank communications and actions
        
        In real implementation, would parse:
        - Fed statements and meeting minutes
        - ECB press conferences
        - BOJ policy statements
        - Central bank speeches
        """
        actions = []
        
        # Generate mock central bank actions
        cb_actions = [
            {
                'bank': 'Federal Reserve',
                'action_type': 'rate_decision',
                'days_ago': 14,
                'rate_change': 0.25,
                'hawkish_score': 0.3
            },
            {
                'bank': 'European Central Bank',
                'action_type': 'guidance',
                'days_ago': 7,
                'rate_change': 0.0,
                'hawkish_score': -0.2
            },
            {
                'bank': 'Bank of Japan',
                'action_type': 'speech',
                'days_ago': 3,
                'rate_change': 0.0,
                'hawkish_score': -0.5
            }
        ]
        
        for action_data in cb_actions:
            if action_data['days_ago'] <= lookback_days:
                action_date = datetime.now() - timedelta(days=action_data['days_ago'])
                
                action = CentralBankAction(
                    bank=action_data['bank'],
                    action_type=action_data['action_type'],
                    date=action_date,
                    description=f"{action_data['bank']} {action_data['action_type']}",
                    key_points=[
                        "Monitoring inflation closely",
                        "Data-dependent approach",
                        "Committed to price stability"
                    ],
                    current_rate=5.25 if 'Federal' in action_data['bank'] else 4.0,
                    previous_rate=5.0 if 'Federal' in action_data['bank'] else 4.0,
                    rate_change=action_data['rate_change'],
                    market_expectation=action_data['rate_change'],
                    surprise_factor=0.0,  # As expected
                    market_impact=MarketImpact.NEUTRAL,
                    currency_strength=action_data['hawkish_score'] * 0.5,
                    bond_yield_impact=action_data['rate_change'] * 25,  # 25bp per rate change
                    hawkish_dovish_score=action_data['hawkish_score']
                )
                
                actions.append(action)
        
        return actions
    
    async def get_economic_calendar_events(self, start_date: datetime,
                                         end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get economic calendar events for a date range
        
        Returns events with timing, importance, and impact assessments
        """
        events = []
        
        # Mock calendar events
        current_date = start_date
        while current_date <= end_date:
            # Check if it's a weekday
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                
                # Generate random events based on day
                if current_date.day % 7 == 0:  # Weekly indicator
                    events.append({
                        'date': current_date,
                        'time': '08:30',
                        'country': 'US',
                        'indicator': 'Initial Jobless Claims',
                        'importance': 'medium',
                        'previous': '220K',
                        'forecast': '225K',
                        'currency_impact': 'USD'
                    })
                
                if current_date.day == 1:  # Monthly PMI
                    events.append({
                        'date': current_date,
                        'time': '09:45',
                        'country': 'US',
                        'indicator': 'Manufacturing PMI',
                        'importance': 'high',
                        'previous': '51.2',
                        'forecast': '51.0',
                        'currency_impact': 'USD'
                    })
                
                if current_date.day == 15:  # Monthly retail sales
                    events.append({
                        'date': current_date,
                        'time': '08:30',
                        'country': 'US',
                        'indicator': 'Retail Sales',
                        'importance': 'medium',
                        'previous': '0.4%',
                        'forecast': '0.3%',
                        'currency_impact': 'USD'
                    })
            
            current_date += timedelta(days=1)
        
        return events
    
    def calculate_economic_surprise_index(self, indicators: List[EconomicIndicator]) -> float:
        """
        Calculate Economic Surprise Index
        
        Measures whether economic data is coming in better or worse than expected
        Positive values indicate data beating expectations
        """
        if not indicators:
            return 0.0
        
        # Weight by indicator importance
        importance_weights = {
            ImpactSeverity.CRITICAL: 4.0,
            ImpactSeverity.HIGH: 3.0,
            ImpactSeverity.MEDIUM: 2.0,
            ImpactSeverity.LOW: 1.0
        }
        
        weighted_surprises = []
        total_weight = 0
        
        for indicator in indicators:
            surprise = indicator.surprise_factor
            weight = importance_weights[indicator.importance]
            
            weighted_surprises.append(surprise * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return sum(weighted_surprises) / total_weight
    
    def assess_macro_momentum(self, indicators: List[EconomicIndicator]) -> Dict[str, float]:
        """
        Assess momentum across different categories of indicators
        
        Returns momentum scores for growth, inflation, employment, etc.
        """
        momentum_by_category = {
            'growth': [],
            'inflation': [],
            'employment': [],
            'confidence': []
        }
        
        # Categorize indicators
        for indicator in indicators:
            momentum = indicator.momentum
            
            if indicator.indicator_type in [EconomicIndicatorType.GDP, 
                                          EconomicIndicatorType.RETAIL_SALES,
                                          EconomicIndicatorType.MANUFACTURING_PMI]:
                momentum_by_category['growth'].append(momentum)
            
            elif indicator.indicator_type == EconomicIndicatorType.INFLATION:
                momentum_by_category['inflation'].append(momentum)
            
            elif indicator.indicator_type == EconomicIndicatorType.UNEMPLOYMENT:
                momentum_by_category['employment'].append(-momentum)  # Invert for employment
            
            elif indicator.indicator_type == EconomicIndicatorType.CONSUMER_CONFIDENCE:
                momentum_by_category['confidence'].append(momentum)
        
        # Calculate average momentum for each category
        result = {}
        for category, momentums in momentum_by_category.items():
            if momentums:
                result[category] = np.mean(momentums)
            else:
                result[category] = 0.0
        
        return result
