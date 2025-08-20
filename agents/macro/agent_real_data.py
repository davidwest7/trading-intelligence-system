"""
Real Data Macro Agent
Uses FRED API adapter for economic indicators and currency data
"""
import asyncio
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')
from common.models import BaseAgent
from common.data_adapters.fred_adapter import FREDAdapter

load_dotenv('env_real_keys.env')

class RealDataMacroAgent(BaseAgent):
    """Macro Agent with real economic data from FRED API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataMacroAgent", config)
        self.fred_adapter = FREDAdapter(config)
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache (economic data changes slowly)
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        return await self.analyze_macro_environment(**kwargs)
    
    async def analyze_macro_environment(self, **kwargs) -> Dict[str, Any]:
        """Analyze macro environment using real FRED data"""
        print(f"🌍 Real Data Macro Agent: Analyzing macro economic environment")
        
        try:
            # Use cache if fresh
            cache_key = 'macro_analysis'
            if cache_key in self.cache and (datetime.now().timestamp() - self.cache[cache_key]['timestamp']) < self.cache_ttl:
                return self.cache[cache_key]['data']
            
            # Get macro analysis from FRED adapter
            macro_analysis = await self.fred_adapter.analyze_macro_environment()
            
            # Generate macro signals
            macro_signals = await self._generate_macro_signals(macro_analysis)
            
            # Create comprehensive macro analysis
            analysis_result = {
                'timestamp': datetime.now(),
                'macro_analysis': macro_analysis,
                'macro_signals': macro_signals,
                'economic_indicators': self._extract_economic_indicators(macro_analysis),
                'risk_assessment': self._assess_macro_risk(macro_analysis),
                'trading_implications': self._generate_trading_implications(macro_signals),
                'data_source': 'FRED API (Real Economic Data)'
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': analysis_result,
                'timestamp': datetime.now().timestamp()
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error analyzing macro environment: {e}")
            return self._create_empty_macro_analysis()
    
    async def _generate_macro_signals(self, macro_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate macro trading signals"""
        try:
            signals = macro_analysis.get('macro_signals', {})
            
            # Enhanced signal generation
            enhanced_signals = {
                'gdp_signal': signals.get('gdp_growth', 'neutral'),
                'inflation_signal': signals.get('inflation_trend', 'neutral'),
                'employment_signal': signals.get('employment_health', 'neutral'),
                'monetary_signal': signals.get('monetary_policy', 'neutral'),
                'overall_macro_signal': signals.get('overall_macro', 'neutral'),
                'confidence_level': self._calculate_confidence_level(macro_analysis),
                'signal_strength': self._calculate_signal_strength(signals)
            }
            
            return enhanced_signals
            
        except Exception as e:
            print(f"Error generating macro signals: {e}")
            return {
                'gdp_signal': 'neutral',
                'inflation_signal': 'neutral',
                'employment_signal': 'neutral',
                'monetary_signal': 'neutral',
                'overall_macro_signal': 'neutral',
                'confidence_level': 'low',
                'signal_strength': 'weak'
            }
    
    def _extract_economic_indicators(self, macro_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key economic indicators"""
        try:
            latest_values = macro_analysis.get('latest_values', {})
            
            indicators = {}
            for indicator, data in latest_values.items():
                indicators[indicator] = {
                    'value': data.get('value'),
                    'date': data.get('date'),
                    'units': data.get('units', ''),
                    'trend': self._calculate_trend(indicator, data.get('value'))
                }
            
            return indicators
            
        except Exception as e:
            print(f"Error extracting economic indicators: {e}")
            return {}
    
    def _assess_macro_risk(self, macro_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess macro economic risks"""
        try:
            signals = macro_analysis.get('macro_signals', {})
            latest_values = macro_analysis.get('latest_values', {})
            
            risk_factors = []
            risk_level = 'low'
            
            # GDP Risk
            if 'gdp' in latest_values:
                gdp_value = float(latest_values['gdp']['value'])
                if gdp_value < 15000:
                    risk_factors.append('Low GDP growth')
                    risk_level = 'high'
            
            # Inflation Risk
            if 'cpi' in latest_values:
                cpi_value = float(latest_values['cpi']['value'])
                if cpi_value > 300:
                    risk_factors.append('High inflation')
                    risk_level = 'high'
            
            # Employment Risk
            if 'unemployment' in latest_values:
                unemp_value = float(latest_values['unemployment']['value'])
                if unemp_value > 6.0:
                    risk_factors.append('High unemployment')
                    risk_level = 'medium'
            
            # Monetary Policy Risk
            if 'fed_funds_rate' in latest_values:
                fed_value = float(latest_values['fed_funds_rate']['value'])
                if fed_value > 5.0:
                    risk_factors.append('Restrictive monetary policy')
                    risk_level = 'medium'
            
            return {
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'risk_score': self._calculate_risk_score(risk_factors),
                'recommendations': self._generate_risk_recommendations(risk_factors)
            }
            
        except Exception as e:
            print(f"Error assessing macro risk: {e}")
            return {
                'risk_level': 'unknown',
                'risk_factors': [],
                'risk_score': 0,
                'recommendations': []
            }
    
    def _generate_trading_implications(self, macro_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading implications from macro signals"""
        try:
            overall_signal = macro_signals.get('overall_macro_signal', 'neutral')
            confidence = macro_signals.get('confidence_level', 'low')
            
            implications = {
                'market_outlook': self._get_market_outlook(overall_signal),
                'sector_preferences': self._get_sector_preferences(macro_signals),
                'asset_allocation': self._get_asset_allocation(overall_signal),
                'risk_management': self._get_risk_management(overall_signal),
                'timing_considerations': self._get_timing_considerations(confidence)
            }
            
            return implications
            
        except Exception as e:
            print(f"Error generating trading implications: {e}")
            return {
                'market_outlook': 'neutral',
                'sector_preferences': [],
                'asset_allocation': 'balanced',
                'risk_management': 'standard',
                'timing_considerations': 'medium-term'
            }
    
    def _calculate_confidence_level(self, macro_analysis: Dict[str, Any]) -> str:
        """Calculate confidence level in macro analysis"""
        try:
            latest_values = macro_analysis.get('latest_values', {})
            
            # Count available indicators
            available_indicators = len(latest_values)
            
            if available_indicators >= 5:
                return 'high'
            elif available_indicators >= 3:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            return 'low'
    
    def _calculate_signal_strength(self, signals: Dict[str, Any]) -> str:
        """Calculate signal strength"""
        try:
            bullish_count = sum(1 for signal in signals.values() if signal == 'bullish')
            bearish_count = sum(1 for signal in signals.values() if signal == 'bearish')
            
            total_signals = len(signals)
            if total_signals == 0:
                return 'weak'
            
            max_count = max(bullish_count, bearish_count)
            strength_ratio = max_count / total_signals
            
            if strength_ratio >= 0.8:
                return 'strong'
            elif strength_ratio >= 0.6:
                return 'moderate'
            else:
                return 'weak'
                
        except Exception as e:
            return 'weak'
    
    def _calculate_trend(self, indicator: str, value: Any) -> str:
        """Calculate trend for an indicator"""
        try:
            if not value:
                return 'unknown'
            
            # This is a simplified trend calculation
            # In a real implementation, you'd compare with historical values
            return 'stable'
            
        except Exception as e:
            return 'unknown'
    
    def _calculate_risk_score(self, risk_factors: List[str]) -> int:
        """Calculate numerical risk score"""
        try:
            score = len(risk_factors) * 10
            return min(score, 100)
        except Exception as e:
            return 0
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if 'Low GDP growth' in risk_factors:
            recommendations.append('Consider defensive sectors (utilities, consumer staples)')
        
        if 'High inflation' in risk_factors:
            recommendations.append('Consider inflation-protected assets (TIPS, commodities)')
        
        if 'High unemployment' in risk_factors:
            recommendations.append('Reduce exposure to consumer discretionary stocks')
        
        if 'Restrictive monetary policy' in risk_factors:
            recommendations.append('Consider fixed income and defensive positioning')
        
        if not recommendations:
            recommendations.append('Maintain balanced portfolio allocation')
        
        return recommendations
    
    def _get_market_outlook(self, signal: str) -> str:
        """Get market outlook based on macro signal"""
        if signal == 'bullish':
            return 'positive'
        elif signal == 'bearish':
            return 'negative'
        else:
            return 'neutral'
    
    def _get_sector_preferences(self, signals: Dict[str, Any]) -> List[str]:
        """Get sector preferences based on macro signals"""
        preferences = []
        
        if signals.get('gdp_signal') == 'bullish':
            preferences.append('Technology')
            preferences.append('Consumer Discretionary')
        
        if signals.get('inflation_signal') == 'bearish':
            preferences.append('Energy')
            preferences.append('Materials')
        
        if signals.get('employment_signal') == 'bullish':
            preferences.append('Financials')
            preferences.append('Industrials')
        
        if not preferences:
            preferences.append('Balanced sector allocation')
        
        return preferences
    
    def _get_asset_allocation(self, signal: str) -> str:
        """Get asset allocation recommendation"""
        if signal == 'bullish':
            return 'equity-heavy'
        elif signal == 'bearish':
            return 'defensive'
        else:
            return 'balanced'
    
    def _get_risk_management(self, signal: str) -> str:
        """Get risk management recommendation"""
        if signal == 'bearish':
            return 'increase-hedging'
        elif signal == 'bullish':
            return 'reduce-hedging'
        else:
            return 'maintain-current'
    
    def _get_timing_considerations(self, confidence: str) -> str:
        """Get timing considerations"""
        if confidence == 'high':
            return 'immediate-action'
        elif confidence == 'medium':
            return 'gradual-implementation'
        else:
            return 'wait-and-see'
    
    def _create_empty_macro_analysis(self) -> Dict[str, Any]:
        """Create empty macro analysis when data is unavailable"""
        return {
            'timestamp': datetime.now(),
            'macro_analysis': {},
            'macro_signals': {
                'gdp_signal': 'neutral',
                'inflation_signal': 'neutral',
                'employment_signal': 'neutral',
                'monetary_signal': 'neutral',
                'overall_macro_signal': 'neutral',
                'confidence_level': 'low',
                'signal_strength': 'weak'
            },
            'economic_indicators': {},
            'risk_assessment': {
                'risk_level': 'unknown',
                'risk_factors': [],
                'risk_score': 0,
                'recommendations': ['No data available']
            },
            'trading_implications': {
                'market_outlook': 'neutral',
                'sector_preferences': ['No data available'],
                'asset_allocation': 'balanced',
                'risk_management': 'standard',
                'timing_considerations': 'wait-and-see'
            },
            'data_source': 'FRED API (no data available)'
        }
