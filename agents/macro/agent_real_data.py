"""
Real Data Macro Agent
Uses Polygon.io adapter for economic indicators and currency data
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
from common.data_adapters.polygon_adapter import PolygonAdapter

load_dotenv('env_real_keys.env')

class RealDataMacroAgent(BaseAgent):
    """Macro Agent with real market data from Polygon.io"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataMacroAgent", config)
        self.polygon_adapter = PolygonAdapter(config)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        return await self.analyze_macro_environment(**kwargs)
    
    async def analyze_macro_environment(self, **kwargs) -> Dict[str, Any]:
        """Analyze macro environment using real market data"""
        print(f"🌍 Real Data Macro Agent: Analyzing macro environment")
        
        try:
            # Get economic indicators
            economic_indicators = await self.polygon_adapter.get_economic_indicators()
            
            # Get currency data
            currency_data = await self.polygon_adapter.get_currency_data()
            
            # Analyze macro trends
            macro_trends = await self._analyze_macro_trends(economic_indicators, currency_data)
            
            # Generate macro signals
            macro_signals = self._generate_macro_signals(economic_indicators, currency_data)
            
            return {
                'timestamp': datetime.now(),
                'economic_indicators': economic_indicators,
                'currency_data': currency_data,
                'macro_trends': macro_trends,
                'macro_signals': macro_signals,
                'data_source': 'Polygon.io (Real Market Data)'
            }
            
        except Exception as e:
            print(f"❌ Error analyzing macro environment: {e}")
            return self._create_empty_macro_analysis()
    
    async def _analyze_macro_trends(self, economic_indicators: Dict[str, Any], 
                                  currency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze macro trends from real data"""
        
        trends = {
            'timestamp': datetime.now()
        }
        
        # Market trend analysis
        market_trends = []
        for index_name, data in economic_indicators.items():
            if isinstance(data, dict) and 'change_percent' in data:
                market_trends.append({
                    'index': index_name,
                    'change_percent': data['change_percent'],
                    'price': data['price']
                })
        
        # Calculate market breadth
        positive_markets = [m for m in market_trends if m['change_percent'] > 0]
        negative_markets = [m for m in market_trends if m['change_percent'] < 0]
        
        trends['market_breadth'] = {
            'positive_count': len(positive_markets),
            'negative_count': len(negative_markets),
            'total_count': len(market_trends),
            'breadth_ratio': len(positive_markets) / len(market_trends) if market_trends else 0.5
        }
        
        # Risk-on/risk-off analysis
        risk_assets = ['SPY', 'QQQ', 'IWM']  # Stocks
        safe_assets = ['GLD', 'TLT']  # Gold and bonds
        
        risk_performance = np.mean([m['change_percent'] for m in market_trends if m['index'].upper() in risk_assets])
        safe_performance = np.mean([m['change_percent'] for m in market_trends if m['index'].upper() in safe_assets])
        
        trends['risk_sentiment'] = {
            'risk_assets_performance': risk_performance,
            'safe_assets_performance': safe_performance,
            'risk_on_off_ratio': risk_performance - safe_performance,
            'sentiment': 'risk_on' if risk_performance > safe_performance else 'risk_off'
        }
        
        # Currency correlation analysis
        currency_trends = []
        for currency, data in currency_data.get('currencies', {}).items():
            if isinstance(data, dict) and 'change_percent' in data:
                currency_trends.append({
                    'currency': currency,
                    'change_percent': data['change_percent']
                })
        
        trends['currency_trends'] = currency_trends
        
        # Dollar strength analysis
        usd_data = currency_data.get('currencies', {}).get('USD', {})
        if usd_data and 'change_percent' in usd_data:
            trends['dollar_strength'] = {
                'usd_change': usd_data['change_percent'],
                'dollar_regime': 'strong' if usd_data['change_percent'] > 0.5 else 'weak' if usd_data['change_percent'] < -0.5 else 'neutral'
            }
        
        return trends
    
    def _generate_macro_signals(self, economic_indicators: Dict[str, Any], 
                              currency_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate macro-based trading signals"""
        signals = []
        
        # Market breadth signals
        positive_count = 0
        total_count = 0
        
        for index_name, data in economic_indicators.items():
            if isinstance(data, dict) and 'change_percent' in data:
                total_count += 1
                if data['change_percent'] > 0:
                    positive_count += 1
        
        breadth_ratio = positive_count / total_count if total_count > 0 else 0.5
        
        if breadth_ratio > 0.7:
            signals.append({
                'type': 'BROAD_MARKET_STRENGTH',
                'strength': 'strong',
                'message': f"Broad market strength: {positive_count}/{total_count} indices positive"
            })
        elif breadth_ratio < 0.3:
            signals.append({
                'type': 'BROAD_MARKET_WEAKNESS',
                'strength': 'strong',
                'message': f"Broad market weakness: {positive_count}/{total_count} indices positive"
            })
        
        # Risk sentiment signals
        risk_assets = ['SPY', 'QQQ', 'IWM']
        safe_assets = ['GLD', 'TLT']
        
        risk_performance = np.mean([economic_indicators.get(asset, {}).get('change_percent', 0) for asset in risk_assets])
        safe_performance = np.mean([economic_indicators.get(asset, {}).get('change_percent', 0) for asset in safe_assets])
        
        if risk_performance > safe_performance + 1.0:
            signals.append({
                'type': 'RISK_ON_ENVIRONMENT',
                'strength': 'strong',
                'message': f"Risk-on environment: stocks +{risk_performance:.1f}% vs safe assets +{safe_performance:.1f}%"
            })
        elif safe_performance > risk_performance + 1.0:
            signals.append({
                'type': 'RISK_OFF_ENVIRONMENT',
                'strength': 'strong',
                'message': f"Risk-off environment: safe assets +{safe_performance:.1f}% vs stocks +{risk_performance:.1f}%"
            })
        
        # Currency signals
        usd_data = currency_data.get('currencies', {}).get('USD', {})
        if usd_data and 'change_percent' in usd_data:
            if usd_data['change_percent'] > 0.5:
                signals.append({
                    'type': 'STRONG_DOLLAR',
                    'strength': 'medium',
                    'message': f"Strong dollar: USD +{usd_data['change_percent']:.1f}%"
                })
            elif usd_data['change_percent'] < -0.5:
                signals.append({
                    'type': 'WEAK_DOLLAR',
                    'strength': 'medium',
                    'message': f"Weak dollar: USD {usd_data['change_percent']:.1f}%"
                })
        
        # Gold signals
        gold_data = economic_indicators.get('gold', {})
        if gold_data and 'change_percent' in gold_data:
            if gold_data['change_percent'] > 2.0:
                signals.append({
                    'type': 'GOLD_RALLY',
                    'strength': 'strong',
                    'message': f"Gold rally: +{gold_data['change_percent']:.1f}% - safe haven demand"
                })
        
        # Bond signals
        bonds_data = economic_indicators.get('bonds', {})
        if bonds_data and 'change_percent' in bonds_data:
            if bonds_data['change_percent'] > 1.0:
                signals.append({
                    'type': 'BOND_RALLY',
                    'strength': 'medium',
                    'message': f"Bond rally: +{bonds_data['change_percent']:.1f}% - flight to safety"
                })
        
        return signals
    
    def _create_empty_macro_analysis(self) -> Dict[str, Any]:
        """Create empty macro analysis when data is unavailable"""
        return {
            'timestamp': datetime.now(),
            'economic_indicators': {},
            'currency_data': {'currencies': {}},
            'macro_trends': {
                'timestamp': datetime.now(),
                'market_breadth': {
                    'positive_count': 0,
                    'negative_count': 0,
                    'total_count': 0,
                    'breadth_ratio': 0.5
                },
                'risk_sentiment': {
                    'risk_assets_performance': 0.0,
                    'safe_assets_performance': 0.0,
                    'risk_on_off_ratio': 0.0,
                    'sentiment': 'neutral'
                },
                'currency_trends': [],
                'dollar_strength': {
                    'usd_change': 0.0,
                    'dollar_regime': 'neutral'
                }
            },
            'macro_signals': [],
            'data_source': 'Polygon.io (Real Market Data)'
        }
