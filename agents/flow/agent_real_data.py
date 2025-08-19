"""
Real Data Flow Agent
Uses Polygon.io adapter for Level 2 data and institutional flow analysis
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

class RealDataFlowAgent(BaseAgent):
    """Flow Analysis Agent with real market data from Polygon.io"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataFlowAgent", config)
        self.polygon_adapter = PolygonAdapter(config)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        tickers = kwargs.get('tickers', args[0] if args else ['AAPL', 'TSLA', 'SPY'])
        return await self.analyze_market_flow(tickers, **kwargs)
    
    async def analyze_market_flow(self, tickers: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze market flow using real data"""
        print(f"🌊 Real Data Flow Agent: Analyzing flow for {len(tickers)} tickers")
        
        results = {}
        
        for ticker in tickers:
            try:
                # Get Level 2 market data
                level2_data = await self.polygon_adapter.get_level2_data(ticker)
                
                # Get institutional flow data
                institutional_flow = await self.polygon_adapter.get_institutional_flow(ticker)
                
                # Get unusual options activity
                unusual_options = await self.polygon_adapter.get_unusual_options_activity(ticker)
                
                # Get real-time quote for additional context
                quote = await self.polygon_adapter.get_real_time_quote(ticker)
                
                # Analyze flow patterns
                flow_analysis = await self._analyze_flow_patterns(
                    ticker, level2_data, institutional_flow, unusual_options, quote
                )
                
                results[ticker] = flow_analysis
                
            except Exception as e:
                print(f"❌ Error analyzing flow for {ticker}: {e}")
                results[ticker] = self._create_empty_flow_analysis(ticker)
        
        # Generate overall flow signals
        overall_flow = await self._generate_overall_flow_signals(results)
        
        return {
            'timestamp': datetime.now(),
            'tickers_analyzed': len(tickers),
            'flow_analysis': results,
            'overall_flow': overall_flow,
            'data_source': 'Polygon.io (Real Market Data)'
        }
    
    async def _analyze_flow_patterns(self, ticker: str, level2_data: Dict[str, Any],
                                   institutional_flow: Dict[str, Any],
                                   unusual_options: List[Dict[str, Any]],
                                   quote: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze flow patterns from real data"""
        
        analysis = {
            'ticker': ticker,
            'current_price': quote['price'],
            'timestamp': datetime.now()
        }
        
        # Level 2 analysis
        analysis['bid_ask_spread'] = level2_data['ask'] - level2_data['bid']
        analysis['spread_percentage'] = (analysis['bid_ask_spread'] / quote['price']) * 100
        analysis['bid_size'] = level2_data['bid_size']
        analysis['ask_size'] = level2_data['ask_size']
        analysis['order_imbalance'] = analysis['bid_size'] - analysis['ask_size']
        
        # Institutional flow analysis
        analysis['total_volume'] = institutional_flow['total_volume']
        analysis['large_trades_count'] = institutional_flow['large_trades_count']
        analysis['avg_trade_size'] = institutional_flow['avg_trade_size']
        analysis['institutional_score'] = institutional_flow['institutional_flow_score']
        
        # Options flow analysis
        analysis['unusual_options_count'] = len(unusual_options)
        analysis['options_flow_value'] = sum(opt.get('premium', 0) for opt in unusual_options)
        
        # Flow signals
        analysis['flow_signals'] = self._generate_flow_signals(analysis)
        
        # Flow regime detection
        analysis['flow_regime'] = self._detect_flow_regime(analysis)
        
        return analysis
    
    def _generate_flow_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate flow-based trading signals"""
        signals = []
        
        # Bid-ask spread signals
        if analysis['spread_percentage'] < 0.1:
            signals.append({
                'type': 'TIGHT_SPREAD',
                'strength': 'strong',
                'message': f"Tight bid-ask spread ({analysis['spread_percentage']:.3f}%) - high liquidity"
            })
        elif analysis['spread_percentage'] > 0.5:
            signals.append({
                'type': 'WIDE_SPREAD',
                'strength': 'medium',
                'message': f"Wide bid-ask spread ({analysis['spread_percentage']:.3f}%) - low liquidity"
            })
        
        # Order imbalance signals
        if analysis['order_imbalance'] > 1000:
            signals.append({
                'type': 'BULLISH_IMBALANCE',
                'strength': 'strong',
                'message': f"Bullish order imbalance (+{analysis['order_imbalance']:,} shares)"
            })
        elif analysis['order_imbalance'] < -1000:
            signals.append({
                'type': 'BEARISH_IMBALANCE',
                'strength': 'strong',
                'message': f"Bearish order imbalance ({analysis['order_imbalance']:,} shares)"
            })
        
        # Institutional flow signals
        if analysis['institutional_score'] > 0.3:
            signals.append({
                'type': 'HIGH_INSTITUTIONAL_FLOW',
                'strength': 'strong',
                'message': f"High institutional flow ({analysis['institutional_score']:.1%})"
            })
        
        if analysis['large_trades_count'] > 10:
            signals.append({
                'type': 'LARGE_TRADE_ACTIVITY',
                'strength': 'medium',
                'message': f"High large trade activity ({analysis['large_trades_count']} trades)"
            })
        
        # Options flow signals
        if analysis['unusual_options_count'] > 5:
            signals.append({
                'type': 'UNUSUAL_OPTIONS_ACTIVITY',
                'strength': 'strong',
                'message': f"Unusual options activity ({analysis['unusual_options_count']} contracts)"
            })
        
        if analysis['options_flow_value'] > 1000000:
            signals.append({
                'type': 'HIGH_OPTIONS_FLOW',
                'strength': 'strong',
                'message': f"High options flow value (${analysis['options_flow_value']:,.0f})"
            })
        
        return signals
    
    def _detect_flow_regime(self, analysis: Dict[str, Any]) -> str:
        """Detect the current flow regime"""
        bullish_signals = 0
        bearish_signals = 0
        
        for signal in analysis['flow_signals']:
            if 'BULLISH' in signal['type']:
                bullish_signals += 1
            elif 'BEARISH' in signal['type']:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish_flow'
        elif bearish_signals > bullish_signals:
            return 'bearish_flow'
        else:
            return 'neutral_flow'
    
    async def _generate_overall_flow_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall flow signals"""
        bullish_flow_count = 0
        bearish_flow_count = 0
        neutral_flow_count = 0
        total_volume = 0
        total_institutional_score = 0
        
        for ticker, analysis in results.items():
            total_volume += analysis.get('total_volume', 0)
            total_institutional_score += analysis.get('institutional_score', 0)
            
            if analysis.get('flow_regime') == 'bullish_flow':
                bullish_flow_count += 1
            elif analysis.get('flow_regime') == 'bearish_flow':
                bearish_flow_count += 1
            else:
                neutral_flow_count += 1
        
        overall_regime = 'neutral'
        if bullish_flow_count > bearish_flow_count:
            overall_regime = 'bullish'
        elif bearish_flow_count > bullish_flow_count:
            overall_regime = 'bearish'
        
        avg_institutional_score = total_institutional_score / len(results) if results else 0
        
        return {
            'overall_regime': overall_regime,
            'bullish_flow_tickers': bullish_flow_count,
            'bearish_flow_tickers': bearish_flow_count,
            'neutral_flow_tickers': neutral_flow_count,
            'total_volume': total_volume,
            'avg_institutional_score': avg_institutional_score,
            'confidence': min(bullish_flow_count + bearish_flow_count, 10) / 10
        }
    
    def _create_empty_flow_analysis(self, ticker: str) -> Dict[str, Any]:
        """Create empty flow analysis for failed tickers"""
        return {
            'ticker': ticker,
            'current_price': 0.0,
            'bid_ask_spread': 0.0,
            'spread_percentage': 0.0,
            'bid_size': 0,
            'ask_size': 0,
            'order_imbalance': 0,
            'total_volume': 0,
            'large_trades_count': 0,
            'avg_trade_size': 0,
            'institutional_score': 0.0,
            'unusual_options_count': 0,
            'options_flow_value': 0,
            'flow_signals': [],
            'flow_regime': 'neutral',
            'timestamp': datetime.now()
        }
