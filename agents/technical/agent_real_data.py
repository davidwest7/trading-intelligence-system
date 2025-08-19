"""
Real Data Technical Agent
Uses Polygon.io adapter for real market data
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

class RealDataTechnicalAgent(BaseAgent):
    """Technical Analysis Agent with real market data from Polygon.io"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataTechnicalAgent", config)
        self.polygon_adapter = PolygonAdapter(config)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        tickers = kwargs.get('tickers', args[0] if args else ['AAPL', 'TSLA', 'SPY'])
        return await self.analyze_technical_indicators(tickers, **kwargs)
    
    async def analyze_technical_indicators(self, tickers: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze technical indicators using real market data"""
        print(f"🔧 Real Data Technical Agent: Analyzing {len(tickers)} tickers")
        
        results = {}
        
        for ticker in tickers:
            try:
                # Get real-time quote
                quote = await self.polygon_adapter.get_real_time_quote(ticker)
                
                # Get intraday data for technical analysis
                intraday_data = await self.polygon_adapter.get_intraday_data(
                    ticker, interval="5", limit=100
                )
                
                # Get options data
                options_data = await self.polygon_adapter.get_options_data(ticker)
                
                # Calculate technical indicators
                technical_analysis = await self._calculate_technical_indicators(
                    ticker, quote, intraday_data, options_data
                )
                
                results[ticker] = technical_analysis
                
            except Exception as e:
                print(f"❌ Error analyzing {ticker}: {e}")
                results[ticker] = self._create_empty_analysis(ticker)
        
        # Generate overall signals
        overall_signals = await self._generate_overall_signals(results)
        
        return {
            'timestamp': datetime.now(),
            'tickers_analyzed': len(tickers),
            'technical_analysis': results,
            'overall_signals': overall_signals,
            'data_source': 'Polygon.io (Real Market Data)'
        }
    
    async def _calculate_technical_indicators(self, ticker: str, quote: Dict[str, Any], 
                                            intraday_data: pd.DataFrame, 
                                            options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators from real data"""
        
        analysis = {
            'ticker': ticker,
            'current_price': quote['price'],
            'change_percent': quote['change_percent'],
            'volume': quote['volume'],
            'timestamp': datetime.now()
        }
        
        # Calculate indicators from intraday data
        if not intraday_data.empty and len(intraday_data) > 20:
            # RSI
            analysis['rsi'] = self._calculate_rsi(intraday_data['close'])
            
            # Moving averages
            analysis['sma_20'] = intraday_data['close'].rolling(20).mean().iloc[-1]
            analysis['sma_50'] = intraday_data['close'].rolling(50).mean().iloc[-1]
            analysis['ema_12'] = intraday_data['close'].ewm(span=12).mean().iloc[-1]
            analysis['ema_26'] = intraday_data['close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            analysis['macd'] = analysis['ema_12'] - analysis['ema_26']
            analysis['macd_signal'] = intraday_data['close'].ewm(span=9).mean().iloc[-1]
            analysis['macd_histogram'] = analysis['macd'] - analysis['macd_signal']
            
            # Bollinger Bands
            sma_20 = intraday_data['close'].rolling(20).mean()
            std_20 = intraday_data['close'].rolling(20).std()
            analysis['bb_upper'] = sma_20.iloc[-1] + (std_20.iloc[-1] * 2)
            analysis['bb_lower'] = sma_20.iloc[-1] - (std_20.iloc[-1] * 2)
            analysis['bb_position'] = (quote['price'] - analysis['bb_lower']) / (analysis['bb_upper'] - analysis['bb_lower'])
            
            # Volume analysis
            analysis['volume_sma'] = intraday_data['volume'].rolling(20).mean().iloc[-1]
            analysis['volume_ratio'] = quote['volume'] / analysis['volume_sma'] if analysis['volume_sma'] > 0 else 0
            
            # Support and resistance
            analysis['support_level'] = intraday_data['low'].rolling(20).min().iloc[-1]
            analysis['resistance_level'] = intraday_data['high'].rolling(20).max().iloc[-1]
            
            # Trend analysis
            analysis['trend'] = 'bullish' if analysis['sma_20'] > analysis['sma_50'] else 'bearish'
            analysis['trend_strength'] = abs(analysis['sma_20'] - analysis['sma_50']) / analysis['sma_50']
            
        else:
            # Use default values if not enough data
            analysis.update(self._get_default_indicators())
        
        # Options analysis
        analysis['options_count'] = options_data.get('options_count', 0)
        analysis['options_activity'] = 'high' if analysis['options_count'] > 100 else 'low'
        
        # Generate signals
        analysis['signals'] = self._generate_technical_signals(analysis)
        
        return analysis
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI from price data"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _get_default_indicators(self) -> Dict[str, Any]:
        """Get default indicator values when data is insufficient"""
        return {
            'rsi': 50.0,
            'sma_20': 0.0,
            'sma_50': 0.0,
            'ema_12': 0.0,
            'ema_26': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'bb_position': 0.5,
            'volume_sma': 0.0,
            'volume_ratio': 1.0,
            'support_level': 0.0,
            'resistance_level': 0.0,
            'trend': 'neutral',
            'trend_strength': 0.0
        }
    
    def _generate_technical_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate technical trading signals"""
        signals = []
        
        # RSI signals
        if analysis['rsi'] < 30:
            signals.append({
                'type': 'RSI_OVERSOLD',
                'strength': 'strong',
                'message': f"RSI oversold ({analysis['rsi']:.1f}) - potential buy signal"
            })
        elif analysis['rsi'] > 70:
            signals.append({
                'type': 'RSI_OVERBOUGHT',
                'strength': 'strong',
                'message': f"RSI overbought ({analysis['rsi']:.1f}) - potential sell signal"
            })
        
        # MACD signals
        if analysis['macd'] > analysis['macd_signal'] and analysis['macd_histogram'] > 0:
            signals.append({
                'type': 'MACD_BULLISH',
                'strength': 'medium',
                'message': "MACD bullish crossover - upward momentum"
            })
        elif analysis['macd'] < analysis['macd_signal'] and analysis['macd_histogram'] < 0:
            signals.append({
                'type': 'MACD_BEARISH',
                'strength': 'medium',
                'message': "MACD bearish crossover - downward momentum"
            })
        
        # Bollinger Bands signals
        if analysis['bb_position'] < 0.2:
            signals.append({
                'type': 'BB_OVERSOLD',
                'strength': 'medium',
                'message': "Price near lower Bollinger Band - potential bounce"
            })
        elif analysis['bb_position'] > 0.8:
            signals.append({
                'type': 'BB_OVERBOUGHT',
                'strength': 'medium',
                'message': "Price near upper Bollinger Band - potential reversal"
            })
        
        # Volume signals
        if analysis['volume_ratio'] > 2.0:
            signals.append({
                'type': 'HIGH_VOLUME',
                'strength': 'strong',
                'message': f"High volume ({analysis['volume_ratio']:.1f}x average) - strong move"
            })
        
        # Trend signals
        if analysis['trend'] == 'bullish' and analysis['trend_strength'] > 0.05:
            signals.append({
                'type': 'STRONG_UPTREND',
                'strength': 'strong',
                'message': f"Strong uptrend - momentum continuing"
            })
        elif analysis['trend'] == 'bearish' and analysis['trend_strength'] > 0.05:
            signals.append({
                'type': 'STRONG_DOWNTREND',
                'strength': 'strong',
                'message': f"Strong downtrend - momentum continuing"
            })
        
        return signals
    
    async def _generate_overall_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market signals"""
        bullish_count = 0
        bearish_count = 0
        total_signals = 0
        
        for ticker, analysis in results.items():
            for signal in analysis.get('signals', []):
                total_signals += 1
                if 'BULLISH' in signal['type'] or 'OVERSOLD' in signal['type']:
                    bullish_count += 1
                elif 'BEARISH' in signal['type'] or 'OVERBOUGHT' in signal['type']:
                    bearish_count += 1
        
        overall_sentiment = 'neutral'
        if bullish_count > bearish_count:
            overall_sentiment = 'bullish'
        elif bearish_count > bullish_count:
            overall_sentiment = 'bearish'
        
        return {
            'overall_sentiment': overall_sentiment,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'total_signals': total_signals,
            'confidence': min(bullish_count + bearish_count, 10) / 10
        }
    
    def _create_empty_analysis(self, ticker: str) -> Dict[str, Any]:
        """Create empty analysis for failed tickers"""
        return {
            'ticker': ticker,
            'current_price': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'signals': [],
            'timestamp': datetime.now()
        }
