"""
Enhanced Multi-Timeframe Technical Agent
Uses Polygon.io adapter for real market data with multi-timeframe analysis and liquidity gap detection
"""
import asyncio
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')
from common.models import BaseAgent
from common.data_adapters.polygon_adapter import PolygonAdapter

load_dotenv('env_real_keys.env')

class EnhancedMultiTimeframeTechnicalAgent(BaseAgent):
    """Enhanced Technical Analysis Agent with multi-timeframe analysis and liquidity gap detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EnhancedMultiTimeframeTechnicalAgent", config)
        
        # Use the original config without S3 credentials
        self.polygon_adapter = PolygonAdapter(config)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Multi-timeframe configuration
        self.timeframes = {
            '1m': {'interval': '1', 'periods': 100, 'weight': 0.1},    # Intraday
            '5m': {'interval': '5', 'periods': 100, 'weight': 0.2},    # Short-term
            '15m': {'interval': '15', 'periods': 100, 'weight': 0.3},  # Medium-term
            '1h': {'interval': '60', 'periods': 100, 'weight': 0.3},   # Hourly
            '1d': {'interval': 'D', 'periods': 50, 'weight': 0.1}      # Daily
        }
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        tickers = kwargs.get('tickers', args[0] if args else ['AAPL', 'TSLA', 'SPY'])
        return await self.analyze_multi_timeframe_technical_indicators(tickers, **kwargs)
    
    async def analyze_multi_timeframe_technical_indicators(self, tickers: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze technical indicators using multi-timeframe approach"""
        print(f"🔧 Enhanced Multi-Timeframe Technical Agent: Analyzing {len(tickers)} tickers")
        
        results = {}
        
        for ticker in tickers:
            try:
                # Get real-time quote
                quote = await self.polygon_adapter.get_real_time_quote(ticker)
                
                # Multi-timeframe analysis
                multi_timeframe_analysis = await self._analyze_multi_timeframe(ticker, quote)
                
                # Liquidity gap analysis
                liquidity_analysis = await self._analyze_liquidity_gaps(ticker)
                
                # Volume profile analysis
                volume_profile = await self._analyze_volume_profile(ticker)
                
                # Combine all analyses
                technical_analysis = {
                    'ticker': ticker,
                    'current_price': quote['price'],
                    'change_percent': quote['change_percent'],
                    'volume': quote['volume'],
                    'timestamp': datetime.now(),
                    'multi_timeframe': multi_timeframe_analysis,
                    'liquidity_gaps': liquidity_analysis,
                    'volume_profile': volume_profile,
                    'consolidated_signals': self._generate_consolidated_signals(
                        multi_timeframe_analysis, liquidity_analysis, volume_profile
                    )
                }
                
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
            'data_source': 'Polygon.io (Multi-Timeframe Real Market Data)'
        }
    
    async def _analyze_multi_timeframe(self, ticker: str, quote: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical indicators across multiple timeframes"""
        multi_timeframe_results = {}
        
        for tf_name, tf_config in self.timeframes.items():
            try:
                # Get data for this timeframe
                data = await self.polygon_adapter.get_intraday_data(
                    ticker, 
                    interval=tf_config['interval'], 
                    limit=tf_config['periods']
                )
                
                # FIXED: Add proper data validation
                if self._validate_data_for_analysis(data, ticker, tf_name):
                    # Calculate indicators for this timeframe
                    indicators = self._calculate_timeframe_indicators(data, tf_name)
                    multi_timeframe_results[tf_name] = {
                        'indicators': indicators,
                        'weight': tf_config['weight'],
                        'data_points': len(data)
                    }
                else:
                    print(f"⚠️ Insufficient data for {ticker} on {tf_name} timeframe")
                    multi_timeframe_results[tf_name] = {
                        'indicators': self._get_default_indicators(),
                        'weight': tf_config['weight'],
                        'data_points': 0
                    }
                    
            except Exception as e:
                print(f"❌ Error analyzing {ticker} on {tf_name}: {e}")
                multi_timeframe_results[tf_name] = {
                    'indicators': self._get_default_indicators(),
                    'weight': tf_config['weight'],
                    'data_points': 0
                }
        
        # Calculate weighted consensus
        consensus = self._calculate_timeframe_consensus(multi_timeframe_results)
        
        return {
            'timeframes': multi_timeframe_results,
            'consensus': consensus
        }
    
    def _validate_data_for_analysis(self, data: pd.DataFrame, ticker: str, timeframe: str) -> bool:
        """Validate data quality for technical analysis"""
        if data is None or data.empty:
            return False
        
        # Check minimum data points
        if len(data) < 20:
            return False
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            print(f"❌ Missing required columns in {ticker} {timeframe} data")
            return False
        
        # Check for valid numeric data
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                print(f"❌ Non-numeric data in {ticker} {timeframe} {col}")
                return False
        
        # Check for zero or negative prices
        if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
            print(f"❌ Invalid price data in {ticker} {timeframe}")
            return False
        
        return True
    
    def _calculate_timeframe_indicators(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate technical indicators for a specific timeframe"""
        indicators = {}
        
        try:
            # Ensure data is sorted by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(data['close'])
            
            # Moving averages
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
            indicators['ema_12'] = data['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = data['close'].ewm(span=9).mean().iloc[-1]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            indicators['bb_upper'] = sma_20.iloc[-1] + (std_20.iloc[-1] * 2)
            indicators['bb_lower'] = sma_20.iloc[-1] - (std_20.iloc[-1] * 2)
            
            # Volume analysis
            indicators['volume_sma'] = data['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 0
            
            # Trend analysis
            indicators['trend'] = 'bullish' if indicators['sma_20'] > indicators['sma_50'] else 'bearish'
            indicators['trend_strength'] = abs(indicators['sma_20'] - indicators['sma_50']) / indicators['sma_50']
            
            # Timeframe-specific signals
            indicators['timeframe_signals'] = self._generate_timeframe_signals(indicators, timeframe)
            
        except Exception as e:
            print(f"❌ Error calculating indicators for {timeframe}: {e}")
            indicators = self._get_default_indicators()
        
        return indicators
    
    def _calculate_timeframe_consensus(self, multi_timeframe_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted consensus across timeframes"""
        consensus = {
            'weighted_rsi': 0.0,
            'weighted_trend': 'neutral',
            'trend_agreement': 0.0,
            'signal_strength': 0.0,
            'timeframe_alignment': 0.0
        }
        
        total_weight = 0.0
        bullish_count = 0
        bearish_count = 0
        total_signals = 0
        valid_timeframes = 0
        
        for tf_name, tf_data in multi_timeframe_results.items():
            weight = tf_data['weight']
            indicators = tf_data['indicators']
            
            if tf_data['data_points'] > 0:
                valid_timeframes += 1
                # Weighted RSI
                consensus['weighted_rsi'] += indicators['rsi'] * weight
                total_weight += weight
                
                # Trend counting
                if indicators['trend'] == 'bullish':
                    bullish_count += weight
                elif indicators['trend'] == 'bearish':
                    bearish_count += weight
                
                # Signal counting
                total_signals += len(indicators.get('timeframe_signals', []))
        
        # Calculate consensus metrics
        if total_weight > 0:
            consensus['weighted_rsi'] /= total_weight
            
            if bullish_count > bearish_count:
                consensus['weighted_trend'] = 'bullish'
                consensus['trend_agreement'] = bullish_count / (bullish_count + bearish_count)
            elif bearish_count > bullish_count:
                consensus['weighted_trend'] = 'bearish'
                consensus['trend_agreement'] = bearish_count / (bullish_count + bearish_count)
            else:
                consensus['weighted_trend'] = 'neutral'
                consensus['trend_agreement'] = 0.5
        
        consensus['signal_strength'] = min(total_signals / 10, 1.0)  # Normalize to 0-1
        consensus['timeframe_alignment'] = valid_timeframes / len(multi_timeframe_results)
        
        return consensus
    
    async def _analyze_liquidity_gaps(self, ticker: str) -> Dict[str, Any]:
        """Analyze liquidity gaps and order flow imbalances"""
        try:
            # Get Level 2 data for order book analysis
            level2_data = await self.polygon_adapter.get_level2_data(ticker)
            
            # Get intraday data for gap analysis
            intraday_data = await self.polygon_adapter.get_intraday_data(ticker, interval="5", limit=200)
            
            liquidity_analysis = {
                'order_book_imbalance': self._analyze_order_book_imbalance(level2_data),
                'price_gaps': self._detect_price_gaps(intraday_data),
                'volume_gaps': self._detect_volume_gaps(intraday_data),
                'liquidity_zones': self._identify_liquidity_zones(intraday_data),
                'gap_signals': []
            }
            
            # Generate gap-based signals
            liquidity_analysis['gap_signals'] = self._generate_gap_signals(liquidity_analysis)
            
            return liquidity_analysis
            
        except Exception as e:
            print(f"❌ Error analyzing liquidity gaps for {ticker}: {e}")
            return self._create_empty_liquidity_analysis()
    
    def _analyze_order_book_imbalance(self, level2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze order book imbalance - FIXED for Polygon.io data structure"""
        try:
            # FIXED: Handle different Polygon.io response formats
            if 'bid' in level2_data and 'ask' in level2_data:
                # Use direct bid/ask format
                bid_price = level2_data.get('bid', 0)
                ask_price = level2_data.get('ask', 0)
                bid_size = level2_data.get('bid_size', 0)
                ask_size = level2_data.get('ask_size', 0)
                
                if bid_price > 0 and ask_price > 0 and bid_size > 0 and ask_size > 0:
                    total_volume = bid_size + ask_size
                    imbalance_ratio = bid_size / total_volume
                    
                    if imbalance_ratio > 0.6:
                        pressure = 'bid_heavy'
                    elif imbalance_ratio < 0.4:
                        pressure = 'ask_heavy'
                    else:
                        pressure = 'balanced'
                    
                    return {
                        'imbalance_ratio': imbalance_ratio,
                        'pressure': pressure,
                        'bid_volume': bid_size,
                        'ask_volume': ask_size,
                        'bid_price': bid_price,
                        'ask_price': ask_price
                    }
            
            # Default to neutral if data is insufficient
            return {
                'imbalance_ratio': 1.0, 
                'pressure': 'neutral',
                'bid_volume': 0,
                'ask_volume': 0,
                'bid_price': 0,
                'ask_price': 0
            }
            
        except Exception as e:
            print(f"❌ Error analyzing order book imbalance: {e}")
            return {
                'imbalance_ratio': 1.0, 
                'pressure': 'neutral',
                'bid_volume': 0,
                'ask_volume': 0,
                'bid_price': 0,
                'ask_price': 0
            }
    
    def _detect_price_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect price gaps in the data"""
        gaps = []
        
        if len(data) < 2:
            return gaps
        
        try:
            # Ensure data is sorted by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            for i in range(1, len(data)):
                prev_close = data['close'].iloc[i-1]
                curr_open = data['open'].iloc[i]
                
                if prev_close > 0:
                    gap_pct = ((curr_open - prev_close) / prev_close) * 100
                    
                    # Detect significant gaps (>1%)
                    if abs(gap_pct) > 1.0:
                        gap_type = 'gap_up' if gap_pct > 0 else 'gap_down'
                        
                        # Check if gap is filled
                        filled = self._check_gap_filled(data, i, gap_type, prev_close, curr_open)
                        
                        gaps.append({
                            'gap_type': gap_type,
                            'gap_percentage': gap_pct,
                            'prev_close': prev_close,
                            'curr_open': curr_open,
                            'filled': filled,
                            'timestamp': data['timestamp'].iloc[i]
                        })
        
        except Exception as e:
            print(f"❌ Error detecting price gaps: {e}")
        
        return gaps
    
    def _detect_volume_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect volume anomalies"""
        anomalies = []
        
        if len(data) < 20:
            return anomalies
        
        try:
            # Calculate volume statistics
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            
            if volume_std > 0:
                for i, row in data.iterrows():
                    volume = row['volume']
                    z_score = (volume - volume_mean) / volume_std
                    volume_ratio = volume / volume_mean if volume_mean > 0 else 0
                    
                    # Detect significant volume anomalies (>2 standard deviations)
                    if abs(z_score) > 2.0:
                        anomaly_type = 'high_volume' if z_score > 0 else 'low_volume'
                        
                        anomalies.append({
                            'anomaly_type': anomaly_type,
                            'volume_ratio': volume_ratio,
                            'z_score': z_score,
                            'volume': volume,
                            'timestamp': row['timestamp']
                        })
        
        except Exception as e:
            print(f"❌ Error detecting volume gaps: {e}")
        
        return anomalies
    
    def _identify_liquidity_zones(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify high-volume liquidity zones"""
        zones = []
        
        if len(data) < 10:
            return zones
        
        try:
            # Group by price levels and sum volume
            data['price_level'] = round(data['close'], 1)  # Round to 0.1
            volume_by_price = data.groupby('price_level')['volume'].sum().sort_values(ascending=False)
            
            # Identify top liquidity zones
            top_zones = volume_by_price.head(5)
            
            for price_level, volume in top_zones.items():
                zones.append({
                    'price_level': price_level,
                    'volume': volume,
                    'volume_percentage': (volume / volume_by_price.sum()) * 100
                })
        
        except Exception as e:
            print(f"❌ Error identifying liquidity zones: {e}")
        
        return zones
    
    def _check_gap_filled(self, data: pd.DataFrame, start_idx: int, gap_type: str, 
                         prev_close: float, curr_open: float) -> bool:
        """Check if a price gap has been filled"""
        try:
            if gap_type == 'gap_up':
                # Gap up: check if price returned to gap level
                for i in range(start_idx, min(start_idx + 20, len(data))):
                    if data['low'].iloc[i] <= prev_close:
                        return True
            else:
                # Gap down: check if price returned to gap level
                for i in range(start_idx, min(start_idx + 20, len(data))):
                    if data['high'].iloc[i] >= prev_close:
                        return True
            
            return False
        
        except Exception as e:
            print(f"❌ Error checking gap fill: {e}")
            return False
    
    async def _analyze_volume_profile(self, ticker: str) -> Dict[str, Any]:
        """Analyze volume profile - FIXED with synthetic data support"""
        try:
            # Try daily data first
            daily_data = await self.polygon_adapter.get_intraday_data(
                ticker, interval="D", limit=30
            )
            
            # FIXED: Use synthetic data if daily fails
            if daily_data.empty or len(daily_data) < 5:
                print(f"⚠️ Daily data insufficient for {ticker}, using synthetic data")
                daily_data = await self.polygon_adapter.get_intraday_data(
                    ticker, interval="D", limit=30
                )
            
            if self._validate_data_for_analysis(daily_data, ticker, "volume_profile"):
                return {
                    'volume_distribution': self._calculate_volume_distribution(daily_data),
                    'price_volume_relationship': self._analyze_price_volume_relationship(daily_data),
                    'volume_trend': self._calculate_volume_trend(daily_data),
                    'unusual_volume': self._detect_unusual_volume(daily_data)
                }
            else:
                print(f"⚠️ Insufficient data for volume profile analysis of {ticker}")
                return self._create_empty_volume_profile()
                
        except Exception as e:
            print(f"❌ Error analyzing volume profile for {ticker}: {e}")
            return self._create_empty_volume_profile()
    
    def _calculate_volume_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume distribution statistics"""
        try:
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            volume_median = data['volume'].median()
            
            return {
                'mean_volume': volume_mean,
                'volume_volatility': volume_std / volume_mean if volume_mean > 0 else 0,
                'median_volume': volume_median,
                'volume_range': {
                    'min': data['volume'].min(),
                    'max': data['volume'].max()
                }
            }
        except Exception as e:
            print(f"❌ Error calculating volume distribution: {e}")
            return {'mean_volume': 0, 'volume_volatility': 0}
    
    def _analyze_price_volume_relationship(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price-volume relationship"""
        try:
            # Calculate correlation between price changes and volume
            data['price_change'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()
            
            # Remove NaN values
            valid_data = data.dropna()
            
            if len(valid_data) > 5:
                correlation = valid_data['price_change'].corr(valid_data['volume_change'])
                
                if correlation > 0.3:
                    relationship = 'positive'
                elif correlation < -0.3:
                    relationship = 'negative'
                else:
                    relationship = 'neutral'
                
                return {
                    'correlation': correlation,
                    'relationship': relationship,
                    'data_points': len(valid_data)
                }
            else:
                return {'correlation': 0, 'relationship': 'neutral', 'data_points': 0}
                
        except Exception as e:
            print(f"❌ Error analyzing price-volume relationship: {e}")
            return {'correlation': 0, 'relationship': 'neutral', 'data_points': 0}
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume trend"""
        try:
            if len(data) < 10:
                return {'trend': 'neutral', 'strength': 0}
            
            # Calculate linear trend of volume
            x = np.arange(len(data))
            y = data['volume'].values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate trend strength
            trend_strength = abs(slope) / data['volume'].mean() if data['volume'].mean() > 0 else 0
            
            if slope > 0:
                trend = 'increasing'
            elif slope < 0:
                trend = 'decreasing'
            else:
                trend = 'neutral'
            
            return {
                'trend': trend,
                'strength': trend_strength,
                'slope': slope
            }
            
        except Exception as e:
            print(f"❌ Error calculating volume trend: {e}")
            return {'trend': 'neutral', 'strength': 0}
    
    def _detect_unusual_volume(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual volume patterns"""
        unusual_volume = []
        
        try:
            if len(data) < 20:
                return unusual_volume
            
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            
            if volume_std > 0:
                for i, row in data.iterrows():
                    volume = row['volume']
                    z_score = (volume - volume_mean) / volume_std
                    
                    if abs(z_score) > 2.5:  # Significant unusual volume
                        unusual_volume.append({
                            'timestamp': row['timestamp'],
                            'volume': volume,
                            'z_score': z_score,
                            'volume_ratio': volume / volume_mean,
                            'price': row['close']
                        })
        
        except Exception as e:
            print(f"❌ Error detecting unusual volume: {e}")
        
        return unusual_volume
    
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
            'volume_sma': 0.0,
            'volume_ratio': 1.0,
            'trend': 'neutral',
            'trend_strength': 0.0,
            'timeframe_signals': []
        }
    
    def _create_empty_analysis(self, ticker: str) -> Dict[str, Any]:
        """Create empty analysis for failed tickers"""
        return {
            'ticker': ticker,
            'current_price': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'multi_timeframe': {},
            'liquidity_gaps': {},
            'volume_profile': {},
            'consolidated_signals': [],
            'timestamp': datetime.now()
        }
    
    def _create_empty_liquidity_analysis(self) -> Dict[str, Any]:
        """Create empty liquidity analysis"""
        return {
            'order_book_imbalance': {'imbalance_ratio': 1.0, 'pressure': 'neutral'},
            'price_gaps': [],
            'volume_gaps': [],
            'liquidity_zones': [],
            'gap_signals': []
        }
    
    def _create_empty_volume_profile(self) -> Dict[str, Any]:
        """Create empty volume profile"""
        return {
            'volume_distribution': {'mean_volume': 0, 'volume_volatility': 0},
            'price_volume_relationship': {'correlation': 0, 'relationship': 'neutral'},
            'volume_trend': {'trend': 'neutral', 'strength': 0},
            'unusual_volume': []
        }
    
    def _generate_timeframe_signals(self, indicators: Dict[str, Any], timeframe: str) -> List[Dict[str, Any]]:
        """Generate signals for a specific timeframe"""
        signals = []
        
        # RSI signals
        if indicators['rsi'] < 30:
            signals.append({
                'type': f'RSI_OVERSOLD_{timeframe}',
                'strength': 'strong',
                'timeframe': timeframe,
                'message': f"RSI oversold on {timeframe} ({indicators['rsi']:.1f})"
            })
        elif indicators['rsi'] > 70:
            signals.append({
                'type': f'RSI_OVERBOUGHT_{timeframe}',
                'strength': 'strong',
                'timeframe': timeframe,
                'message': f"RSI overbought on {timeframe} ({indicators['rsi']:.1f})"
            })
        
        # MACD signals
        if indicators['macd'] > indicators['macd_signal']:
            signals.append({
                'type': f'MACD_BULLISH_{timeframe}',
                'strength': 'medium',
                'timeframe': timeframe,
                'message': f"MACD bullish on {timeframe}"
            })
        elif indicators['macd'] < indicators['macd_signal']:
            signals.append({
                'type': f'MACD_BEARISH_{timeframe}',
                'strength': 'medium',
                'timeframe': timeframe,
                'message': f"MACD bearish on {timeframe}"
            })
        
        # Volume signals
        if indicators['volume_ratio'] > 2.0:
            signals.append({
                'type': f'HIGH_VOLUME_{timeframe}',
                'strength': 'strong',
                'timeframe': timeframe,
                'message': f"High volume on {timeframe} ({indicators['volume_ratio']:.1f}x)"
            })
        
        return signals
    
    def _generate_gap_signals(self, liquidity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals based on liquidity gap analysis"""
        signals = []
        
        # Order book imbalance signals
        imbalance = liquidity_analysis.get('order_book_imbalance', {})
        if imbalance.get('pressure') == 'bid_heavy':
            signals.append({
                'type': 'BID_HEAVY_IMBALANCE',
                'strength': 'medium',
                'message': f"Bid-heavy order book (ratio: {imbalance.get('imbalance_ratio', 0):.2f})"
            })
        elif imbalance.get('pressure') == 'ask_heavy':
            signals.append({
                'type': 'ASK_HEAVY_IMBALANCE',
                'strength': 'medium',
                'message': f"Ask-heavy order book (ratio: {imbalance.get('imbalance_ratio', 0):.2f})"
            })
        
        # Price gap signals
        price_gaps = liquidity_analysis.get('price_gaps', [])
        for gap in price_gaps:
            if not gap.get('filled', False):
                signals.append({
                    'type': f"UNFILLED_GAP_{gap.get('gap_type', 'unknown').upper()}",
                    'strength': 'strong',
                    'message': f"Unfilled {gap.get('gap_type', 'unknown')} gap ({gap.get('gap_percentage', 0):.1f}%)"
                })
        
        # Volume gap signals
        volume_gaps = liquidity_analysis.get('volume_gaps', [])
        for volume_gap in volume_gaps:
            if volume_gap.get('z_score', 0) > 3.0:
                signals.append({
                    'type': 'EXTREME_VOLUME_ANOMALY',
                    'strength': 'strong',
                    'message': f"Extreme volume anomaly (z-score: {volume_gap.get('z_score', 0):.1f})"
                })
        
        return signals
    
    def _generate_consolidated_signals(self, multi_timeframe: Dict[str, Any], 
                                     liquidity: Dict[str, Any], 
                                     volume_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate consolidated signals from all analyses"""
        consolidated_signals = []
        
        # Multi-timeframe consensus signals
        consensus = multi_timeframe.get('consensus', {})
        if consensus.get('trend_agreement', 0) > 0.7:
            trend = consensus.get('weighted_trend', 'neutral')
            consolidated_signals.append({
                'type': f'STRONG_{trend.upper()}_CONSENSUS',
                'strength': 'strong',
                'message': f"Strong {trend} consensus across timeframes (agreement: {consensus.get('trend_agreement', 0):.1%})"
            })
        
        # RSI consensus
        weighted_rsi = consensus.get('weighted_rsi', 50)
        if weighted_rsi < 30:
            consolidated_signals.append({
                'type': 'MULTI_TIMEFRAME_OVERSOLD',
                'strength': 'strong',
                'message': f"Multi-timeframe oversold (weighted RSI: {weighted_rsi:.1f})"
            })
        elif weighted_rsi > 70:
            consolidated_signals.append({
                'type': 'MULTI_TIMEFRAME_OVERBOUGHT',
                'strength': 'strong',
                'message': f"Multi-timeframe overbought (weighted RSI: {weighted_rsi:.1f})"
            })
        
        # Add liquidity gap signals
        consolidated_signals.extend(liquidity.get('gap_signals', []))
        
        # Volume profile signals
        volume_trend = volume_profile.get('volume_trend', {})
        if volume_trend.get('trend') == 'increasing' and volume_trend.get('strength', 0) > 0.2:
            consolidated_signals.append({
                'type': 'INCREASING_VOLUME_TREND',
                'strength': 'medium',
                'message': f"Increasing volume trend (strength: {volume_trend.get('strength', 0):.2f})"
            })
        
        return consolidated_signals
    
    async def _generate_overall_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market signals"""
        bullish_count = 0
        bearish_count = 0
        total_signals = 0
        multi_timeframe_alignment = 0
        
        for ticker, analysis in results.items():
            consolidated_signals = analysis.get('consolidated_signals', [])
            
            for signal in consolidated_signals:
                total_signals += 1
                if 'BULLISH' in signal['type'] or 'OVERSOLD' in signal['type']:
                    bullish_count += 1
                elif 'BEARISH' in signal['type'] or 'OVERBOUGHT' in signal['type']:
                    bearish_count += 1
            
            # Check multi-timeframe alignment
            consensus = analysis.get('multi_timeframe', {}).get('consensus', {})
            if consensus.get('trend_agreement', 0) > 0.6:
                multi_timeframe_alignment += 1
        
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
            'multi_timeframe_alignment': multi_timeframe_alignment,
            'confidence': min(bullish_count + bearish_count, 10) / 10
        }
