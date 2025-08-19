"""
Optimized Technical Analysis Agent

Advanced technical analysis with:
- Multi-timeframe pattern recognition
- Advanced indicator calculations
- Performance optimization
- Error handling and resilience
- Real-time signal generation
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import talib
from dataclasses import dataclass

from .models import (
    TechnicalOpportunity, AnalysisPayload, AnalysisMetadata, 
    MarketRegime, Direction
)
from .strategies import (
    ImbalanceStrategy, FairValueGapStrategy, LiquiditySweepStrategy,
    IDFPStrategy, TrendStrategy, BreakoutStrategy, MeanReversionStrategy
)
from .backtest import PurgedCrossValidationBacktester


@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    symbol: str
    timeframe: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    indicators: Dict[str, float]
    pattern: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "indicators": self.indicators,
            "pattern": self.pattern
        }


class OptimizedTechnicalAgent:
    """
    Optimized Technical Analysis Agent for financial markets
    
    Enhanced Capabilities:
    ✅ Multi-timeframe technical analysis with parallel processing
    ✅ Advanced indicator calculations (RSI, MACD, Bollinger Bands, etc.)
    ✅ Pattern recognition (candlestick patterns, chart patterns)
    ✅ Support/resistance level detection
    ✅ Volume profile analysis
    ✅ Performance optimization and caching
    ✅ Error handling and resilience
    ✅ Real-time signal generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuration with defaults
        self.config = config or {}
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.lookback_periods = self.config.get('lookback_periods', 200)
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize strategies
        self.strategies = {
            "imbalance": ImbalanceStrategy(),
            "fvg": FairValueGapStrategy(),
            "liquidity_sweep": LiquiditySweepStrategy(),
            "idfp": IDFPStrategy(),
            "trend": TrendStrategy(),
            "breakout": BreakoutStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
        
        # Technical indicators configuration
        self.indicators_config = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'atr': {'period': 14},
            'ema': {'periods': [9, 21, 50, 200]},
            'volume_sma': {'period': 20}
        }
        
        # Performance metrics
        self.metrics = {
            'total_analyses': 0,
            'signals_generated': 0,
            'processing_time_avg': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0
        }
        
        # Error tracking
        self.error_count = 0
        self.total_requests = 0
        
        logging.info("Optimized Technical Agent initialized successfully")
    
    async def analyze_technical_signals(
        self,
        symbols: List[str],
        timeframes: List[str] = None,
        strategies: List[str] = None,
        include_indicators: bool = True,
        include_patterns: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized technical analysis with caching and parallel processing
        
        Args:
            symbols: List of symbols to analyze
            timeframes: List of timeframes to analyze
            strategies: List of strategies to apply
            include_indicators: Include technical indicators
            include_patterns: Include pattern recognition
            use_cache: Use cached results if available
        
        Returns:
            Complete technical analysis results
        """
        
        if timeframes is None:
            timeframes = ["15m", "1h", "4h", "1d"]
        
        if strategies is None:
            strategies = ["trend", "breakout", "mean_reversion"]
        
        # Check cache first
        cache_key = f"{','.join(sorted(symbols))}_{','.join(sorted(timeframes))}_{','.join(sorted(strategies))}"
        if use_cache and self._is_cache_valid(cache_key):
            self.metrics['cache_hit_rate'] += 1
            return self.cache[cache_key]
        
        try:
            self.total_requests += 1
            start_time = time.time()
            
            # Analyze each symbol in parallel
            analysis_tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    self._analyze_symbol_technical(
                        symbol, timeframes, strategies, include_indicators, include_patterns
                    )
                )
                analysis_tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            all_signals = []
            symbol_analyses = {}
            
            for i, result in enumerate(results):
                symbol = symbols[i]
                if isinstance(result, Exception):
                    logging.error(f"Error analyzing {symbol}: {result}")
                    self.error_count += 1
                elif result is not None:
                    symbol_analyses[symbol] = result
                    all_signals.extend(result.get('signals', []))
            
            # Generate summary
            summary = self._create_technical_summary(symbol_analyses, all_signals)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            self.metrics['processing_time_avg'] = (
                (self.metrics['processing_time_avg'] * (self.total_requests - 1) + processing_time) 
                / self.total_requests
            )
            self.metrics['total_analyses'] += len(symbols)
            self.metrics['signals_generated'] += len(all_signals)
            
            # Create results
            results = {
                "symbol_analyses": symbol_analyses,
                "all_signals": [signal.to_dict() for signal in all_signals],
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "processing_info": {
                    "total_symbols": len(symbols),
                    "processing_time": processing_time,
                    "cache_hit_rate": self.metrics['cache_hit_rate']
                }
            }
            
            # Cache results
            if use_cache:
                self._cache_result(cache_key, results)
            
            return results
            
        except Exception as e:
            self.error_count += 1
            self.metrics['error_rate'] = self.error_count / self.total_requests
            logging.error(f"Error in technical analysis: {e}")
            raise
    
    async def _analyze_symbol_technical(
        self,
        symbol: str,
        timeframes: List[str],
        strategies: List[str],
        include_indicators: bool,
        include_patterns: bool
    ) -> Dict[str, Any]:
        """Analyze technical signals for a single symbol"""
        
        try:
            # Get market data
            market_data = await self._get_market_data_optimized(symbol, timeframes)
            
            # Analyze each timeframe
            timeframe_analyses = {}
            all_signals = []
            
            for timeframe in timeframes:
                if timeframe in market_data:
                    df = market_data[timeframe]
                    
                    # Calculate technical indicators
                    indicators = {}
                    if include_indicators:
                        indicators = await self._calculate_indicators_optimized(df)
                    
                    # Detect patterns
                    patterns = {}
                    if include_patterns:
                        patterns = await self._detect_patterns_optimized(df)
                    
                    # Apply strategies
                    strategy_signals = await self._apply_strategies_optimized(
                        df, symbol, timeframe, strategies, indicators, patterns
                    )
                    
                    # Generate support/resistance levels
                    support_resistance = await self._calculate_support_resistance_optimized(df)
                    
                    # Volume profile analysis
                    volume_profile = await self._analyze_volume_profile_optimized(df)
                    
                    timeframe_analyses[timeframe] = {
                        'indicators': indicators,
                        'patterns': patterns,
                        'signals': strategy_signals,
                        'support_resistance': support_resistance,
                        'volume_profile': volume_profile,
                        'market_regime': self._determine_market_regime_optimized(df)
                    }
                    
                    all_signals.extend(strategy_signals)
            
            # Multi-timeframe analysis
            multi_timeframe_signals = self._analyze_multi_timeframe_optimized(
                symbol, timeframe_analyses, all_signals
            )
            
            return {
                'symbol': symbol,
                'timeframe_analyses': timeframe_analyses,
                'signals': all_signals,
                'multi_timeframe_signals': multi_timeframe_signals,
                'overall_bias': self._determine_overall_bias_optimized(all_signals),
                'confidence': self._calculate_analysis_confidence(all_signals)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing technical signals for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timeframe_analyses': {},
                'signals': [],
                'multi_timeframe_signals': [],
                'overall_bias': 'neutral',
                'confidence': 0.0
            }
    
    async def _get_market_data_optimized(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get optimized market data"""
        
        # Generate realistic mock data for demo
        data = {}
        base_price = 100.0 + np.random.random() * 50
        
        for timeframe in timeframes:
            # Determine number of periods based on timeframe
            periods = {
                "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
                "1h": 24, "4h": 6, "1d": 1
            }.get(timeframe, 100)
            
            # Generate realistic OHLCV data
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='1h')
            
            # Random walk with trend and volatility
            returns = np.random.normal(0.0001, 0.015, periods)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV DataFrame
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = df['close'].shift(1).fillna(prices[0])
            
            # Add realistic high/low with volatility
            volatility = np.random.uniform(0.005, 0.02, periods)
            high_noise = np.random.uniform(0, 1, periods) * volatility
            low_noise = np.random.uniform(0, 1, periods) * volatility
            
            df['high'] = df['close'] * (1 + high_noise)
            df['low'] = df['close'] * (1 - low_noise)
            
            # Ensure OHLC consistency
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            # Generate volume with some correlation to price movement
            base_volume = 1000000
            volume_multiplier = 1 + np.abs(df['close'].pct_change().fillna(0))
            df['volume'] = base_volume * volume_multiplier * np.random.exponential(1, periods)
            
            data[timeframe] = df
        
        return data
    
    async def _calculate_indicators_optimized(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators efficiently"""
        
        try:
            indicators = {}
            
            # RSI
            rsi_period = self.indicators_config['rsi']['period']
            indicators['rsi'] = talib.RSI(df['close'].values, timeperiod=rsi_period)[-1]
            
            # MACD
            macd_config = self.indicators_config['macd']
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values, 
                fastperiod=macd_config['fast'],
                slowperiod=macd_config['slow'],
                signalperiod=macd_config['signal']
            )
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]
            
            # Bollinger Bands
            bb_period = self.indicators_config['bollinger']['period']
            bb_std = self.indicators_config['bollinger']['std_dev']
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'].values, 
                timeperiod=bb_period,
                nbdevup=bb_std,
                nbdevdn=bb_std
            )
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            
            # Stochastic
            stoch_config = self.indicators_config['stochastic']
            stoch_k, stoch_d = talib.STOCH(
                df['high'].values, df['low'].values, df['close'].values,
                fastk_period=stoch_config['k_period'],
                slowk_period=stoch_config['d_period'],
                slowd_period=stoch_config['d_period']
            )
            indicators['stoch_k'] = stoch_k[-1]
            indicators['stoch_d'] = stoch_d[-1]
            
            # ATR
            atr_period = self.indicators_config['atr']['period']
            indicators['atr'] = talib.ATR(
                df['high'].values, df['low'].values, df['close'].values,
                timeperiod=atr_period
            )[-1]
            
            # EMAs
            for period in self.indicators_config['ema']['periods']:
                ema = talib.EMA(df['close'].values, timeperiod=period)
                indicators[f'ema_{period}'] = ema[-1]
            
            # Volume SMA
            volume_period = self.indicators_config['volume_sma']['period']
            indicators['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=volume_period)[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return {}
    
    async def _detect_patterns_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect candlestick and chart patterns"""
        
        try:
            patterns = {}
            
            # Candlestick patterns
            candlestick_patterns = [
                'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
                'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
                'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLDARKCLOUDCOVER',
                'CDLDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
                'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
                'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
                'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING',
                'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI',
                'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
                'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
                'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
                'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
                'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
                'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
            ]
            
            for pattern_name in candlestick_patterns:
                try:
                    pattern_func = getattr(talib, pattern_name)
                    result = pattern_func(
                        df['open'].values, df['high'].values, 
                        df['low'].values, df['close'].values
                    )
                    if result[-1] != 0:
                        patterns[pattern_name] = result[-1]
                except:
                    continue
            
            # Chart patterns (simplified)
            patterns['trend'] = self._detect_trend_pattern(df)
            patterns['support_resistance'] = self._detect_support_resistance_pattern(df)
            patterns['breakout'] = self._detect_breakout_pattern(df)
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting patterns: {e}")
            return {}
    
    def _detect_trend_pattern(self, df: pd.DataFrame) -> str:
        """Detect trend pattern"""
        if len(df) < 20:
            return "insufficient_data"
        
        # Calculate trend using linear regression
        x = np.arange(len(df))
        y = df['close'].values
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "uptrend"
        elif slope < -0.1:
            return "downtrend"
        else:
            return "sideways"
    
    def _detect_support_resistance_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels"""
        if len(df) < 20:
            return {}
        
        # Find local minima and maxima
        highs = df['high'].rolling(window=5, center=True).max()
        lows = df['low'].rolling(window=5, center=True).min()
        
        # Identify support and resistance levels
        resistance_levels = highs.nlargest(3).unique()
        support_levels = lows.nsmallest(3).unique()
        
        return {
            'resistance_levels': resistance_levels.tolist(),
            'support_levels': support_levels.tolist()
        }
    
    def _detect_breakout_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect breakout patterns"""
        if len(df) < 20:
            return {}
        
        # Calculate recent high/low
        recent_high = df['high'].rolling(window=20).max().iloc[-1]
        recent_low = df['low'].rolling(window=20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Check for breakouts
        breakout_up = current_price > recent_high * 1.01
        breakout_down = current_price < recent_low * 0.99
        
        return {
            'breakout_up': breakout_up,
            'breakout_down': breakout_down,
            'recent_high': recent_high,
            'recent_low': recent_low
        }
    
    async def _apply_strategies_optimized(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategies: List[str],
        indicators: Dict[str, float],
        patterns: Dict[str, Any]
    ) -> List[TechnicalSignal]:
        """Apply technical strategies and generate signals"""
        
        signals = []
        
        try:
            # Trend strategy
            if 'trend' in strategies:
                trend_signals = self._apply_trend_strategy(df, symbol, timeframe, indicators)
                signals.extend(trend_signals)
            
            # Breakout strategy
            if 'breakout' in strategies:
                breakout_signals = self._apply_breakout_strategy(df, symbol, timeframe, patterns)
                signals.extend(breakout_signals)
            
            # Mean reversion strategy
            if 'mean_reversion' in strategies:
                mean_reversion_signals = self._apply_mean_reversion_strategy(df, symbol, timeframe, indicators)
                signals.extend(mean_reversion_signals)
            
            # RSI strategy
            if 'rsi' in indicators:
                rsi_signals = self._apply_rsi_strategy(df, symbol, timeframe, indicators)
                signals.extend(rsi_signals)
            
            # MACD strategy
            if 'macd' in indicators:
                macd_signals = self._apply_macd_strategy(df, symbol, timeframe, indicators)
                signals.extend(macd_signals)
            
        except Exception as e:
            logging.error(f"Error applying strategies: {e}")
        
        return signals
    
    def _apply_trend_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[TechnicalSignal]:
        """Apply trend following strategy"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Check EMA alignment
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            
            # Bullish trend: price above EMAs and EMAs aligned
            if (current_price > ema_9 > ema_21 > ema_50):
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="trend",
                    direction="long",
                    strength=0.8,
                    confidence=0.75,
                    entry_price=current_price,
                    stop_loss=ema_21,
                    take_profit=current_price * 1.03,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="bullish_trend"
                )
                signals.append(signal)
            
            # Bearish trend: price below EMAs and EMAs aligned
            elif (current_price < ema_9 < ema_21 < ema_50):
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="trend",
                    direction="short",
                    strength=0.8,
                    confidence=0.75,
                    entry_price=current_price,
                    stop_loss=ema_21,
                    take_profit=current_price * 0.97,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="bearish_trend"
                )
                signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error in trend strategy: {e}")
        
        return signals
    
    def _apply_breakout_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str, patterns: Dict[str, Any]) -> List[TechnicalSignal]:
        """Apply breakout strategy"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            breakout_pattern = patterns.get('breakout', {})
            
            if breakout_pattern.get('breakout_up', False):
                recent_high = breakout_pattern.get('recent_high', current_price)
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="breakout",
                    direction="long",
                    strength=0.9,
                    confidence=0.8,
                    entry_price=current_price,
                    stop_loss=recent_high * 0.98,
                    take_profit=current_price * 1.05,
                    timestamp=datetime.now(),
                    indicators={},
                    pattern="breakout_up"
                )
                signals.append(signal)
            
            elif breakout_pattern.get('breakout_down', False):
                recent_low = breakout_pattern.get('recent_low', current_price)
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="breakout",
                    direction="short",
                    strength=0.9,
                    confidence=0.8,
                    entry_price=current_price,
                    stop_loss=recent_low * 1.02,
                    take_profit=current_price * 0.95,
                    timestamp=datetime.now(),
                    indicators={},
                    pattern="breakout_down"
                )
                signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error in breakout strategy: {e}")
        
        return signals
    
    def _apply_mean_reversion_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[TechnicalSignal]:
        """Apply mean reversion strategy"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            rsi = indicators.get('rsi', 50)
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            
            # Oversold conditions
            if rsi < 30 and current_price < bb_lower:
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="mean_reversion",
                    direction="long",
                    strength=0.7,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=bb_lower * 0.98,
                    take_profit=current_price * 1.02,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="oversold"
                )
                signals.append(signal)
            
            # Overbought conditions
            elif rsi > 70 and current_price > bb_upper:
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="mean_reversion",
                    direction="short",
                    strength=0.7,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=bb_upper * 1.02,
                    take_profit=current_price * 0.98,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="overbought"
                )
                signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error in mean reversion strategy: {e}")
        
        return signals
    
    def _apply_rsi_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[TechnicalSignal]:
        """Apply RSI strategy"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            rsi = indicators.get('rsi', 50)
            
            # RSI divergence and extreme levels
            if rsi < 20:
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="rsi",
                    direction="long",
                    strength=0.6,
                    confidence=0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 0.98,
                    take_profit=current_price * 1.02,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="rsi_oversold"
                )
                signals.append(signal)
            
            elif rsi > 80:
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="rsi",
                    direction="short",
                    strength=0.6,
                    confidence=0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 1.02,
                    take_profit=current_price * 0.98,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="rsi_overbought"
                )
                signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error in RSI strategy: {e}")
        
        return signals
    
    def _apply_macd_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: Dict[str, float]) -> List[TechnicalSignal]:
        """Apply MACD strategy"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_hist = indicators.get('macd_histogram', 0)
            
            # MACD crossover signals
            if macd > macd_signal and macd_hist > 0:
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="macd",
                    direction="long",
                    strength=0.7,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price * 0.98,
                    take_profit=current_price * 1.03,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="macd_bullish"
                )
                signals.append(signal)
            
            elif macd < macd_signal and macd_hist < 0:
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type="macd",
                    direction="short",
                    strength=0.7,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price * 1.02,
                    take_profit=current_price * 0.97,
                    timestamp=datetime.now(),
                    indicators=indicators,
                    pattern="macd_bearish"
                )
                signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error in MACD strategy: {e}")
        
        return signals
    
    async def _calculate_support_resistance_optimized(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        try:
            if len(df) < 20:
                return {'support': [], 'resistance': []}
            
            # Find pivot points
            highs = df['high'].values
            lows = df['low'].values
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(df) - 2):
                # Resistance (local maxima)
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_levels.append(highs[i])
                
                # Support (local minima)
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_levels.append(lows[i])
            
            # Get top levels
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)))[:5]
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
        
        except Exception as e:
            logging.error(f"Error calculating support/resistance: {e}")
            return {'support': [], 'resistance': []}
    
    async def _analyze_volume_profile_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile"""
        try:
            if len(df) < 20:
                return {}
            
            # Calculate volume-weighted average price (VWAP)
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            
            # Volume distribution
            price_bins = pd.cut(df['close'], bins=10)
            volume_profile = df.groupby(price_bins)['volume'].sum()
            
            # High volume nodes
            high_volume_nodes = volume_profile.nlargest(3)
            
            return {
                'vwap': vwap,
                'volume_profile': volume_profile.to_dict(),
                'high_volume_nodes': high_volume_nodes.to_dict(),
                'volume_trend': 'increasing' if df['volume'].iloc[-5:].mean() > df['volume'].iloc[-20:-5].mean() else 'decreasing'
            }
        
        except Exception as e:
            logging.error(f"Error analyzing volume profile: {e}")
            return {}
    
    def _determine_market_regime_optimized(self, df: pd.DataFrame) -> str:
        """Determine market regime"""
        try:
            if len(df) < 20:
                return "insufficient_data"
            
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Calculate trend strength
            trend_strength = abs(df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            if volatility > 0.02:
                return "volatile"
            elif trend_strength > 0.05:
                return "trending"
            elif volatility < 0.01:
                return "calm"
            else:
                return "ranging"
        
        except Exception as e:
            logging.error(f"Error determining market regime: {e}")
            return "unknown"
    
    def _analyze_multi_timeframe_optimized(
        self,
        symbol: str,
        timeframe_analyses: Dict[str, Any],
        all_signals: List[TechnicalSignal]
    ) -> List[TechnicalSignal]:
        """Analyze multi-timeframe alignment"""
        multi_timeframe_signals = []
        
        try:
            # Group signals by direction
            long_signals = [s for s in all_signals if s.direction == "long"]
            short_signals = [s for s in all_signals if s.direction == "short"]
            
            # Check for multi-timeframe alignment
            if len(long_signals) >= 2:
                # Multiple timeframes showing bullish signals
                avg_confidence = np.mean([s.confidence for s in long_signals])
                avg_strength = np.mean([s.strength for s in long_signals])
                
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe="multi",
                    signal_type="multi_timeframe",
                    direction="long",
                    strength=avg_strength * 1.2,  # Boost for alignment
                    confidence=avg_confidence * 1.1,
                    entry_price=long_signals[0].entry_price,
                    stop_loss=min([s.stop_loss for s in long_signals]),
                    take_profit=max([s.take_profit for s in long_signals]),
                    timestamp=datetime.now(),
                    indicators={},
                    pattern="multi_timeframe_bullish"
                )
                multi_timeframe_signals.append(signal)
            
            elif len(short_signals) >= 2:
                # Multiple timeframes showing bearish signals
                avg_confidence = np.mean([s.confidence for s in short_signals])
                avg_strength = np.mean([s.strength for s in short_signals])
                
                signal = TechnicalSignal(
                    symbol=symbol,
                    timeframe="multi",
                    signal_type="multi_timeframe",
                    direction="short",
                    strength=avg_strength * 1.2,  # Boost for alignment
                    confidence=avg_confidence * 1.1,
                    entry_price=short_signals[0].entry_price,
                    stop_loss=max([s.stop_loss for s in short_signals]),
                    take_profit=min([s.take_profit for s in short_signals]),
                    timestamp=datetime.now(),
                    indicators={},
                    pattern="multi_timeframe_bearish"
                )
                multi_timeframe_signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error in multi-timeframe analysis: {e}")
        
        return multi_timeframe_signals
    
    def _determine_overall_bias_optimized(self, signals: List[TechnicalSignal]) -> str:
        """Determine overall bias from signals"""
        if not signals:
            return "neutral"
        
        long_count = len([s for s in signals if s.direction == "long"])
        short_count = len([s for s in signals if s.direction == "short"])
        
        if long_count > short_count * 1.5:
            return "bullish"
        elif short_count > long_count * 1.5:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_analysis_confidence(self, signals: List[TechnicalSignal]) -> float:
        """Calculate overall analysis confidence"""
        if not signals:
            return 0.0
        
        # Weight by signal strength and confidence
        weighted_confidence = sum(s.confidence * s.strength for s in signals)
        total_weight = sum(s.strength for s in signals)
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _create_technical_summary(self, symbol_analyses: Dict[str, Any], all_signals: List[TechnicalSignal]) -> Dict[str, Any]:
        """Create technical analysis summary"""
        try:
            # Count signals by type
            signal_types = defaultdict(int)
            directions = defaultdict(int)
            patterns = defaultdict(int)
            
            for signal in all_signals:
                signal_types[signal.signal_type] += 1
                directions[signal.direction] += 1
                patterns[signal.pattern] += 1
            
            # Calculate average metrics
            avg_confidence = np.mean([s.confidence for s in all_signals]) if all_signals else 0.0
            avg_strength = np.mean([s.strength for s in all_signals]) if all_signals else 0.0
            
            # Market bias distribution
            biases = [analysis.get('overall_bias', 'neutral') for analysis in symbol_analyses.values()]
            bias_distribution = defaultdict(int)
            for bias in biases:
                bias_distribution[bias] += 1
            
            return {
                'total_signals': len(all_signals),
                'symbols_analyzed': len(symbol_analyses),
                'signal_types': dict(signal_types),
                'directions': dict(directions),
                'patterns': dict(patterns),
                'average_confidence': avg_confidence,
                'average_strength': avg_strength,
                'bias_distribution': dict(bias_distribution),
                'top_signals': sorted(all_signals, key=lambda x: x.confidence * x.strength, reverse=True)[:5]
            }
        
        except Exception as e:
            logging.error(f"Error creating technical summary: {e}")
            return {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        age = time.time() - self.cache_timestamps[cache_key]
        return age < self.cache_ttl
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result with timestamp"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logging.info("Optimized Technical Agent cleanup completed")
