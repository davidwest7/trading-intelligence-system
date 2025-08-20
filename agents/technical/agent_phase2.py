"""
Technical Agent - Phase 2 Standardized

Technical analysis agent with uncertainty quantification (μ, σ, horizon).
Emits standardized signals for technical patterns and indicators.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from common.models import BaseAgent
from schemas.contracts import Signal, SignalType, RegimeType, HorizonType, DirectionType


logger = logging.getLogger(__name__)


class TechnicalAgentPhase2(BaseAgent):
    """
    Technical Analysis Agent with Uncertainty Quantification
    
    Features:
    - Multiple timeframe analysis (1m, 5m, 15m, 1h, 4h, 1d)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Pattern recognition (support/resistance, breakouts)
    - Uncertainty quantification based on indicator confluence
    - Regime-aware signal generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("technical_agent_v2", SignalType.TECHNICAL, config)
        
        # Technical analysis parameters
        self.timeframes = config.get('timeframes', ['5m', '15m', '1h', '4h']) if config else ['5m', '15m', '1h', '4h']
        self.min_confidence = config.get('min_confidence', 0.3) if config else 0.3
        self.lookback_periods = config.get('lookback_periods', 100) if config else 100
        
        # Indicator parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Pattern recognition parameters
        self.support_resistance_strength = 3
        self.breakout_confirmation = 0.02  # 2% breakout threshold
        
        # Performance tracking
        self.signals_accuracy = {}
        self.confidence_calibration = {}
        
    async def generate_signals(self, symbols: List[str], **kwargs) -> List[Signal]:
        """
        Generate technical signals with uncertainty quantification
        
        Args:
            symbols: List of symbols to analyze
            **kwargs: Additional parameters (market_data, trace_id, etc.)
            
        Returns:
            List of standardized Signal objects
        """
        try:
            signals = []
            market_data = kwargs.get('market_data', {})
            trace_id = kwargs.get('trace_id')
            
            for symbol in symbols:
                symbol_signals = await self._analyze_symbol(
                    symbol, market_data, trace_id
                )
                signals.extend(symbol_signals)
            
            logger.info(f"Generated {len(signals)} technical signals for {len(symbols)} symbols")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")
            return []
    
    async def _analyze_symbol(self, symbol: str, market_data: Dict[str, Any],
                            trace_id: Optional[str] = None) -> List[Signal]:
        """Analyze a single symbol across multiple timeframes"""
        try:
            signals = []
            
            # Get market data for symbol
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                # Generate synthetic data for demo
                symbol_data = self._generate_synthetic_data(symbol)
            
            # Analyze each timeframe
            for timeframe in self.timeframes:
                timeframe_data = symbol_data.get(timeframe, symbol_data)
                
                signal = await self._analyze_timeframe(
                    symbol, timeframe, timeframe_data, trace_id
                )
                
                if signal and signal.confidence >= self.min_confidence:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return []
    
    async def _analyze_timeframe(self, symbol: str, timeframe: str,
                               data: Dict[str, Any], trace_id: Optional[str] = None) -> Optional[Signal]:
        """Analyze a single timeframe"""
        try:
            # Convert data to DataFrame for analysis
            df = self._prepare_dataframe(data)
            if df is None or len(df) < self.lookback_periods:
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Detect patterns
            patterns = self._detect_patterns(df, indicators)
            
            # Generate signal
            signal_strength, confidence = self._generate_signal_strength(
                indicators, patterns, timeframe
            )
            
            if abs(signal_strength) < 0.005:  # Minimum signal threshold
                return None
            
            # Determine market conditions for uncertainty calculation
            market_conditions = self._assess_market_conditions(df, indicators)
            
            # Create standardized signal
            signal = self.create_signal(
                symbol=symbol,
                mu=signal_strength,
                confidence=confidence,
                market_conditions=market_conditions,
                trace_id=trace_id,
                metadata={
                    'timeframe': timeframe,
                    'indicators': indicators,
                    'patterns': patterns,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe} for {symbol}: {e}")
            return None
    
    def _prepare_dataframe(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare DataFrame from market data"""
        try:
            if 'ohlcv' in data:
                # Standard OHLCV data
                df = pd.DataFrame(data['ohlcv'])
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                # Create from individual fields
                df = pd.DataFrame({
                    'open': data.get('open', []),
                    'high': data.get('high', []),
                    'low': data.get('low', []),
                    'close': data.get('close', []),
                    'volume': data.get('volume', [])
                })
            
            # Ensure we have enough data
            if len(df) < self.lookback_periods:
                return None
            
            return df.tail(self.lookback_periods)
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic market data for demo"""
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate realistic price data
        base_price = 100.0
        num_periods = self.lookback_periods
        
        returns = np.random.normal(0.001, 0.02, num_periods)  # Daily returns
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])
        
        # Generate OHLCV data
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, num_periods)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, num_periods)))
        open_prices = np.roll(prices, 1)
        open_prices[0] = prices[0]
        
        volume = np.random.lognormal(15, 0.5, num_periods)
        
        return {
            'close': prices.tolist(),
            'open': open_prices.tolist(),
            'high': high.tolist(),
            'low': low.tolist(),
            'volume': volume.tolist()
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
            
            # MACD
            exp1 = df['close'].ewm(span=self.macd_fast).mean()
            exp2 = df['close'].ewm(span=self.macd_slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=self.macd_signal).mean()
            histogram = macd - signal_line
            
            indicators['macd'] = macd.iloc[-1] if not macd.empty else 0
            indicators['macd_signal'] = signal_line.iloc[-1] if not signal_line.empty else 0
            indicators['macd_histogram'] = histogram.iloc[-1] if not histogram.empty else 0
            
            # Bollinger Bands
            sma = df['close'].rolling(window=self.bb_period).mean()
            std = df['close'].rolling(window=self.bb_period).std()
            bb_upper = sma + (std * self.bb_std)
            bb_lower = sma - (std * self.bb_std)
            bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            indicators['bb_position'] = bb_position.iloc[-1] if not bb_position.empty else 0.5
            indicators['bb_squeeze'] = (std.iloc[-1] / sma.iloc[-1]) if not std.empty and not sma.empty else 0.02
            
            # Moving Averages
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()
            
            indicators['sma_20'] = sma_20.iloc[-1] if not sma_20.empty else df['close'].iloc[-1]
            indicators['sma_50'] = sma_50.iloc[-1] if not sma_50.empty else df['close'].iloc[-1]
            indicators['price_vs_sma20'] = (df['close'].iloc[-1] - indicators['sma_20']) / indicators['sma_20']
            indicators['price_vs_sma50'] = (df['close'].iloc[-1] - indicators['sma_50']) / indicators['sma_50']
            
            # Volume indicators
            volume_sma = df['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['volume'].iloc[-1] / volume_sma.iloc[-1] if not volume_sma.empty else 1.0
            
            # Volatility
            returns = df['close'].pct_change()
            indicators['volatility'] = returns.rolling(window=20).std().iloc[-1] if not returns.empty else 0.02
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _detect_patterns(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect chart patterns"""
        patterns = {}
        
        try:
            # Support/Resistance levels
            highs = df['high'].rolling(window=10).max()
            lows = df['low'].rolling(window=10).min()
            
            current_price = df['close'].iloc[-1]
            resistance_level = highs.iloc[-20:].max()
            support_level = lows.iloc[-20:].min()
            
            patterns['near_resistance'] = abs(current_price - resistance_level) / current_price < 0.02
            patterns['near_support'] = abs(current_price - support_level) / current_price < 0.02
            
            # Breakout detection
            recent_high = df['high'].iloc[-5:].max()
            recent_low = df['low'].iloc[-5:].min()
            
            patterns['breakout_up'] = current_price > recent_high * (1 + self.breakout_confirmation)
            patterns['breakdown'] = current_price < recent_low * (1 - self.breakout_confirmation)
            
            # Trend analysis
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            
            patterns['uptrend'] = sma_20 > sma_50 and current_price > sma_20
            patterns['downtrend'] = sma_20 < sma_50 and current_price < sma_20
            
            # Momentum patterns
            rsi = indicators.get('rsi', 50)
            patterns['oversold'] = rsi < self.rsi_oversold
            patterns['overbought'] = rsi > self.rsi_overbought
            
            # MACD patterns
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            patterns['macd_bullish'] = macd > macd_signal and macd > 0
            patterns['macd_bearish'] = macd < macd_signal and macd < 0
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _generate_signal_strength(self, indicators: Dict[str, Any], 
                                patterns: Dict[str, Any], timeframe: str) -> Tuple[float, float]:
        """Generate signal strength and confidence"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            signal_weights = []
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi < self.rsi_oversold:
                bullish_signals += 1
                signal_weights.append(0.8)
            elif rsi > self.rsi_overbought:
                bearish_signals += 1
                signal_weights.append(0.8)
            total_signals += 1
            
            # MACD signals
            if patterns.get('macd_bullish', False):
                bullish_signals += 1
                signal_weights.append(0.7)
            elif patterns.get('macd_bearish', False):
                bearish_signals += 1
                signal_weights.append(0.7)
            total_signals += 1
            
            # Bollinger Band signals
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.1:  # Near lower band
                bullish_signals += 1
                signal_weights.append(0.6)
            elif bb_position > 0.9:  # Near upper band
                bearish_signals += 1
                signal_weights.append(0.6)
            total_signals += 1
            
            # Trend signals
            if patterns.get('uptrend', False):
                bullish_signals += 1
                signal_weights.append(0.5)
            elif patterns.get('downtrend', False):
                bearish_signals += 1
                signal_weights.append(0.5)
            total_signals += 1
            
            # Breakout signals
            if patterns.get('breakout_up', False):
                bullish_signals += 1
                signal_weights.append(0.9)
            elif patterns.get('breakdown', False):
                bearish_signals += 1
                signal_weights.append(0.9)
            total_signals += 1
            
            # Calculate signal strength
            net_bullish = bullish_signals - bearish_signals
            signal_strength = (net_bullish / total_signals) * 0.05  # Max 5% expected return
            
            # Calculate confidence based on signal confluence
            if bullish_signals > 0 or bearish_signals > 0:
                dominant_signals = max(bullish_signals, bearish_signals)
                confidence = (dominant_signals / total_signals) * np.mean(signal_weights) if signal_weights else 0.5
            else:
                confidence = 0.2  # Low confidence for neutral signals
            
            # Adjust for timeframe
            timeframe_multiplier = self._get_timeframe_multiplier(timeframe)
            signal_strength *= timeframe_multiplier
            
            # Ensure reasonable ranges
            signal_strength = np.clip(signal_strength, -0.10, 0.10)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return signal_strength, confidence
            
        except Exception as e:
            logger.error(f"Error generating signal strength: {e}")
            return 0.0, 0.0
    
    def _get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get signal multiplier based on timeframe"""
        multipliers = {
            '5m': 0.5,
            '15m': 0.7,
            '1h': 1.0,
            '4h': 1.2,
            '1d': 1.5
        }
        return multipliers.get(timeframe, 1.0)
    
    def _assess_market_conditions(self, df: pd.DataFrame, 
                                indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current market conditions for uncertainty calculation"""
        return {
            'volatility': indicators.get('volatility', 0.02),
            'liquidity': min(indicators.get('volume_ratio', 1.0), 2.0),  # Cap at 2x
            'trend_strength': abs(indicators.get('price_vs_sma20', 0.0)),
            'momentum': abs(indicators.get('macd_histogram', 0.0)),
        }
    
    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeType:
        """Detect market regime based on technical indicators"""
        try:
            volatility = market_data.get('volatility', 0.15)
            trend_strength = market_data.get('trend_strength', 0.0)
            momentum = market_data.get('momentum', 0.0)
            
            # High volatility regime
            if volatility > 0.25:
                return RegimeType.HIGH_VOL
            
            # Low volatility regime
            if volatility < 0.10:
                return RegimeType.LOW_VOL
            
            # Trending regime
            if trend_strength > 0.05 and momentum > 0.01:
                return RegimeType.TRENDING
            
            # Mean reverting regime
            if trend_strength < 0.02:
                return RegimeType.MEAN_REVERTING
            
            # Default to risk-on
            return RegimeType.RISK_ON
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return RegimeType.RISK_ON
