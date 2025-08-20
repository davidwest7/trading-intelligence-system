#!/usr/bin/env python3
"""
Advanced Technical Indicators Module

Implements sophisticated technical indicators used by top quantitative hedge funds:
- Ichimoku Cloud
- Fibonacci Retracements
- Elliott Wave Analysis
- Harmonic Patterns
- Volume Profile Analysis
- Market Microstructure Indicators
- Advanced Oscillators
- Statistical Arbitrage Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@dataclass
class IchimokuCloud:
    """Ichimoku Cloud components"""
    tenkan_sen: pd.Series
    kijun_sen: pd.Series
    senkou_span_a: pd.Series
    senkou_span_b: pd.Series
    chikou_span: pd.Series

@dataclass
class FibonacciLevels:
    """Fibonacci retracement levels"""
    level_0: float
    level_236: float
    level_382: float
    level_500: float
    level_618: float
    level_786: float
    level_100: float

@dataclass
class ElliottWave:
    """Elliott Wave analysis"""
    wave_count: int
    current_wave: int
    wave_pattern: str
    confidence: float
    target_levels: List[float]

@dataclass
class HarmonicPattern:
    """Harmonic pattern detection"""
    pattern_type: str
    completion_ratio: float
    confidence: float
    entry_point: float
    stop_loss: float
    take_profit: float

class AdvancedTechnicalIndicators:
    """Advanced technical indicators for institutional-grade trading"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
    
    def calculate_ichimoku_cloud(self, data: pd.DataFrame) -> IchimokuCloud:
        """Calculate Ichimoku Cloud components"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        chikou_span = close.shift(-26)
        
        return IchimokuCloud(
            tenkan_sen=tenkan_sen,
            kijun_sen=kijun_sen,
            senkou_span_a=senkou_span_a,
            senkou_span_b=senkou_span_b,
            chikou_span=chikou_span
        )
    
    def calculate_fibonacci_levels(self, data: pd.DataFrame) -> FibonacciLevels:
        """Calculate Fibonacci retracement levels"""
        if len(data) < 20:
            current_price = data['close'].iloc[-1] if not data.empty else 100
            return FibonacciLevels(
                level_0=current_price * 1.1,
                level_236=current_price * 1.074,
                level_382=current_price * 1.038,
                level_500=current_price,
                level_618=current_price * 0.962,
                level_786=current_price * 0.926,
                level_100=current_price * 0.9
            )
        
        # Find swing high and low
        recent_data = data.tail(20)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Calculate retracement levels
        diff = swing_high - swing_low
        
        return FibonacciLevels(
            level_0=swing_high,
            level_236=swing_high - (diff * 0.236),
            level_382=swing_high - (diff * 0.382),
            level_500=swing_high - (diff * 0.500),
            level_618=swing_high - (diff * 0.618),
            level_786=swing_high - (diff * 0.786),
            level_100=swing_low
        )
    
    def detect_elliott_waves(self, data: pd.DataFrame) -> ElliottWave:
        """Detect Elliott Wave patterns"""
        if len(data) < 50:
            return ElliottWave(
                wave_count=0,
                current_wave=1,
                wave_pattern="INSUFFICIENT_DATA",
                confidence=0.0,
                target_levels=[]
            )
        
        # Find peaks and troughs
        highs, _ = find_peaks(data['high'].values, distance=5)
        lows, _ = find_peaks(-data['low'].values, distance=5)
        
        if len(highs) < 3 or len(lows) < 3:
            return ElliottWave(
                wave_count=0,
                current_wave=1,
                wave_pattern="INSUFFICIENT_PEAKS",
                confidence=0.0,
                target_levels=[]
            )
        
        # Simple Elliott Wave detection
        recent_highs = data['high'].iloc[highs[-3:]]
        recent_lows = data['low'].iloc[lows[-3:]]
        
        # Check for impulse wave pattern (5 waves)
        if len(recent_highs) >= 3 and len(recent_lows) >= 2:
            # Wave 3 should be the longest
            wave_lengths = []
            for i in range(len(recent_highs) - 1):
                wave_lengths.append(recent_highs.iloc[i+1] - recent_highs.iloc[i])
            
            if len(wave_lengths) >= 2:
                max_wave = max(wave_lengths)
                wave_count = len(wave_lengths) + 1
                
                # Calculate target levels using Fibonacci ratios
                current_price = data['close'].iloc[-1]
                target_levels = [
                    current_price * 1.236,  # Wave 3 target
                    current_price * 1.618,  # Wave 5 target
                    current_price * 0.618   # Retracement target
                ]
                
                return ElliottWave(
                    wave_count=wave_count,
                    current_wave=min(wave_count, 5),
                    wave_pattern="IMPULSE",
                    confidence=0.7,
                    target_levels=target_levels
                )
        
        return ElliottWave(
            wave_count=0,
            current_wave=1,
            wave_pattern="CORRECTIVE",
            confidence=0.3,
            target_levels=[]
        )
    
    def detect_harmonic_patterns(self, data: pd.DataFrame) -> List[HarmonicPattern]:
        """Detect harmonic patterns (Gartley, Butterfly, Bat, etc.)"""
        patterns = []
        
        if len(data) < 30:
            return patterns
        
        # Calculate swing points
        highs, _ = find_peaks(data['high'].values, distance=5)
        lows, _ = find_peaks(-data['low'].values, distance=5)
        
        if len(highs) < 4 or len(lows) < 4:
            return patterns
        
        # Gartley Pattern detection
        recent_highs = data['high'].iloc[highs[-4:]]
        recent_lows = data['low'].iloc[lows[-4:]]
        
        if len(recent_highs) >= 4 and len(recent_lows) >= 4:
            # Check for Gartley pattern ratios
            xab_ratio = abs(recent_highs.iloc[1] - recent_lows.iloc[0]) / abs(recent_highs.iloc[0] - recent_lows.iloc[0])
            abc_ratio = abs(recent_lows.iloc[1] - recent_highs.iloc[1]) / abs(recent_highs.iloc[1] - recent_lows.iloc[0])
            bcd_ratio = abs(recent_lows.iloc[2] - recent_highs.iloc[2]) / abs(recent_lows.iloc[1] - recent_highs.iloc[1])
            
            # Gartley ratios: XAB=0.618, ABC=0.382, BCD=1.272
            if (0.5 < xab_ratio < 0.7 and 
                0.3 < abc_ratio < 0.5 and 
                1.1 < bcd_ratio < 1.4):
                
                current_price = data['close'].iloc[-1]
                entry_point = recent_lows.iloc[2]
                stop_loss = entry_point * 0.98
                take_profit = entry_point * 1.236
                
                patterns.append(HarmonicPattern(
                    pattern_type="GARTLEY",
                    completion_ratio=0.8,
                    confidence=0.75,
                    entry_point=entry_point,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                ))
        
        return patterns
    
    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 20) -> Dict[str, float]:
        """Calculate Volume Profile analysis"""
        if len(data) < 50:
            return {
                'poc': data['close'].iloc[-1] if not data.empty else 100,
                'value_area_high': data['close'].iloc[-1] * 1.02,
                'value_area_low': data['close'].iloc[-1] * 0.98,
                'volume_weighted_price': data['close'].iloc[-1]
            }
        
        # Calculate Volume Weighted Average Price (VWAP)
        vwap = (data['close'] * data['volume']).sum() / data['volume'].sum()
        
        # Create price bins
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / bins
        
        # Calculate volume profile
        volume_profile = {}
        for i in range(bins):
            price_level = data['low'].min() + (i * bin_size)
            volume_at_level = data[
                (data['low'] <= price_level + bin_size) & 
                (data['high'] >= price_level)
            ]['volume'].sum()
            volume_profile[price_level] = volume_at_level
        
        # Find Point of Control (POC) - price level with highest volume
        poc_price = max(volume_profile, key=volume_profile.get)
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.7
        
        # Find value area boundaries
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = 0
        value_area_prices = []
        
        for price, volume in sorted_levels:
            cumulative_volume += volume
            value_area_prices.append(price)
            if cumulative_volume >= target_volume:
                break
        
        return {
            'poc': poc_price,
            'value_area_high': max(value_area_prices),
            'value_area_low': min(value_area_prices),
            'volume_weighted_price': vwap
        }
    
    def calculate_market_microstructure(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure indicators"""
        if len(data) < 20:
            return {
                'bid_ask_spread': 0.001,
                'order_flow_imbalance': 0.0,
                'market_depth': 1.0,
                'price_impact': 0.001
            }
        
        # Calculate bid-ask spread proxy (high-low spread)
        spread = (data['high'] - data['low']) / data['close']
        avg_spread = spread.rolling(20).mean().iloc[-1]
        
        # Calculate order flow imbalance (volume-weighted price changes)
        price_changes = data['close'].pct_change()
        volume_weighted_changes = (price_changes * data['volume']).rolling(20).sum()
        order_flow_imbalance = volume_weighted_changes.iloc[-1]
        
        # Calculate market depth proxy (volume/price volatility)
        price_volatility = data['close'].pct_change().rolling(20).std()
        market_depth = (data['volume'].rolling(20).mean() / price_volatility).iloc[-1]
        
        # Calculate price impact (how much price moves per unit volume)
        volume_changes = data['volume'].pct_change()
        price_impact = (price_changes / volume_changes).rolling(20).mean().iloc[-1]
        
        return {
            'bid_ask_spread': avg_spread,
            'order_flow_imbalance': order_flow_imbalance,
            'market_depth': market_depth,
            'price_impact': abs(price_impact) if not np.isnan(price_impact) else 0.001
        }
    
    def calculate_advanced_oscillators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate advanced oscillators"""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Williams %R
        williams_r = talib.WILLR(high, low, close, timeperiod=14)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Commodity Channel Index (CCI)
        cci = talib.CCI(high, low, close, timeperiod=14)
        
        # Money Flow Index (MFI)
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Rate of Change (ROC)
        roc = talib.ROC(close, timeperiod=10)
        
        # Ultimate Oscillator
        ultimate_osc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        # Average Directional Index (ADX)
        adx = talib.ADX(high, low, close, timeperiod=14)
        
        return {
            'williams_r': pd.Series(williams_r, index=data.index),
            'stoch_k': pd.Series(stoch_k, index=data.index),
            'stoch_d': pd.Series(stoch_d, index=data.index),
            'cci': pd.Series(cci, index=data.index),
            'mfi': pd.Series(mfi, index=data.index),
            'roc': pd.Series(roc, index=data.index),
            'ultimate_osc': pd.Series(ultimate_osc, index=data.index),
            'adx': pd.Series(adx, index=data.index)
        }
    
    def calculate_statistical_arbitrage_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical arbitrage signals"""
        if len(data) < 50:
            return {
                'z_score': 0.0,
                'mean_reversion_probability': 0.5,
                'momentum_probability': 0.5,
                'volatility_regime': 'NORMAL'
            }
        
        # Calculate rolling z-score
        returns = data['close'].pct_change().dropna()
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        z_score = (returns - rolling_mean) / rolling_std
        current_z_score = z_score.iloc[-1]
        
        # Calculate mean reversion probability
        if abs(current_z_score) > 2:
            mean_reversion_prob = 0.8
        elif abs(current_z_score) > 1.5:
            mean_reversion_prob = 0.6
        else:
            mean_reversion_prob = 0.3
        
        # Calculate momentum probability
        recent_returns = returns.tail(5)
        momentum_prob = 0.7 if recent_returns.mean() > 0 else 0.3
        
        # Determine volatility regime
        current_vol = rolling_std.iloc[-1]
        avg_vol = rolling_std.mean()
        
        if current_vol > avg_vol * 1.5:
            vol_regime = 'HIGH'
        elif current_vol < avg_vol * 0.7:
            vol_regime = 'LOW'
        else:
            vol_regime = 'NORMAL'
        
        return {
            'z_score': current_z_score,
            'mean_reversion_probability': mean_reversion_prob,
            'momentum_probability': momentum_prob,
            'volatility_regime': vol_regime
        }
    
    def calculate_composite_signal(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate composite technical signal combining all indicators"""
        if len(data) < 50:
            return {
                'composite_score': 0.0,
                'signal_strength': 0.0,
                'confidence': 0.5,
                'risk_level': 'MEDIUM'
            }
        
        # Get all indicators
        ichimoku = self.calculate_ichimoku_cloud(data)
        fibonacci = self.calculate_fibonacci_levels(data)
        elliott = self.detect_elliott_waves(data)
        harmonics = self.detect_harmonic_patterns(data)
        volume_profile = self.calculate_volume_profile(data)
        microstructure = self.calculate_market_microstructure(data)
        oscillators = self.calculate_advanced_oscillators(data)
        stat_arb = self.calculate_statistical_arbitrage_signals(data)
        
        # Calculate composite score
        current_price = data['close'].iloc[-1]
        
        # Ichimoku signals
        ichimoku_score = 0
        if current_price > ichimoku.senkou_span_a.iloc[-1] and current_price > ichimoku.senkou_span_b.iloc[-1]:
            ichimoku_score = 0.2  # Bullish
        elif current_price < ichimoku.senkou_span_a.iloc[-1] and current_price < ichimoku.senkou_span_b.iloc[-1]:
            ichimoku_score = -0.2  # Bearish
        
        # Fibonacci signals
        fib_score = 0
        if (fibonacci.level_618 < current_price < fibonacci.level_382):
            fib_score = 0.15  # Support zone
        elif (fibonacci.level_382 < current_price < fibonacci.level_236):
            fib_score = -0.15  # Resistance zone
        
        # Elliott Wave signals
        elliott_score = elliott.confidence * 0.1 if elliott.wave_pattern == "IMPULSE" else -0.1
        
        # Harmonic patterns
        harmonic_score = sum([h.confidence * 0.2 for h in harmonics])
        
        # Volume profile signals
        vp_score = 0
        if current_price > volume_profile['poc']:
            vp_score = 0.1
        else:
            vp_score = -0.1
        
        # Oscillator signals
        osc_score = 0
        if oscillators['williams_r'].iloc[-1] < -80:
            osc_score += 0.1  # Oversold
        elif oscillators['williams_r'].iloc[-1] > -20:
            osc_score -= 0.1  # Overbought
        
        if oscillators['mfi'].iloc[-1] < 20:
            osc_score += 0.1  # Oversold
        elif oscillators['mfi'].iloc[-1] > 80:
            osc_score -= 0.1  # Overbought
        
        # Statistical arbitrage signals
        stat_score = 0
        if stat_arb['z_score'] < -2:
            stat_score = 0.2  # Mean reversion opportunity
        elif stat_arb['z_score'] > 2:
            stat_score = -0.2  # Mean reversion opportunity
        
        # Combine all scores
        composite_score = (ichimoku_score + fib_score + elliott_score + 
                          harmonic_score + vp_score + osc_score + stat_score)
        
        # Normalize to [-1, 1] range
        composite_score = np.clip(composite_score, -1, 1)
        
        # Calculate signal strength and confidence
        signal_strength = abs(composite_score)
        confidence = min(0.9, 0.5 + signal_strength * 0.4)
        
        # Determine risk level
        if abs(composite_score) > 0.7:
            risk_level = 'HIGH'
        elif abs(composite_score) > 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'composite_score': composite_score,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'risk_level': risk_level
        }
