"""
Money Flow Indicators Calculator

Calculates various money flow and volume-based indicators:
- Money Flow Index (MFI)
- Accumulation/Distribution Line
- On-Balance Volume (OBV)
- Volume Price Trend (VPT)
- Chaikin Money Flow (CMF)
- Ease of Movement (EMV)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class MoneyFlowCalculator:
    """
    Calculator for money flow and volume-based technical indicators
    
    All indicators help identify buying/selling pressure and money flow direction
    """
    
    def __init__(self):
        self.default_periods = {
            'mfi': 14,
            'cmf': 21,
            'emv': 14
        }
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all money flow indicators
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary with all calculated indicators
        """
        if len(data) < 20:
            return self._default_indicators()
        
        try:
            results = {}
            
            # Money Flow Index
            results['mfi'] = self.calculate_mfi(data)
            
            # Accumulation/Distribution Line
            results['ad_line'] = self.calculate_ad_line(data)
            
            # On-Balance Volume
            results['obv'] = self.calculate_obv(data)
            
            # Volume Price Trend
            results['vpt'] = self.calculate_vpt(data)
            
            # Chaikin Money Flow
            results['cmf'] = self.calculate_cmf(data)
            
            # Ease of Movement
            results['emv'] = self.calculate_emv(data)
            
            return results
            
        except Exception as e:
            print(f"Error calculating money flow indicators: {e}")
            return self._default_indicators()
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = None) -> float:
        """
        Calculate Money Flow Index (MFI)
        
        MFI is like RSI but incorporates volume.
        Values above 80 indicate overbought, below 20 indicate oversold.
        """
        if period is None:
            period = self.default_periods['mfi']
        
        if len(data) < period + 1:
            return 50.0  # Neutral
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        money_flow = typical_price * data['volume']
        
        # Identify positive and negative money flows
        price_changes = typical_price.diff()
        
        positive_flow = money_flow.where(price_changes > 0, 0)
        negative_flow = money_flow.where(price_changes < 0, 0)
        
        # Calculate money flow ratio
        positive_mf_sum = positive_flow.rolling(window=period).sum()
        negative_mf_sum = negative_flow.rolling(window=period).sum()
        
        # Avoid division by zero
        money_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        return float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50.0
    
    def calculate_ad_line(self, data: pd.DataFrame) -> float:
        """
        Calculate Accumulation/Distribution Line
        
        AD Line measures cumulative flow of money into and out of a security.
        Rising AD Line suggests accumulation, falling suggests distribution.
        """
        if len(data) < 2:
            return 0.0
        
        # Calculate Money Flow Multiplier
        close_location_value = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        
        # Handle division by zero (when high == low)
        close_location_value = close_location_value.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = close_location_value * data['volume']
        
        # Calculate cumulative AD Line
        ad_line = money_flow_volume.cumsum()
        
        return float(ad_line.iloc[-1])
    
    def calculate_obv(self, data: pd.DataFrame) -> float:
        """
        Calculate On-Balance Volume (OBV)
        
        OBV adds volume on up days and subtracts volume on down days.
        Rising OBV suggests buying pressure, falling suggests selling pressure.
        """
        if len(data) < 2:
            return 0.0
        
        # Calculate price changes
        price_changes = data['close'].diff()
        
        # Create OBV series
        obv_changes = np.where(
            price_changes > 0, data['volume'],  # Up day: add volume
            np.where(price_changes < 0, -data['volume'], 0)  # Down day: subtract volume
        )
        
        # Calculate cumulative OBV
        obv = pd.Series(obv_changes, index=data.index).cumsum()
        
        return float(obv.iloc[-1])
    
    def calculate_vpt(self, data: pd.DataFrame) -> float:
        """
        Calculate Volume Price Trend (VPT)
        
        VPT is similar to OBV but uses percentage price changes.
        More sensitive to price changes than OBV.
        """
        if len(data) < 2:
            return 0.0
        
        # Calculate percentage price changes
        price_change_pct = data['close'].pct_change()
        
        # Calculate VPT changes
        vpt_changes = price_change_pct * data['volume']
        
        # Calculate cumulative VPT
        vpt = vpt_changes.cumsum()
        
        return float(vpt.iloc[-1]) if not np.isnan(vpt.iloc[-1]) else 0.0
    
    def calculate_cmf(self, data: pd.DataFrame, period: int = None) -> float:
        """
        Calculate Chaikin Money Flow (CMF)
        
        CMF oscillates between -1 and +1.
        Values above 0 indicate buying pressure, below 0 indicate selling pressure.
        """
        if period is None:
            period = self.default_periods['cmf']
        
        if len(data) < period:
            return 0.0
        
        # Calculate Money Flow Multiplier (same as AD Line)
        close_location_value = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        close_location_value = close_location_value.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = close_location_value * data['volume']
        
        # Calculate CMF as ratio of money flow volume to total volume
        cmf = money_flow_volume.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        
        return float(cmf.iloc[-1]) if not np.isnan(cmf.iloc[-1]) else 0.0
    
    def calculate_emv(self, data: pd.DataFrame, period: int = None) -> float:
        """
        Calculate Ease of Movement (EMV)
        
        EMV shows the relationship between price change and volume.
        High positive values indicate easy upward movement.
        High negative values indicate easy downward movement.
        """
        if period is None:
            period = self.default_periods['emv']
        
        if len(data) < period + 1:
            return 0.0
        
        # Calculate Distance Moved
        high_low_mid = (data['high'] + data['low']) / 2
        distance_moved = high_low_mid.diff()
        
        # Calculate Box Height (high - low)
        box_height = data['high'] - data['low']
        
        # Calculate Box Ratio (volume / box_height)
        # Add small constant to avoid division by zero
        box_ratio = data['volume'] / (box_height + 1e-10)
        
        # Calculate 1-period EMV
        emv_1period = distance_moved / (box_ratio + 1e-10)
        
        # Calculate EMV as moving average
        emv = emv_1period.rolling(window=period).mean()
        
        return float(emv.iloc[-1]) if not np.isnan(emv.iloc[-1]) else 0.0
    
    def calculate_force_index(self, data: pd.DataFrame, period: int = 13) -> float:
        """
        Calculate Force Index
        
        Force Index measures the force used to move the price of an asset.
        Positive values indicate buying force, negative indicate selling force.
        """
        if len(data) < period + 1:
            return 0.0
        
        # Calculate raw force index
        price_change = data['close'].diff()
        force_index_raw = price_change * data['volume']
        
        # Calculate exponential moving average
        force_index = force_index_raw.ewm(span=period).mean()
        
        return float(force_index.iloc[-1]) if not np.isnan(force_index.iloc[-1]) else 0.0
    
    def calculate_klinger_oscillator(self, data: pd.DataFrame, 
                                   fast_period: int = 34, slow_period: int = 55) -> Dict[str, float]:
        """
        Calculate Klinger Volume Oscillator
        
        Compares volume flow through a security with the security's price movements.
        """
        if len(data) < slow_period + 1:
            return {'klinger': 0.0, 'signal': 0.0}
        
        # Calculate typical price
        hlc3 = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate trend
        trend = np.where(hlc3 > hlc3.shift(1), 1, -1)
        
        # Calculate volume force
        volume_force = data['volume'] * trend * 100
        
        # Calculate Klinger Oscillator
        fast_ema = volume_force.ewm(span=fast_period).mean()
        slow_ema = volume_force.ewm(span=slow_period).mean()
        
        klinger = fast_ema - slow_ema
        
        # Calculate signal line (13-period EMA of Klinger)
        signal = klinger.ewm(span=13).mean()
        
        return {
            'klinger': float(klinger.iloc[-1]) if not np.isnan(klinger.iloc[-1]) else 0.0,
            'signal': float(signal.iloc[-1]) if not np.isnan(signal.iloc[-1]) else 0.0
        }
    
    def calculate_price_volume_trend(self, data: pd.DataFrame) -> float:
        """
        Alternative implementation of Price Volume Trend
        """
        return self.calculate_vpt(data)
    
    def get_money_flow_summary(self, indicators: Dict[str, float]) -> Dict[str, str]:
        """
        Generate interpretation summary for money flow indicators
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            Dictionary with interpretations
        """
        summary = {}
        
        # MFI interpretation
        mfi = indicators.get('mfi', 50)
        if mfi > 80:
            summary['mfi'] = "Overbought - Potential selling pressure"
        elif mfi < 20:
            summary['mfi'] = "Oversold - Potential buying opportunity"
        elif mfi > 50:
            summary['mfi'] = "Bullish money flow"
        else:
            summary['mfi'] = "Bearish money flow"
        
        # CMF interpretation
        cmf = indicators.get('cmf', 0)
        if cmf > 0.1:
            summary['cmf'] = "Strong buying pressure"
        elif cmf < -0.1:
            summary['cmf'] = "Strong selling pressure"
        else:
            summary['cmf'] = "Neutral money flow"
        
        # OBV interpretation (simplified)
        obv = indicators.get('obv', 0)
        if obv > 0:
            summary['obv'] = "Cumulative buying pressure"
        else:
            summary['obv'] = "Cumulative selling pressure"
        
        # AD Line interpretation
        ad_line = indicators.get('ad_line', 0)
        if ad_line > 0:
            summary['ad_line'] = "Accumulation phase"
        else:
            summary['ad_line'] = "Distribution phase"
        
        return summary
    
    def _default_indicators(self) -> Dict[str, float]:
        """Return default indicator values when calculation fails"""
        return {
            'mfi': 50.0,
            'ad_line': 0.0,
            'obv': 0.0,
            'vpt': 0.0,
            'cmf': 0.0,
            'emv': 0.0
        }
    
    def detect_divergences(self, price_data: pd.Series, indicator_data: pd.Series, 
                          lookback: int = 20) -> Dict[str, Any]:
        """
        Detect bullish/bearish divergences between price and money flow indicators
        
        Args:
            price_data: Price series (usually close prices)
            indicator_data: Money flow indicator series
            lookback: Periods to look back for divergence detection
            
        Returns:
            Dictionary with divergence analysis
        """
        if len(price_data) < lookback or len(indicator_data) < lookback:
            return {'divergence_detected': False, 'type': None}
        
        # Get recent data
        recent_price = price_data.tail(lookback)
        recent_indicator = indicator_data.tail(lookback)
        
        # Find local highs and lows
        price_highs = recent_price.rolling(3).max() == recent_price
        price_lows = recent_price.rolling(3).min() == recent_price
        
        indicator_highs = recent_indicator.rolling(3).max() == recent_indicator
        indicator_lows = recent_indicator.rolling(3).min() == recent_indicator
        
        # Simple divergence detection
        price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
        indicator_trend = recent_indicator.iloc[-1] - recent_indicator.iloc[0]
        
        # Detect divergence
        if price_trend > 0 and indicator_trend < 0:
            return {'divergence_detected': True, 'type': 'bearish'}
        elif price_trend < 0 and indicator_trend > 0:
            return {'divergence_detected': True, 'type': 'bullish'}
        else:
            return {'divergence_detected': False, 'type': None}
