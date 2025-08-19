"""
Order Flow Analysis for Market Microstructure

Analyzes order flow to detect:
- Buying vs selling pressure
- Institutional vs retail flow
- Market impact and liquidity
- Flow persistence and momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import scipy.stats as stats
from dataclasses import dataclass

from .models import (
    OrderFlowMetrics, FlowMetrics, MarketTick, 
    FlowDirection, VolumeProfile
)


class OrderFlowAnalyzer:
    """
    Advanced order flow analysis using market microstructure data
    
    Features:
    - Tick-by-tick flow analysis
    - Buying/selling pressure detection
    - Market impact measurement
    - Liquidity analysis
    - Volume profile construction
    """
    
    def __init__(self):
        self.tick_buffer = []
        self.flow_history = []
        
        # Thresholds for flow classification
        self.strong_flow_threshold = 0.3
        self.moderate_flow_threshold = 0.15
        
        # Volume profile parameters
        self.price_buckets = 50
        
    def analyze_order_flow(self, ticks: List[MarketTick], 
                          timeframe: str = "1h") -> Dict[str, Any]:
        """
        Comprehensive order flow analysis
        
        Args:
            ticks: List of market ticks
            timeframe: Analysis timeframe
            
        Returns:
            Complete order flow analysis
        """
        if not ticks:
            return self._default_flow_analysis()
        
        # Convert ticks to DataFrame for easier processing
        tick_df = self._ticks_to_dataframe(ticks)
        
        # Classify ticks as buy/sell
        classified_ticks = self._classify_ticks(tick_df)
        
        # Calculate order flow metrics
        flow_metrics = self._calculate_flow_metrics(classified_ticks)
        
        # Analyze buying/selling pressure
        pressure_analysis = self._analyze_pressure(classified_ticks)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(classified_ticks)
        
        # Build volume profile
        volume_profile = self._build_volume_profile(classified_ticks)
        
        # Detect flow persistence
        persistence = self._calculate_flow_persistence(classified_ticks)
        
        # Overall flow direction
        overall_direction = self._determine_overall_direction(pressure_analysis)
        
        return {
            "flow_metrics": flow_metrics,
            "pressure_analysis": pressure_analysis,
            "market_impact": market_impact,
            "volume_profile": volume_profile,
            "flow_persistence": persistence,
            "overall_direction": overall_direction.value,
            "tick_count": len(ticks),
            "analysis_period": {
                "start": ticks[0].timestamp.isoformat(),
                "end": ticks[-1].timestamp.isoformat(),
                "duration_minutes": (ticks[-1].timestamp - ticks[0].timestamp).total_seconds() / 60
            }
        }
    
    def _ticks_to_dataframe(self, ticks: List[MarketTick]) -> pd.DataFrame:
        """Convert tick data to DataFrame"""
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'bid': tick.bid,
                'ask': tick.ask,
                'bid_size': tick.bid_size,
                'ask_size': tick.ask_size
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _classify_ticks(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify ticks as buy/sell/neutral using multiple methods
        
        Uses:
        1. Lee-Ready algorithm (price vs bid/ask)
        2. Tick rule (price change direction)
        3. Quote rule (price relative to midpoint)
        """
        df = tick_df.copy()
        
        # Method 1: Lee-Ready algorithm
        df['midpoint'] = (df['bid'] + df['ask']) / 2
        df['lee_ready_side'] = np.where(
            df['price'] > df['midpoint'], 'buy',
            np.where(df['price'] < df['midpoint'], 'sell', 'neutral')
        )
        
        # Method 2: Tick rule
        df['price_change'] = df['price'].diff()
        df['tick_rule_side'] = np.where(
            df['price_change'] > 0, 'buy',
            np.where(df['price_change'] < 0, 'sell', 'neutral')
        )
        
        # Method 3: Quote rule (distance from bid/ask)
        df['quote_rule_side'] = np.where(
            (df['price'] - df['bid']) < (df['ask'] - df['price']), 'sell',
            np.where((df['price'] - df['bid']) > (df['ask'] - df['price']), 'buy', 'neutral')
        )
        
        # Combine methods with weights
        df['buy_score'] = (
            (df['lee_ready_side'] == 'buy').astype(int) * 0.5 +
            (df['tick_rule_side'] == 'buy').astype(int) * 0.3 +
            (df['quote_rule_side'] == 'buy').astype(int) * 0.2
        )
        
        df['sell_score'] = (
            (df['lee_ready_side'] == 'sell').astype(int) * 0.5 +
            (df['tick_rule_side'] == 'sell').astype(int) * 0.3 +
            (df['quote_rule_side'] == 'sell').astype(int) * 0.2
        )
        
        # Final classification
        df['side'] = np.where(
            df['buy_score'] > df['sell_score'], 'buy',
            np.where(df['sell_score'] > df['buy_score'], 'sell', 'neutral')
        )
        
        # Confidence in classification
        df['classification_confidence'] = np.abs(df['buy_score'] - df['sell_score'])
        
        return df
    
    def _calculate_flow_metrics(self, df: pd.DataFrame) -> FlowMetrics:
        """Calculate comprehensive flow metrics"""
        total_volume = df['volume'].sum()
        
        # Separate by side
        buy_trades = df[df['side'] == 'buy']
        sell_trades = df[df['side'] == 'sell']
        neutral_trades = df[df['side'] == 'neutral']
        
        buy_volume = buy_trades['volume'].sum()
        sell_volume = sell_trades['volume'].sum()
        neutral_volume = neutral_trades['volume'].sum()
        
        # Flow metrics
        net_buying_pressure = (buy_volume - sell_volume) / max(total_volume, 1)
        
        # VWAP
        vwap = (df['price'] * df['volume']).sum() / max(total_volume, 1)
        
        # Price-volume correlation
        price_vol_corr = df['price'].corr(df['volume']) if len(df) > 2 else 0.0
        
        # Tick metrics
        uptick_count = len(df[df['price_change'] > 0])
        downtick_count = len(df[df['price_change'] < 0])
        total_ticks = len(df)
        
        uptick_ratio = uptick_count / max(total_ticks, 1)
        downtick_ratio = downtick_count / max(total_ticks, 1)
        tick_imbalance = uptick_ratio - downtick_ratio
        
        return FlowMetrics(
            period_start=df.index[0],
            period_end=df.index[-1],
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            neutral_volume=neutral_volume,
            net_buying_pressure=net_buying_pressure,
            volume_weighted_price=vwap,
            price_volume_correlation=price_vol_corr,
            uptick_ratio=uptick_ratio,
            downtick_ratio=downtick_ratio,
            tick_imbalance=tick_imbalance
        )
    
    def _analyze_pressure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze buying and selling pressure"""
        buy_trades = df[df['side'] == 'buy']
        sell_trades = df[df['side'] == 'sell']
        
        # Volume-weighted pressure
        total_volume = df['volume'].sum()
        buy_volume_weighted = (buy_trades['volume'] * buy_trades['price']).sum()
        sell_volume_weighted = (sell_trades['volume'] * sell_trades['price']).sum()
        
        # Trade size analysis
        avg_buy_size = buy_trades['volume'].mean() if len(buy_trades) > 0 else 0
        avg_sell_size = sell_trades['volume'].mean() if len(sell_trades) > 0 else 0
        
        # Large trade detection (top 10% by volume)
        volume_90th = df['volume'].quantile(0.9)
        large_trades = df[df['volume'] >= volume_90th]
        large_buy_volume = large_trades[large_trades['side'] == 'buy']['volume'].sum()
        large_sell_volume = large_trades[large_trades['side'] == 'sell']['volume'].sum()
        
        # Pressure persistence
        pressure_persistence = self._calculate_pressure_persistence(df)
        
        return {
            "buy_pressure": {
                "volume": buy_trades['volume'].sum(),
                "trade_count": len(buy_trades),
                "avg_trade_size": avg_buy_size,
                "volume_weighted_price": buy_volume_weighted / max(buy_trades['volume'].sum(), 1),
                "large_trade_volume": large_buy_volume
            },
            "sell_pressure": {
                "volume": sell_trades['volume'].sum(),
                "trade_count": len(sell_trades),
                "avg_trade_size": avg_sell_size,
                "volume_weighted_price": sell_volume_weighted / max(sell_trades['volume'].sum(), 1),
                "large_trade_volume": large_sell_volume
            },
            "pressure_ratio": (buy_trades['volume'].sum()) / max(sell_trades['volume'].sum(), 1),
            "persistence": pressure_persistence,
            "institutional_activity": {
                "large_trade_ratio": len(large_trades) / max(len(df), 1),
                "institutional_buy_bias": large_buy_volume / max(large_buy_volume + large_sell_volume, 1)
            }
        }
    
    def _calculate_pressure_persistence(self, df: pd.DataFrame) -> float:
        """Calculate how persistent buying/selling pressure is"""
        # Rolling pressure over small windows
        window_size = max(10, len(df) // 10)
        
        rolling_pressure = []
        for i in range(window_size, len(df)):
            window_df = df.iloc[i-window_size:i]
            buy_vol = window_df[window_df['side'] == 'buy']['volume'].sum()
            sell_vol = window_df[window_df['side'] == 'sell']['volume'].sum()
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                pressure = (buy_vol - sell_vol) / total_vol
                rolling_pressure.append(pressure)
        
        if len(rolling_pressure) < 2:
            return 0.0
        
        # Autocorrelation as persistence measure
        pressure_series = pd.Series(rolling_pressure)
        persistence = pressure_series.autocorr(lag=1)
        
        return persistence if not np.isnan(persistence) else 0.0
    
    def _calculate_market_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market impact metrics"""
        if len(df) < 5:
            return {"error": "Insufficient data for market impact calculation"}
        
        # Price impact per unit volume
        df['returns'] = df['price'].pct_change()
        df['log_volume'] = np.log(df['volume'] + 1)
        
        # Linear regression: returns ~ volume
        if df['log_volume'].std() > 0:
            impact_coefficient = np.corrcoef(df['returns'].dropna(), 
                                           df['log_volume'][1:].dropna())[0, 1]
        else:
            impact_coefficient = 0.0
        
        # Temporary vs permanent impact (simplified)
        short_term_impact = df['returns'].rolling(3).sum().std()
        long_term_impact = df['returns'].rolling(10).sum().std()
        
        # Kyle's lambda (price impact parameter)
        if df['volume'].std() > 0:
            kyle_lambda = abs(impact_coefficient) * df['volume'].std()
        else:
            kyle_lambda = 0.0
        
        # Amihud illiquidity ratio
        amihud_illiquidity = (abs(df['returns']) / (df['volume'] + 1)).mean()
        
        return {
            "impact_coefficient": impact_coefficient,
            "kyle_lambda": kyle_lambda,
            "amihud_illiquidity": amihud_illiquidity,
            "short_term_impact": short_term_impact,
            "long_term_impact": long_term_impact,
            "impact_ratio": short_term_impact / max(long_term_impact, 0.001)
        }
    
    def _build_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build volume profile (Volume at Price)"""
        if len(df) == 0:
            return {"error": "No data for volume profile"}
        
        price_min = df['price'].min()
        price_max = df['price'].max()
        
        if price_max == price_min:
            return {"error": "No price variation"}
        
        # Create price buckets
        price_bins = np.linspace(price_min, price_max, self.price_buckets)
        df['price_bucket'] = pd.cut(df['price'], bins=price_bins, include_lowest=True)
        
        # Volume at each price level
        volume_profile = df.groupby('price_bucket')['volume'].sum()
        
        # Point of Control (POC) - price with highest volume
        poc_price = volume_profile.idxmax().mid if len(volume_profile) > 0 else price_min
        
        # Value Area (70% of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        
        # Find value area
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumulative_volume = 0
        value_area_prices = []
        
        for price_bucket, volume in sorted_profile.items():
            cumulative_volume += volume
            value_area_prices.append(price_bucket.mid)
            if cumulative_volume >= target_volume:
                break
        
        value_area_high = max(value_area_prices) if value_area_prices else price_max
        value_area_low = min(value_area_prices) if value_area_prices else price_min
        
        # Profile type classification
        profile_type = self._classify_volume_profile(volume_profile, df)
        
        return {
            "poc": poc_price,
            "value_area_high": value_area_high,
            "value_area_low": value_area_low,
            "profile_type": profile_type.value,
            "price_levels": [bucket.mid for bucket in volume_profile.index],
            "volume_at_price": volume_profile.values.tolist(),
            "total_volume": total_volume,
            "value_area_volume_pct": 70.0
        }
    
    def _classify_volume_profile(self, volume_profile: pd.Series, 
                               df: pd.DataFrame) -> VolumeProfile:
        """Classify the type of volume profile"""
        # Check volume distribution
        max_volume = volume_profile.max()
        mean_volume = volume_profile.mean()
        
        # Single vs multiple peaks
        peaks = volume_profile[volume_profile > mean_volume * 1.5]
        
        if len(peaks) == 1:
            # Single peak - normal distribution
            return VolumeProfile.NEUTRAL_VOLUME
        elif len(peaks) > 2:
            # Multiple peaks - complex profile
            return VolumeProfile.NEUTRAL_VOLUME
        
        # Check if volume is higher on up or down moves
        price_changes = df['price'].diff()
        up_moves = df[price_changes > 0]['volume'].sum()
        down_moves = df[price_changes < 0]['volume'].sum()
        
        total_directional_volume = up_moves + down_moves
        
        if total_directional_volume > 0:
            up_ratio = up_moves / total_directional_volume
            
            if up_ratio > 0.6:
                return VolumeProfile.BULLISH_VOLUME
            elif up_ratio < 0.4:
                return VolumeProfile.BEARISH_VOLUME
        
        # Check for climax volume
        recent_volume = df['volume'].tail(len(df) // 10).mean()  # Last 10% of data
        avg_volume = df['volume'].mean()
        
        if recent_volume > avg_volume * 2:
            return VolumeProfile.CLIMAX_VOLUME
        elif recent_volume < avg_volume * 0.5:
            return VolumeProfile.LOW_VOLUME
        
        return VolumeProfile.NEUTRAL_VOLUME
    
    def _calculate_flow_persistence(self, df: pd.DataFrame) -> float:
        """Calculate how persistent the order flow is"""
        if len(df) < 10:
            return 0.0
        
        # Create flow imbalance series
        window_size = max(5, len(df) // 20)
        flow_imbalances = []
        
        for i in range(window_size, len(df)):
            window_df = df.iloc[i-window_size:i]
            buy_vol = window_df[window_df['side'] == 'buy']['volume'].sum()
            sell_vol = window_df[window_df['side'] == 'sell']['volume'].sum()
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                imbalance = (buy_vol - sell_vol) / total_vol
                flow_imbalances.append(imbalance)
        
        if len(flow_imbalances) < 2:
            return 0.0
        
        # Measure persistence as autocorrelation
        flow_series = pd.Series(flow_imbalances)
        persistence = flow_series.autocorr(lag=1)
        
        return persistence if not np.isnan(persistence) else 0.0
    
    def _determine_overall_direction(self, pressure_analysis: Dict[str, Any]) -> FlowDirection:
        """Determine overall flow direction from pressure analysis"""
        buy_volume = pressure_analysis['buy_pressure']['volume']
        sell_volume = pressure_analysis['sell_pressure']['volume']
        pressure_ratio = pressure_analysis['pressure_ratio']
        
        # Strong thresholds
        if pressure_ratio > 1.5:
            return FlowDirection.BULLISH
        elif pressure_ratio < 0.67:
            return FlowDirection.BEARISH
        
        # Check institutional activity
        institutional_bias = pressure_analysis['institutional_activity']['institutional_buy_bias']
        
        if institutional_bias > 0.6:
            return FlowDirection.ACCUMULATION
        elif institutional_bias < 0.4:
            return FlowDirection.DISTRIBUTION
        
        return FlowDirection.NEUTRAL
    
    def _default_flow_analysis(self) -> Dict[str, Any]:
        """Return default analysis when no ticks available"""
        return {
            "error": "No tick data available",
            "flow_metrics": None,
            "pressure_analysis": None,
            "market_impact": None,
            "volume_profile": None,
            "flow_persistence": 0.0,
            "overall_direction": FlowDirection.NEUTRAL.value,
            "tick_count": 0
        }
    
    def get_flow_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable flow summary"""
        if "error" in analysis:
            return f"Flow analysis error: {analysis['error']}"
        
        direction = analysis['overall_direction']
        tick_count = analysis['tick_count']
        
        if analysis['flow_metrics']:
            net_pressure = analysis['flow_metrics']['net_buying_pressure']
            
            summary = f"Flow Direction: {direction.upper()}"
            summary += f"\nNet Buying Pressure: {net_pressure:.2%}"
            summary += f"\nTicks Analyzed: {tick_count:,}"
            
            if analysis['pressure_analysis']:
                pressure_ratio = analysis['pressure_analysis']['pressure_ratio']
                summary += f"\nBuy/Sell Ratio: {pressure_ratio:.2f}"
            
            return summary
        
        return "Insufficient data for flow analysis"
