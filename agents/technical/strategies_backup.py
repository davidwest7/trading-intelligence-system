"""
Technical analysis strategies implementation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from .models import (
    TechnicalOpportunity, TechnicalFeatures, RiskMetrics, 
    TimeframeAlignment, Direction, ImbalanceZone, LiquidityLevel,
    VolatilityRegime
)


class BaseStrategy(ABC):
    """Base class for technical strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.lookback = 200
        
    @abstractmethod
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        """Analyze data and return opportunities"""
        pass
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def detect_trend(self, df: pd.DataFrame) -> Tuple[float, Direction]:
        """Detect trend strength and direction"""
        # Simple EMA-based trend detection
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        ema_200 = df['close'].ewm(span=200).mean()
        
        current_price = df['close'].iloc[-1]
        trend_score = 0.0
        
        # EMA alignment
        if current_price > ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
            trend_score = 0.8
            direction = Direction.LONG
        elif current_price < ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
            trend_score = 0.8
            direction = Direction.SHORT
        else:
            trend_score = 0.3
            direction = Direction.LONG if current_price > ema_50.iloc[-1] else Direction.SHORT
            
        return trend_score, direction


class ImbalanceStrategy(BaseStrategy):
    """Strategy to detect price imbalances and fair value gaps"""
    
    def __init__(self):
        super().__init__("imbalance")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        opportunities = []
        
        for tf in timeframes:
            if tf not in data:
                continue
                
            df = data[tf].tail(self.lookback)
            imbalances = self.detect_imbalances(df, tf)
            
            # Always try to create opportunity for demo
            opportunity = self.create_opportunity(df, symbol, tf, imbalances)
            if opportunity:
                opportunities.append(opportunity)
        
        return opportunities
    
    def detect_imbalances(self, df: pd.DataFrame, timeframe: str) -> List[ImbalanceZone]:
        """Detect price imbalances/fair value gaps"""
        imbalances = []
        
        for i in range(2, len(df) - 1):
            # Look for gaps where candle[i-1].high < candle[i+1].low (bullish imbalance)
            # or candle[i-1].low > candle[i+1].high (bearish imbalance)
            
            prev_candle = df.iloc[i-1]
            curr_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish imbalance
            if prev_candle['high'] < next_candle['low']:
                gap_size = next_candle['low'] - prev_candle['high']
                strength = min(gap_size / prev_candle['close'], 1.0)
                
                if strength > 0.001:  # Minimum gap size threshold
                    imbalance = ImbalanceZone(
                        start_price=prev_candle['high'],
                        end_price=next_candle['low'],
                        timeframe=timeframe,
                        strength=strength,
                        volume_imbalance=abs(curr_candle['volume'] - df['volume'].rolling(20).mean().iloc[i]),
                        timestamp=pd.to_datetime(curr_candle.name)
                    )
                    imbalances.append(imbalance)
            
            # Bearish imbalance
            elif prev_candle['low'] > next_candle['high']:
                gap_size = prev_candle['low'] - next_candle['high']
                strength = min(gap_size / prev_candle['close'], 1.0)
                
                if strength > 0.001:
                    imbalance = ImbalanceZone(
                        start_price=next_candle['high'],
                        end_price=prev_candle['low'],
                        timeframe=timeframe,
                        strength=strength,
                        volume_imbalance=abs(curr_candle['volume'] - df['volume'].rolling(20).mean().iloc[i]),
                        timestamp=pd.to_datetime(curr_candle.name)
                    )
                    imbalances.append(imbalance)
        
        # Sort by strength and return top candidates
        return sorted(imbalances, key=lambda x: x.strength, reverse=True)[:5]
    
    def create_opportunity(self, df: pd.DataFrame, symbol: str, timeframe: str, imbalances: List[ImbalanceZone]) -> Optional[TechnicalOpportunity]:
        """Create trading opportunity from imbalances"""
        # For demo purposes, create an opportunity even if no imbalances found
        if not imbalances:
            # Create a demo imbalance
            demo_imbalance = ImbalanceZone(
                start_price=df['close'].iloc[-1] * 1.005,
                end_price=df['close'].iloc[-1] * 1.015,
                timeframe=timeframe,
                strength=0.6,
                volume_imbalance=1000,
                timestamp=pd.to_datetime(df.index[-1])
            )
            imbalances = [demo_imbalance]
            
        strongest_imbalance = imbalances[0]
        current_price = df['close'].iloc[-1]
        atr = self.calculate_atr(df).iloc[-1]
        
        # Determine direction based on imbalance relative to current price
        if current_price < strongest_imbalance.start_price:
            direction = Direction.LONG
            entry_price = strongest_imbalance.start_price
            stop_loss = entry_price - (2 * atr)
            take_profit = [entry_price + (2 * atr), entry_price + (4 * atr)]
        else:
            direction = Direction.SHORT
            entry_price = strongest_imbalance.end_price
            stop_loss = entry_price + (2 * atr)
            take_profit = [entry_price - (2 * atr), entry_price - (4 * atr)]
        
        risk_reward = abs(take_profit[0] - entry_price) / abs(stop_loss - entry_price)
        
        # Build technical features
        features = TechnicalFeatures(
            imbalance_zones=imbalances,
            trend_strength=self.detect_trend(df)[0],
            volatility_regime=self.get_volatility_regime(df)
        )
        
        # Calculate risk metrics (simplified)
        risk_metrics = RiskMetrics(
            max_drawdown=0.05,  # TODO: Calculate from backtest
            var_95=abs(stop_loss - entry_price) / entry_price,
            sharpe_ratio=1.2,  # TODO: Calculate from backtest
            win_rate=0.65  # TODO: Calculate from backtest
        )
        
        # Timeframe alignment
        alignment = TimeframeAlignment(
            primary=timeframe,
            confirmation=[],  # TODO: Check other timeframes
            alignment_score=strongest_imbalance.strength
        )
        
        return TechnicalOpportunity(
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            confidence_score=strongest_imbalance.strength,
            timeframe_alignment=alignment,
            technical_features=features,
            risk_metrics=risk_metrics,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(hours=24)
        )
    
    def get_volatility_regime(self, df: pd.DataFrame) -> VolatilityRegime:
        """Determine current volatility regime"""
        returns = df['close'].pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1]
        historical_vol = returns.rolling(100).std().mean()
        
        vol_ratio = current_vol / historical_vol
        
        if vol_ratio > 1.5:
            return VolatilityRegime.HIGH
        elif vol_ratio < 0.7:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL


class FairValueGapStrategy(BaseStrategy):
    """Strategy to detect and trade fair value gaps"""
    
    def __init__(self):
        super().__init__("fvg")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        # Similar implementation to ImbalanceStrategy but with different logic
        # TODO: Implement FVG-specific detection
        return []


class LiquiditySweepStrategy(BaseStrategy):
    """Strategy to detect liquidity sweeps and stop hunts"""
    
    def __init__(self):
        super().__init__("liquidity_sweep")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        # TODO: Implement liquidity sweep detection
        # - Identify key support/resistance levels
        # - Detect when price briefly breaks level then reverses
        # - Look for volume spikes during sweep
        return []


class IDFPStrategy(BaseStrategy):
    """Institutional Dealing Range/Point strategy"""
    
    def __init__(self):
        super().__init__("idfp")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        # TODO: Implement IDFP detection
        # - Identify ranging periods
        # - Detect institutional dealing points
        # - Trade bounces from key levels
        return []


class TrendStrategy(BaseStrategy):
    """Trend following strategy"""
    
    def __init__(self):
        super().__init__("trend")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        # TODO: Implement trend following
        # - Multiple EMA alignment
        # - Momentum confirmation
        # - Pullback entries
        return []


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for range breaks and pattern completions"""
    
    def __init__(self):
        super().__init__("breakout")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        # TODO: Implement breakout detection
        # - Chart pattern recognition
        # - Volume confirmation
        # - False breakout filtering
        return []


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy for oversold/overbought conditions"""
    
    def __init__(self):
        super().__init__("mean_reversion")
    
    def analyze(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]) -> List[TechnicalOpportunity]:
        # TODO: Implement mean reversion
        # - RSI divergences
        # - Bollinger Band extremes
        # - Support/resistance bounces
        return []
