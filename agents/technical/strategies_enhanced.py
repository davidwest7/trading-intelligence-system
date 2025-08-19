"""
Enhanced Technical Analysis Strategies with Realistic Market Data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from .models import TechnicalOpportunity, Direction, VolatilityRegime, TimeframeAlignment, TechnicalFeatures, RiskMetrics
from common.data_adapters.yfinance_adapter import YFinanceAdapter


@dataclass
class ImbalanceLevel:
    """Represents a market imbalance level"""
    price: float
    strength: float  # 0-1, how strong the imbalance is
    volume: int
    timestamp: datetime
    direction: Direction


@dataclass
class TrendDirection:
    """Represents trend direction and strength"""
    direction: Direction
    strength: float  # 0-1, trend strength
    slope: float
    support: float
    resistance: float


class EnhancedTechnicalStrategies:
    """
    Enhanced technical analysis strategies with realistic market data
    """
    
    def __init__(self):
        self.data_adapter = YFinanceAdapter({})
        self.min_volume_threshold = 100000  # Minimum volume for valid signals
        self.min_price_threshold = 1.0  # Minimum price for analysis
        
    async def find_imbalances(self, symbol: str, timeframe: str = '1h', 
                            lookback_days: int = 5) -> List[ImbalanceLevel]:
        """
        Find market imbalances using realistic price action
        """
        try:
            # Get realistic market data
            since = datetime.now() - timedelta(days=lookback_days)
            df = await self.data_adapter.get_ohlcv(symbol, timeframe, since, 1000)
            
            if df.empty or len(df) < 20:
                return []
            
            imbalances = []
            
            # Calculate volume-weighted average price (VWAP)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Find gaps and imbalances
            for i in range(1, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                # Calculate gap size
                gap_up = current['Low'] - previous['High']
                gap_down = previous['Low'] - current['High']
                
                # Volume analysis
                avg_volume = df['Volume'].rolling(20).mean().iloc[i]
                volume_ratio = current['Volume'] / avg_volume if avg_volume > 0 else 1
                
                # Gap up imbalance
                if gap_up > 0 and gap_up > current['Close'] * 0.005:  # 0.5% gap
                    strength = min(1.0, (gap_up / current['Close']) * 10 * volume_ratio)
                    if strength > 0.3:  # Minimum strength threshold
                        imbalances.append(ImbalanceLevel(
                            price=previous['High'],
                            strength=strength,
                            volume=int(current['Volume']),
                            timestamp=current['Date'],
                            direction=Direction.LONG
                        ))
                
                # Gap down imbalance
                elif gap_down > 0 and gap_down > current['Close'] * 0.005:  # 0.5% gap
                    strength = min(1.0, (gap_down / current['Close']) * 10 * volume_ratio)
                    if strength > 0.3:  # Minimum strength threshold
                        imbalances.append(ImbalanceLevel(
                            price=previous['Low'],
                            strength=strength,
                            volume=int(current['Volume']),
                            timestamp=current['Date'],
                            direction=Direction.SHORT
                        ))
            
            return imbalances
            
        except Exception as e:
            print(f"Error finding imbalances for {symbol}: {e}")
            return []
    
    async def detect_trends(self, symbol: str, timeframe: str = '1h', 
                          lookback_days: int = 10) -> TrendDirection:
        """
        Detect trend direction and strength using multiple indicators
        """
        try:
            # Get realistic market data
            since = datetime.now() - timedelta(days=lookback_days)
            df = await self.data_adapter.get_ohlcv(symbol, timeframe, since, 1000)
            
            if df.empty or len(df) < 50:
                return TrendDirection(Direction.LONG, 0.5, 0.0, 0.0, 0.0)
            
            # Calculate technical indicators
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Determine trend direction
            sma_trend = latest['SMA_20'] > latest['SMA_50']
            ema_trend = latest['EMA_12'] > latest['EMA_26']
            macd_trend = latest['MACD'] > latest['MACD_Signal']
            price_trend = latest['Close'] > latest['SMA_20']
            
            # Calculate trend strength (0-1)
            trend_signals = [sma_trend, ema_trend, macd_trend, price_trend]
            bullish_signals = sum(trend_signals)
            trend_strength = bullish_signals / len(trend_signals)
            
            # Determine direction
            if trend_strength > 0.5:
                direction = Direction.LONG
            else:
                direction = Direction.SHORT
                trend_strength = 1 - trend_strength
            
            # Calculate slope
            recent_prices = df['Close'].tail(20)
            if len(recent_prices) >= 2:
                slope = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            else:
                slope = 0.0
            
            # Calculate support and resistance
            support = df['Low'].tail(20).min()
            resistance = df['High'].tail(20).max()
            
            return TrendDirection(
                direction=direction,
                strength=trend_strength,
                slope=slope,
                support=support,
                resistance=resistance
            )
            
        except Exception as e:
            print(f"Error detecting trends for {symbol}: {e}")
            return TrendDirection(Direction.LONG, 0.5, 0.0, 0.0, 0.0)
    
    async def find_liquidity_sweeps(self, symbol: str, timeframe: str = '1h',
                                  lookback_days: int = 3) -> List[Dict[str, Any]]:
        """
        Find liquidity sweeps using realistic volume analysis
        """
        try:
            # Get realistic market data
            since = datetime.now() - timedelta(days=lookback_days)
            df = await self.data_adapter.get_ohlcv(symbol, timeframe, since, 1000)
            
            if df.empty or len(df) < 20:
                return []
            
            sweeps = []
            
            # Calculate volume metrics
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Find high volume candles
            high_volume_threshold = 2.0  # 2x average volume
            
            for i in range(1, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                # Check for volume spike
                if current['Volume_Ratio'] > high_volume_threshold:
                    # Check for wick (liquidity sweep)
                    body_size = abs(current['Close'] - current['Open'])
                    total_range = current['High'] - current['Low']
                    
                    if total_range > 0:
                        wick_ratio = (total_range - body_size) / total_range
                        
                        if wick_ratio > 0.3:  # Significant wick
                            # Determine sweep direction
                            upper_wick = current['High'] - max(current['Open'], current['Close'])
                            lower_wick = min(current['Open'], current['Close']) - current['Low']
                            
                            if upper_wick > lower_wick:
                                # Bullish sweep (swept highs)
                                sweep_type = 'bullish'
                                sweep_price = current['High']
                                direction = Direction.LONG
                            else:
                                # Bearish sweep (swept lows)
                                sweep_type = 'bearish'
                                sweep_price = current['Low']
                                direction = Direction.SHORT
                            
                            sweeps.append({
                                'timestamp': current['Date'],
                                'price': sweep_price,
                                'type': sweep_type,
                                'direction': direction,
                                'volume_ratio': current['Volume_Ratio'],
                                'wick_ratio': wick_ratio,
                                'strength': min(1.0, current['Volume_Ratio'] * wick_ratio)
                            })
            
            return sweeps
            
        except Exception as e:
            print(f"Error finding liquidity sweeps for {symbol}: {e}")
            return []
    
    async def analyze_multi_timeframe(self, symbol: str, 
                                    timeframes: List[str] = ['1h', '4h', '1d']) -> TimeframeAlignment:
        """
        Analyze multi-timeframe alignment for stronger signals
        """
        try:
            alignments = []
            
            for timeframe in timeframes:
                trend = await self.detect_trends(symbol, timeframe, 10)
                alignments.append({
                    'timeframe': timeframe,
                    'direction': trend.direction,
                    'strength': trend.strength
                })
            
            # Calculate alignment score
            bullish_count = sum(1 for a in alignments if a['direction'] == Direction.LONG)
            alignment_score = bullish_count / len(alignments)
            
            # Determine primary timeframe
            primary_timeframe = timeframes[0]
            
            return TimeframeAlignment(
                primary_timeframe=primary_timeframe,
                aligned_timeframes=[tf for tf in timeframes if alignment_score > 0.5],
                alignment_score=alignment_score
            )
            
        except Exception as e:
            print(f"Error analyzing multi-timeframe for {symbol}: {e}")
            return TimeframeAlignment('1h', ['1h'], 0.5)
    
    async def create_opportunity(self, symbol: str, strategy: str, 
                               imbalance: Optional[ImbalanceLevel] = None,
                               trend: Optional[TrendDirection] = None,
                               sweeps: Optional[List[Dict[str, Any]]] = None) -> TechnicalOpportunity:
        """
        Create a technical opportunity with realistic parameters
        """
        try:
            # Get current market data
            quote = await self.data_adapter.get_quote(symbol)
            current_price = quote['price']
            
            if current_price < self.min_price_threshold:
                return None
            
            # Determine direction and entry
            if strategy == 'imbalance' and imbalance:
                direction = imbalance.direction
                entry_price = imbalance.price
                confidence = imbalance.strength
            elif strategy == 'trend' and trend:
                direction = trend.direction
                entry_price = current_price
                confidence = trend.strength
            else:
                # Default to current price analysis
                direction = Direction.LONG if current_price > 0 else Direction.SHORT
                entry_price = current_price
                confidence = 0.5
            
            # Calculate realistic stop loss and take profit
            volatility = 0.02  # 2% volatility
            atr = current_price * volatility
            
            if direction == Direction.LONG:
                stop_loss = entry_price - (atr * 2)  # 2 ATR below entry
                take_profit = [entry_price + (atr * 3), entry_price + (atr * 5)]  # 3 and 5 ATR above
            else:
                stop_loss = entry_price + (atr * 2)  # 2 ATR above entry
                take_profit = [entry_price - (atr * 3), entry_price - (atr * 5)]  # 3 and 5 ATR below
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit[0] - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            
            # Multi-timeframe analysis
            timeframes = ['1h', '4h', '1d']
            timeframe_alignment = await self.analyze_multi_timeframe(symbol, timeframes)
            
            # Technical features
            technical_features = TechnicalFeatures(
                rsi=50.0,  # Will be calculated in real implementation
                macd=0.0,
                volume_ratio=1.0,
                volatility=volatility
            )
            
            # Risk metrics
            risk_metrics = RiskMetrics(
                max_loss=risk,
                position_size=0.01,  # 1% of portfolio
                sharpe_ratio=1.0,
                max_drawdown=0.05
            )
            
            return TechnicalOpportunity(
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                confidence_score=confidence,
                timeframe_alignment=timeframe_alignment,
                technical_features=technical_features,
                risk_metrics=risk_metrics,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error creating opportunity for {symbol}: {e}")
            return None
