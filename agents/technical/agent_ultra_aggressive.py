"""
Ultra-Aggressive Technical Analysis Agent with Optimized Signal Detection
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .models import TechnicalOpportunity, Direction, VolatilityRegime, TimeframeAlignment, TechnicalFeatures, RiskMetrics
from common.data_adapters.yfinance_adapter_fixed import FixedYFinanceAdapter
from common.opportunity_store import OpportunityStore
from common.unified_opportunity_scorer_enhanced import EnhancedUnifiedOpportunityScorer


class UltraAggressiveTechnicalAgent:
    """
    Ultra-Aggressive Technical Analysis Agent with optimized signal detection
    """
    
    def __init__(self):
        self.data_adapter = FixedYFinanceAdapter({})
        self.opportunity_store = OpportunityStore()
        self.scorer = EnhancedUnifiedOpportunityScorer()
        self.min_confidence = 0.1  # Ultra-aggressive: lowered from 0.2 to 0.1
        self.max_opportunities_per_symbol = 10  # Increased from 5 to 10
        
        # Enhanced ML models
        self.pattern_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        
        # Optimized caching
        self.data_cache = {}
        self.cache_ttl = 180  # 3 minutes cache
        
    async def find_opportunities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Find opportunities with ultra-aggressive algorithms"""
        try:
            start_time = time.time()
            
            symbols = payload.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'AMD', 'INTC'])
            timeframes = payload.get('timeframes', ['1h', '4h', '1d'])
            strategies = payload.get('strategies', ['imbalance', 'trend', 'liquidity', 'breakout', 'reversal', 'momentum', 'support_resistance'])
            
            print(f"🔍 Ultra-Aggressive Analysis: {len(symbols)} symbols, {len(strategies)} strategies")
            
            all_opportunities = []
            
            for symbol in symbols:
                symbol_opportunities = await self._analyze_symbol_ultra_aggressive(symbol, timeframes, strategies)
                all_opportunities.extend(symbol_opportunities)
                await asyncio.sleep(0.02)  # Reduced rate limiting for faster processing
            
            # Calculate scores and store
            for opportunity in all_opportunities:
                opportunity.priority_score = self.scorer.calculate_priority_score(opportunity)
                self.opportunity_store.add_opportunity(opportunity)
            
            all_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            
            analysis_time = time.time() - start_time
            avg_confidence = np.mean([opp.confidence_score for opp in all_opportunities]) if all_opportunities else 0
            
            return {
                'opportunities': [self._opportunity_to_dict(opp) for opp in all_opportunities],
                'metadata': {
                    'analysis_time': analysis_time,
                    'opportunities_found': len(all_opportunities),
                    'average_confidence': avg_confidence,
                    'symbols_analyzed': len(symbols),
                    'strategies_used': strategies
                },
                'success': True
            }
            
        except Exception as e:
            print(f"Error in ultra-aggressive analysis: {e}")
            return {'opportunities': [], 'metadata': {'error': str(e)}, 'success': False}
    
    async def _analyze_symbol_ultra_aggressive(self, symbol: str, timeframes: List[str], 
                                            strategies: List[str]) -> List[TechnicalOpportunity]:
        """Ultra-aggressive symbol analysis with optimized data processing"""
        opportunities = []
        
        try:
            # Optimized data fetching with caching
            df = await self._get_optimized_market_data(symbol, timeframes)
            
            if df.empty or len(df) < 30:  # Reduced minimum data requirement
                return opportunities
            
            # Enhanced technical indicators
            df = self._add_enhanced_indicators(df)
            
            # Advanced pattern recognition
            patterns = self._detect_enhanced_patterns(df)
            
            # Multi-strategy analysis with reduced filtering
            for strategy in strategies:
                try:
                    if strategy == 'imbalance':
                        opps = await self._analyze_imbalances_ultra_aggressive(symbol, df, patterns)
                    elif strategy == 'trend':
                        opps = await self._analyze_trends_ultra_aggressive(symbol, df, patterns)
                    elif strategy == 'liquidity':
                        opps = await self._analyze_liquidity_ultra_aggressive(symbol, df, patterns)
                    elif strategy == 'breakout':
                        opps = await self._analyze_breakouts_ultra_aggressive(symbol, df, patterns)
                    elif strategy == 'reversal':
                        opps = await self._analyze_reversals_ultra_aggressive(symbol, df, patterns)
                    elif strategy == 'momentum':
                        opps = await self._analyze_momentum_ultra_aggressive(symbol, df, patterns)
                    elif strategy == 'support_resistance':
                        opps = await self._analyze_support_resistance_ultra_aggressive(symbol, df, patterns)
                    else:
                        continue
                    
                    if opps:
                        opportunities.extend(opps)
                        
                except Exception as e:
                    print(f"Error in {strategy} analysis for {symbol}: {e}")
                    continue
            
            return opportunities[:self.max_opportunities_per_symbol]
            
        except Exception as e:
            print(f"Error in ultra-aggressive analysis for {symbol}: {e}")
            return opportunities
    
    async def _get_optimized_market_data(self, symbol: str, timeframes: List[str]) -> pd.DataFrame:
        """Optimized market data fetching with enhanced caching"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframes[0]}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.data_cache:
                cached_data, timestamp = self.data_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Fetch data with optimized parameters
            since = datetime.now() - timedelta(days=15)  # Increased lookback
            df = await self.data_adapter.get_ohlcv(symbol, timeframes[0], since, 2000)  # Increased data points
            
            # Cache the result
            self.data_cache[cache_key] = (df, time.time())
            
            return df
            
        except Exception as e:
            print(f"Error fetching optimized market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators with more sensitivity"""
        try:
            # Basic indicators
            df['RSI'] = self._calculate_rsi(df['Close'], 14)
            df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
            df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
            df['ATR'] = self._calculate_atr(df)
            
            # Enhanced indicators
            df['RSI_7'] = self._calculate_rsi(df['Close'], 7)  # Faster RSI
            df['RSI_21'] = self._calculate_rsi(df['Close'], 21)  # Slower RSI
            df['EMA_9'] = df['Close'].ewm(span=9).mean()
            df['EMA_21'] = df['Close'].ewm(span=21).mean()
            df['SMA_10'] = df['Close'].rolling(10).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            
            # Momentum indicators
            df['ROC'] = self._calculate_rate_of_change(df['Close'], 10)
            df['Williams_R'] = self._calculate_williams_r(df)
            df['CCI'] = self._calculate_cci(df)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['OBV'] = self._calculate_obv(df)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Price action
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Upper_Wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Body_Ratio'] = df['Body_Size'] / (df['High'] - df['Low'])
            
            # Volatility indicators
            df['Volatility'] = df['Close'].rolling(20).std()
            df['Volatility_Ratio'] = df['Volatility'] / df['Close'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            print(f"Error adding enhanced indicators: {e}")
            return df
    
    async def _analyze_imbalances_ultra_aggressive(self, symbol: str, df: pd.DataFrame, 
                                                 patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        """Ultra-aggressive imbalance analysis with reduced thresholds"""
        opportunities = []
        
        try:
            for i in range(1, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                # Reduced gap threshold from 0.3% to 0.1%
                gap_up = current['Low'] - previous['High']
                gap_down = previous['Low'] - current['High']
                
                # Ultra-aggressive volume threshold
                volume_surge = current['Volume_Ratio'] > 1.5  # Reduced from 2.0
                
                # Gap up imbalance with reduced threshold
                if gap_up > 0 and gap_up > current['Close'] * 0.001:  # 0.1% gap
                    strength = min(1.0, (gap_up / current['Close']) * 20 * current['Volume_Ratio'])
                    if strength > 0.1:  # Ultra-aggressive threshold
                        opportunity = await self._create_opportunity(symbol, 'imbalance', {
                            'type': 'gap_up',
                            'price': previous['High'],
                            'strength': strength,
                            'volume_surge': volume_surge
                        }, df)
                        if opportunity:
                            opportunities.append(opportunity)
                
                # Gap down imbalance with reduced threshold
                elif gap_down > 0 and gap_down > current['Close'] * 0.001:
                    strength = min(1.0, (gap_down / current['Close']) * 20 * current['Volume_Ratio'])
                    if strength > 0.1:
                        opportunity = await self._create_opportunity(symbol, 'imbalance', {
                            'type': 'gap_down',
                            'price': previous['Low'],
                            'strength': strength,
                            'volume_surge': volume_surge
                        }, df)
                        if opportunity:
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            print(f"Error in ultra-aggressive imbalance analysis: {e}")
            return opportunities
    
    async def _analyze_trends_ultra_aggressive(self, symbol: str, df: pd.DataFrame, 
                                             patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        """Ultra-aggressive trend analysis with enhanced sensitivity"""
        opportunities = []
        
        try:
            # Multiple trend signals with reduced thresholds
            trend_signals = []
            
            # Price trend signals
            current_price = df['Close'].iloc[-1]
            current_ema9 = df['EMA_9'].iloc[-1]
            current_ema21 = df['EMA_21'].iloc[-1]
            current_sma10 = df['SMA_10'].iloc[-1]
            current_sma50 = df['SMA_50'].iloc[-1]
            
            # Reduced trend thresholds
            price_above_ema9 = current_price > current_ema9
            price_above_ema21 = current_price > current_ema21
            price_above_sma10 = current_price > current_sma10
            price_above_sma50 = current_price > current_sma50
            ema9_above_ema21 = current_ema9 > current_ema21
            sma10_above_sma50 = current_sma10 > current_sma50
            
            # RSI trend signals
            rsi_trend = df['RSI'].iloc[-1] > 45  # Reduced from 50
            rsi7_trend = df['RSI_7'].iloc[-1] > 40  # Even more aggressive
            
            # MACD trend signals
            macd_trend = df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]
            
            # Volume trend signals
            volume_trend = df['Volume_Ratio'].iloc[-3:].mean() > 1.1  # Reduced threshold
            
            trend_signals = [price_above_ema9, price_above_ema21, price_above_sma10, 
                           price_above_sma50, ema9_above_ema21, sma10_above_sma50,
                           rsi_trend, rsi7_trend, macd_trend, volume_trend]
            
            bullish_signals = sum(trend_signals)
            trend_strength = bullish_signals / len(trend_signals)
            
            # Ultra-aggressive trend detection
            if trend_strength > 0.4:  # Reduced from 0.5
                direction = Direction.LONG
            else:
                direction = Direction.SHORT
                trend_strength = 1 - trend_strength
            
            if trend_strength > 0.2:  # Ultra-aggressive threshold
                opportunity = await self._create_opportunity(symbol, 'trend', {
                    'direction': direction,
                    'strength': trend_strength,
                    'signals': trend_signals
                }, df)
                if opportunity:
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            print(f"Error in ultra-aggressive trend analysis: {e}")
            return opportunities
    
    async def _analyze_momentum_ultra_aggressive(self, symbol: str, df: pd.DataFrame, 
                                               patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        """Ultra-aggressive momentum analysis"""
        opportunities = []
        
        try:
            # Momentum signals with reduced thresholds
            current_roc = df['ROC'].iloc[-1]
            current_williams_r = df['Williams_R'].iloc[-1]
            current_cci = df['CCI'].iloc[-1]
            current_stoch_k = df['Stoch_K'].iloc[-1]
            
            # Ultra-aggressive momentum thresholds
            if current_roc > 0.5:  # Reduced from 1.0
                opportunity = await self._create_opportunity(symbol, 'momentum', {
                    'type': 'momentum_up',
                    'strength': min(1.0, current_roc / 2.0),
                    'indicator': 'ROC'
                }, df)
                if opportunity:
                    opportunities.append(opportunity)
            
            if current_williams_r < -70:  # Reduced from -80
                opportunity = await self._create_opportunity(symbol, 'momentum', {
                    'type': 'momentum_up',
                    'strength': min(1.0, abs(current_williams_r) / 100.0),
                    'indicator': 'Williams_R'
                }, df)
                if opportunity:
                    opportunities.append(opportunity)
            
            if current_cci > 100:  # Reduced from 150
                opportunity = await self._create_opportunity(symbol, 'momentum', {
                    'type': 'momentum_up',
                    'strength': min(1.0, current_cci / 200.0),
                    'indicator': 'CCI'
                }, df)
                if opportunity:
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            print(f"Error in ultra-aggressive momentum analysis: {e}")
            return opportunities
    
    async def _create_opportunity(self, symbol: str, strategy: str, 
                                analysis_data: Dict[str, Any], df: pd.DataFrame) -> Optional[TechnicalOpportunity]:
        """Create technical opportunity with ultra-aggressive parameters"""
        try:
            current_price = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            # Determine direction and confidence
            if strategy == 'imbalance':
                direction = Direction.LONG if analysis_data['type'] == 'gap_up' else Direction.SHORT
                entry_price = analysis_data['price']
                confidence = analysis_data['strength']
            elif strategy == 'trend':
                direction = analysis_data['direction']
                entry_price = current_price
                confidence = analysis_data['strength']
            elif strategy == 'momentum':
                direction = Direction.LONG if analysis_data['type'] == 'momentum_up' else Direction.SHORT
                entry_price = current_price
                confidence = analysis_data['strength']
            else:
                direction = Direction.LONG
                entry_price = current_price
                confidence = 0.3  # Default confidence
            
            # Ultra-aggressive risk management
            if direction == Direction.LONG:
                stop_loss = entry_price - (atr * 1.0)  # Tighter stops
                take_profit = [entry_price + (atr * 2.0), entry_price + (atr * 3.0)]
            else:
                stop_loss = entry_price + (atr * 1.0)
                take_profit = [entry_price - (atr * 2.0), entry_price - (atr * 3.0)]
            
            # Calculate risk-reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit[0] - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            
            # Create timeframe alignment
            timeframe_alignment = TimeframeAlignment()
            timeframe_alignment.primary = '1h'
            timeframe_alignment.confirmation = ['4h', '1d']
            timeframe_alignment.alignment_score = confidence
            
            # Technical features
            technical_features = TechnicalFeatures()
            technical_features.trend_strength = confidence
            technical_features.volatility_regime = VolatilityRegime.NORMAL
            
            # Risk metrics
            risk_metrics = RiskMetrics(
                max_loss=risk,
                position_size=0.03,  # 3% position size
                sharpe_ratio=1.5,
                max_drawdown=0.02
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
            print(f"Error creating opportunity: {e}")
            return None
    
    # Enhanced technical indicator calculations
    def _calculate_rate_of_change(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change"""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        return ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['Volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    # Standard indicator calculations (from previous implementation)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> tuple:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _detect_enhanced_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced pattern detection with more patterns"""
        patterns = {}
        
        # Enhanced candlestick patterns
        patterns['doji'] = self._detect_doji(df)
        patterns['hammer'] = self._detect_hammer(df)
        patterns['engulfing'] = self._detect_engulfing(df)
        patterns['shooting_star'] = self._detect_shooting_star(df)
        patterns['morning_star'] = self._detect_morning_star(df)
        patterns['evening_star'] = self._detect_evening_star(df)
        
        # Enhanced chart patterns
        patterns['double_top'] = self._detect_double_top(df)
        patterns['double_bottom'] = self._detect_double_bottom(df)
        patterns['head_shoulders'] = self._detect_head_shoulders(df)
        patterns['triangle'] = self._detect_triangle(df)
        
        # Support/Resistance
        patterns['support_levels'] = self._find_support_levels(df)
        patterns['resistance_levels'] = self._find_resistance_levels(df)
        
        return patterns
    
    # Enhanced pattern detection methods
    def _detect_shooting_star(self, df: pd.DataFrame) -> List[int]:
        shooting_star_indices = []
        for i in range(len(df)):
            body_size = abs(df.iloc[i]['Close'] - df.iloc[i]['Open'])
            upper_wick = df.iloc[i]['High'] - max(df.iloc[i]['Open'], df.iloc[i]['Close'])
            lower_wick = min(df.iloc[i]['Open'], df.iloc[i]['Close']) - df.iloc[i]['Low']
            
            if upper_wick > body_size * 2 and lower_wick < body_size * 0.5:
                shooting_star_indices.append(i)
        return shooting_star_indices
    
    def _detect_morning_star(self, df: pd.DataFrame) -> List[int]:
        morning_star_indices = []
        for i in range(2, len(df)):
            # Simplified morning star detection
            if df.iloc[i-2]['Close'] < df.iloc[i-2]['Open'] and \
               abs(df.iloc[i-1]['Close'] - df.iloc[i-1]['Open']) < df.iloc[i-1]['High'] - df.iloc[i-1]['Low'] * 0.1 and \
               df.iloc[i]['Close'] > df.iloc[i]['Open']:
                morning_star_indices.append(i)
        return morning_star_indices
    
    def _detect_evening_star(self, df: pd.DataFrame) -> List[int]:
        evening_star_indices = []
        for i in range(2, len(df)):
            # Simplified evening star detection
            if df.iloc[i-2]['Close'] > df.iloc[i-2]['Open'] and \
               abs(df.iloc[i-1]['Close'] - df.iloc[i-1]['Open']) < df.iloc[i-1]['High'] - df.iloc[i-1]['Low'] * 0.1 and \
               df.iloc[i]['Close'] < df.iloc[i]['Open']:
                evening_star_indices.append(i)
        return evening_star_indices
    
    def _detect_triangle(self, df: pd.DataFrame) -> List[int]:
        # Simplified triangle detection
        return []  # Complex pattern, simplified for now
    
    # Standard pattern detection methods (from previous implementation)
    def _detect_doji(self, df: pd.DataFrame) -> List[int]:
        doji_indices = []
        for i in range(len(df)):
            body_size = abs(df.iloc[i]['Close'] - df.iloc[i]['Open'])
            total_range = df.iloc[i]['High'] - df.iloc[i]['Low']
            if total_range > 0 and body_size / total_range < 0.1:
                doji_indices.append(i)
        return doji_indices
    
    def _detect_hammer(self, df: pd.DataFrame) -> List[int]:
        hammer_indices = []
        for i in range(len(df)):
            body_size = abs(df.iloc[i]['Close'] - df.iloc[i]['Open'])
            lower_wick = min(df.iloc[i]['Open'], df.iloc[i]['Close']) - df.iloc[i]['Low']
            upper_wick = df.iloc[i]['High'] - max(df.iloc[i]['Open'], df.iloc[i]['Close'])
            
            if lower_wick > body_size * 2 and upper_wick < body_size * 0.5:
                hammer_indices.append(i)
        return hammer_indices
    
    def _detect_engulfing(self, df: pd.DataFrame) -> List[int]:
        engulfing_indices = []
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            prev_body = abs(prev['Close'] - prev['Open'])
            curr_body = abs(curr['Close'] - curr['Open'])
            
            if curr_body > prev_body * 1.5:
                if curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']:
                    engulfing_indices.append(i)
        return engulfing_indices
    
    def _detect_double_top(self, df: pd.DataFrame) -> List[int]:
        peaks = []
        for i in range(1, len(df)-1):
            if df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['High'] > df.iloc[i+1]['High']:
                peaks.append(i)
        return peaks
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[int]:
        troughs = []
        for i in range(1, len(df)-1):
            if df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and df.iloc[i]['Low'] < df.iloc[i+1]['Low']:
                troughs.append(i)
        return troughs
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[int]:
        return []  # Complex pattern, simplified for now
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        supports = []
        for i in range(2, len(df)-2):
            if df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and df.iloc[i]['Low'] < df.iloc[i-2]['Low'] and \
               df.iloc[i]['Low'] < df.iloc[i+1]['Low'] and df.iloc[i]['Low'] < df.iloc[i+2]['Low']:
                supports.append(df.iloc[i]['Low'])
        return supports
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        resistances = []
        for i in range(2, len(df)-2):
            if df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['High'] > df.iloc[i-2]['High'] and \
               df.iloc[i]['High'] > df.iloc[i+1]['High'] and df.iloc[i]['High'] > df.iloc[i+2]['High']:
                resistances.append(df.iloc[i]['High'])
        return resistances
    
    # Placeholder methods for other strategies
    async def _analyze_liquidity_ultra_aggressive(self, symbol: str, df: pd.DataFrame, patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        return []
    
    async def _analyze_breakouts_ultra_aggressive(self, symbol: str, df: pd.DataFrame, patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        return []
    
    async def _analyze_reversals_ultra_aggressive(self, symbol: str, df: pd.DataFrame, patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        return []
    
    async def _analyze_support_resistance_ultra_aggressive(self, symbol: str, df: pd.DataFrame, patterns: Dict[str, Any]) -> List[TechnicalOpportunity]:
        return []
    
    def _opportunity_to_dict(self, opportunity: TechnicalOpportunity) -> Dict[str, Any]:
        """Convert opportunity to dictionary"""
        return {
            'symbol': opportunity.symbol,
            'strategy': opportunity.strategy,
            'direction': opportunity.direction.value,
            'entry_price': opportunity.entry_price,
            'stop_loss': opportunity.stop_loss,
            'take_profit': opportunity.take_profit,
            'risk_reward_ratio': opportunity.risk_reward_ratio,
            'confidence_score': opportunity.confidence_score,
            'priority_score': opportunity.priority_score,
            'timestamp': opportunity.timestamp.isoformat()
        }
