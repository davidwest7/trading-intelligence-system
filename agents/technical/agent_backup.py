"""
Technical Strategy Agent - Main implementation
"""

import time
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import (
    TechnicalOpportunity, AnalysisPayload, AnalysisMetadata, 
    MarketRegime, Direction
)
from .strategies import (
    ImbalanceStrategy, FairValueGapStrategy, LiquiditySweepStrategy,
    IDFPStrategy, TrendStrategy, BreakoutStrategy, MeanReversionStrategy
)
from .backtest import PurgedCrossValidationBacktester


class TechnicalAgent:
    """
    Technical Strategy Agent for multi-timeframe technical analysis
    
    Features:
    - Imbalance/FVG detection
    - Liquidity sweep identification  
    - IDFP (Institutional Dealing Range/Point) analysis
    - Multi-timeframe alignment
    - Trend/breakout/mean-reversion ensemble
    - Purged cross-validation backtesting
    """
    
    def __init__(self, data_adapter=None):
        self.strategies = {
            "imbalance": ImbalanceStrategy(),
            "fvg": FairValueGapStrategy(),
            "liquidity_sweep": LiquiditySweepStrategy(),
            "idfp": IDFPStrategy(),
            "trend": TrendStrategy(),
            "breakout": BreakoutStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
        
        self.data_adapter = data_adapter
        self.backtester = PurgedCrossValidationBacktester()
        
    async def find_opportunities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for finding technical opportunities
        
        Args:
            payload: Analysis payload containing symbols, timeframes, strategies, etc.
            
        Returns:
            Dictionary with opportunities and metadata
        """
        start_time = time.time()
        
        # Parse payload
        analysis_payload = AnalysisPayload(**payload)
        
        # Validate strategies
        valid_strategies = [s for s in analysis_payload.strategies if s in self.strategies]
        if not valid_strategies:
            valid_strategies = ["imbalance", "trend"]  # Default strategies
            
        # Get market data for all symbols and timeframes
        market_data = await self._get_market_data(
            analysis_payload.symbols, 
            analysis_payload.timeframes,
            analysis_payload.lookback_periods
        )
        
        # Run analysis across all strategies and symbols
        all_opportunities = []
        for symbol in analysis_payload.symbols:
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                continue
                
            for strategy_name in valid_strategies:
                strategy = self.strategies[strategy_name]
                opportunities = strategy.analyze(symbol_data, symbol, analysis_payload.timeframes)
                
                # Filter by confidence score
                filtered_opportunities = [
                    opp for opp in opportunities 
                    if opp.confidence_score >= analysis_payload.min_score
                ]
                
                all_opportunities.extend(filtered_opportunities)
        
        # Apply risk filtering
        risk_filtered_opportunities = self._apply_risk_filter(
            all_opportunities, 
            analysis_payload.max_risk
        )
        
        # Calculate metadata
        analysis_time = int((time.time() - start_time) * 1000)
        metadata = AnalysisMetadata(
            analysis_time_ms=analysis_time,
            symbols_analyzed=len(analysis_payload.symbols),
            opportunities_found=len(risk_filtered_opportunities),
            market_regime=self._determine_market_regime(market_data),
            overall_bias=self._determine_overall_bias(risk_filtered_opportunities)
        )
        
        return {
            "opportunities": [opp.to_dict() for opp in risk_filtered_opportunities],
            "metadata": metadata.to_dict()
        }
    
    async def _get_market_data(self, symbols: List[str], timeframes: List[str], lookback: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get market data for analysis"""
        if not self.data_adapter:
            # Return mock data for testing
            return self._generate_mock_data(symbols, timeframes, lookback)
            
        # TODO: Implement data adapter integration
        # data = {}
        # for symbol in symbols:
        #     data[symbol] = {}
        #     for tf in timeframes:
        #         df = await self.data_adapter.get_ohlcv(symbol, tf, lookback)
        #         data[symbol][tf] = df
        # return data
        
        return self._generate_mock_data(symbols, timeframes, lookback)
    
    def _generate_mock_data(self, symbols: List[str], timeframes: List[str], lookback: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate mock OHLCV data for testing"""
        import numpy as np
        
        data = {}
        for symbol in symbols:
            data[symbol] = {}
            for tf in timeframes:
                # Generate realistic price data
                dates = pd.date_range(end=datetime.now(), periods=lookback, freq='1h')
                
                # Random walk with some trend
                returns = np.random.normal(0.0001, 0.01, lookback)
                prices = 100 * np.exp(np.cumsum(returns))
                
                # Create OHLCV data
                df = pd.DataFrame(index=dates)
                df['close'] = prices
                df['open'] = df['close'].shift(1).fillna(prices[0])
                
                # Add some noise for high/low
                noise = np.random.normal(0, 0.005, lookback)
                df['high'] = df['close'] * (1 + np.abs(noise))
                df['low'] = df['close'] * (1 - np.abs(noise))
                
                # Ensure OHLC consistency
                df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
                df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
                
                df['volume'] = np.random.exponential(1000000, lookback)
                
                data[symbol][tf] = df
                
        return data
    
    def _apply_risk_filter(self, opportunities: List[TechnicalOpportunity], max_risk: float) -> List[TechnicalOpportunity]:
        """Filter opportunities by maximum risk threshold"""
        filtered = []
        for opp in opportunities:
            risk_pct = abs(opp.stop_loss - opp.entry_price) / opp.entry_price
            if risk_pct <= max_risk:
                filtered.append(opp)
        return filtered
    
    def _determine_market_regime(self, market_data: Dict[str, Dict[str, pd.DataFrame]]) -> MarketRegime:
        """Determine overall market regime from price data"""
        # Simplified regime detection
        # TODO: Implement proper regime detection using HMM or similar
        
        total_volatility = 0
        count = 0
        
        for symbol_data in market_data.values():
            for df in symbol_data.values():
                if len(df) > 20:
                    returns = df['close'].pct_change().dropna()
                    vol = returns.rolling(20).std().iloc[-1]
                    total_volatility += vol
                    count += 1
        
        if count == 0:
            return MarketRegime.CALM
            
        avg_vol = total_volatility / count
        
        if avg_vol > 0.02:
            return MarketRegime.VOLATILE
        elif avg_vol > 0.015:
            return MarketRegime.TRENDING
        elif avg_vol > 0.01:
            return MarketRegime.RANGING
        else:
            return MarketRegime.CALM
    
    def _determine_overall_bias(self, opportunities: List[TechnicalOpportunity]) -> str:
        """Determine overall directional bias from opportunities"""
        if not opportunities:
            return "neutral"
            
        long_count = sum(1 for opp in opportunities if opp.direction == Direction.LONG)
        short_count = len(opportunities) - long_count
        
        if long_count > short_count * 1.5:
            return "bullish"
        elif short_count > long_count * 1.5:
            return "bearish"
        else:
            return "neutral"
    
    async def backtest_strategy(self, strategy_name: str, symbols: List[str], 
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Backtest a specific strategy using purged cross-validation
        
        Args:
            strategy_name: Name of strategy to backtest
            symbols: List of symbols to test
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results with performance metrics
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        strategy = self.strategies[strategy_name]
        
        # Get historical data
        historical_data = await self._get_historical_data(symbols, start_date, end_date)
        
        # Run purged cross-validation
        results = await self.backtester.run_backtest(
            strategy=strategy,
            data=historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        return results
    
    async def _get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get historical data for backtesting"""
        # TODO: Implement historical data fetching
        # For now, return mock data
        return {}


# Example usage and testing
if __name__ == "__main__":
    async def test_technical_agent():
        agent = TechnicalAgent()
        
        payload = {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframes": ["15m", "1h", "4h"],
            "strategies": ["imbalance", "trend"],
            "min_score": 0.6,
            "max_risk": 0.02,
            "lookback_periods": 200
        }
        
        results = await agent.find_opportunities(payload)
        print("Opportunities found:", len(results["opportunities"]))
        print("Analysis time:", results["metadata"]["analysis_time_ms"], "ms")
        
        if results["opportunities"]:
            print("\nFirst opportunity:")
            for key, value in results["opportunities"][0].items():
                print(f"  {key}: {value}")
    
    # Run test
    asyncio.run(test_technical_agent())
