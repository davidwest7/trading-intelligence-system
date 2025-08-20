"""
Comprehensive Test Script for Top Performers Agent

Tests all components:
- UniverseConstructor
- PerformanceCalculator
- MomentumModel
- SectorAnalyzer
- TopPerformersAgent
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.top_performers.agent_complete import (
    TopPerformersAgent, UniverseConstructor, PerformanceCalculator, 
    MomentumModel, SectorAnalyzer, AssetClass, PerformanceMetrics,
    MomentumIndicators, VolumeProfile, RankingData
)


def test_universe_constructor():
    """Test UniverseConstructor functionality"""
    print("🧪 Testing UniverseConstructor...")
    
    constructor = UniverseConstructor()
    
    try:
        # Test universe construction
        asset_classes = ["equities", "fx", "crypto"]
        regions = ["US", "EU", "UK"]
        
        universe = constructor.construct_universe(asset_classes, regions)
        
        # Validate universe
        assert isinstance(universe, list), "Universe should be a list"
        assert len(universe) > 0, "Universe should not be empty"
        assert len(universe) <= constructor.max_symbols, f"Universe should not exceed {constructor.max_symbols} symbols"
        
        # Check for expected symbols
        expected_symbols = {
            'equities': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'fx': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'crypto': ['BTC-USD', 'ETH-USD']
        }
        
        for asset_class, symbols in expected_symbols.items():
            for symbol in symbols:
                if symbol in universe:
                    print(f"  ✅ Found {symbol} in {asset_class} universe")
        
        print(f"  ✅ Universe constructed with {len(universe)} symbols")
        
    except Exception as e:
        print(f"  ❌ Error testing universe constructor: {e}")
        return False
    
    print("  ✅ UniverseConstructor tests passed")
    return True


def test_performance_calculator():
    """Test PerformanceCalculator functionality"""
    print("🧪 Testing PerformanceCalculator...")
    
    calculator = PerformanceCalculator(risk_free_rate=0.02)
    
    try:
        # Generate test price data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Test case 1: Trending up
        trend_prices = pd.Series(100 * np.exp(np.linspace(0, 0.2, 252)), index=dates)
        performance = calculator.calculate_performance(trend_prices)
        
        # Validate performance metrics
        assert isinstance(performance, PerformanceMetrics), "Should return PerformanceMetrics"
        assert performance.return_pct > 0, "Trending up should have positive return"
        assert performance.volatility > 0, "Volatility should be positive"
        assert performance.sharpe_ratio > 0, "Trending up should have positive Sharpe ratio"
        assert 0 <= performance.win_rate <= 1, "Win rate should be [0,1]"
        
        print(f"  ✅ Trending performance: return={performance.return_pct:.2%}, sharpe={performance.sharpe_ratio:.2f}")
        
        # Test case 2: Sideways
        sideways_prices = pd.Series(100 + 5 * np.sin(np.linspace(0, 4*np.pi, 252)), index=dates)
        performance = calculator.calculate_performance(sideways_prices)
        
        assert abs(performance.return_pct) < 0.1, "Sideways should have low return"
        print(f"  ✅ Sideways performance: return={performance.return_pct:.2%}, sharpe={performance.sharpe_ratio:.2f}")
        
        # Test case 3: Volatile
        volatile_prices = pd.Series(100 + np.cumsum(np.random.normal(0, 0.02, 252)), index=dates)
        performance = calculator.calculate_performance(volatile_prices)
        
        assert performance.volatility > 0.1, "Volatile should have high volatility"
        print(f"  ✅ Volatile performance: volatility={performance.volatility:.2%}, max_dd={performance.max_drawdown:.2%}")
        
        # Test empty performance
        empty_perf = calculator._empty_performance()
        assert isinstance(empty_perf, PerformanceMetrics), "Should return PerformanceMetrics"
        assert empty_perf.return_pct == 0.0, "Empty performance should have zero return"
        
        print("  ✅ Empty performance handling works")
        
    except Exception as e:
        print(f"  ❌ Error testing performance calculator: {e}")
        return False
    
    print("  ✅ PerformanceCalculator tests passed")
    return True


def test_momentum_model():
    """Test MomentumModel functionality"""
    print("🧪 Testing MomentumModel...")
    
    model = MomentumModel()
    
    try:
        # Generate test price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Test case 1: Strong uptrend
        uptrend_prices = pd.Series(100 * np.exp(np.linspace(0, 0.3, 100)), index=dates)
        momentum = model.calculate_momentum(uptrend_prices)
        
        # Validate momentum indicators
        assert isinstance(momentum, MomentumIndicators), "Should return MomentumIndicators"
        assert 0 <= momentum.rsi <= 100, "RSI should be [0,100]"
        assert momentum.momentum_score > 0.5, "Uptrend should have high momentum score"
        assert momentum.trend_strength > 0.5, "Uptrend should have high trend strength"
        
        print(f"  ✅ Uptrend momentum: RSI={momentum.rsi:.1f}, score={momentum.momentum_score:.2f}, strength={momentum.trend_strength:.2f}")
        
        # Test case 2: Downtrend
        downtrend_prices = pd.Series(100 * np.exp(-np.linspace(0, 0.2, 100)), index=dates)
        momentum = model.calculate_momentum(downtrend_prices)
        
        assert momentum.momentum_score < 0.5, "Downtrend should have low momentum score"
        print(f"  ✅ Downtrend momentum: RSI={momentum.rsi:.1f}, score={momentum.momentum_score:.2f}")
        
        # Test case 3: Sideways
        sideways_prices = pd.Series(100 + 2 * np.sin(np.linspace(0, 4*np.pi, 100)), index=dates)
        momentum = model.calculate_momentum(sideways_prices)
        
        assert abs(momentum.momentum_score - 0.5) < 0.3, "Sideways should have neutral momentum"
        print(f"  ✅ Sideways momentum: RSI={momentum.rsi:.1f}, score={momentum.momentum_score:.2f}")
        
        # Test with universe data
        universe_prices = {
            'AAPL': uptrend_prices,
            'TSLA': downtrend_prices,
            'MSFT': sideways_prices
        }
        momentum_with_universe = model.calculate_momentum(uptrend_prices, universe_prices)
        
        assert 0 <= momentum_with_universe.relative_strength <= 1, "Relative strength should be [0,1]"
        print(f"  ✅ Relative strength calculation: {momentum_with_universe.relative_strength:.2f}")
        
        # Test empty momentum
        empty_momentum = model._empty_momentum()
        assert isinstance(empty_momentum, MomentumIndicators), "Should return MomentumIndicators"
        assert empty_momentum.rsi == 50.0, "Empty momentum should have neutral RSI"
        
        print("  ✅ Empty momentum handling works")
        
    except Exception as e:
        print(f"  ❌ Error testing momentum model: {e}")
        return False
    
    print("  ✅ MomentumModel tests passed")
    return True


def test_sector_analyzer():
    """Test SectorAnalyzer functionality"""
    print("🧪 Testing SectorAnalyzer...")
    
    analyzer = SectorAnalyzer()
    
    try:
        # Create test rankings
        test_rankings = []
        
        # Technology stocks
        for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
            ranking = RankingData(
                symbol=symbol,
                rank=i+1,
                asset_class=AssetClass.EQUITIES,
                region='US',
                performance=PerformanceMetrics(
                    return_pct=0.15 + i*0.02,
                    volatility=0.20,
                    sharpe_ratio=1.5 + i*0.1,
                    max_drawdown=-0.05,
                    sortino_ratio=1.8,
                    information_ratio=0.5,
                    calmar_ratio=3.0,
                    win_rate=0.65
                ),
                momentum=MomentumIndicators(
                    rsi=70 + i*5,
                    macd_signal="bullish",
                    trend_strength=0.8 + i*0.05,
                    relative_strength=0.7 + i*0.05,
                    momentum_score=0.8 + i*0.05,
                    momentum_decay=0.0
                ),
                volume=VolumeProfile(
                    avg_daily_volume=1000000 * (10 - i),
                    volume_trend="increasing",
                    relative_volume=1.2 - i*0.1,
                    volume_momentum=0.1
                ),
                score=0.9 - i*0.1,
                confidence=0.8,
                timestamp=datetime.now()
            )
            test_rankings.append(ranking)
        
        # Consumer stocks
        for i, symbol in enumerate(['AMZN', 'TSLA']):
            ranking = RankingData(
                symbol=symbol,
                rank=len(test_rankings)+1,
                asset_class=AssetClass.EQUITIES,
                region='US',
                performance=PerformanceMetrics(
                    return_pct=0.10 + i*0.02,
                    volatility=0.25,
                    sharpe_ratio=1.2 + i*0.1,
                    max_drawdown=-0.08,
                    sortino_ratio=1.5,
                    information_ratio=0.3,
                    calmar_ratio=2.5,
                    win_rate=0.60
                ),
                momentum=MomentumIndicators(
                    rsi=65 + i*5,
                    macd_signal="bullish",
                    trend_strength=0.7 + i*0.05,
                    relative_strength=0.6 + i*0.05,
                    momentum_score=0.7 + i*0.05,
                    momentum_decay=0.0
                ),
                volume=VolumeProfile(
                    avg_daily_volume=800000 * (10 - i),
                    volume_trend="stable",
                    relative_volume=1.0 - i*0.1,
                    volume_momentum=0.05
                ),
                score=0.8 - i*0.1,
                confidence=0.75,
                timestamp=datetime.now()
            )
            test_rankings.append(ranking)
        
        # Analyze sector performance
        sector_analysis = analyzer.analyze_sector_performance(test_rankings)
        
        # Validate sector analysis
        assert isinstance(sector_analysis, dict), "Should return dictionary"
        assert 'Technology' in sector_analysis, "Should have Technology sector"
        assert 'Consumer Discretionary' in sector_analysis, "Should have Consumer Discretionary sector"
        
        # Validate Technology sector
        tech = sector_analysis['Technology']
        assert tech['count'] == 3, "Should have 3 technology stocks"
        assert tech['avg_return'] > 0, "Technology should have positive average return"
        assert len(tech['top_performers']) == 3, "Should have 3 top performers"
        
        # Validate Consumer Discretionary sector
        consumer = sector_analysis['Consumer Discretionary']
        assert consumer['count'] == 2, "Should have 2 consumer stocks"
        assert consumer['avg_return'] > 0, "Consumer should have positive average return"
        
        print(f"  ✅ Sector analysis completed:")
        print(f"     Technology: {tech['count']} stocks, avg return: {tech['avg_return']:.2%}")
        print(f"     Consumer: {consumer['count']} stocks, avg return: {consumer['avg_return']:.2%}")
        
    except Exception as e:
        print(f"  ❌ Error testing sector analyzer: {e}")
        return False
    
    print("  ✅ SectorAnalyzer tests passed")
    return True


async def test_top_performers_agent():
    """Test TopPerformersAgent functionality"""
    print("🧪 Testing TopPerformersAgent...")
    
    agent = TopPerformersAgent()
    
    try:
        # Test ranking with different parameters
        test_cases = [
            ("1w", ["equities"], ["US"], 1000000, 10),
            ("1m", ["equities", "fx"], ["US", "EU"], 500000, 20),
            ("3m", ["equities", "crypto"], ["US"], 2000000, 15)
        ]
        
        for horizon, asset_classes, regions, min_volume, limit in test_cases:
            results = await agent.rank(
                horizon=horizon,
                asset_classes=asset_classes,
                regions=regions,
                min_volume=min_volume,
                limit=limit
            )
            
            # Validate result structure
            assert 'timestamp' in results, "Missing timestamp"
            assert 'horizon' in results, "Missing horizon"
            assert 'rankings' in results, "Missing rankings"
            assert 'sector_analysis' in results, "Missing sector analysis"
            assert 'benchmark_performance' in results, "Missing benchmark performance"
            assert 'market_regime' in results, "Missing market regime"
            assert 'metadata' in results, "Missing metadata"
            
            # Validate rankings
            rankings = results['rankings']
            assert isinstance(rankings, list), "Rankings should be list"
            assert len(rankings) <= limit, f"Should not exceed limit of {limit}"
            
            if len(rankings) > 0:
                # Validate first ranking
                first_ranking = rankings[0]
                assert 'rank' in first_ranking, "Missing rank"
                assert 'symbol' in first_ranking, "Missing symbol"
                assert 'asset_class' in first_ranking, "Missing asset_class"
                assert 'performance' in first_ranking, "Missing performance"
                assert 'momentum_indicators' in first_ranking, "Missing momentum_indicators"
                assert 'volume_profile' in first_ranking, "Missing volume_profile"
                assert 'score' in first_ranking, "Missing score"
                
                # Validate performance metrics
                performance = first_ranking['performance']
                assert 'return_pct' in performance, "Missing return_pct"
                assert 'volatility' in performance, "Missing volatility"
                assert 'sharpe_ratio' in performance, "Missing sharpe_ratio"
                assert 'max_drawdown' in performance, "Missing max_drawdown"
                
                # Validate momentum indicators
                momentum = first_ranking['momentum_indicators']
                assert 'rsi' in momentum, "Missing RSI"
                assert 'momentum_score' in momentum, "Missing momentum_score"
                assert 'relative_strength' in momentum, "Missing relative_strength"
                
                # Validate volume profile
                volume = first_ranking['volume_profile']
                assert 'avg_daily_volume' in volume, "Missing avg_daily_volume"
                assert 'relative_volume' in volume, "Missing relative_volume"
                
                # Check ranking order
                scores = [r['score'] for r in rankings]
                assert scores == sorted(scores, reverse=True), "Rankings should be sorted by score"
                
                print(f"  ✅ {horizon} ranking: {len(rankings)} assets, top score: {first_ranking['score']:.3f}")
            
            # Validate benchmark performance
            benchmark = results['benchmark_performance']
            assert 'return' in benchmark, "Missing benchmark return"
            assert 'volatility' in benchmark, "Missing benchmark volatility"
            assert 'sharpe' in benchmark, "Missing benchmark sharpe"
            
            # Validate market regime
            regime = results['market_regime']
            assert 'regime' in regime, "Missing regime"
            assert 'confidence' in regime, "Missing confidence"
            assert 'avg_return' in regime, "Missing avg_return"
            
            # Validate metadata
            metadata = results['metadata']
            assert 'total_analyzed' in metadata, "Missing total_analyzed"
            assert 'rankings_generated' in metadata, "Missing rankings_generated"
            assert 'ranking_time' in metadata, "Missing ranking_time"
        
        # Test momentum signals
        symbols = ["AAPL", "TSLA", "MSFT"]
        signals = await agent.get_momentum_signals(symbols, "1w")
        
        assert isinstance(signals, list), "Signals should be list"
        for signal in signals:
            assert hasattr(signal, 'symbol'), "Signal missing symbol"
            assert hasattr(signal, 'signal_type'), "Signal missing signal_type"
            assert hasattr(signal, 'mu'), "Signal missing mu"
            assert hasattr(signal, 'sigma'), "Signal missing sigma"
            assert hasattr(signal, 'confidence'), "Signal missing confidence"
        
        print(f"  ✅ Generated {len(signals)} momentum signals")
        
    except Exception as e:
        print(f"  ❌ Error testing top performers agent: {e}")
        return False
    
    print("  ✅ TopPerformersAgent tests passed")
    return True


async def test_integration():
    """Test integration between components"""
    print("🧪 Testing Integration...")
    
    agent = TopPerformersAgent()
    
    try:
        # Test end-to-end workflow
        results = await agent.rank(
            horizon="1w",
            asset_classes=["equities", "fx"],
            regions=["US", "EU"],
            min_volume=1000000,
            limit=20
        )
        
        # Validate comprehensive results
        assert len(results['rankings']) > 0, "Should have rankings"
        assert len(results['sector_analysis']) > 0, "Should have sector analysis"
        assert results['market_regime']['regime'] in ['bull_market', 'bear_market', 'volatile', 'sideways'], "Invalid regime"
        
        # Check data quality
        for ranking in results['rankings']:
            assert ranking['score'] >= 0 and ranking['score'] <= 1, "Score should be [0,1]"
            assert ranking['performance']['return_pct'] > -1, "Return should be > -100%"
            assert ranking['performance']['volatility'] >= 0, "Volatility should be >= 0"
            assert ranking['momentum_indicators']['rsi'] >= 0 and ranking['momentum_indicators']['rsi'] <= 100, "RSI should be [0,100]"
        
        print(f"  ✅ Integration test completed:")
        print(f"     Rankings: {len(results['rankings'])}")
        print(f"     Sectors: {len(results['sector_analysis'])}")
        print(f"     Regime: {results['market_regime']['regime']}")
        print(f"     Benchmark return: {results['benchmark_performance']['return']:.2%}")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False
    
    print("  ✅ Integration tests passed")
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Top Performers Agent Tests\n")
    
    tests = [
        ("UniverseConstructor", test_universe_constructor()),
        ("PerformanceCalculator", test_performance_calculator()),
        ("MomentumModel", test_momentum_model()),
        ("SectorAnalyzer", test_sector_analyzer()),
        ("TopPerformersAgent", test_top_performers_agent()),
        ("Integration", test_integration())
    ]
    
    results = {}
    
    for test_name, test_coro in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Tests")
        print(f"{'='*50}")
        
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Top Performers Agent tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
