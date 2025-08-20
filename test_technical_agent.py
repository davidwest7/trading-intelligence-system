"""
Comprehensive Test Script for Technical Agent

Tests all components:
- DataAdapter
- RegimeDetector
- TechnicalAnalyzer
- TechnicalAgent
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.technical.agent_complete import (
    TechnicalAgent, DataAdapter, RegimeDetector, TechnicalAnalyzer
)


async def test_data_adapter():
    """Test DataAdapter functionality"""
    print("🧪 Testing DataAdapter...")
    
    adapter = DataAdapter()
    
    # Test different symbols and timeframes
    test_cases = [
        ("AAPL", "1h", 100),
        ("EURUSD", "15m", 200),
        ("BTC-USD", "4h", 50)
    ]
    
    for symbol, timeframe, lookback in test_cases:
        try:
            data = await adapter.get_ohlcv(symbol, timeframe, lookback)
            
            # Validate data structure
            assert isinstance(data, pd.DataFrame), f"Data should be DataFrame for {symbol}"
            assert len(data) == lookback, f"Data length should be {lookback} for {symbol}"
            assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']), f"Missing OHLCV columns for {symbol}"
            
            # Validate data quality
            assert not data.isnull().any().any(), f"Data contains nulls for {symbol}"
            assert (data['high'] >= data['low']).all(), f"High should be >= low for {symbol}"
            assert (data['high'] >= data['close']).all(), f"High should be >= close for {symbol}"
            assert (data['high'] >= data['open']).all(), f"High should be >= open for {symbol}"
            
            print(f"  ✅ {symbol} {timeframe} data: {len(data)} rows, price range: {data['close'].min():.2f}-{data['close'].max():.2f}")
            
        except Exception as e:
            print(f"  ❌ Error testing {symbol} {timeframe}: {e}")
            return False
    
    # Test caching
    try:
        data1 = await adapter.get_ohlcv("AAPL", "1h", 50)
        data2 = await adapter.get_ohlcv("AAPL", "1h", 50)
        assert data1.equals(data2), "Cached data should be identical"
        print("  ✅ Caching works correctly")
    except Exception as e:
        print(f"  ❌ Caching test failed: {e}")
        return False
    
    print("  ✅ DataAdapter tests passed")
    return True


def test_regime_detector():
    """Test RegimeDetector functionality"""
    print("🧪 Testing RegimeDetector...")
    
    detector = RegimeDetector()
    
    # Generate test data for different regimes
    test_cases = [
        ("trending", np.linspace(100, 120, 100)),  # Upward trend
        ("ranging", 100 + 5 * np.sin(np.linspace(0, 4*np.pi, 100))),  # Sideways
        ("volatile", 100 + np.cumsum(np.random.normal(0, 0.02, 100)))  # Volatile
    ]
    
    for regime_name, prices in test_cases:
        try:
            df = pd.DataFrame({
                'close': prices,
                'open': prices * 0.999,
                'high': prices * 1.002,
                'low': prices * 0.998,
                'volume': np.random.exponential(1000000, len(prices))
            })
            
            result = detector.detect_regime(df)
            
            # Validate result structure
            assert 'regime' in result, f"Missing regime for {regime_name}"
            assert 'confidence' in result, f"Missing confidence for {regime_name}"
            assert 'indicators' in result, f"Missing indicators for {regime_name}"
            assert 0 <= result['confidence'] <= 1, f"Confidence should be [0,1] for {regime_name}"
            
            # Validate indicators
            indicators = result['indicators']
            assert 'volatility' in indicators, f"Missing volatility indicator for {regime_name}"
            assert 'trend' in indicators, f"Missing trend indicator for {regime_name}"
            assert 'momentum' in indicators, f"Missing momentum indicator for {regime_name}"
            assert 'volume' in indicators, f"Missing volume indicator for {regime_name}"
            
            print(f"  ✅ {regime_name} regime detected: {result['regime']} (confidence: {result['confidence']:.2f})")
            
        except Exception as e:
            print(f"  ❌ Error testing {regime_name} regime: {e}")
            return False
    
    # Test with insufficient data
    try:
        small_df = pd.DataFrame({'close': [100, 101, 102]})
        result = detector.detect_regime(small_df)
        assert result['regime'] == 'ranging', "Should return default regime for insufficient data"
        print("  ✅ Insufficient data handling works")
    except Exception as e:
        print(f"  ❌ Insufficient data test failed: {e}")
        return False
    
    print("  ✅ RegimeDetector tests passed")
    return True


def test_technical_analyzer():
    """Test TechnicalAnalyzer functionality"""
    print("🧪 Testing TechnicalAnalyzer...")
    
    analyzer = TechnicalAnalyzer()
    
    # Generate test data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'volume': np.random.exponential(1000000, 100)
    }, index=dates)
    
    try:
        patterns = analyzer.analyze_patterns(df)
        
        # Validate pattern structure
        assert 'support_resistance' in patterns, "Missing support/resistance analysis"
        assert 'chart_patterns' in patterns, "Missing chart patterns"
        assert 'indicators' in patterns, "Missing technical indicators"
        assert 'candlestick' in patterns, "Missing candlestick patterns"
        
        # Test support/resistance
        sr = patterns['support_resistance']
        assert 'support' in sr, "Missing support levels"
        assert 'resistance' in sr, "Missing resistance levels"
        assert isinstance(sr['support'], list), "Support should be list"
        assert isinstance(sr['resistance'], list), "Resistance should be list"
        
        # Test indicators
        indicators = patterns['indicators']
        assert 'rsi' in indicators, "Missing RSI"
        assert 'macd' in indicators, "Missing MACD"
        assert 'bollinger' in indicators, "Missing Bollinger Bands"
        assert 'stochastic' in indicators, "Missing Stochastic"
        
        # Validate RSI
        rsi = indicators['rsi']
        assert 0 <= rsi <= 100, f"RSI should be [0,100], got {rsi}"
        
        # Validate MACD
        macd = indicators['macd']
        assert 'macd' in macd, "MACD missing macd value"
        assert 'signal' in macd, "MACD missing signal value"
        assert 'histogram' in macd, "MACD missing histogram value"
        
        # Validate Bollinger Bands
        bb = indicators['bollinger']
        assert 'upper' in bb, "Bollinger missing upper band"
        assert 'middle' in bb, "Bollinger missing middle band"
        assert 'lower' in bb, "Bollinger missing lower band"
        
        print(f"  ✅ Technical analysis completed:")
        print(f"     RSI: {rsi:.1f}")
        print(f"     Support levels: {len(sr['support'])}")
        print(f"     Resistance levels: {len(sr['resistance'])}")
        
    except Exception as e:
        print(f"  ❌ Error testing technical analyzer: {e}")
        return False
    
    print("  ✅ TechnicalAnalyzer tests passed")
    return True


async def test_technical_agent():
    """Test TechnicalAgent functionality"""
    print("🧪 Testing TechnicalAgent...")
    
    agent = TechnicalAgent()
    
    # Test symbols
    symbols = ["AAPL", "EURUSD", "BTC-USD"]
    timeframes = ["15m", "1h"]
    strategies = ["patterns", "indicators", "regime"]
    
    try:
        # Test full analysis
        results = await agent.analyze(symbols, timeframes, strategies)
        
        # Validate result structure
        assert 'timestamp' in results, "Missing timestamp"
        assert 'symbols_analyzed' in results, "Missing symbols_analyzed"
        assert 'timeframes' in results, "Missing timeframes"
        assert 'analysis' in results, "Missing analysis"
        assert 'regime_analysis' in results, "Missing regime_analysis"
        assert 'signals' in results, "Missing signals"
        assert 'performance' in results, "Missing performance"
        
        # Validate analysis for each symbol
        for symbol in symbols:
            assert symbol in results['analysis'], f"Missing analysis for {symbol}"
            symbol_analysis = results['analysis'][symbol]
            
            assert 'symbol' in symbol_analysis, f"Missing symbol in analysis for {symbol}"
            assert 'timeframes' in symbol_analysis, f"Missing timeframes in analysis for {symbol}"
            assert 'overall_regime' in symbol_analysis, f"Missing overall_regime in analysis for {symbol}"
            assert 'consensus_signals' in symbol_analysis, f"Missing consensus_signals in analysis for {symbol}"
            
            # Validate timeframe analysis
            for tf in timeframes:
                if tf in symbol_analysis['timeframes']:
                    tf_analysis = symbol_analysis['timeframes'][tf]
                    if 'regime' in strategies and 'regime' in tf_analysis:
                        assert 'regime' in tf_analysis['regime'], f"Missing regime for {symbol} {tf}"
                    if 'patterns' in strategies and 'patterns' in tf_analysis:
                        assert 'support_resistance' in tf_analysis['patterns'], f"Missing patterns for {symbol} {tf}"
                    if 'indicators' in strategies and 'indicators' in tf_analysis:
                        assert 'rsi' in tf_analysis['indicators'], f"Missing indicators for {symbol} {tf}"
        
        # Validate regime analysis
        regime_analysis = results['regime_analysis']
        assert 'regime' in regime_analysis, "Missing regime in overall analysis"
        assert 'confidence' in regime_analysis, "Missing confidence in overall analysis"
        
        # Validate performance metrics
        performance = results['performance']
        assert 'analysis_time' in performance, "Missing analysis_time"
        assert 'avg_analysis_time' in performance, "Missing avg_analysis_time"
        assert 'analysis_count' in performance, "Missing analysis_count"
        
        print(f"  ✅ Technical analysis completed:")
        print(f"     Symbols analyzed: {len(results['symbols_analyzed'])}")
        print(f"     Signals generated: {len(results['signals'])}")
        print(f"     Overall regime: {regime_analysis['regime']} (confidence: {regime_analysis['confidence']:.2f})")
        print(f"     Analysis time: {performance['analysis_time']:.3f}s")
        
    except Exception as e:
        print(f"  ❌ Error testing technical agent: {e}")
        return False
    
    # Test individual methods
    try:
        # Test support/resistance
        sr = await agent.get_support_resistance("AAPL", "1h")
        assert 'support' in sr, "Missing support levels"
        assert 'resistance' in sr, "Missing resistance levels"
        print("  ✅ Support/resistance analysis works")
        
        # Test regime analysis
        regime = await agent.get_regime_analysis("AAPL", "1h")
        assert 'regime' in regime, "Missing regime"
        assert 'confidence' in regime, "Missing confidence"
        print("  ✅ Individual regime analysis works")
        
    except Exception as e:
        print(f"  ❌ Error testing individual methods: {e}")
        return False
    
    print("  ✅ TechnicalAgent tests passed")
    return True


async def test_integration():
    """Test integration between components"""
    print("🧪 Testing Integration...")
    
    agent = TechnicalAgent()
    
    try:
        # Test end-to-end workflow
        symbols = ["AAPL", "TSLA"]
        timeframes = ["1h", "4h"]
        
        # Full analysis
        results = await agent.analyze(symbols, timeframes)
        
        # Validate signals
        signals = results['signals']
        for signal in signals:
            assert hasattr(signal, 'symbol'), "Signal missing symbol"
            assert hasattr(signal, 'signal_type'), "Signal missing signal_type"
            assert hasattr(signal, 'mu'), "Signal missing mu"
            assert hasattr(signal, 'sigma'), "Signal missing sigma"
            assert hasattr(signal, 'confidence'), "Signal missing confidence"
            assert hasattr(signal, 'horizon'), "Signal missing horizon"
            assert hasattr(signal, 'timestamp'), "Signal missing timestamp"
        
        print(f"  ✅ Integration test completed:")
        print(f"     Generated {len(signals)} signals")
        print(f"     Analysis successful for {len(results['symbols_analyzed'])} symbols")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False
    
    print("  ✅ Integration tests passed")
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Technical Agent Tests\n")
    
    tests = [
        ("DataAdapter", test_data_adapter()),
        ("RegimeDetector", test_regime_detector()),
        ("TechnicalAnalyzer", test_technical_analyzer()),
        ("TechnicalAgent", test_technical_agent()),
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
        print("🎉 All Technical Agent tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
