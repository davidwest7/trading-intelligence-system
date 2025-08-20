"""
Comprehensive Integration Test for Medium Priority Agents

Tests both agents together:
- Technical Agent
- Top Performers Agent
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.technical.agent_complete import TechnicalAgent
from agents.top_performers.agent_complete import TopPerformersAgent
from schemas.contracts import HorizonType


async def test_technical_agent_integration():
    """Test Technical Agent integration"""
    print("🧪 Testing Technical Agent Integration...")
    
    agent = TechnicalAgent()
    
    try:
        # Test symbols across different asset classes
        symbols = ["AAPL", "EURUSD", "BTC-USD"]
        timeframes = ["15m", "1h", "4h"]
        
        # Full analysis
        results = await agent.analyze(symbols, timeframes)
        
        # Validate results
        assert len(results['symbols_analyzed']) == 3, "Should analyze 3 symbols"
        assert len(results['timeframes']) == 3, "Should use 3 timeframes"
        assert len(results['analysis']) == 3, "Should have analysis for 3 symbols"
        
        # Check each symbol analysis
        for symbol in symbols:
            assert symbol in results['analysis'], f"Missing analysis for {symbol}"
            symbol_analysis = results['analysis'][symbol]
            
            # Check timeframes
            for tf in timeframes:
                if tf in symbol_analysis['timeframes']:
                    tf_analysis = symbol_analysis['timeframes'][tf]
                    if 'indicators' in tf_analysis:
                        assert 'rsi' in tf_analysis['indicators'], f"Missing RSI for {symbol} {tf}"
                    if 'patterns' in tf_analysis:
                        assert 'support_resistance' in tf_analysis['patterns'], f"Missing patterns for {symbol} {tf}"
        
        # Check signals
        signals = results['signals']
        assert len(signals) > 0, "Should generate signals"
        
        # Validate signal structure
        for signal in signals:
            assert hasattr(signal, 'symbol'), "Signal missing symbol"
            assert hasattr(signal, 'agent_type'), "Signal missing agent_type"
            assert hasattr(signal, 'mu'), "Signal missing mu"
            assert hasattr(signal, 'sigma'), "Signal missing sigma"
            assert hasattr(signal, 'confidence'), "Signal missing confidence"
            assert hasattr(signal, 'horizon'), "Signal missing horizon"
            assert hasattr(signal, 'timestamp'), "Signal missing timestamp"
        
        print(f"  ✅ Technical analysis completed:")
        print(f"     Symbols analyzed: {len(results['symbols_analyzed'])}")
        print(f"     Timeframes used: {len(results['timeframes'])}")
        print(f"     Signals generated: {len(signals)}")
        print(f"     Overall regime: {results['regime_analysis']['regime']}")
        print(f"     Analysis time: {results['performance']['analysis_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Technical Agent integration test failed: {e}")
        return False


async def test_top_performers_agent_integration():
    """Test Top Performers Agent integration"""
    print("🧪 Testing Top Performers Agent Integration...")
    
    agent = TopPerformersAgent()
    
    try:
        # Test different ranking scenarios
        test_cases = [
            {
                'horizon': '1w',
                'asset_classes': ['equities', 'fx'],
                'regions': ['US', 'EU'],
                'min_volume': 1000000,
                'limit': 15
            },
            {
                'horizon': '1m',
                'asset_classes': ['equities', 'crypto'],
                'regions': ['US'],
                'min_volume': 500000,
                'limit': 20
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            results = await agent.rank(**test_case)
            
            # Validate results
            assert 'rankings' in results, "Missing rankings"
            assert 'sector_analysis' in results, "Missing sector analysis"
            assert 'benchmark_performance' in results, "Missing benchmark performance"
            assert 'market_regime' in results, "Missing market regime"
            
            rankings = results['rankings']
            assert len(rankings) <= test_case['limit'], f"Should not exceed limit of {test_case['limit']}"
            
            if len(rankings) > 0:
                # Validate ranking structure
                first_ranking = rankings[0]
                assert 'rank' in first_ranking, "Missing rank"
                assert 'symbol' in first_ranking, "Missing symbol"
                assert 'score' in first_ranking, "Missing score"
                assert 'performance' in first_ranking, "Missing performance"
                assert 'momentum_indicators' in first_ranking, "Missing momentum indicators"
                
                # Check ranking order
                scores = [r['score'] for r in rankings]
                assert scores == sorted(scores, reverse=True), "Rankings should be sorted by score"
                
                print(f"  ✅ Test case {i+1} completed:")
                print(f"     Horizon: {test_case['horizon']}")
                print(f"     Rankings: {len(rankings)}")
                print(f"     Top score: {first_ranking['score']:.3f}")
                print(f"     Top symbol: {first_ranking['symbol']}")
                print(f"     Market regime: {results['market_regime']['regime']}")
        
        # Test momentum signals
        symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
        signals = await agent.get_momentum_signals(symbols, "1w")
        
        assert isinstance(signals, list), "Signals should be list"
        for signal in signals:
            assert hasattr(signal, 'symbol'), "Signal missing symbol"
            assert hasattr(signal, 'agent_type'), "Signal missing agent_type"
            assert hasattr(signal, 'mu'), "Signal missing mu"
            assert hasattr(signal, 'sigma'), "Signal missing sigma"
        
        print(f"  ✅ Generated {len(signals)} momentum signals")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Top Performers Agent integration test failed: {e}")
        return False


async def test_cross_agent_integration():
    """Test integration between Technical and Top Performers agents"""
    print("🧪 Testing Cross-Agent Integration...")
    
    technical_agent = TechnicalAgent()
    top_performers_agent = TopPerformersAgent()
    
    try:
        # Get top performers first
        rankings_result = await top_performers_agent.rank(
            horizon="1w",
            asset_classes=["equities"],
            regions=["US"],
            min_volume=1000000,
            limit=10
        )
        
        # Extract top symbols for technical analysis
        top_symbols = [r['symbol'] for r in rankings_result['rankings'][:5]]
        
        print(f"  📊 Top performers: {top_symbols}")
        
        # Perform technical analysis on top performers
        technical_results = await technical_agent.analyze(
            symbols=top_symbols,
            timeframes=["1h", "4h"],
            strategies=["patterns", "indicators", "regime"]
        )
        
        # Validate cross-agent results
        assert len(technical_results['symbols_analyzed']) == len(top_symbols), "Should analyze all top symbols"
        assert len(technical_results['signals']) > 0, "Should generate technical signals"
        
        # Check for consensus between agents
        technical_signals = technical_results['signals']
        momentum_signals = await top_performers_agent.get_momentum_signals(top_symbols, "1w")
        
        # Find symbols with both technical and momentum signals
        technical_symbols = {s.symbol for s in technical_signals}
        momentum_symbols = {s.symbol for s in momentum_signals}
        consensus_symbols = technical_symbols.intersection(momentum_symbols)
        
        print(f"  ✅ Cross-agent analysis completed:")
        print(f"     Top performers analyzed: {len(top_symbols)}")
        print(f"     Technical signals: {len(technical_signals)}")
        print(f"     Momentum signals: {len(momentum_signals)}")
        print(f"     Consensus symbols: {len(consensus_symbols)}")
        print(f"     Consensus: {list(consensus_symbols)}")
        
        # Validate signal quality
        for signal in technical_signals + momentum_signals:
            assert signal.confidence >= 0 and signal.confidence <= 1, "Confidence should be [0,1]"
            assert signal.sigma > 0, "Sigma should be positive"
            assert signal.horizon in [HorizonType.INTRADAY, HorizonType.SHORT_TERM, HorizonType.MEDIUM_TERM, HorizonType.LONG_TERM], "Invalid horizon"
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cross-agent integration test failed: {e}")
        return False


async def test_performance_metrics():
    """Test performance metrics and data quality"""
    print("🧪 Testing Performance Metrics...")
    
    technical_agent = TechnicalAgent()
    top_performers_agent = TopPerformersAgent()
    
    try:
        # Test technical agent performance
        start_time = asyncio.get_event_loop().time()
        technical_results = await technical_agent.analyze(
            symbols=["AAPL", "TSLA", "MSFT"],
            timeframes=["1h", "4h"]
        )
        technical_time = asyncio.get_event_loop().time() - start_time
        
        # Test top performers agent performance
        start_time = asyncio.get_event_loop().time()
        rankings_results = await top_performers_agent.rank(
            horizon="1w",
            asset_classes=["equities"],
            limit=20
        )
        rankings_time = asyncio.get_event_loop().time() - start_time
        
        # Validate performance
        assert technical_time < 10.0, f"Technical analysis too slow: {technical_time:.2f}s"
        assert rankings_time < 10.0, f"Rankings too slow: {rankings_time:.2f}s"
        
        # Validate data quality
        technical_signals = technical_results['signals']
        rankings = rankings_results['rankings']
        
        # Check for reasonable signal distribution
        if len(technical_signals) > 0:
            confidences = [s.confidence for s in technical_signals]
            avg_confidence = np.mean(confidences)
            assert 0.3 <= avg_confidence <= 0.9, f"Average confidence {avg_confidence:.2f} outside reasonable range"
        
        # Check for reasonable ranking distribution
        if len(rankings) > 0:
            scores = [r['score'] for r in rankings]
            avg_score = np.mean(scores)
            assert 0.3 <= avg_score <= 0.9, f"Average score {avg_score:.2f} outside reasonable range"
            
            # Check score distribution
            score_std = np.std(scores)
            assert score_std > 0.02, f"Score distribution too narrow: {score_std:.3f}"
        
        print(f"  ✅ Performance metrics validated:")
        print(f"     Technical analysis time: {technical_time:.3f}s")
        print(f"     Rankings time: {rankings_time:.3f}s")
        print(f"     Technical signals: {len(technical_signals)}")
        print(f"     Rankings: {len(rankings)}")
        
        if len(technical_signals) > 0:
            avg_confidence = np.mean([s.confidence for s in technical_signals])
            print(f"     Average signal confidence: {avg_confidence:.2f}")
        
        if len(rankings) > 0:
            avg_score = np.mean([r['score'] for r in rankings])
            print(f"     Average ranking score: {avg_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance metrics test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and edge cases"""
    print("🧪 Testing Error Handling...")
    
    technical_agent = TechnicalAgent()
    top_performers_agent = TopPerformersAgent()
    
    try:
        # Test with empty symbols
        empty_results = await technical_agent.analyze(symbols=[], timeframes=["1h"])
        assert len(empty_results['signals']) == 0, "Should handle empty symbols"
        
        # Test with invalid symbols
        invalid_results = await technical_agent.analyze(symbols=["INVALID_SYMBOL"], timeframes=["1h"])
        assert len(invalid_results['analysis']) == 1, "Should handle invalid symbols gracefully"
        
        # Test with empty asset classes
        empty_rankings = await top_performers_agent.rank(
            horizon="1w",
            asset_classes=[],
            limit=10
        )
        assert len(empty_rankings['rankings']) == 0, "Should handle empty asset classes"
        
        # Test with very high volume filter
        high_volume_rankings = await top_performers_agent.rank(
            horizon="1w",
            asset_classes=["equities"],
            min_volume=1000000000,  # $1B minimum
            limit=10
        )
        assert len(high_volume_rankings['rankings']) <= 10, "Should handle high volume filter"
        
        print("  ✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all medium priority agent tests"""
    print("🚀 Starting Medium Priority Agents Integration Tests\n")
    
    tests = [
        ("Technical Agent Integration", test_technical_agent_integration()),
        ("Top Performers Agent Integration", test_top_performers_agent_integration()),
        ("Cross-Agent Integration", test_cross_agent_integration()),
        ("Performance Metrics", test_performance_metrics()),
        ("Error Handling", test_error_handling())
    ]
    
    results = {}
    
    for test_name, test_coro in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name}")
        print(f"{'='*60}")
        
        try:
            result = await test_coro
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("MEDIUM PRIORITY AGENTS TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Medium Priority Agents tests passed!")
        print("\n📋 SUMMARY:")
        print("✅ Technical Agent: Data adapter integration, regime detection, pattern recognition")
        print("✅ Top Performers Agent: Ranking system, momentum analysis, sector analysis")
        print("✅ Cross-Agent Integration: Technical analysis on top performers")
        print("✅ Performance Metrics: Speed and data quality validation")
        print("✅ Error Handling: Edge cases and invalid inputs")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
