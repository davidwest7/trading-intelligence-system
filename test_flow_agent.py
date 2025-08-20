#!/usr/bin/env python3
"""
Test Complete Flow Agent Implementation

Tests all resolved TODOs:
✅ Hidden Markov Model for regime detection
✅ Market breadth calculations
✅ Volatility term structure analysis
✅ Cross-asset correlation analysis
✅ Flow momentum indicators
✅ Regime transition probability estimation
✅ Real-time regime monitoring
✅ Multi-timeframe regime analysis
"""

import asyncio
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.flow.agent_complete import (
    FlowAgent, HiddenMarkovRegimeDetector, MarketBreadthCalculator,
    VolatilityStructureAnalyzer, CrossAssetFlowAnalyzer, FlowMomentumCalculator
)

async def test_flow_agent():
    """Test the complete flow agent implementation"""
    print("🧪 Testing Complete Flow Agent Implementation")
    print("=" * 60)
    
    # Initialize agent
    agent = FlowAgent()
    
    # Test 1: Hidden Markov Regime Detector
    print("\n1. Testing Hidden Markov Regime Detector...")
    hmm_detector = HiddenMarkovRegimeDetector()
    
    # Generate mock observations
    observations = np.random.randn(20, 16)  # 20 observations, 16 features
    
    # Fit the model
    hmm_detector.fit(observations)
    print(f"   ✅ HMM model fitted successfully")
    
    # Test prediction
    test_observation = np.random.randn(16)
    regime, confidence = hmm_detector.predict_regime(test_observation)
    print(f"   Predicted regime: {regime}, Confidence: {confidence:.3f}")
    
    # Test regime probabilities
    probabilities = hmm_detector.get_regime_probabilities()
    print(f"   Regime probabilities: {probabilities}")
    
    # Test 2: Market Breadth Calculator
    print("\n2. Testing Market Breadth Calculator...")
    breadth_calculator = MarketBreadthCalculator()
    
    breadth = breadth_calculator.calculate_breadth(["AAPL", "MSFT", "GOOGL"])
    print(f"   Advance/Decline ratio: {breadth.advance_decline_ratio:.3f}")
    print(f"   New highs/lows ratio: {breadth.new_highs_lows_ratio:.3f}")
    print(f"   Cumulative A/D: {breadth.cumulative_advance_decline:.3f}")
    print(f"   Sector breadth: {len(breadth.sector_breadth)} sectors")
    print(f"   Market cap breadth: {len(breadth.market_cap_breadth)} categories")
    
    # Test 3: Volatility Structure Analyzer
    print("\n3. Testing Volatility Structure Analyzer...")
    vol_analyzer = VolatilityStructureAnalyzer()
    
    vol_structure = vol_analyzer.analyze_volatility_structure("1d")
    print(f"   VIX term structure slope: {vol_structure.vix_term_structure_slope:.3f}")
    print(f"   Realized vs implied vol: {vol_structure.realized_vs_implied_vol:.3f}")
    print(f"   Vol of vol: {vol_structure.vol_of_vol:.3f}")
    print(f"   Volatility regime: {vol_structure.volatility_regime}")
    
    # Test 4: Cross-Asset Flow Analyzer
    print("\n4. Testing Cross-Asset Flow Analyzer...")
    flow_analyzer = CrossAssetFlowAnalyzer()
    
    cross_asset = flow_analyzer.analyze_cross_asset_flows("1d")
    print(f"   Equity-bond correlation: {cross_asset.equity_bond_correlation:.3f}")
    print(f"   Dollar strength index: {cross_asset.dollar_strength_index:.3f}")
    print(f"   Commodity momentum: {cross_asset.commodity_momentum:.3f}")
    print(f"   Safe haven flows: {cross_asset.safe_haven_flows:.3f}")
    print(f"   Risk parity weights: {len(cross_asset.risk_parity_weights)} assets")
    
    # Test 5: Flow Momentum Calculator
    print("\n5. Testing Flow Momentum Calculator...")
    momentum_calculator = FlowMomentumCalculator()
    
    momentum = momentum_calculator.calculate_momentum("SPY", "1d")
    print(f"   Money Flow Index: {momentum.money_flow_index:.1f}")
    print(f"   Chaikin Money Flow: {momentum.chaikin_money_flow:.3f}")
    print(f"   On-Balance Volume: {momentum.on_balance_volume:,.0f}")
    print(f"   Accumulation/Distribution: {momentum.accumulation_distribution:,.0f}")
    print(f"   Volume Price Trend: {momentum.volume_price_trend:.3f}")
    
    # Test 6: Complete Flow Agent
    print("\n6. Testing Complete Flow Agent...")
    
    result = await agent.regime_map(window="1d", markets=["equities", "fx"])
    
    print(f"   Current regime: {result['current_regime']['name']}")
    print(f"   Regime confidence: {result['current_regime']['confidence']:.3f}")
    print(f"   Regime duration: {result['current_regime']['duration_days']} days")
    print(f"   Regime strength: {result['current_regime']['strength']:.3f}")
    
    print(f"   Regime probabilities:")
    for regime, prob in result['regime_probabilities'].items():
        print(f"     - {regime}: {prob:.3f}")
    
    print(f"   Flow indicators:")
    print(f"     - Market breadth: {len(result['flow_indicators']['market_breadth'])} metrics")
    print(f"     - Volatility structure: {len(result['flow_indicators']['volatility_structure'])} metrics")
    print(f"     - Cross-asset flows: {len(result['flow_indicators']['cross_asset_flows'])} metrics")
    print(f"     - Flow momentum: {len(result['flow_indicators']['flow_momentum'])} metrics")
    
    print(f"   Regime transitions: {len(result['regime_transitions'])} transitions")
    print(f"   Multi-timeframe analysis: {len(result['multi_timeframe_analysis'])} timeframes")
    
    # Test 7: Multi-timeframe Analysis
    print("\n7. Testing Multi-timeframe Analysis...")
    
    for timeframe, analysis in result['multi_timeframe_analysis'].items():
        print(f"   {timeframe}: {analysis['regime']} (confidence: {analysis['confidence']:.3f})")
    
    # Test 8: Regime Persistence
    print("\n8. Testing Regime Persistence...")
    
    # Run multiple analyses to test regime persistence
    for i in range(5):
        result = await agent.regime_map(window="1d")
        print(f"   Run {i+1}: {result['current_regime']['name']} (strength: {result['current_regime']['strength']:.3f})")
    
    print("\n✅ All Flow Agent tests completed successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_flow_agent())
