"""
Test script for the newly implemented agents:
- Sentiment Agent
- Flow Agent  
- Macro Agent

Shows the complete multi-agent system in action
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.sentiment.agent import SentimentAgent
from agents.flow.agent import FlowAgent
from agents.macro.agent import MacroAgent
from common.scoring.unified_score import UnifiedScorer


async def test_sentiment_agent():
    """Test the Sentiment Agent"""
    print("🎯 TESTING SENTIMENT AGENT")
    print("=" * 50)
    
    try:
        agent = SentimentAgent()
        
        # Test sentiment analysis
        result = await agent.stream(
            tickers=["AAPL", "TSLA"],
            window="1h",
            sources=["twitter", "reddit", "news"],
            min_confidence=0.6
        )
        
        print("✅ Sentiment Analysis Results:")
        for sentiment_data in result["sentiment_data"]:
            print(f"  📊 {sentiment_data['ticker']}")
            print(f"     Sentiment: {sentiment_data['sentiment_score']:.3f}")
            print(f"     Confidence: {sentiment_data['confidence']:.3f}")
            print(f"     Volume: {sentiment_data['volume']}")
            print(f"     Bot Ratio: {sentiment_data['bot_ratio']:.1%}")
            print(f"     Sentiment Label: {sentiment_data['sentiment_label']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment Agent Error: {e}")
        traceback.print_exc()
        return False


async def test_flow_agent():
    """Test the Flow Agent"""
    print("🎯 TESTING FLOW AGENT")
    print("=" * 50)
    
    try:
        agent = FlowAgent()
        
        # Test flow analysis
        result = await agent.analyze_flow(
            tickers=["AAPL", "EURUSD"],
            timeframes=["1h", "4h"],
            include_regime=True,
            include_microstructure=True
        )
        
        print("✅ Flow Analysis Results:")
        for flow_data in result["flow_analyses"]:
            print(f"  📈 {flow_data['ticker']}")
            print(f"     Overall Direction: {flow_data['overall_direction']}")
            print(f"     Flow Strength: {flow_data['flow_strength']:.3f}")
            print(f"     Confidence: {flow_data['confidence']:.3f}")
            print(f"     Current Regime: {flow_data['current_regime']['regime_type']}")
            print(f"     Net Flow: {flow_data['net_flow']:.3f}")
            print(f"     Flow Persistence: {flow_data['flow_persistence']:.3f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Flow Agent Error: {e}")
        traceback.print_exc()
        return False


async def test_macro_agent():
    """Test the Macro Agent"""
    print("🎯 TESTING MACRO AGENT")
    print("=" * 50)
    
    try:
        agent = MacroAgent()
        
        # Test macro analysis
        result = await agent.analyze_macro_environment(
            horizon="medium_term",
            regions=["global"],
            include_geopolitical=True,
            include_central_banks=True
        )
        
        macro_analysis = result["macro_analysis"]
        
        print("✅ Macro Analysis Results:")
        print(f"  🌍 Analysis Horizon: {macro_analysis['analysis_horizon']}")
        print(f"  📊 Global Growth Outlook: {macro_analysis['global_growth_outlook']}")
        print(f"  💰 Inflation Environment: {macro_analysis['inflation_environment']}")
        print(f"  🏦 Interest Rate Cycle: {macro_analysis['interest_rate_cycle']}")
        print(f"  ⚠️  Risk Environment: {macro_analysis['risk_environment']}")
        print(f"  📈 Market Regime: {macro_analysis['market_regime']}")
        print(f"  💵 USD Strength Outlook: {macro_analysis['usd_strength_outlook']}")
        print(f"  🛡️  Safe Haven Demand: {macro_analysis['safe_haven_demand']:.1%}")
        print(f"  🎯 Analysis Confidence: {macro_analysis['analysis_confidence']:.1%}")
        
        print(f"\n  📊 Recent Indicators: {len(macro_analysis['recent_indicators'])}")
        print(f"  📅 Upcoming Indicators: {len(macro_analysis['upcoming_indicators'])}")
        print(f"  🌐 Active Events: {len(macro_analysis['active_events'])}")
        print(f"  ⚡ Emerging Risks: {len(macro_analysis['emerging_risks'])}")
        
        print(f"\n  🔍 Key Risks:")
        for risk in macro_analysis['key_risks'][:3]:
            print(f"     • {risk}")
        
        print(f"\n  🚀 Key Opportunities:")
        for opp in macro_analysis['key_opportunities'][:3]:
            print(f"     • {opp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Macro Agent Error: {e}")
        traceback.print_exc()
        return False


async def test_multi_agent_integration():
    """Test multi-agent integration with unified scoring"""
    print("\n🎯 TESTING MULTI-AGENT INTEGRATION")
    print("=" * 50)
    
    try:
        # Initialize agents
        sentiment_agent = SentimentAgent()
        flow_agent = FlowAgent()
        macro_agent = MacroAgent()
        unified_scorer = UnifiedScorer()
        
        print("🔄 Running multi-agent analysis for AAPL...")
        
        # Run agents in parallel
        sentiment_task = sentiment_agent.stream(
            tickers=["AAPL"], window="1h", min_confidence=0.5
        )
        flow_task = flow_agent.analyze_flow(
            tickers=["AAPL"], timeframes=["1h"], include_regime=True
        )
        macro_task = macro_agent.analyze_macro_environment(
            horizon="short_term", include_geopolitical=True
        )
        
        # Wait for all agents
        sentiment_result, flow_result, macro_result = await asyncio.gather(
            sentiment_task, flow_task, macro_task
        )
        
        print("✅ All agents completed successfully!")
        
        # Create integrated opportunity
        opportunity = {
            "id": "aapl_multi_agent_demo",
            "symbol": "AAPL",
            "strategy": "multi_agent_integration",
            "timestamp": datetime.now().isoformat(),
            "raw_signals": {
                "likelihood": 0.72,
                "expected_return": 0.035,
                "risk": 0.025,
                "liquidity": 0.95,
                "conviction": 0.68,
                "recency": 1.0,
                "regime_fit": 0.75
            },
            "metadata": {
                "asset_class": "equities",
                "agent_consensus": "bullish"
            },
            "agent_inputs": {
                "sentiment": {
                    "score": sentiment_result["sentiment_data"][0]["sentiment_score"],
                    "confidence": sentiment_result["sentiment_data"][0]["confidence"],
                    "volume": sentiment_result["sentiment_data"][0]["volume"]
                },
                "flow": {
                    "direction": flow_result["flow_analyses"][0]["overall_direction"],
                    "strength": flow_result["flow_analyses"][0]["flow_strength"],
                    "regime": flow_result["flow_analyses"][0]["current_regime"]["regime_type"]
                },
                "macro": {
                    "growth_outlook": macro_result["macro_analysis"]["global_growth_outlook"],
                    "market_regime": macro_result["macro_analysis"]["market_regime"],
                    "risk_environment": macro_result["macro_analysis"]["risk_environment"]
                }
            }
        }
        
        # Score the integrated opportunity
        scored_opportunities = unified_scorer.score_opportunities([opportunity])
        
        print("\n📊 INTEGRATED ANALYSIS RESULT:")
        scored_opp = scored_opportunities[0]
        print(f"  🎯 Symbol: {scored_opp['symbol']}")
        print(f"  ⭐ Unified Score: {scored_opp['unified_score']:.3f}")
        print(f"  📈 Calibrated Probability: {scored_opp['calibrated_probability']:.1%}")
        print(f"  🎖️  Percentile Rank: {scored_opp['percentile_rank']:.1f}%")
        print(f"  📊 Confidence Interval: [{scored_opp['confidence_interval'][0]:.3f}, {scored_opp['confidence_interval'][1]:.3f}]")
        
        print(f"\n🤖 AGENT CONTRIBUTIONS:")
        agent_inputs = opportunity["agent_inputs"]
        print(f"  😊 Sentiment: {agent_inputs['sentiment']['score']:.3f} (confidence: {agent_inputs['sentiment']['confidence']:.1%})")
        print(f"  🌊 Flow: {agent_inputs['flow']['direction']} (strength: {agent_inputs['flow']['strength']:.3f})")
        print(f"  🌍 Macro: {agent_inputs['macro']['growth_outlook']} growth, {agent_inputs['macro']['market_regime']} regime")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-Agent Integration Error: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run all agent tests"""
    print("🚀 TRADING INTELLIGENCE SYSTEM - NEW AGENTS TEST")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test individual agents
    results = {}
    
    results["sentiment"] = await test_sentiment_agent()
    print()
    
    results["flow"] = await test_flow_agent()
    print()
    
    results["macro"] = await test_macro_agent()
    print()
    
    # Test integration
    results["integration"] = await test_multi_agent_integration()
    print()
    
    # Summary
    print("🎉 TEST SUMMARY")
    print("=" * 30)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.title()} Agent: {status}")
    
    print(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎊 ALL TESTS PASSED! The multi-agent system is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print(f"\n🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
