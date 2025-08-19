#!/usr/bin/env python3
"""
Trading Intelligence System - Demo Script

Comprehensive demonstration of the multi-agent trading intelligence system.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.technical.agent import TechnicalAgent
from common.scoring.unified_score import UnifiedScorer
from common.event_bus.bus import EventBus, EventType
from common.feature_store.store import FeatureStore
import pandas as pd
import numpy as np


async def demo_technical_analysis():
    """Demo technical analysis capabilities"""
    print("=" * 60)
    print("🔧 TECHNICAL ANALYSIS DEMO")
    print("=" * 60)
    
    agent = TechnicalAgent()
    
    # Analysis request
    payload = {
        "symbols": ["AAPL", "TSLA", "EURUSD"],
        "timeframes": ["1h", "4h"],
        "strategies": ["imbalance"],
        "min_score": 0.01,  # Low threshold for demo
        "max_risk": 0.05
    }
    
    print(f"📈 Analyzing {len(payload['symbols'])} symbols: {', '.join(payload['symbols'])}")
    print(f"⏱️  Timeframes: {', '.join(payload['timeframes'])}")
    print(f"🎯 Strategies: {', '.join(payload['strategies'])}")
    
    # Run analysis
    result = await agent.find_opportunities(payload)
    
    print(f"\n📊 RESULTS:")
    print(f"   Analysis time: {result['metadata']['analysis_time_ms']} ms")
    print(f"   Opportunities found: {len(result['opportunities'])}")
    print(f"   Market regime: {result['metadata']['market_regime']}")
    print(f"   Overall bias: {result['metadata']['overall_bias']}")
    
    # Show opportunities
    if result['opportunities']:
        print(f"\n🎯 OPPORTUNITIES:")
        for i, opp in enumerate(result['opportunities'][:3], 1):
            print(f"   {i}. {opp['symbol']} ({opp['strategy']})")
            print(f"      Entry: ${opp['entry_price']:.4f}")
            print(f"      Stop Loss: ${opp['stop_loss']:.4f}")
            print(f"      Take Profit: ${opp['take_profit'][0]:.4f}")
            print(f"      R/R Ratio: {opp['risk_reward_ratio']:.2f}")
            print(f"      Confidence: {opp['confidence_score']:.1%}")
            print()
    else:
        print("   No opportunities found with current criteria")
    
    return result


async def demo_unified_scoring():
    """Demo unified scoring system"""
    print("=" * 60)
    print("🏆 UNIFIED SCORING DEMO")
    print("=" * 60)
    
    scorer = UnifiedScorer()
    
    # Sample opportunities for scoring
    opportunities = [
        {
            'id': 'opp_1',
            'symbol': 'AAPL',
            'strategy': 'technical',
            'raw_signals': {
                'likelihood': 0.8,
                'expected_return': 0.05,
                'risk': 0.02,
                'liquidity': 0.95,
                'conviction': 0.7,
                'recency': 0.9,
                'regime_fit': 0.6
            },
            'metadata': {'asset_class': 'equities'}
        },
        {
            'id': 'opp_2',
            'symbol': 'EURUSD',
            'strategy': 'sentiment',
            'raw_signals': {
                'likelihood': 0.6,
                'expected_return': 0.02,
                'risk': 0.015,
                'liquidity': 1.0,
                'conviction': 0.5,
                'recency': 1.0,
                'regime_fit': 0.8
            },
            'metadata': {'asset_class': 'fx'}
        },
        {
            'id': 'opp_3',
            'symbol': 'BTC-USD',
            'strategy': 'flow',
            'raw_signals': {
                'likelihood': 0.7,
                'expected_return': 0.08,
                'risk': 0.05,
                'liquidity': 0.8,
                'conviction': 0.8,
                'recency': 0.9,
                'regime_fit': 0.4
            },
            'metadata': {'asset_class': 'crypto'}
        }
    ]
    
    print(f"📊 Scoring {len(opportunities)} opportunities")
    print("💡 Using asset class-specific weights:")
    print("   - Equities: Higher emphasis on likelihood & returns")
    print("   - FX: Higher risk weighting due to leverage")
    print("   - Crypto: Higher return potential, higher risk")
    
    # Score opportunities
    scored_opportunities = scorer.score_opportunities(opportunities)
    
    print(f"\n🏆 SCORED RESULTS:")
    for opp in scored_opportunities:
        print(f"   Rank {opp.rank}: {opp.symbol} ({opp.strategy})")
        print(f"      Unified Score: {opp.unified_score:.3f}")
        print(f"      Calibrated Prob: {opp.calibrated_probability:.3f}")
        print(f"      Percentile: {opp.percentile_rank:.1%}")
        print(f"      Confidence Interval: [{opp.confidence_interval[0]:.3f}, {opp.confidence_interval[1]:.3f}]")
        print()
    
    return scored_opportunities


async def demo_event_bus():
    """Demo event bus capabilities"""
    print("=" * 60)
    print("📡 EVENT BUS DEMO")
    print("=" * 60)
    
    bus = EventBus(persist_events=True)
    
    # Event storage
    events_received = []
    
    # Event handlers
    async def market_handler(event):
        events_received.append(f"Market Tick: {event.data['symbol']} @ ${event.data['price']}")
        
    async def signal_handler(event):
        events_received.append(f"Agent Signal: {event.data['agent_name']} - {event.data['signal_type']} (confidence: {event.data['confidence']:.1%})")
    
    # Subscribe to events
    bus.subscribe(EventType.MARKET_TICK, market_handler)
    bus.subscribe(EventType.AGENT_SIGNAL, signal_handler)
    
    print("📡 Starting event bus...")
    await bus.start()
    
    # Publish sample events
    print("📤 Publishing sample events...")
    await bus.publish_market_tick('ibkr', 'AAPL', 150.25, 1500)
    await bus.publish_market_tick('ibkr', 'TSLA', 245.80, 2300)
    await bus.publish_agent_signal('technical', 'technical', 'buy_signal', 0.85)
    await bus.publish_agent_signal('sentiment', 'sentiment', 'bullish_sentiment', 0.72)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    print(f"\n📨 EVENTS PROCESSED ({len(events_received)}):")
    for event in events_received:
        print(f"   • {event}")
    
    # Show event history
    history = bus.get_event_history()
    print(f"\n📚 Event history contains {len(history)} events")
    
    await bus.stop()
    print("🛑 Event bus stopped")
    
    return events_received


async def demo_feature_store():
    """Demo feature store capabilities"""
    print("=" * 60)
    print("💾 FEATURE STORE DEMO")
    print("=" * 60)
    
    # Initialize feature store
    store = FeatureStore(db_path='data/demo_features.db', parquet_path='data/demo_parquet/')
    
    # Generate sample features
    print("📊 Generating sample technical indicators...")
    
    symbols = ['AAPL', 'TSLA', 'MSFT']
    sample_data_list = []
    
    for symbol in symbols:
        # Generate 48 hours of hourly data
        dates = pd.date_range(end=datetime.now(), periods=48, freq='1h')
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'rsi': np.random.uniform(30, 70, 48),
            'sma_20': np.random.uniform(140, 160, 48),
            'bb_upper': np.random.uniform(160, 170, 48),
            'bb_lower': np.random.uniform(130, 140, 48),
            'volume_ma': np.random.uniform(1000000, 5000000, 48),
            'momentum': np.random.uniform(-0.05, 0.05, 48)
        })
        
        sample_data_list.append(sample_data)
    
    # Combine all data
    all_features = pd.concat(sample_data_list, ignore_index=True)
    
    print(f"📥 Writing {len(all_features)} feature records to store...")
    
    # Write to feature store
    success = await store.write_features(
        'technical_indicators', 
        all_features,
        metadata={'description': 'Demo technical indicators', 'version': '1.0'}
    )
    
    if success:
        print("✅ Features written successfully")
        
        # Read back some features
        print("📤 Reading features for AAPL...")
        aapl_features = await store.read_features(
            'technical_indicators',
            features=['rsi', 'sma_20', 'momentum'],
            symbols=['AAPL']
        )
        
        print(f"📊 Retrieved {len(aapl_features)} AAPL feature records")
        if len(aapl_features) > 0:
            print("Sample features:")
            print(aapl_features[['timestamp', 'symbol', 'rsi', 'sma_20', 'momentum']].head(3).to_string(index=False))
    else:
        print("❌ Failed to write features")
    
    store.close()
    return all_features


async def demo_full_pipeline():
    """Demo the complete trading intelligence pipeline"""
    print("=" * 60)
    print("🚀 FULL PIPELINE DEMO")
    print("=" * 60)
    
    print("🔄 Running complete trading intelligence pipeline...")
    
    # Step 1: Technical Analysis
    print("\n1️⃣ Technical Analysis...")
    tech_result = await demo_technical_analysis()
    
    # Step 2: Create opportunities for scoring if none found
    opportunities = tech_result['opportunities']
    if not opportunities:
        print("   Creating mock opportunities for demo...")
        opportunities = [
            {
                'id': 'mock_1',
                'symbol': 'AAPL',
                'strategy': 'imbalance',
                'confidence_score': 0.75,
                'entry_price': 150.0,
                'stop_loss': 147.0,
                'take_profit': [156.0],
                'raw_signals': {
                    'likelihood': 0.75,
                    'expected_return': 0.04,
                    'risk': 0.02,
                    'liquidity': 0.95,
                    'conviction': 0.8,
                    'recency': 1.0,
                    'regime_fit': 0.7
                },
                'metadata': {'asset_class': 'equities'}
            }
        ]
    
    # Step 3: Unified Scoring
    print("\n2️⃣ Unified Scoring...")
    scorer = UnifiedScorer()
    scored_opportunities = scorer.score_opportunities(opportunities)
    
    # Step 4: Event Publishing
    print("\n3️⃣ Event Publishing...")
    bus = EventBus(persist_events=True)
    await bus.start()
    
    await bus.publish_agent_signal(
        'pipeline', 'pipeline', 'analysis_complete', 1.0,
        additional_data={
            'opportunities_found': len(scored_opportunities),
            'top_score': scored_opportunities[0].unified_score if scored_opportunities else 0
        }
    )
    
    await asyncio.sleep(0.2)
    await bus.stop()
    
    print(f"\n🎯 PIPELINE SUMMARY:")
    print(f"   Total opportunities: {len(scored_opportunities)}")
    if scored_opportunities:
        top_opp = scored_opportunities[0]
        print(f"   Top opportunity: {top_opp.symbol} (score: {top_opp.unified_score:.3f})")
        print(f"   Calibrated probability: {top_opp.calibrated_probability:.1%}")
    
    print(f"   Analysis complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'technical_results': tech_result,
        'scored_opportunities': scored_opportunities
    }


async def main():
    """Main demo runner"""
    print("🚀 TRADING INTELLIGENCE SYSTEM DEMO")
    print("🤖 Multi-Agent Trading Intelligence for Research-Grade Analysis")
    print("⚡ Built with BMAD Architecture + Unified Scoring")
    print()
    
    try:
        # Run individual component demos
        await demo_technical_analysis()
        await demo_unified_scoring()
        await demo_event_bus()
        await demo_feature_store()
        
        # Run full pipeline
        await demo_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("🔗 Next Steps:")
        print("   • Run the API server: python main.py")
        print("   • Access docs: http://localhost:8000/docs")
        print("   • Start Docker services: make docker-up")
        print("   • Complete other agents (see TODO lists)")
        print()
        print("🏗️ System Architecture:")
        print("   • Technical Agent: ✅ Fully implemented")
        print("   • Unified Scorer: ✅ Fully implemented") 
        print("   • Event Bus: ✅ Fully implemented")
        print("   • Feature Store: ✅ Core functionality working")
        print("   • Other 9 agents: 🔧 Stubs with detailed TODOs")
        print()
        print("📊 System Capabilities:")
        print("   • Multi-timeframe technical analysis")
        print("   • Imbalance/FVG detection")
        print("   • Purged cross-validation backtesting")
        print("   • Regime-aware unified scoring")
        print("   • Real-time event processing")
        print("   • Point-in-time feature storage")
        print()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🎬 Starting Trading Intelligence System Demo...")
    asyncio.run(main())
