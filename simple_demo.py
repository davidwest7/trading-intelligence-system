#!/usr/bin/env python3
"""
Simple Multi-Agent Demo - Focus on working agents
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.technical.agent import TechnicalAgent
from agents.moneyflows.agent import MoneyFlowsAgent
from agents.undervalued.agent import UndervaluedAgent
from agents.insider.agent import InsiderAgent
from agents.causal.agent import CausalAgent
from agents.hedging.agent import HedgingAgent
from agents.learning.agent import LearningAgent
from common.scoring.unified_score import UnifiedScorer

async def main():
    print("�� MULTI-AGENT TRADING SYSTEM DEMO")
    print("=" * 50)
    print(f"🕒 Started at: {datetime.now()}")
    
    demo_tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']
    
    # Technical Analysis
    print("\n📈 TECHNICAL ANALYSIS")
    print("-" * 30)
    tech_agent = TechnicalAgent()
    tech_result = await tech_agent.analyze(
        symbols=demo_tickers[:2],
        timeframes=['1h'],
        strategies=['imbalance']
    )
    print(f"✅ Found {len(tech_result.get('opportunities', []))} technical opportunities")
    
    # Money Flows Analysis
    print("\n💰 MONEY FLOWS ANALYSIS")
    print("-" * 30)
    flows_agent = MoneyFlowsAgent()
    flows_result = await flows_agent.analyze_money_flows(
        tickers=demo_tickers[:2]
    )
    print(f"✅ Analyzed money flows for {len(flows_result.get('money_flow_analyses', []))} assets")
    
    # Undervalued Analysis
    print("\n💎 VALUE ANALYSIS")
    print("-" * 30)
    value_agent = UndervaluedAgent()
    value_result = await value_agent.analyze_undervalued_stocks(
        universe=demo_tickers[:3]
    )
    opportunities = value_result.get('undervalued_analysis', {}).get('identified_opportunities', [])
    print(f"✅ Found {len(opportunities)} value opportunities")
    
    # Insider Analysis
    print("\n👥 INSIDER ANALYSIS")
    print("-" * 30)
    insider_agent = InsiderAgent()
    insider_result = await insider_agent.analyze_insider_activity(
        tickers=demo_tickers[:2]
    )
    print(f"✅ Analyzed insider activity for {len(insider_result.get('insider_analyses', []))} assets")
    
    # Causal Analysis
    print("\n🔍 CAUSAL IMPACT ANALYSIS")
    print("-" * 30)
    causal_agent = CausalAgent()
    causal_result = await causal_agent.analyze_causal_impact(
        tickers=demo_tickers[:2]
    )
    print(f"✅ Analyzed causal impact for {len(causal_result.get('causal_analyses', []))} assets")
    
    # Hedging Analysis
    print("\n🛡️ HEDGING ANALYSIS")
    print("-" * 30)
    hedge_agent = HedgingAgent()
    hedge_result = await hedge_agent.analyze_hedging_strategies(
        portfolio_holdings={'AAPL': 0.4, 'TSLA': 0.3, 'GOOGL': 0.3}
    )
    hedges = hedge_result.get('hedging_analysis', {}).get('recommended_hedges', [])
    print(f"✅ Generated {len(hedges)} hedge recommendations")
    
    # Learning Analysis
    print("\n🧠 LEARNING SYSTEM ANALYSIS")
    print("-" * 30)
    learning_agent = LearningAgent()
    learning_result = await learning_agent.analyze_learning_system()
    models = learning_result.get('learning_analysis', {}).get('active_models', [])
    print(f"✅ Analyzed {len(models)} active learning models")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    print("✅ Technical Agent: Fully operational")
    print("✅ Money Flows Agent: Fully operational") 
    print("✅ Undervalued Agent: Fully operational")
    print("✅ Insider Agent: Fully operational")
    print("✅ Causal Agent: Fully operational")
    print("✅ Hedging Agent: Fully operational")
    print("✅ Learning Agent: Fully operational")
    print(f"\n🎯 Total: 7 agents successfully demonstrated!")
    print(f"🕒 Completed at: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
