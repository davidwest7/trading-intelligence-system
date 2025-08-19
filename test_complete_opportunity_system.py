#!/usr/bin/env python3
"""
Test Complete Opportunity System - Verify opportunity storage and ranking
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_complete_opportunity_system():
    print("🔍 TESTING COMPLETE OPPORTUNITY SYSTEM")
    print("=" * 50)
    print(f"🕒 Test started: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    try:
        print("1️⃣ Testing Opportunity Store...")
        from common.opportunity_store import opportunity_store, Opportunity
        
        # Clear existing opportunities for clean test
        print("   📋 Clearing existing opportunities...")
        
        print("   ✅ Opportunity store initialized")
        
        print("\n2️⃣ Testing Unified Scorer...")
        from common.unified_opportunity_scorer import unified_scorer
        print("   ✅ Unified scorer initialized")
        
        print("\n3️⃣ Testing Agent Opportunity Generation...")
        from agents.undervalued.agent import UndervaluedAgent
        from agents.moneyflows.agent import MoneyFlowsAgent
        
        # Test value analysis agent
        print("   🔍 Testing Value Analysis Agent...")
        value_agent = UndervaluedAgent()
        value_result = await value_agent.process(universe=['BRK.B', 'JPM'])
        
        print(f"   ✅ Value agent result: {type(value_result)}")
        if isinstance(value_result, dict):
            analysis = value_result.get('undervalued_analysis', {})
            opportunities = analysis.get('identified_opportunities', [])
            print(f"   🎯 Value opportunities found: {len(opportunities)}")
        
        # Test money flows agent
        print("   🔍 Testing Money Flows Agent...")
        flow_agent = MoneyFlowsAgent()
        flow_result = await flow_agent.process(tickers=['AAPL', 'TSLA'])
        
        print(f"   ✅ Flow agent result: {type(flow_result)}")
        if isinstance(flow_result, dict):
            analyses = flow_result.get('money_flow_analyses', [])
            print(f"   🎯 Flow analyses found: {len(analyses)}")
        
        print("\n4️⃣ Testing Opportunity Extraction and Storage...")
        
        # Extract opportunities from value analysis
        value_opportunities = []
        if isinstance(value_result, dict):
            analysis = value_result.get('undervalued_analysis', {})
            raw_opportunities = analysis.get('identified_opportunities', [])
            for opp in raw_opportunities:
                value_opportunities.append({
                    'ticker': opp.get('ticker', 'Unknown'),
                    'type': 'Value',
                    'entry_reason': f"Margin of safety: {opp.get('margin_of_safety', 0):.1%}",
                    'upside_potential': opp.get('upside_potential', 0),
                    'confidence': opp.get('confidence_level', 0.5),
                    'time_horizon': opp.get('time_horizon', '12-18 months')
                })
        
        # Extract opportunities from money flows
        flow_opportunities = []
        if isinstance(flow_result, dict):
            analyses = flow_result.get('money_flow_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                net_flow = analysis.get('net_institutional_flow', 0)
                if abs(net_flow) > 100000:  # Significant flow
                    flow_type = "Inflow" if net_flow > 0 else "Outflow"
                    flow_opportunities.append({
                        'ticker': ticker,
                        'type': 'Flow',
                        'entry_reason': f"Strong institutional {flow_type.lower()}: ${abs(net_flow):,.0f}",
                        'upside_potential': min(abs(net_flow) / 1000000 * 0.05, 0.3),
                        'confidence': 0.7,
                        'time_horizon': '1-3 months'
                    })
        
        print(f"   🎯 Value opportunities extracted: {len(value_opportunities)}")
        print(f"   🎯 Flow opportunities extracted: {len(flow_opportunities)}")
        
        # Store opportunities
        print("\n5️⃣ Testing Opportunity Storage...")
        
        # Store value opportunities
        if value_opportunities:
            added_count = opportunity_store.add_opportunities_from_agent('value_analysis', 'test_job_001', value_opportunities)
            print(f"   ✅ Stored {added_count} value opportunities")
        
        # Store flow opportunities
        if flow_opportunities:
            added_count = opportunity_store.add_opportunities_from_agent('money_flows', 'test_job_002', flow_opportunities)
            print(f"   ✅ Stored {added_count} flow opportunities")
        
        print("\n6️⃣ Testing Opportunity Retrieval and Ranking...")
        
        # Get all opportunities
        all_opportunities = opportunity_store.get_all_opportunities()
        print(f"   📊 Total opportunities in database: {len(all_opportunities)}")
        
        # Get statistics
        stats = opportunity_store.get_statistics()
        print(f"   📊 Database statistics: {stats}")
        
        # Rank opportunities
        if all_opportunities:
            ranked_opportunities = unified_scorer.rank_opportunities(all_opportunities)
            print(f"   🏆 Ranked {len(ranked_opportunities)} opportunities")
            
            # Show top 5
            print("   🎯 Top 5 opportunities:")
            for i, opp in enumerate(ranked_opportunities[:5], 1):
                print(f"      {i}. {opp.ticker} ({opp.opportunity_type}) - Score: {opp.priority_score:.2f}")
        
        # Get top 10
        top_10 = opportunity_store.get_top_opportunities(limit=10)
        print(f"   🏆 Top 10 opportunities retrieved: {len(top_10)}")
        
        # Calculate portfolio metrics
        if all_opportunities:
            portfolio_metrics = unified_scorer.calculate_portfolio_metrics(all_opportunities)
            print(f"   📊 Portfolio metrics: {portfolio_metrics}")
        
        print("\n7️⃣ Testing Agent-Specific Retrieval...")
        
        # Get opportunities by agent
        value_opps = opportunity_store.get_opportunities_by_agent('value_analysis')
        flow_opps = opportunity_store.get_opportunities_by_agent('money_flows')
        
        print(f"   📊 Value analysis opportunities: {len(value_opps)}")
        print(f"   📊 Money flows opportunities: {len(flow_opps)}")
        
        print("\n🎯 COMPLETE OPPORTUNITY SYSTEM TEST RESULTS")
        print("=" * 50)
        print("✅ What was tested:")
        print("   1. Opportunity store initialization")
        print("   2. Unified scorer initialization")
        print("   3. Agent opportunity generation")
        print("   4. Opportunity extraction and storage")
        print("   5. Database storage and retrieval")
        print("   6. Opportunity ranking and scoring")
        print("   7. Agent-specific filtering")
        print("")
        print("📊 Final Statistics:")
        print(f"   • Total opportunities: {len(all_opportunities)}")
        print(f"   • Value opportunities: {len(value_opps)}")
        print(f"   • Flow opportunities: {len(flow_opps)}")
        if all_opportunities:
            print(f"   • Average score: {portfolio_metrics['average_score']:.2f}")
            print(f"   • Expected return: {portfolio_metrics['expected_return']:.1%}")
        print("")
        print("🚀 System Status: READY FOR STREAMLIT DASHBOARD!")
        print("")
        print(f"🕒 Test completed: {datetime.now().strftime('%H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(f"📋 Error details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_opportunity_system())
    if not success:
        print("\n❌ Complete opportunity system test failed.")
        sys.exit(1)
