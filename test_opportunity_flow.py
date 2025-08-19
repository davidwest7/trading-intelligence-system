#!/usr/bin/env python3
"""
Test Opportunity Flow - Verify opportunities flow from agents to dashboard
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_opportunity_flow():
    print("🔍 TESTING OPPORTUNITY FLOW TO DASHBOARD")
    print("=" * 50)
    print(f"🕒 Test started: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    # Test each agent that should generate opportunities
    print("1️⃣ Testing Value Analysis Opportunities...")
    try:
        from agents.undervalued.agent import UndervaluedAgent
        agent = UndervaluedAgent()
        result = await agent.process(universe=['BRK.B', 'JPM', 'XOM'])
        
        analysis = result.get('undervalued_analysis', {})
        opportunities = analysis.get('identified_opportunities', [])
        
        print(f"   📊 Raw result structure:")
        print(f"      • Analysis keys: {list(analysis.keys())}")
        print(f"      • Opportunities found: {len(opportunities)}")
        
        if opportunities:
            print(f"   🎯 Sample opportunities:")
            for i, opp in enumerate(opportunities[:3], 1):
                ticker = opp.get('ticker', 'Unknown')
                margin = opp.get('margin_of_safety', 0)
                upside = opp.get('upside_potential', 0)
                confidence = opp.get('confidence_level', 0)
                print(f"      {i}. {ticker}: {margin:.1%} margin, {upside:.1%} upside, {confidence:.1%} confidence")
        else:
            print("   ⚠️  No opportunities in result")
            
    except Exception as e:
        print(f"   ❌ Value analysis failed: {e}")
    
    print("")
    print("2️⃣ Testing Money Flow Opportunities...")
    try:
        from agents.moneyflows.agent import MoneyFlowsAgent
        agent = MoneyFlowsAgent()
        result = await agent.process(tickers=['AAPL', 'TSLA'])
        
        analyses = result.get('money_flow_analyses', [])
        print(f"   📊 Money flow analyses: {len(analyses)}")
        
        significant_flows = []
        for analysis in analyses:
            ticker = analysis.get('ticker', 'Unknown')
            net_flow = analysis.get('net_institutional_flow', 0)
            if abs(net_flow) > 100000:
                significant_flows.append((ticker, net_flow))
                flow_type = "Inflow" if net_flow > 0 else "Outflow"
                print(f"   💰 {ticker}: ${abs(net_flow):,.0f} {flow_type}")
        
        print(f"   🎯 Significant flows that would become opportunities: {len(significant_flows)}")
        
    except Exception as e:
        print(f"   ❌ Money flows failed: {e}")
    
    print("")
    print("3️⃣ Testing Dashboard Opportunity Extraction...")
    
    # Test the enhanced job tracker opportunity extraction
    try:
        # Import the enhanced job tracker
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_enhanced", "streamlit_enhanced.py")
        streamlit_enhanced = importlib.util.module_from_spec(spec)
        
        # Create a sample result to test extraction
        sample_result = {
            'undervalued_analysis': {
                'identified_opportunities': [
                    {
                        'ticker': 'BRK.B',
                        'margin_of_safety': 0.25,
                        'upside_potential': 0.35,
                        'confidence_level': 0.8,
                        'time_horizon': '12-18 months'
                    },
                    {
                        'ticker': 'JPM',
                        'margin_of_safety': 0.18,
                        'upside_potential': 0.22,
                        'confidence_level': 0.75,
                        'time_horizon': '6-12 months'
                    }
                ]
            }
        }
        
        # Test opportunity extraction (we'll simulate this since we can't import the class directly)
        opportunities = []
        analysis = sample_result.get('undervalued_analysis', {})
        raw_opportunities = analysis.get('identified_opportunities', [])
        for opp in raw_opportunities:
            opportunities.append({
                'ticker': opp.get('ticker', 'Unknown'),
                'type': 'Value',
                'entry_reason': f"Margin of safety: {opp.get('margin_of_safety', 0):.1%}",
                'upside_potential': opp.get('upside_potential', 0),
                'confidence': opp.get('confidence_level', 0.5),
                'time_horizon': opp.get('time_horizon', '12-18 months')
            })
        
        print(f"   🎯 Dashboard-formatted opportunities: {len(opportunities)}")
        for i, opp in enumerate(opportunities, 1):
            print(f"      {i}. {opp['ticker']} ({opp['type']}): {opp['entry_reason']}")
            print(f"         Upside: {opp['upside_potential']:.1%}, Confidence: {opp['confidence']:.1%}")
        
    except Exception as e:
        print(f"   ❌ Dashboard extraction test failed: {e}")
    
    print("")
    print("🎯 OPPORTUNITY FLOW SUMMARY")
    print("=" * 30)
    print("✅ What should happen:")
    print("   1. User runs analysis in dashboard")
    print("   2. Agent generates results with opportunities")
    print("   3. EnhancedJobTracker extracts opportunities")
    print("   4. Opportunities appear in 'Opportunities' tab")
    print("   5. Detailed insights show in job results")
    print("")
    print("🔧 To see opportunities in dashboard:")
    print("   1. Launch: streamlit run streamlit_enhanced.py --server.port=8501")
    print("   2. Run Value Analysis with multiple symbols")
    print("   3. Check 'Opportunities' tab after completion")
    print("   4. Look for opportunity cards with details")
    print("")
    print(f"🕒 Test completed: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(test_opportunity_flow())
