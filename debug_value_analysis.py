#!/usr/bin/env python3
"""
Debug Value Analysis - Identify the specific error in value analysis
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def debug_value_analysis():
    print("🔍 DEBUGGING VALUE ANALYSIS ERROR")
    print("=" * 40)
    print(f"🕒 Debug started: {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    try:
        print("1️⃣ Importing UndervaluedAgent...")
        from agents.undervalued.agent import UndervaluedAgent
        print("   ✅ UndervaluedAgent imported successfully")
        
        print("\n2️⃣ Creating agent instance...")
        agent = UndervaluedAgent()
        print("   ✅ Agent instance created")
        
        print("\n3️⃣ Testing agent.process() with error handling...")
        try:
            result = await agent.process(universe=['BRK.B', 'JPM'])
            print(f"   ✅ Agent.process() completed successfully")
            print(f"   📊 Result type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"   📊 Result keys: {list(result.keys())}")
                
                if 'undervalued_analysis' in result:
                    analysis = result['undervalued_analysis']
                    print(f"   📊 Analysis keys: {list(analysis.keys())}")
                    
                    opportunities = analysis.get('identified_opportunities', [])
                    print(f"   🎯 Opportunities found: {len(opportunities)}")
                    
                    if opportunities:
                        print("   🎯 Sample opportunities:")
                        for i, opp in enumerate(opportunities[:2], 1):
                            print(f"      {i}. {opp.get('ticker', 'Unknown')}: {opp.get('margin_of_safety', 0):.1%} margin")
                    else:
                        print("   ⚠️  No opportunities in result")
                else:
                    print("   ❌ Missing 'undervalued_analysis' key in result")
            else:
                print(f"   ❌ Result is not a dict: {type(result)}")
                
        except Exception as e:
            print(f"   ❌ Agent.process() failed: {e}")
            print(f"   📋 Full error traceback:")
            print(traceback.format_exc())
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        print(f"📋 Full error traceback:")
        print(traceback.format_exc())
        return False
    
    print("\n🎯 DEBUG SUMMARY")
    print("=" * 20)
    print("✅ If successful: Value analysis works, check Streamlit integration")
    print("❌ If failed: Error details shown above")
    print(f"🕒 Debug completed: {datetime.now().strftime('%H:%M:%S')}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(debug_value_analysis())
    if not success:
        print("\n❌ Debug failed. Check the error details above.")
        sys.exit(1)
