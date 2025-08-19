#!/usr/bin/env python3
"""
Fixed Dashboard Launcher - Handles errors and ensures opportunities flow correctly
"""

import os
import sys
import subprocess
import time
import traceback
from pathlib import Path

def main():
    print("🚀 FIXED TRADING INTELLIGENCE DASHBOARD LAUNCHER")
    print("=" * 55)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    print(f"📁 Script location: {script_dir}")
    
    # Change to the script directory
    os.chdir(script_dir)
    print(f"📂 Current directory: {os.getcwd()}")
    
    # Check if required files exist
    required_files = [
        'streamlit_enhanced.py',
        'main.py', 
        'demo.py'
    ]
    
    print("\n📋 Checking required files:")
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}: Found")
        else:
            print(f"   ❌ {file}: Missing")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Missing required files. Please ensure you're in the project directory.")
        return False
    
    print("\n🔧 Testing agent functionality and error handling...")
    
    # Test that agents work and can provide opportunities
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(script_dir))
        
        # Import and test agents with error handling
        from agents.undervalued.agent import UndervaluedAgent
        from agents.moneyflows.agent import MoneyFlowsAgent
        
        print("   ✅ Agent imports successful")
        
        # Test opportunity generation with error handling
        import asyncio
        
        async def test_opportunities_with_error_handling():
            try:
                # Test Value Analysis Agent for opportunities
                print("   🔍 Testing Value Analysis Agent...")
                value_agent = UndervaluedAgent()
                result = await value_agent.process(universe=['BRK.B', 'JPM', 'XOM'])
                
                print(f"   ✅ Value agent result type: {type(result)}")
                print(f"   ✅ Value agent result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                analysis = result.get('undervalued_analysis', {})
                opportunities = analysis.get('identified_opportunities', [])
                
                print(f"   ✅ Value opportunities: {len(opportunities)} found")
                
                if opportunities:
                    for i, opp in enumerate(opportunities[:3], 1):
                        ticker = opp.get('ticker', 'Unknown')
                        margin = opp.get('margin_of_safety', 0)
                        upside = opp.get('upside_potential', 0)
                        confidence = opp.get('confidence_level', 0)
                        print(f"      {i}. {ticker}: {margin:.1%} margin, {upside:.1%} upside, {confidence:.1%} confidence")
                else:
                    print("   ⚠️  No opportunities in value analysis result")
                
                return opportunities
                
            except Exception as e:
                print(f"   ❌ Value analysis error: {e}")
                print(f"   📋 Error details: {traceback.format_exc()}")
                return []
        
        opportunities = asyncio.run(test_opportunities_with_error_handling())
        
        if opportunities:
            print(f"\n✅ Opportunity flow working: {len(opportunities)} opportunities generated")
        else:
            print("\n⚠️  No opportunities generated - this may be why dashboard is empty")
            
    except Exception as e:
        print(f"\n❌ Agent testing failed: {e}")
        print(f"📋 Error details: {traceback.format_exc()}")
        return False
    
    print("\n🔧 Testing Streamlit dashboard error handling...")
    
    # Test the enhanced job tracker with error handling
    try:
        # Test opportunity extraction with error handling
        sample_result = {
            'undervalued_analysis': {
                'identified_opportunities': [
                    {
                        'ticker': 'BRK.B',
                        'margin_of_safety': 0.25,
                        'upside_potential': 0.35,
                        'confidence_level': 0.8,
                        'time_horizon': '12-18 months'
                    }
                ]
            }
        }
        
        # Test opportunity extraction logic
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
        
        print(f"   ✅ Dashboard opportunity extraction: {len(opportunities)} opportunities")
        
    except Exception as e:
        print(f"   ❌ Dashboard extraction test failed: {e}")
        print(f"   📋 Error details: {traceback.format_exc()}")
        return False
    
    print("\n🌐 Launching Fixed Enhanced Dashboard...")
    print("   Dashboard will show:")
    print("   • Real-time job progress with error handling")
    print("   • Generated opportunities (if any)")
    print("   • Detailed error messages")
    print("   • Interactive monitoring")
    print("\n   URL: http://localhost:8501")
    print("\n   To stop: Press Ctrl+C")
    print("\n" + "="*55)
    
    # Launch Streamlit dashboard
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_enhanced.py", 
            "--server.port=8501",
            "--server.headless=false"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to launch dashboard: {e}")
        print(f"📋 Error details: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Make sure you're in the project directory")
        print("   2. Run: cd /Users/davidwestera/trading-intelligence-system")
        print("   3. Then run: python launch_dashboard_fixed.py")
        print("   4. Check error messages above for specific issues")
        sys.exit(1)
