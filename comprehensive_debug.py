#!/usr/bin/env python3
"""
Comprehensive System Debug - Test all components and identify issues
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SystemDebugger:
    """Comprehensive system debugger"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
    
    def log_success(self, message):
        """Log a successful test"""
        self.successes.append(message)
        print(f"   ✅ {message}")
    
    def log_warning(self, message):
        """Log a warning"""
        self.warnings.append(message)
        print(f"   ⚠️  {message}")
    
    def log_error(self, message, error=None):
        """Log an error"""
        error_msg = f"{message}"
        if error:
            error_msg += f" - {str(error)}"
        self.errors.append(error_msg)
        print(f"   ❌ {error_msg}")
    
    async def debug_imports(self):
        """Debug all imports"""
        print("\n1️⃣ DEBUGGING IMPORTS...")
        
        # Test basic imports
        try:
            import streamlit as st
            self.log_success("Streamlit imported")
        except Exception as e:
            self.log_error("Streamlit import failed", e)
        
        try:
            import pandas as pd
            self.log_success("Pandas imported")
        except Exception as e:
            self.log_error("Pandas import failed", e)
        
        try:
            import plotly.express as px
            self.log_success("Plotly imported")
        except Exception as e:
            self.log_error("Plotly import failed", e)
        
        # Test agent imports
        try:
            from agents.undervalued.agent import UndervaluedAgent
            self.log_success("UndervaluedAgent imported")
        except Exception as e:
            self.log_error("UndervaluedAgent import failed", e)
        
        try:
            from agents.moneyflows.agent import MoneyFlowsAgent
            self.log_success("MoneyFlowsAgent imported")
        except Exception as e:
            self.log_error("MoneyFlowsAgent import failed", e)
        
        try:
            from agents.technical.agent import TechnicalAgent
            self.log_success("TechnicalAgent imported")
        except Exception as e:
            self.log_error("TechnicalAgent import failed", e)
        
        try:
            from agents.insider.agent import InsiderAgent
            self.log_success("InsiderAgent imported")
        except Exception as e:
            self.log_error("InsiderAgent import failed", e)
        
        # Test common imports
        try:
            from common.opportunity_store import opportunity_store, Opportunity
            self.log_success("Opportunity store imported")
        except Exception as e:
            self.log_error("Opportunity store import failed", e)
        
        try:
            from common.unified_opportunity_scorer import unified_scorer
            self.log_success("Unified scorer imported")
        except Exception as e:
            self.log_error("Unified scorer import failed", e)
    
    async def debug_opportunity_store(self):
        """Debug opportunity store functionality"""
        print("\n2️⃣ DEBUGGING OPPORTUNITY STORE...")
        
        try:
            from common.opportunity_store import opportunity_store, Opportunity
            
            # Test store initialization
            self.log_success("Opportunity store initialized")
            
            # Test adding opportunities
            test_opportunity = Opportunity(
                id="test_001",
                ticker="TEST",
                agent_type="test_agent",
                opportunity_type="Test",
                entry_reason="Test opportunity",
                upside_potential=0.15,
                confidence=0.8,
                time_horizon="1-3 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={"test": "data"}
            )
            
            success = opportunity_store.add_opportunity(test_opportunity)
            if success:
                self.log_success("Opportunity added to store")
            else:
                self.log_error("Failed to add opportunity to store")
            
            # Test retrieving opportunities
            opportunities = opportunity_store.get_all_opportunities()
            self.log_success(f"Retrieved {len(opportunities)} opportunities from store")
            
            # Test statistics
            stats = opportunity_store.get_statistics()
            self.log_success(f"Store statistics: {stats}")
            
        except Exception as e:
            self.log_error("Opportunity store debug failed", e)
    
    async def debug_unified_scorer(self):
        """Debug unified scorer functionality"""
        print("\n3️⃣ DEBUGGING UNIFIED SCORER...")
        
        try:
            from common.unified_opportunity_scorer import unified_scorer
            from common.opportunity_store import Opportunity
            
            # Test scorer initialization
            self.log_success("Unified scorer initialized")
            
            # Test scoring
            test_opportunity = Opportunity(
                id="test_score_001",
                ticker="AAPL",
                agent_type="value_analysis",
                opportunity_type="Value",
                entry_reason="Test scoring",
                upside_potential=0.25,
                confidence=0.85,
                time_horizon="6-12 months",
                discovered_at=datetime.now(),
                job_id="test_job",
                raw_data={"test": "data"}
            )
            
            score = unified_scorer.calculate_priority_score(test_opportunity)
            self.log_success(f"Priority score calculated: {score:.3f}")
            
            # Test ranking
            opportunities = [test_opportunity]
            ranked = unified_scorer.rank_opportunities(opportunities)
            self.log_success(f"Ranked {len(ranked)} opportunities")
            
            # Test portfolio metrics
            metrics = unified_scorer.calculate_portfolio_metrics(opportunities)
            self.log_success(f"Portfolio metrics calculated: {metrics}")
            
        except Exception as e:
            self.log_error("Unified scorer debug failed", e)
    
    async def debug_agents(self):
        """Debug agent functionality"""
        print("\n4️⃣ DEBUGGING AGENTS...")
        
        # Test UndervaluedAgent
        try:
            from agents.undervalued.agent import UndervaluedAgent
            agent = UndervaluedAgent()
            result = await agent.process(universe=['BRK.B', 'JPM'])
            
            if isinstance(result, dict) and 'undervalued_analysis' in result:
                opportunities = result['undervalued_analysis'].get('identified_opportunities', [])
                self.log_success(f"UndervaluedAgent: {len(opportunities)} opportunities generated")
            else:
                self.log_error("UndervaluedAgent: Invalid result format")
                
        except Exception as e:
            self.log_error("UndervaluedAgent debug failed", e)
        
        # Test MoneyFlowsAgent
        try:
            from agents.moneyflows.agent import MoneyFlowsAgent
            agent = MoneyFlowsAgent()
            result = await agent.process(tickers=['AAPL', 'TSLA'])
            
            if isinstance(result, dict) and 'money_flow_analyses' in result:
                analyses = result['money_flow_analyses']
                self.log_success(f"MoneyFlowsAgent: {len(analyses)} analyses generated")
            else:
                self.log_error("MoneyFlowsAgent: Invalid result format")
                
        except Exception as e:
            self.log_error("MoneyFlowsAgent debug failed", e)
        
        # Test TechnicalAgent
        try:
            from agents.technical.agent import TechnicalAgent
            agent = TechnicalAgent()
            
            # Create proper payload for TechnicalAgent
            payload = {
                'symbols': ['AAPL', 'TSLA'],
                'timeframes': ['1h', '4h', '1d'],
                'strategies': ['imbalance', 'trend'],
                'min_score': 0.01,
                'max_risk': 0.02,
                'lookback_periods': 30
            }
            
            result = await agent.find_opportunities(payload)
            
            if isinstance(result, dict) and 'opportunities' in result:
                opportunities = result['opportunities']
                self.log_success(f"TechnicalAgent: {len(opportunities)} opportunities generated")
            else:
                self.log_error("TechnicalAgent: Invalid result format")
                
        except Exception as e:
            self.log_error("TechnicalAgent debug failed", e)
    
    async def debug_streamlit_components(self):
        """Debug Streamlit components"""
        print("\n5️⃣ DEBUGGING STREAMLIT COMPONENTS...")
        
        try:
            # Test if functions exist by reading the file
            with open('streamlit_enhanced.py', 'r') as f:
                content = f.read()
            
            # Check for function definitions
            if 'def enhanced_dashboard_view():' in content:
                self.log_success("enhanced_dashboard_view function exists")
            else:
                self.log_error("enhanced_dashboard_view function missing")
            
            if 'def opportunities_view():' in content:
                self.log_success("opportunities_view function exists")
            else:
                self.log_error("opportunities_view function missing")
            
            if 'def top_opportunities_view():' in content:
                self.log_success("top_opportunities_view function exists")
            else:
                self.log_error("top_opportunities_view function missing")
            
            if 'def main_dashboard():' in content:
                self.log_success("main_dashboard function exists")
            else:
                self.log_error("main_dashboard function missing")
            
            # Test if the file can be imported
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("streamlit_enhanced", "streamlit_enhanced.py")
                streamlit_enhanced = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(streamlit_enhanced)
                self.log_success("Streamlit enhanced module can be imported")
            except Exception as e:
                self.log_error("Streamlit enhanced module import failed", e)
            
        except Exception as e:
            self.log_error("Streamlit components debug failed", e)
    
    async def debug_opportunity_flow(self):
        """Debug complete opportunity flow"""
        print("\n6️⃣ DEBUGGING COMPLETE OPPORTUNITY FLOW...")
        
        try:
            from common.opportunity_store import opportunity_store
            from common.unified_opportunity_scorer import unified_scorer
            from agents.undervalued.agent import UndervaluedAgent
            
            # Clear existing opportunities for clean test
            print("   📋 Testing with clean opportunity store...")
            
            # Generate opportunities
            agent = UndervaluedAgent()
            result = await agent.process(universe=['BRK.B', 'JPM'])
            
            if isinstance(result, dict) and 'undervalued_analysis' in result:
                analysis = result['undervalued_analysis']
                raw_opportunities = analysis.get('identified_opportunities', [])
                
                # Extract opportunities
                opportunities = []
                for opp in raw_opportunities:
                    opportunities.append({
                        'ticker': opp.get('ticker', 'Unknown'),
                        'type': 'Value',
                        'entry_reason': f"Margin of safety: {opp.get('margin_of_safety', 0):.1%}",
                        'upside_potential': opp.get('upside_potential', 0),
                        'confidence': opp.get('confidence_level', 0.5),
                        'time_horizon': opp.get('time_horizon', '12-18 months')
                    })
                
                self.log_success(f"Extracted {len(opportunities)} opportunities")
                
                # Store opportunities
                if opportunities:
                    added_count = opportunity_store.add_opportunities_from_agent('value_analysis', 'debug_job', opportunities)
                    self.log_success(f"Stored {added_count} opportunities in database")
                    
                    # Test retrieval and ranking
                    all_opportunities = opportunity_store.get_all_opportunities()
                    self.log_success(f"Retrieved {len(all_opportunities)} opportunities from database")
                    
                    if all_opportunities:
                        # Update scores
                        for opp in all_opportunities:
                            opp.priority_score = unified_scorer.calculate_priority_score(opp)
                        
                        # Get top opportunities
                        top_opportunities = opportunity_store.get_top_opportunities(limit=5)
                        self.log_success(f"Top {len(top_opportunities)} opportunities retrieved")
                        
                        # Show sample
                        if top_opportunities:
                            sample = top_opportunities[0]
                            self.log_success(f"Sample opportunity: {sample.ticker} (Score: {sample.priority_score:.3f})")
                
            else:
                self.log_error("Failed to generate opportunities from agent")
                
        except Exception as e:
            self.log_error("Opportunity flow debug failed", e)
    
    def debug_file_structure(self):
        """Debug file structure and permissions"""
        print("\n7️⃣ DEBUGGING FILE STRUCTURE...")
        
        required_files = [
            'streamlit_enhanced.py',
            'common/opportunity_store.py',
            'common/unified_opportunity_scorer.py',
            'agents/undervalued/agent.py',
            'agents/moneyflows/agent.py',
            'agents/technical/agent.py',
            'agents/insider/agent.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.log_success(f"File exists: {file_path}")
                
                # Check if readable
                try:
                    with open(file_path, 'r') as f:
                        f.read(100)  # Read first 100 chars
                    self.log_success(f"File readable: {file_path}")
                except Exception as e:
                    self.log_error(f"File not readable: {file_path}", e)
            else:
                self.log_error(f"File missing: {file_path}")
        
        # Check database file
        if os.path.exists('opportunities.db'):
            self.log_success("Opportunities database exists")
        else:
            self.log_warning("Opportunities database not found (will be created)")
    
    def print_summary(self):
        """Print debug summary"""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE DEBUG SUMMARY")
        print("="*60)
        
        print(f"\n✅ SUCCESSES ({len(self.successes)}):")
        for success in self.successes:
            print(f"   • {success}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   • Total tests: {len(self.successes) + len(self.warnings) + len(self.errors)}")
        print(f"   • Successful: {len(self.successes)}")
        print(f"   • Warnings: {len(self.warnings)}")
        print(f"   • Errors: {len(self.errors)}")
        
        if self.errors:
            print(f"\n🔧 RECOMMENDATIONS:")
            print(f"   • Fix {len(self.errors)} critical errors before proceeding")
            print(f"   • Address {len(self.warnings)} warnings for optimal performance")
        else:
            print(f"\n🎉 SYSTEM STATUS: READY!")
            print(f"   • All critical components working")
            print(f"   • Opportunity flow functional")
            print(f"   • Dashboard ready to launch")

async def main():
    """Main debug function"""
    print("🔍 COMPREHENSIVE SYSTEM DEBUG")
    print("="*60)
    print(f"🕒 Debug started: {datetime.now().strftime('%H:%M:%S')}")
    
    debugger = SystemDebugger()
    
    # Run all debug tests
    await debugger.debug_imports()
    await debugger.debug_opportunity_store()
    await debugger.debug_unified_scorer()
    await debugger.debug_agents()
    await debugger.debug_streamlit_components()
    await debugger.debug_opportunity_flow()
    debugger.debug_file_structure()
    
    # Print summary
    debugger.print_summary()
    
    print(f"\n🕒 Debug completed: {datetime.now().strftime('%H:%M:%S')}")
    
    return len(debugger.errors) == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        print("\n❌ System has critical errors that need to be fixed.")
        sys.exit(1)
    else:
        print("\n✅ System is ready for use!")
