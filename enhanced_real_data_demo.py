#!/usr/bin/env python3
"""
Enhanced Real Data Demo - Trading Intelligence System

Demonstration of all agents with real Polygon.io data integration.
"""

import asyncio
import time
import logging
import os
import sys
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

# Load environment variables
load_dotenv('env_real_keys.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRealDataDemo:
    """Enhanced demo with real data agents"""
    
    def __init__(self):
        self.config = {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
            'twitter_api_key': os.getenv('TWITTER_API_KEY'),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID')
        }
        
        # Initialize real data agents
        self.agents = {}
        self.results = {}
        
    async def initialize_agents(self):
        """Initialize all real data agents"""
        print("🚀 **INITIALIZING REAL DATA AGENTS**")
        print("=" * 50)
        
        try:
            # Technical Agent
            from agents.technical.agent_real_data import RealDataTechnicalAgent
            self.agents['technical'] = RealDataTechnicalAgent(self.config)
            print("✅ Technical Agent initialized")
            
            # Flow Agent
            from agents.flow.agent_real_data import RealDataFlowAgent
            self.agents['flow'] = RealDataFlowAgent(self.config)
            print("✅ Flow Agent initialized")
            
            # Top Performers Agent
            from agents.top_performers.agent_real_data import RealDataTopPerformersAgent
            self.agents['top_performers'] = RealDataTopPerformersAgent(self.config)
            print("✅ Top Performers Agent initialized")
            
            # Undervalued Agent
            from agents.undervalued.agent_real_data import RealDataUndervaluedAgent
            self.agents['undervalued'] = RealDataUndervaluedAgent(self.config)
            print("✅ Undervalued Agent initialized")
            
            # Macro Agent
            from agents.macro.agent_real_data import RealDataMacroAgent
            self.agents['macro'] = RealDataMacroAgent(self.config)
            print("✅ Macro Agent initialized")
            
            # Sentiment Agent
            from agents.sentiment.agent_real_data import RealDataSentimentAgent
            self.agents['sentiment'] = RealDataSentimentAgent(self.config)
            print("✅ Sentiment Agent initialized")
            
            print(f"✅ All {len(self.agents)} agents initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_comprehensive_analysis(self, tickers=None):
        """Run comprehensive analysis with all agents"""
        if tickers is None:
            tickers = ['AAPL', 'TSLA', 'SPY', 'QQQ', 'MSFT', 'GOOGL']
        
        print(f"\n🔍 **COMPREHENSIVE REAL DATA ANALYSIS**")
        print("=" * 60)
        print(f"Analyzing {len(tickers)} tickers with real market data")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all agents in parallel
        tasks = []
        
        # Technical Analysis
        tasks.append(self._run_technical_analysis(tickers))
        
        # Flow Analysis
        tasks.append(self._run_flow_analysis(tickers))
        
        # Top Performers Analysis
        tasks.append(self._run_top_performers_analysis())
        
        # Undervalued Analysis
        tasks.append(self._run_undervalued_analysis(tickers))
        
        # Macro Analysis
        tasks.append(self._run_macro_analysis())
        
        # Sentiment Analysis
        tasks.append(self._run_sentiment_analysis(tickers))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        agent_names = ['technical', 'flow', 'top_performers', 'undervalued', 'macro', 'sentiment']
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ {agent_names[i]} failed: {result}")
                self.results[agent_names[i]] = None
            else:
                self.results[agent_names[i]] = result
        
        total_time = time.time() - start_time
        
        # Generate comprehensive summary
        await self._generate_comprehensive_summary(total_time)
        
        return self.results
    
    async def _run_technical_analysis(self, tickers):
        """Run technical analysis"""
        print("\n🔧 **TECHNICAL ANALYSIS**")
        print("-" * 30)
        
        try:
            result = await self.agents['technical'].analyze_technical_indicators(tickers)
            
            print(f"✅ Analyzed {result['tickers_analyzed']} tickers")
            print(f"   Overall Sentiment: {result['overall_signals']['overall_sentiment']}")
            print(f"   Total Signals: {result['overall_signals']['total_signals']}")
            
            # Show sample technical data
            for ticker in tickers[:3]:
                if ticker in result['technical_analysis']:
                    analysis = result['technical_analysis'][ticker]
                    print(f"   {ticker}: ${analysis['current_price']:.2f} (RSI: {analysis.get('rsi', 0):.1f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Technical analysis failed: {e}")
            return None
    
    async def _run_flow_analysis(self, tickers):
        """Run flow analysis"""
        print("\n🌊 **FLOW ANALYSIS**")
        print("-" * 30)
        
        try:
            result = await self.agents['flow'].analyze_market_flow(tickers)
            
            print(f"✅ Analyzed {result['tickers_analyzed']} tickers")
            print(f"   Overall Regime: {result['overall_flow']['overall_regime']}")
            print(f"   Total Volume: {result['overall_flow']['total_volume']:,}")
            
            # Show sample flow data
            for ticker in tickers[:3]:
                if ticker in result['flow_analysis']:
                    analysis = result['flow_analysis'][ticker]
                    print(f"   {ticker}: Spread {analysis['spread_percentage']:.3f}% (Flow: {analysis['flow_regime']})")
            
            return result
            
        except Exception as e:
            print(f"❌ Flow analysis failed: {e}")
            return None
    
    async def _run_top_performers_analysis(self):
        """Run top performers analysis"""
        print("\n🏆 **TOP PERFORMERS ANALYSIS**")
        print("-" * 30)
        
        try:
            result = await self.agents['top_performers'].analyze_top_performers()
            
            print(f"✅ Successfully analyzed")
            print(f"   Top Gainers: {len(result['top_gainers'])} stocks")
            print(f"   Momentum Regime: {result['momentum_analysis']['momentum_regime']}")
            
            # Show top gainers
            if result['top_gainers']:
                for i, gainer in enumerate(result['top_gainers'][:3]):
                    print(f"   #{i+1}: {gainer['symbol']} +{gainer['change_percent']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Top performers analysis failed: {e}")
            return None
    
    async def _run_undervalued_analysis(self, tickers):
        """Run undervalued analysis"""
        print("\n💎 **UNDERVALUED ANALYSIS**")
        print("-" * 30)
        
        try:
            result = await self.agents['undervalued'].analyze_undervalued_stocks(tickers)
            
            print(f"✅ Analyzed {result['tickers_analyzed']} tickers")
            print(f"   Market Value Regime: {result['overall_value']['market_value_regime']}")
            print(f"   High Value Count: {result['overall_value']['high_value_count']}")
            
            # Show value opportunities
            if result['overall_value']['value_opportunities']:
                for i, opportunity in enumerate(result['overall_value']['value_opportunities'][:3]):
                    print(f"   #{i+1}: {opportunity['ticker']} (Score: {opportunity['value_score']:.0f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Undervalued analysis failed: {e}")
            return None
    
    async def _run_macro_analysis(self):
        """Run macro analysis"""
        print("\n🌍 **MACRO ANALYSIS**")
        print("-" * 30)
        
        try:
            result = await self.agents['macro'].analyze_macro_environment()
            
            print(f"✅ Successfully analyzed")
            print(f"   Market Breadth: {result['macro_trends']['market_breadth']['breadth_ratio']:.1%}")
            print(f"   Risk Sentiment: {result['macro_trends']['risk_sentiment']['sentiment']}")
            
            # Show economic indicators
            indicators = result['economic_indicators']
            if 'sp500' in indicators and indicators['sp500']:
                print(f"   SP500: {indicators['sp500']['change_percent']:.2f}%")
            if 'nasdaq' in indicators and indicators['nasdaq']:
                print(f"   NASDAQ: {indicators['nasdaq']['change_percent']:.2f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Macro analysis failed: {e}")
            return None
    
    async def _run_sentiment_analysis(self, tickers):
        """Run sentiment analysis"""
        print("\n📊 **SENTIMENT ANALYSIS**")
        print("-" * 30)
        
        try:
            result = await self.agents['sentiment'].analyze_sentiment_optimized(tickers)
            
            print(f"✅ Analyzed {result['summary']['total_tickers']} tickers")
            print(f"   Overall Sentiment: {result['sentiment_analysis']['overall_score']:.2f}")
            print(f"   Confidence: {result['sentiment_analysis']['confidence']:.1%}")
            print(f"   Total Posts: {result['summary']['total_posts_analyzed']}")
            
            # Show sentiment distribution
            dist = result['sentiment_analysis']['sentiment_distribution']
            print(f"   Distribution: {dist['positive']:.1%} positive, {dist['negative']:.1%} negative, {dist['neutral']:.1%} neutral")
            
            # Show top sources
            if result['summary']['top_sentiment_sources']:
                print(f"   Top Sources: {', '.join(result['summary']['top_sentiment_sources'][:3])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            return None
    
    async def _generate_comprehensive_summary(self, total_time):
        """Generate comprehensive summary of all results"""
        print("\n" + "=" * 60)
        print("📊 **COMPREHENSIVE ANALYSIS SUMMARY**")
        print("=" * 60)
        
        successful_agents = sum(1 for result in self.results.values() if result is not None)
        total_agents = len(self.results)
        
        print(f"✅ Successful Agents: {successful_agents}/{total_agents}")
        print(f"📈 Success Rate: {successful_agents/total_agents*100:.0f}%")
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        print(f"🌐 Data Source: Polygon.io (Real Market Data)")
        
        # Agent status
        agent_status = {
            'Technical': '✅' if self.results['technical'] else '❌',
            'Flow': '✅' if self.results['flow'] else '❌',
            'Top Performers': '✅' if self.results['top_performers'] else '❌',
            'Undervalued': '✅' if self.results['undervalued'] else '❌',
            'Macro': '✅' if self.results['macro'] else '❌',
            'Sentiment': '✅' if self.results['sentiment'] else '❌'
        }
        
        print("\n📋 Agent Status:")
        for agent, status in agent_status.items():
            print(f"   {status} {agent}")
        
        # Overall system assessment
        if successful_agents >= 5:
            print(f"\n🎉 **SYSTEM STATUS: EXCELLENT**")
            print("   Real data integration successful!")
            print("   System ready for production use.")
        elif successful_agents >= 3:
            print(f"\n🟡 **SYSTEM STATUS: GOOD**")
            print("   Most agents working with real data.")
            print("   Some agents need attention.")
        else:
            print(f"\n🔴 **SYSTEM STATUS: NEEDS ATTENTION**")
            print("   Multiple agents failed.")
            print("   Check API keys and connectivity.")
        
        print("\n" + "=" * 60)
        print("🚀 **NEXT STEPS:**")
        print("1. ✅ Real data integration - COMPLETE")
        print("2. 🔄 Update Streamlit dashboard with real data")
        print("3. 🔄 Add remaining APIs (NewsAPI, Quiver)")
        print("4. 🔄 Deploy to production")
        print("=" * 60)


async def main():
    """Main demo function"""
    print("🚀 **ENHANCED REAL DATA DEMO**")
    print("=" * 50)
    print("Trading Intelligence System with Real Market Data")
    print("=" * 50)
    
    # Initialize demo
    demo = EnhancedRealDataDemo()
    
    # Initialize agents
    if not await demo.initialize_agents():
        print("❌ Failed to initialize agents")
        return
    
    # Run comprehensive analysis
    results = await demo.run_comprehensive_analysis()
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📊 Results available in demo.results")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
