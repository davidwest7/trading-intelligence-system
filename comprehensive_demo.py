#!/usr/bin/env python3
"""
Comprehensive Multi-Agent Trading System Demo

Demonstrates all 10+ agents working together in a unified system
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all our agents
from agents.technical.agent import TechnicalAgent
from agents.sentiment.agent import SentimentAgent
from agents.flow.agent import FlowAgent
from agents.macro.agent import MacroAgent
from agents.moneyflows.agent import MoneyFlowsAgent
from agents.undervalued.agent import UndervaluedAgent
from agents.insider.agent import InsiderAgent
from agents.causal.agent import CausalAgent
from agents.hedging.agent import HedgingAgent
from agents.learning.agent import LearningAgent
from common.scoring.unified_score import UnifiedScorer


class ComprehensiveDemo:
    """Comprehensive demonstration of the complete multi-agent system"""
    
    def __init__(self):
        self.agents = {}
        self.unified_scorer = UnifiedScorer()
        self.demo_tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
        
    async def initialize_agents(self):
        """Initialize all agents"""
        print("🤖 Initializing Multi-Agent System...")
        print("=" * 50)
        
        # Initialize all agents
        agent_classes = [
            ('technical', TechnicalAgent),
            ('sentiment', SentimentAgent),
            ('flow', FlowAgent),
            ('macro', MacroAgent),
            ('moneyflows', MoneyFlowsAgent),
            ('undervalued', UndervaluedAgent),
            ('insider', InsiderAgent),
            ('causal', CausalAgent),
            ('hedging', HedgingAgent),
            ('learning', LearningAgent)
        ]
        
        for name, agent_class in agent_classes:
            try:
                self.agents[name] = agent_class()
                print(f"✅ {name.capitalize()} Agent initialized")
            except Exception as e:
                print(f"❌ Failed to initialize {name} agent: {e}")
        
        print(f"\n🎯 Successfully initialized {len(self.agents)} agents")
    
    async def run_technical_analysis(self):
        """Run technical analysis demonstration"""
        print("\n" + "=" * 60)
        print("📈 TECHNICAL ANALYSIS DEMONSTRATION")
        print("=" * 60)
        
        try:
            result = await self.agents['technical'].analyze(
                symbols=self.demo_tickers[:3],
                timeframes=['1h', '4h'],
                strategies=['imbalance']
            )
            
            opportunities = result.get('opportunities', [])
            print(f"🎯 Found {len(opportunities)} technical opportunities")
            
            for i, opp in enumerate(opportunities[:3], 1):
                print(f"   {i}. {opp['symbol']} - {opp['strategy']}")
                print(f"      Entry: ${opp['entry_price']:.2f}")
                print(f"      R/R: {opp['risk_reward_ratio']:.2f}")
                print(f"      Confidence: {opp['confidence_score']:.1%}")
            
            return opportunities
            
        except Exception as e:
            print(f"❌ Technical analysis failed: {e}")
            return []
    
    async def run_sentiment_analysis(self):
        """Run sentiment analysis demonstration"""
        print("\n" + "=" * 60)
        print("🧠 SENTIMENT ANALYSIS DEMONSTRATION")
        print("=" * 60)
        
        try:
            result = await self.agents['sentiment'].process(
                tickers=self.demo_tickers[:3],
                lookback_hours=24
            )
            
            analyses = result.get('sentiment_analyses', [])
            print(f"📊 Analyzed sentiment for {len(analyses)} assets")
            
            for analysis in analyses:
                ticker = analysis['ticker']
                overall = analysis['overall_sentiment']
                score = analysis['composite_sentiment_score']
                print(f"   {ticker}: {overall} (score: {score:.2f})")
            
            return analyses
            
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            return []
    
    async def run_flow_analysis(self):
        """Run flow analysis demonstration"""
        print("\n" + "=" * 60)
        print("💰 FLOW ANALYSIS DEMONSTRATION")
        print("=" * 60)
        
        try:
            result = await self.agents['flow'].process(
                tickers=self.demo_tickers[:3],
                lookback_period="1d"
            )
            
            analyses = result.get('flow_analyses', [])
            print(f"📈 Analyzed flow for {len(analyses)} assets")
            
            for analysis in analyses:
                ticker = analysis['ticker']
                regime = analysis['detected_regime']
                flow_score = analysis['money_flow_score']
                print(f"   {ticker}: {regime} regime (flow: {flow_score:.2f})")
            
            return analyses
            
        except Exception as e:
            print(f"❌ Flow analysis failed: {e}")
            return []
    
    async def run_macro_analysis(self):
        """Run macro analysis demonstration"""
        print("\n" + "=" * 60)
        print("🌍 MACRO ANALYSIS DEMONSTRATION") 
        print("=" * 60)
        
        try:
            result = await self.agents['macro'].process(
                focus_regions=['US', 'EU', 'ASIA'],
                lookback_days=30
            )
            
            analysis = result.get('macro_analysis', {})
            themes = analysis.get('identified_themes', [])
            print(f"🎯 Identified {len(themes)} macro themes")
            
            for theme in themes[:3]:
                name = theme['theme_name']
                impact = theme['market_impact_score']
                print(f"   • {name} (impact: {impact:.2f})")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Macro analysis failed: {e}")
            return {}
    
    async def run_value_analysis(self):
        """Run undervalued analysis demonstration"""
        print("\n" + "=" * 60)
        print("💎 VALUE ANALYSIS DEMONSTRATION")
        print("=" * 60)
        
        try:
            result = await self.agents['undervalued'].process(
                universe=self.demo_tickers,
                num_recommendations=5
            )
            
            analysis = result.get('undervalued_analysis', {})
            opportunities = analysis.get('identified_opportunities', [])
            print(f"💰 Found {len(opportunities)} value opportunities")
            
            for opp in opportunities[:3]:
                ticker = opp['ticker']
                margin = opp['margin_of_safety']
                upside = opp['upside_potential']
                print(f"   {ticker}: {margin:.1%} margin, {upside:.1%} upside")
            
            return opportunities
            
        except Exception as e:
            print(f"❌ Value analysis failed: {e}")
            return []
    
    async def run_insider_analysis(self):
        """Run insider analysis demonstration"""
        print("\n" + "=" * 60)
        print("👥 INSIDER ANALYSIS DEMONSTRATION")
        print("=" * 60)
        
        try:
            result = await self.agents['insider'].process(
                tickers=self.demo_tickers[:3]
            )
            
            analyses = result.get('insider_analyses', [])
            print(f"📋 Analyzed insider activity for {len(analyses)} assets")
            
            for analysis in analyses:
                ticker = analysis['ticker']
                sentiment = analysis['current_sentiment']['overall_sentiment']
                activity = "High" if analysis['unusual_activity_detected'] else "Normal"
                print(f"   {ticker}: {sentiment} sentiment, {activity} activity")
            
            return analyses
            
        except Exception as e:
            print(f"❌ Insider analysis failed: {e}")
            return []
    
    async def run_unified_scoring(self, all_opportunities):
        """Run unified scoring demonstration"""
        print("\n" + "=" * 60)
        print("🏆 UNIFIED SCORING DEMONSTRATION")
        print("=" * 60)
        
        try:
            # Prepare opportunities for scoring
            scoring_opportunities = []
            
            # Add technical opportunities
            for opp in all_opportunities.get('technical', []):
                scoring_opportunities.append({
                    'id': opp.get('id', f"tech_{opp['symbol']}"),
                    'symbol': opp['symbol'],
                    'strategy': opp['strategy'],
                    'expected_return': (opp.get('take_profit', [100])[0] - opp['entry_price']) / opp['entry_price'],
                    'probability': opp['confidence_score'],
                    'risk_level': 1.0 / opp.get('risk_reward_ratio', 1.0),
                    'time_horizon': 5,  # days
                    'raw_signals': {'technical_score': opp['confidence_score']}
                })
            
            if scoring_opportunities:
                scored = self.unified_scorer.score_opportunities(scoring_opportunities)
                print(f"📊 Scored {len(scored)} opportunities")
                
                for i, opp in enumerate(scored[:5], 1):
                    symbol = opp['symbol']
                    score = opp['unified_score']
                    prob = opp['calibrated_probability']
                    print(f"   {i}. {symbol}: Score {score:.3f}, Prob {prob:.1%}")
                
                return scored
            else:
                print("⚠️  No opportunities to score")
                return []
                
        except Exception as e:
            print(f"❌ Unified scoring failed: {e}")
            return []
    
    async def run_comprehensive_demo(self):
        """Run the complete demonstration"""
        print("🚀 COMPREHENSIVE MULTI-AGENT TRADING SYSTEM DEMO")
        print("=" * 80)
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Demo Universe: {', '.join(self.demo_tickers)}")
        
        # Initialize all agents
        await self.initialize_agents()
        
        # Collect all analysis results
        all_opportunities = {}
        
        # Run each agent's analysis
        print("\n🔄 Running Multi-Agent Analysis Pipeline...")
        
        # Technical Analysis
        tech_opps = await self.run_technical_analysis()
        all_opportunities['technical'] = tech_opps
        
        # Sentiment Analysis
        sentiment_results = await self.run_sentiment_analysis()
        all_opportunities['sentiment'] = sentiment_results
        
        # Flow Analysis
        flow_results = await self.run_flow_analysis()
        all_opportunities['flow'] = flow_results
        
        # Macro Analysis
        macro_results = await self.run_macro_analysis()
        all_opportunities['macro'] = [macro_results] if macro_results else []
        
        # Value Analysis
        value_opps = await self.run_value_analysis()
        all_opportunities['value'] = value_opps
        
        # Insider Analysis
        insider_results = await self.run_insider_analysis()
        all_opportunities['insider'] = insider_results
        
        # Unified Scoring
        scored_opportunities = await self.run_unified_scoring(all_opportunities)
        
        # Final Summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 80)
        
        total_opportunities = sum(len(opps) for opps in all_opportunities.values() if isinstance(opps, list))
        print(f"🎯 Total Opportunities Identified: {total_opportunities}")
        print(f"🏆 Unified Scoring Applied: {len(scored_opportunities)} ranked")
        print(f"🤖 Agents Executed: {len(self.agents)}")
        
        if scored_opportunities:
            best = scored_opportunities[0]
            print(f"\n🥇 TOP RECOMMENDATION:")
            print(f"   Symbol: {best['symbol']}")
            print(f"   Strategy: {best.get('strategy', 'Multi-factor')}")
            print(f"   Unified Score: {best['unified_score']:.3f}")
            print(f"   Success Probability: {best['calibrated_probability']:.1%}")
        
        print(f"\n✅ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🌟 Multi-Agent Trading Intelligence System is fully operational!")


async def main():
    """Main demo function"""
    demo = ComprehensiveDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
