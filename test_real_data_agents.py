#!/usr/bin/env python3
"""
Real Data Agents Test
Tests all agents with real Polygon.io data integration
"""
import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

load_dotenv('env_real_keys.env')

async def test_real_data_agents():
    """Test all real data agents"""
    print("🚀 **REAL DATA AGENTS TEST**")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("Testing 6 agents with Polygon.io real market data")
    print("=" * 60)
    
    # Test configuration
    config = {
        'polygon_api_key': os.getenv('POLYGON_API_KEY'),
        'twitter_api_key': os.getenv('TWITTER_API_KEY'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID')
    }
    
    test_tickers = ['AAPL', 'TSLA', 'SPY', 'QQQ']
    
    results = {}
    
    # ==================== 1. TECHNICAL AGENT ====================
    print("\n🔧 **1. REAL DATA TECHNICAL AGENT**")
    print("-" * 40)
    try:
        from agents.technical.agent_real_data import RealDataTechnicalAgent
        technical_agent = RealDataTechnicalAgent(config)
        technical_result = await technical_agent.analyze_technical_indicators(test_tickers)
        
        print(f"✅ Technical Agent: Analyzed {technical_result['tickers_analyzed']} tickers")
        print(f"   Overall Sentiment: {technical_result['overall_signals']['overall_sentiment']}")
        print(f"   Total Signals: {technical_result['overall_signals']['total_signals']}")
        
        # Show sample technical analysis
        for ticker in test_tickers[:2]:
            if ticker in technical_result['technical_analysis']:
                analysis = technical_result['technical_analysis'][ticker]
                print(f"   {ticker}: ${analysis['current_price']:.2f} (RSI: {analysis.get('rsi', 0):.1f})")
        
        results['technical'] = technical_result
        
    except Exception as e:
        print(f"❌ Technical Agent failed: {e}")
        results['technical'] = None
    
    # ==================== 2. FLOW AGENT ====================
    print("\n🌊 **2. REAL DATA FLOW AGENT**")
    print("-" * 40)
    try:
        from agents.flow.agent_real_data import RealDataFlowAgent
        flow_agent = RealDataFlowAgent(config)
        flow_result = await flow_agent.analyze_market_flow(test_tickers)
        
        print(f"✅ Flow Agent: Analyzed {flow_result['tickers_analyzed']} tickers")
        print(f"   Overall Regime: {flow_result['overall_flow']['overall_regime']}")
        print(f"   Total Volume: {flow_result['overall_flow']['total_volume']:,}")
        
        # Show sample flow analysis
        for ticker in test_tickers[:2]:
            if ticker in flow_result['flow_analysis']:
                analysis = flow_result['flow_analysis'][ticker]
                print(f"   {ticker}: Spread {analysis['spread_percentage']:.3f}% (Flow: {analysis['flow_regime']})")
        
        results['flow'] = flow_result
        
    except Exception as e:
        print(f"❌ Flow Agent failed: {e}")
        results['flow'] = None
    
    # ==================== 3. TOP PERFORMERS AGENT ====================
    print("\n🏆 **3. REAL DATA TOP PERFORMERS AGENT**")
    print("-" * 40)
    try:
        from agents.top_performers.agent_real_data import RealDataTopPerformersAgent
        top_performers_agent = RealDataTopPerformersAgent(config)
        top_performers_result = await top_performers_agent.analyze_top_performers()
        
        print(f"✅ Top Performers Agent: Successfully analyzed")
        print(f"   Top Gainers: {len(top_performers_result['top_gainers'])} stocks")
        print(f"   Momentum Regime: {top_performers_result['momentum_analysis']['momentum_regime']}")
        
        # Show top gainers
        if top_performers_result['top_gainers']:
            top_gainer = top_performers_result['top_gainers'][0]
            print(f"   #1 Gainer: {top_gainer['symbol']} +{top_gainer['change_percent']:.1f}%")
        
        results['top_performers'] = top_performers_result
        
    except Exception as e:
        print(f"❌ Top Performers Agent failed: {e}")
        results['top_performers'] = None
    
    # ==================== 4. UNDERVALUED AGENT ====================
    print("\n💎 **4. REAL DATA UNDERVALUED AGENT**")
    print("-" * 40)
    try:
        from agents.undervalued.agent_real_data import RealDataUndervaluedAgent
        undervalued_agent = RealDataUndervaluedAgent(config)
        undervalued_result = await undervalued_agent.analyze_undervalued_stocks(test_tickers)
        
        print(f"✅ Undervalued Agent: Analyzed {undervalued_result['tickers_analyzed']} tickers")
        print(f"   Market Value Regime: {undervalued_result['overall_value']['market_value_regime']}")
        print(f"   High Value Count: {undervalued_result['overall_value']['high_value_count']}")
        
        # Show value opportunities
        if undervalued_result['overall_value']['value_opportunities']:
            top_opportunity = undervalued_result['overall_value']['value_opportunities'][0]
            print(f"   Top Value: {top_opportunity['ticker']} (Score: {top_opportunity['value_score']:.0f})")
        
        results['undervalued'] = undervalued_result
        
    except Exception as e:
        print(f"❌ Undervalued Agent failed: {e}")
        results['undervalued'] = None
    
    # ==================== 5. MACRO AGENT ====================
    print("\n🌍 **5. REAL DATA MACRO AGENT**")
    print("-" * 40)
    try:
        from agents.macro.agent_real_data import RealDataMacroAgent
        macro_agent = RealDataMacroAgent(config)
        macro_result = await macro_agent.analyze_macro_environment()
        
        print(f"✅ Macro Agent: Successfully analyzed")
        print(f"   Market Breadth: {macro_result['macro_trends']['market_breadth']['breadth_ratio']:.1%}")
        print(f"   Risk Sentiment: {macro_result['macro_trends']['risk_sentiment']['sentiment']}")
        
        # Show economic indicators
        indicators = macro_result['economic_indicators']
        if 'sp500' in indicators and indicators['sp500']:
            print(f"   SP500: {indicators['sp500']['change_percent']:.2f}%")
        
        results['macro'] = macro_result
        
    except Exception as e:
        print(f"❌ Macro Agent failed: {e}")
        results['macro'] = None
    
    # ==================== 6. SENTIMENT AGENT ====================
    print("\n📊 **6. REAL DATA SENTIMENT AGENT**")
    print("-" * 40)
    try:
        from agents.sentiment.agent_real_data import RealDataSentimentAgent
        sentiment_agent = RealDataSentimentAgent(config)
        sentiment_result = await sentiment_agent.analyze_sentiment_optimized(test_tickers)
        
        print(f"✅ Sentiment Agent: Analyzed {len(sentiment_result['sentiment_data'])} tickers")
        print(f"   Overall Sentiment: {sentiment_result['overall_sentiment']}")
        print(f"   Confidence: {sentiment_result['confidence']:.1%}")
        
        # Show sample sentiment
        for ticker in test_tickers[:2]:
            if ticker in sentiment_result['sentiment_data']:
                data = sentiment_result['sentiment_data'][ticker]
                print(f"   {ticker}: Score {data['overall_score']:.2f} ({data['market_impact']})")
        
        results['sentiment'] = sentiment_result
        
    except Exception as e:
        print(f"❌ Sentiment Agent failed: {e}")
        results['sentiment'] = None
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("📊 **REAL DATA AGENTS TEST SUMMARY**")
    print("=" * 60)
    
    successful_agents = sum(1 for result in results.values() if result is not None)
    total_agents = len(results)
    
    print(f"✅ Successful Agents: {successful_agents}/{total_agents}")
    print(f"📈 Coverage: {successful_agents/total_agents*100:.0f}%")
    print(f"🌐 Data Source: Polygon.io (Real Market Data)")
    
    # Agent status
    agent_status = {
        'Technical': '✅' if results['technical'] else '❌',
        'Flow': '✅' if results['flow'] else '❌',
        'Top Performers': '✅' if results['top_performers'] else '❌',
        'Undervalued': '✅' if results['undervalued'] else '❌',
        'Macro': '✅' if results['macro'] else '❌',
        'Sentiment': '✅' if results['sentiment'] else '❌'
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
    print("1. Integrate agents into main system")
    print("2. Update Streamlit dashboard with real data")
    print("3. Deploy to production")
    print("4. Add remaining APIs (NewsAPI, Quiver)")
    print("=" * 60)
    
    return results

async def main():
    """Main test function"""
    print("🚀 Starting Real Data Agents Test...")
    
    try:
        results = await test_real_data_agents()
        return results
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())
