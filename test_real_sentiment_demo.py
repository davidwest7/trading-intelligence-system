#!/usr/bin/env python3
"""
Real Data Sentiment Agent Demo
Demonstrates the sentiment agent using real Twitter and Reddit data
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append('.')

from agents.sentiment.agent_real_data import RealDataSentimentAgent

async def demo_real_sentiment():
    """Demo the real data sentiment agent"""
    
    print("🧠 REAL DATA SENTIMENT AGENT DEMONSTRATION")
    print("=" * 60)
    print("Using actual Twitter and Reddit APIs for sentiment analysis")
    print("=" * 60)
    
    # Initialize the real data sentiment agent
    print("\n🚀 Initializing Real Data Sentiment Agent...")
    agent = RealDataSentimentAgent()
    
    # Test tickers
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    print(f"\n📊 Analyzing sentiment for: {', '.join(test_tickers)}")
    print("⏳ This may take a few moments as we fetch real data...")
    
    # Run sentiment analysis
    start_time = asyncio.get_event_loop().time()
    results = await agent.analyze_sentiment_optimized(test_tickers)
    processing_time = asyncio.get_event_loop().time() - start_time
    
    print(f"\n✅ Analysis Complete! ({processing_time:.2f}s)")
    print("=" * 60)
    
    # Display results
    print("\n📈 SENTIMENT ANALYSIS RESULTS:")
    print("-" * 40)
    
    # Overall sentiment
    sentiment_analysis = results['sentiment_analysis']
    print(f"🎯 Overall Sentiment Score: {sentiment_analysis['overall_score']:.3f}")
    print(f"📊 Confidence Level: {sentiment_analysis['confidence']:.1%}")
    print(f"🌍 Market Impact: {sentiment_analysis['market_impact'].upper()}")
    
    # Sentiment distribution
    distribution = sentiment_analysis['sentiment_distribution']
    print(f"\n📊 Sentiment Distribution:")
    print(f"   Positive: {distribution['positive']:.1%}")
    print(f"   Negative: {distribution['negative']:.1%}")
    print(f"   Neutral:  {distribution['neutral']:.1%}")
    
    # Source breakdown
    source_breakdown = sentiment_analysis['source_breakdown']
    if source_breakdown:
        print(f"\n📱 Data Sources:")
        for source, data in source_breakdown.items():
            print(f"   {source.upper()}: {data['count']} posts, avg sentiment: {data['avg_sentiment']:.3f}")
    
    # Summary statistics
    summary = results['summary']
    print(f"\n📋 SUMMARY STATISTICS:")
    print(f"   Total Posts Analyzed: {summary['total_posts_analyzed']}")
    print(f"   Tickers Analyzed: {summary['total_tickers']}")
    print(f"   Average Sentiment: {summary['average_sentiment']:.3f}")
    
    # Top sources
    top_sources = summary['top_sentiment_sources']
    if top_sources:
        print(f"   Top Data Sources: {', '.join(top_sources)}")
    
    # Sentiment signals
    signals = results['sentiment_signals']
    if signals:
        print(f"\n🚨 SENTIMENT SIGNALS GENERATED:")
        for i, signal in enumerate(signals, 1):
            print(f"   {i}. {signal['signal_type'].replace('_', ' ').title()}")
            print(f"      Strength: {signal['strength']:.3f}")
            print(f"      Confidence: {signal['confidence']:.1%}")
            print(f"      Description: {signal['description']}")
    
    # Sample posts
    posts = results['sentiment_posts']
    if posts:
        print(f"\n📝 SAMPLE POSTS (showing first 3):")
        for i, post in enumerate(posts[:3], 1):
            source = post['source'].upper()
            sentiment = post['sentiment_score']
            text_preview = post['text'][:100] + "..." if len(post['text']) > 100 else post['text']
            print(f"   {i}. [{source}] Sentiment: {sentiment:.3f}")
            print(f"      {text_preview}")
            print(f"      Author: {post['author']}, Reach: {post['reach']}")
    
    # Processing info
    processing_info = results['processing_info']
    print(f"\n⚡ PROCESSING INFORMATION:")
    print(f"   Processing Time: {processing_info['processing_time']:.2f}s")
    print(f"   Cache Hit Rate: {processing_info['cache_hit_rate']:.1%}")
    print(f"   Data Sources Used: {', '.join(processing_info['data_sources'])}")
    
    # Agent metrics
    print(f"\n📊 AGENT METRICS:")
    print(f"   Total Posts Analyzed (lifetime): {agent.metrics['total_posts_analyzed']}")
    print(f"   Signals Generated (lifetime): {agent.metrics['sentiment_signals_generated']}")
    print(f"   Average Processing Time: {agent.metrics['processing_time_avg']:.2f}s")
    
    print("\n" + "=" * 60)
    print("🎉 Real Data Sentiment Analysis Complete!")
    print("✅ Successfully integrated Twitter and Reddit APIs")
    print("✅ Real-time sentiment analysis working")
    print("✅ Production-ready sentiment agent")
    print("=" * 60)

async def main():
    """Main demo function"""
    try:
        await demo_real_sentiment()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
