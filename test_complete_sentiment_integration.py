#!/usr/bin/env python3
"""
Complete Sentiment API Integration Test
Tests Twitter, Reddit, and News APIs working together
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.sentiment.agent_complete import SentimentAgent, TwitterAPIClient, RedditAPIClient, NewsAPIClient
from common.observability.telemetry import init_telemetry

async def test_individual_apis():
    """Test each API individually"""
    print("🧪 Testing Individual API Integrations")
    print("=" * 60)
    
    # Set up API credentials
    config = {
        'twitter_bearer_token': "AAAAAAAAAAAAAAAAAAAAAG%2BRzwEAAAAAaE4cyujI%2Ff3w745NUXBcdZI4XYQ%3DM9wbVqpz3XjlyTNvF7UVus9eaAmrf3oSqpTk0b1oHlSKkQYbiU",
        'reddit_client_id': "q-U8WOp6Efy8TYai8rcgGg",
        'reddit_client_secret': "XZDq0Ro6u1c0aoKcQ98x6bYmb-bLBQ",
        'news_api_key': "3b34e71a4c6547ce8af64e18a35305d1"
    }
    
    # Test Twitter API
    print("\n📱 Testing Twitter API...")
    try:
        twitter_client = TwitterAPIClient(config)
        twitter_connected = await twitter_client.connect()
        
        if twitter_connected:
            print("✅ Twitter API connected successfully")
            tweets = await twitter_client.search_tweets("AAPL", max_results=3)
            print(f"✅ Retrieved {len(tweets)} tweets")
            if tweets:
                print(f"   Sample tweet: {tweets[0]['text'][:100]}...")
        else:
            print("❌ Failed to connect to Twitter API")
    except Exception as e:
        print(f"❌ Twitter API error: {e}")
    
    # Test Reddit API
    print("\n🔴 Testing Reddit API...")
    try:
        reddit_client = RedditAPIClient(config)
        reddit_connected = await reddit_client.connect()
        
        if reddit_connected:
            print("✅ Reddit API connected successfully")
            posts = await reddit_client.search_posts("AAPL", ["investing", "stocks"], limit=3)
            print(f"✅ Retrieved {len(posts)} posts")
            if posts:
                print(f"   Sample post: {posts[0]['text'][:100]}...")
        else:
            print("❌ Failed to connect to Reddit API")
    except Exception as e:
        print(f"❌ Reddit API error: {e}")
    
    # Test News API
    print("\n📰 Testing News API...")
    try:
        news_client = NewsAPIClient(config)
        news_connected = await news_client.connect()
        
        if news_connected:
            print("✅ News API connected successfully")
            articles = await news_client.search_articles("AAPL", max_results=3)
            print(f"✅ Retrieved {len(articles)} articles")
            if articles:
                print(f"   Sample article: {articles[0]['text'][:100]}...")
        else:
            print("❌ Failed to connect to News API")
    except Exception as e:
        print(f"❌ News API error: {e}")

async def test_complete_sentiment_agent():
    """Test the complete sentiment agent with all APIs"""
    print("\n🧪 Testing Complete Sentiment Agent Integration")
    print("=" * 60)
    
    try:
        # Initialize telemetry
        telemetry_config = {
            'service_name': 'test_complete_sentiment',
            'environment': 'test'
        }
        init_telemetry(telemetry_config)
        
        # Set up configuration with all API keys
        config = {
            'twitter_bearer_token': "AAAAAAAAAAAAAAAAAAAAAG%2BRzwEAAAAAaE4cyujI%2Ff3w745NUXBcdZI4XYQ%3DM9wbVqpz3XjlyTNvF7UVus9eaAmrf3oSqpTk0b1oHlSKkQYbiU",
            'reddit_client_id': "q-U8WOp6Efy8TYai8rcgGg",
            'reddit_client_secret': "XZDq0Ro6u1c0aoKcQ98x6bYmb-bLBQ",
            'news_api_key': "3b34e71a4c6547ce8af64e18a35305d1",
            'symbols': ['AAPL', 'TSLA', 'NVDA']
        }
        
        # Initialize the sentiment agent
        print("🤖 Initializing Sentiment Agent with all APIs...")
        agent = SentimentAgent(config)
        
        # Connect to all APIs
        print("🔗 Connecting to all social media APIs...")
        initialized = await agent.initialize()
        
        if initialized:
            print("✅ Sentiment Agent initialized successfully with all APIs!")
            
            # Check which APIs are connected
            print("\n📊 API Connection Status:")
            print(f"   Twitter: {'✅' if agent.twitter_client.is_connected else '❌'}")
            print(f"   Reddit:  {'✅' if agent.reddit_client.is_connected else '❌'}")
            print(f"   News:    {'✅' if agent.news_client.is_connected else '❌'}")
            
            # Generate signals
            print("\n📈 Generating sentiment signals...")
            signals = await agent.generate_signals()
            
            if signals:
                print(f"✅ Generated {len(signals)} sentiment signals!")
                print("\n📋 Signal Details:")
                for i, signal in enumerate(signals, 1):
                    print(f"\n{i}. {signal.symbol}")
                    print(f"   Agent Type: {signal.agent_type}")
                    print(f"   Expected Return (μ): {signal.mu:.4f}")
                    print(f"   Uncertainty (σ): {signal.sigma:.4f}")
                    print(f"   Confidence: {signal.confidence:.4f}")
                    print(f"   Direction: {signal.direction}")
                    print(f"   Regime: {signal.regime}")
                    print(f"   Horizon: {signal.horizon}")
                    print(f"   Sources: {signal.metadata.get('sources', [])}")
                    print(f"   Total Posts: {signal.metadata.get('total_posts', 0)}")
                    
                    # Sentiment distribution
                    dist = signal.metadata.get('sentiment_distribution', {})
                    print(f"   Sentiment: +{dist.get('positive', 0)} -{dist.get('negative', 0)} ={dist.get('neutral', 0)}")
            else:
                print("⚠️ No signals generated")
        else:
            print("❌ Failed to initialize Sentiment Agent")
            
    except Exception as e:
        print(f"❌ Error testing complete sentiment agent: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 Complete Sentiment API Integration Test")
    print("=" * 70)
    print("Testing Twitter, Reddit, and News APIs together")
    print("=" * 70)
    
    # Test 1: Individual API tests
    await test_individual_apis()
    
    # Test 2: Complete sentiment agent
    await test_complete_sentiment_agent()
    
    print("\n🎉 Complete Sentiment API Integration Test Finished!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
