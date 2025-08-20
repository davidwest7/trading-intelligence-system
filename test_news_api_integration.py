#!/usr/bin/env python3
"""
Test News API Integration
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.sentiment.agent_complete import NewsAPIClient
from common.observability.telemetry import init_telemetry

async def test_news_api():
    """Test News API integration"""
    print("🧪 Testing News API Integration")
    print("=" * 50)
    
    # Set up the API key
    api_key = "3b34e71a4c6547ce8af64e18a35305d1"
    os.environ['NEWS_API_KEY'] = api_key
    
    config = {
        'news_api_key': api_key
    }
    
    try:
        # Initialize the client
        print("📡 Initializing News API client...")
        client = NewsAPIClient(config)
        
        # Test connection
        print("🔗 Testing connection...")
        connected = await client.connect()
        
        if connected:
            print("✅ News API connection successful!")
            
            # Test search functionality
            print("🔍 Testing search functionality...")
            articles = await client.search_articles("AAPL", max_results=5)
            
            if articles:
                print(f"✅ Found {len(articles)} articles")
                print("\n📰 Sample articles:")
                for i, article in enumerate(articles[:3], 1):
                    print(f"\n{i}. {article.get('source', 'Unknown')}")
                    print(f"   Title: {article.get('text', 'No title')[:100]}...")
                    print(f"   Published: {article.get('published_at', 'Unknown')}")
                    print(f"   Author: {article.get('author', 'Unknown')}")
            else:
                print("⚠️ No articles found")
        else:
            print("❌ News API connection failed")
            
    except Exception as e:
        print(f"❌ Error testing News API: {e}")
        import traceback
        traceback.print_exc()

async def test_sentiment_agent_with_news():
    """Test the full sentiment agent with News API"""
    print("\n🧪 Testing Sentiment Agent with News API")
    print("=" * 50)
    
    try:
        # Initialize telemetry first
        telemetry_config = {
            'service_name': 'test_news_api',
            'environment': 'test'
        }
        init_telemetry(telemetry_config)
        
        from agents.sentiment.agent_complete import SentimentAgent
        
        # Initialize the agent with symbols
        config = {
            'news_api_key': "3b34e71a4c6547ce8af64e18a35305d1",
            'symbols': ['AAPL', 'TSLA']  # Set symbols in config
        }
        
        agent = SentimentAgent(config)
        
        # Initialize the agent (connect to APIs)
        print("🔗 Initializing sentiment agent...")
        initialized = await agent.initialize()
        
        if initialized:
            print("✅ Sentiment agent initialized successfully")
            
            # Test signal generation
            print("📊 Testing signal generation...")
            signals = await agent.generate_signals()  # No parameters needed
            
            if signals:
                print(f"✅ Generated {len(signals)} signals")
                for signal in signals:
                    print(f"   - {signal.symbol}: {signal.agent_type} ({signal.confidence:.2f})")
            else:
                print("⚠️ No signals generated")
        else:
            print("❌ Failed to initialize sentiment agent")
            
    except Exception as e:
        print(f"❌ Error testing sentiment agent: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 Starting News API Integration Tests")
    print("=" * 60)
    
    # Test 1: Direct News API
    await test_news_api()
    
    # Test 2: Sentiment Agent integration
    await test_sentiment_agent_with_news()
    
    print("\n✅ News API Integration Tests Complete!")

if __name__ == "__main__":
    asyncio.run(main())
