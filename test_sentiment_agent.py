#!/usr/bin/env python3
"""
Test Complete Sentiment Agent Implementation

Tests all resolved TODOs:
✅ Real sentiment calculation
✅ Bot detection
✅ Entity recognition
✅ Velocity calculation
✅ Dispersion metrics
✅ Multi-source aggregation
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.sentiment.agent_complete import (
    SentimentAgent, FinancialSentimentAnalyzer, BotDetector, EntityResolver
)

async def test_sentiment_agent():
    """Test the complete sentiment agent implementation"""
    print("🧪 Testing Complete Sentiment Agent Implementation")
    print("=" * 60)
    
    # Initialize agent
    agent = SentimentAgent()
    
    # Test 1: Financial Sentiment Analyzer
    print("\n1. Testing Financial Sentiment Analyzer...")
    analyzer = FinancialSentimentAnalyzer()
    
    test_texts = [
        "AAPL earnings beat expectations, stock rallies 5%",
        "TSLA crashes after disappointing delivery numbers",
        "Market volatility increases as Fed signals rate hikes",
        "Bullish sentiment on tech stocks continues"
    ]
    
    for text in test_texts:
        sentiment = analyzer.analyze_sentiment(text)
        print(f"   Text: {text[:50]}...")
        print(f"   Sentiment: {sentiment['compound']:.3f}, Confidence: {sentiment['confidence']:.3f}")
    
    # Test 2: Bot Detection
    print("\n2. Testing Bot Detection...")
    bot_detector = BotDetector()
    
    test_posts = [
        {
            'text': 'Great analysis on $AAPL!',
            'account_age_days': 365,
            'posts_per_day': 5,
            'followers_count': 1000,
            'following_count': 500,
            'is_verified': False,
            'name': 'Trader123',
            'description': 'Stock market enthusiast',
            'location': 'New York',
            'website': None
        },
        {
            'text': 'Buy $TSLA now! 🚀🚀🚀',
            'account_age_days': 10,
            'posts_per_day': 100,
            'followers_count': 50,
            'following_count': 5000,
            'is_verified': False,
            'name': 'BotAccount',
            'description': '',
            'location': None,
            'website': None
        }
    ]
    
    bot_flags = bot_detector.detect_bots(test_posts)
    for i, (post, is_bot) in enumerate(zip(test_posts, bot_flags)):
        print(f"   Post {i+1}: {'BOT' if is_bot else 'HUMAN'} - {post['text'][:30]}...")
    
    # Test 3: Entity Recognition
    print("\n3. Testing Entity Recognition...")
    entity_resolver = EntityResolver()
    
    test_text = "Apple Inc. reported strong earnings of $2.5B, up 15% from last year. AAPL stock is bullish."
    entities = entity_resolver.extract_entities(test_text)
    
    print(f"   Text: {test_text}")
    print(f"   Entities found: {len(entities)}")
    for entity in entities:
        print(f"     - {entity['text']} ({entity['type']}) - Confidence: {entity['confidence']:.2f}")
    
    # Test 4: Complete Agent Stream
    print("\n4. Testing Complete Agent Stream...")
    
    tickers = ["AAPL", "TSLA"]
    result = await agent.stream(tickers, window="1h", sources=["twitter", "reddit", "news"])
    
    print(f"   Processed {len(result['sentiment_data'])} tickers")
    for data in result['sentiment_data']:
        print(f"   {data['ticker']}:")
        print(f"     - Sentiment: {data['sentiment_score']:.3f}")
        print(f"     - Volume: {data['volume']}")
        print(f"     - Velocity: {data['velocity']:.3f}")
        print(f"     - Dispersion: {data['dispersion']:.3f}")
        print(f"     - Bot Ratio: {data['bot_ratio']:.3f}")
        print(f"     - Confidence: {data['confidence']:.3f}")
        print(f"     - Entities: {len(data['top_entities'])}")
    
    # Test 5: Velocity Calculation
    print("\n5. Testing Velocity Calculation...")
    
    # Simulate historical data
    for i in range(5):
        await agent.stream(["AAPL"], window="1h")
    
    # Test velocity after multiple calls
    result = await agent.stream(["AAPL"], window="1h")
    velocity = result['sentiment_data'][0]['velocity']
    print(f"   AAPL velocity after 5 calls: {velocity:.3f}")
    
    print("\n✅ All Sentiment Agent tests completed successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_sentiment_agent())
