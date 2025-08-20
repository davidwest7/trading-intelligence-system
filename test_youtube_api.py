#!/usr/bin/env python3
"""
Test YouTube API Integration
"""
import asyncio
import os
from dotenv import load_dotenv
from youtube_trends_integration import YouTubeFinancialMonitor

load_dotenv('env_real_keys.env')

async def test_youtube_api():
    print("🧪 Testing YouTube API Integration")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv('YOUTUBE_API_KEY')
    print(f"API Key Status: {'✅ Found' if api_key else '❌ Missing'}")
    if api_key:
        print(f"Key Format: {'✅ Valid' if len(api_key) > 30 else '❌ Invalid'}")
        print(f"Key Preview: {api_key[:10]}...")
    
    # Initialize monitor
    monitor = YouTubeFinancialMonitor()
    print(f"Monitor Status: {'✅ Ready' if monitor.api_key else '❌ Not Ready'}")
    
    # Test API call
    print("\n🔍 Testing YouTube API call...")
    try:
        result = await monitor.get_financial_videos('AAPL', 3)
        
        print(f"API Call Status: {result.get('status')}")
        
        if result.get('status') == 'WORKING':
            print(f"✅ Videos Found: {result.get('videos_found', 0)}")
            print(f"📊 Quota Used: {result.get('quota_used', 0)}/10000")
            
            # Show first video details
            videos = result.get('videos', [])
            if videos:
                first_video = videos[0]
                print(f"\n📺 First Video:")
                print(f"   Title: {first_video.get('title', 'N/A')[:50]}...")
                print(f"   Channel: {first_video.get('channel_title', 'N/A')}")
                print(f"   Published: {first_video.get('published_at', 'N/A')[:10]}")
        
        elif result.get('status') == 'QUOTA_EXCEEDED':
            print("⚠️ Daily quota exceeded")
        elif result.get('status') == 'ERROR':
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❓ Unexpected status: {result.get('status')}")
            
    except Exception as e:
        print(f"❌ Exception during API call: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_youtube_api())
