#!/usr/bin/env python3
"""
YouTube API Setup Test
Test your YouTube API key and show basic functionality
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

class YouTubeAPITester:
    """Test YouTube API setup and functionality"""
    
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
    async def test_api_key(self):
        """Test if the API key is valid"""
        print("🔑 Testing YouTube API Key...")
        
        if not self.api_key or self.api_key == 'your_youtube_api_key_here':
            print("❌ No YouTube API key found!")
            print("📋 Please add your YouTube API key to env_real_keys.env")
            print("   Format: YOUTUBE_API_KEY=your_actual_api_key_here")
            return False
        
        # Test with a simple search
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'q': 'AAPL stock analysis',
            'type': 'video',
            'maxResults': 1,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ YouTube API Key is valid!")
                        print(f"📊 Status: {data.get('pageInfo', {}).get('totalResults', 0)} results found")
                        return True
                    elif response.status == 403:
                        print("❌ YouTube API Key is invalid or quota exceeded")
                        return False
                    else:
                        print(f"⚠️ Unexpected status: {response.status}")
                        return False
            except Exception as e:
                print(f"❌ Error testing API key: {e}")
                return False
    
    async def test_search_functionality(self, symbol: str = 'AAPL'):
        """Test search functionality"""
        print(f"\n🔍 Testing search for {symbol}...")
        
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'q': f'{symbol} stock analysis',
            'type': 'video',
            'order': 'relevance',
            'maxResults': 5,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        
                        print(f"✅ Found {len(items)} videos for {symbol}")
                        
                        # Display first few videos
                        for i, item in enumerate(items[:3], 1):
                            snippet = item['snippet']
                            print(f"\n📹 Video {i}:")
                            print(f"   Title: {snippet.get('title', 'N/A')}")
                            print(f"   Channel: {snippet.get('channelTitle', 'N/A')}")
                            print(f"   Published: {snippet.get('publishedAt', 'N/A')}")
                            print(f"   Description: {snippet.get('description', 'N/A')[:100]}...")
                        
                        return items
                    else:
                        print(f"❌ Search failed: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Error in search: {e}")
                return []
    
    async def test_video_details(self, video_id: str):
        """Test getting video details"""
        print(f"\n📊 Testing video details for {video_id}...")
        
        url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet,statistics',
            'id': video_id,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        
                        if items:
                            video = items[0]
                            snippet = video['snippet']
                            stats = video.get('statistics', {})
                            
                            print(f"✅ Video details retrieved:")
                            print(f"   Title: {snippet.get('title', 'N/A')}")
                            print(f"   Views: {stats.get('viewCount', 'N/A')}")
                            print(f"   Likes: {stats.get('likeCount', 'N/A')}")
                            print(f"   Comments: {stats.get('commentCount', 'N/A')}")
                            
                            return video
                        else:
                            print("❌ No video details found")
                            return None
                    else:
                        print(f"❌ Video details failed: {response.status}")
                        return None
            except Exception as e:
                print(f"❌ Error getting video details: {e}")
                return None
    
    async def test_comments(self, video_id: str):
        """Test getting video comments"""
        print(f"\n💬 Testing comments for {video_id}...")
        
        url = f"{self.base_url}/commentThreads"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': 5,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        
                        print(f"✅ Found {len(items)} comments")
                        
                        # Display first few comments
                        for i, item in enumerate(items[:3], 1):
                            comment = item['snippet']['topLevelComment']['snippet']
                            print(f"\n💭 Comment {i}:")
                            print(f"   Author: {comment.get('authorDisplayName', 'N/A')}")
                            print(f"   Text: {comment.get('textDisplay', 'N/A')[:100]}...")
                            print(f"   Likes: {comment.get('likeCount', 'N/A')}")
                        
                        return items
                    else:
                        print(f"❌ Comments failed: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Error getting comments: {e}")
                return []
    
    async def test_rate_limits(self):
        """Test rate limiting information"""
        print(f"\n⏱️ Testing rate limits...")
        
        # Make multiple requests to check rate limits
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'q': 'AAPL',
            'type': 'video',
            'maxResults': 1,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            for i in range(3):
                try:
                    async with session.get(url, params=params) as response:
                        print(f"   Request {i+1}: Status {response.status}")
                        
                        # Check rate limit headers
                        remaining = response.headers.get('X-RateLimit-Remaining', 'Unknown')
                        reset = response.headers.get('X-RateLimit-Reset', 'Unknown')
                        
                        if remaining != 'Unknown':
                            print(f"   Remaining requests: {remaining}")
                        if reset != 'Unknown':
                            print(f"   Reset time: {reset}")
                            
                except Exception as e:
                    print(f"   Request {i+1} failed: {e}")
                
                # Small delay between requests
                await asyncio.sleep(1)

async def main():
    """Main test function"""
    print("🎬 YouTube API Setup Test")
    print("="*50)
    
    tester = YouTubeAPITester()
    
    # Test API key
    if not await tester.test_api_key():
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable YouTube Data API v3")
        print("4. Create API credentials (API Key)")
        print("5. Add the key to env_real_keys.env")
        print("6. Run this test again")
        return
    
    # Test search functionality
    videos = await tester.test_search_functionality('AAPL')
    
    if videos:
        # Test video details with first video
        video_id = videos[0]['id']['videoId']
        await tester.test_video_details(video_id)
        
        # Test comments
        await tester.test_comments(video_id)
    
    # Test rate limits
    await tester.test_rate_limits()
    
    print(f"\n🎉 YouTube API setup test complete!")
    print(f"✅ Ready to integrate with sentiment analysis system")

if __name__ == "__main__":
    asyncio.run(main())
