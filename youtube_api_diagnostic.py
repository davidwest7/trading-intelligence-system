#!/usr/bin/env python3
"""
YouTube API Diagnostic Tool
Help troubleshoot YouTube API setup issues
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

class YouTubeAPIDiagnostic:
    """Diagnose YouTube API setup issues"""
    
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
    async def diagnose_api_key(self):
        """Diagnose API key issues"""
        print("🔍 YouTube API Diagnostic")
        print("="*50)
        
        # Check if API key exists
        if not self.api_key:
            print("❌ No API key found in environment")
            return False
        
        print(f"✅ API key found: {self.api_key[:10]}...")
        
        # Test basic connectivity
        await self.test_basic_connectivity()
        
        # Test different endpoints
        await self.test_search_endpoint()
        await self.test_videos_endpoint()
        await self.test_channels_endpoint()
        
        return True
    
    async def test_basic_connectivity(self):
        """Test basic API connectivity"""
        print(f"\n🌐 Testing basic connectivity...")
        
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'q': 'test',
            'type': 'video',
            'maxResults': 1,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    print(f"   Status Code: {response.status}")
                    print(f"   Response Headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Success! Found {data.get('pageInfo', {}).get('totalResults', 0)} results")
                        return True
                    elif response.status == 403:
                        error_data = await response.json()
                        print(f"   ❌ 403 Error: {error_data}")
                        await self.analyze_403_error(error_data)
                        return False
                    elif response.status == 400:
                        error_data = await response.json()
                        print(f"   ❌ 400 Error: {error_data}")
                        await self.analyze_400_error(error_data)
                        return False
                    else:
                        print(f"   ❌ Unexpected status: {response.status}")
                        return False
                        
            except Exception as e:
                print(f"   ❌ Connection error: {e}")
                return False
    
    async def analyze_403_error(self, error_data):
        """Analyze 403 errors"""
        print(f"\n🔍 Analyzing 403 Error:")
        
        error_info = error_data.get('error', {})
        reason = error_info.get('errors', [{}])[0].get('reason', 'unknown')
        
        if reason == 'accessNotConfigured':
            print("   ❌ YouTube Data API v3 is not enabled!")
            print("   📋 Solution: Enable YouTube Data API v3 in Google Cloud Console")
            print("   🔗 Go to: APIs & Services → Library → Search 'YouTube Data API v3' → Enable")
        elif reason == 'quotaExceeded':
            print("   ❌ API quota exceeded!")
            print("   📋 Solution: Check quota usage or wait for reset")
            print("   🔗 Go to: APIs & Services → Quotas")
        elif reason == 'forbidden':
            print("   ❌ API key is invalid or restricted!")
            print("   📋 Solution: Check API key and restrictions")
            print("   🔗 Go to: APIs & Services → Credentials")
        else:
            print(f"   ❌ Unknown 403 reason: {reason}")
            print(f"   📋 Full error: {error_data}")
    
    async def analyze_400_error(self, error_data):
        """Analyze 400 errors"""
        print(f"\n🔍 Analyzing 400 Error:")
        print(f"   ❌ Bad request: {error_data}")
        
        error_info = error_data.get('error', {})
        message = error_info.get('message', '')
        
        if 'invalid' in message.lower():
            print("   📋 Solution: Check API key format and parameters")
        elif 'required' in message.lower():
            print("   📋 Solution: Missing required parameters")
        else:
            print("   📋 Solution: Check request format and parameters")
    
    async def test_search_endpoint(self):
        """Test search endpoint specifically"""
        print(f"\n🔍 Testing search endpoint...")
        
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'q': 'AAPL stock',
            'type': 'video',
            'maxResults': 1,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Search endpoint working")
                        print(f"   📊 Results: {data.get('pageInfo', {}).get('totalResults', 0)}")
                    else:
                        print(f"   ❌ Search endpoint failed: {response.status}")
            except Exception as e:
                print(f"   ❌ Search endpoint error: {e}")
    
    async def test_videos_endpoint(self):
        """Test videos endpoint specifically"""
        print(f"\n📹 Testing videos endpoint...")
        
        # Use a known video ID for testing
        test_video_id = "dQw4w9WgXcQ"  # Rick Roll (always available)
        
        url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet',
            'id': test_video_id,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        if items:
                            print(f"   ✅ Videos endpoint working")
                            print(f"   📊 Video title: {items[0]['snippet']['title']}")
                        else:
                            print(f"   ❌ No video data returned")
                    else:
                        print(f"   ❌ Videos endpoint failed: {response.status}")
            except Exception as e:
                print(f"   ❌ Videos endpoint error: {e}")
    
    async def test_channels_endpoint(self):
        """Test channels endpoint specifically"""
        print(f"\n📺 Testing channels endpoint...")
        
        # Use a known channel ID for testing
        test_channel_id = "UCBR8-60-B28hp2BmDPdntcQ"  # YouTube channel
        
        url = f"{self.base_url}/channels"
        params = {
            'part': 'snippet',
            'id': test_channel_id,
            'key': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        if items:
                            print(f"   ✅ Channels endpoint working")
                            print(f"   📊 Channel title: {items[0]['snippet']['title']}")
                        else:
                            print(f"   ❌ No channel data returned")
                    else:
                        print(f"   ❌ Channels endpoint failed: {response.status}")
            except Exception as e:
                print(f"   ❌ Channels endpoint error: {e}")
    
    async def provide_setup_instructions(self):
        """Provide detailed setup instructions"""
        print(f"\n📋 DETAILED SETUP INSTRUCTIONS:")
        print("="*50)
        
        print("1. GOOGLE CLOUD CONSOLE:")
        print("   🔗 https://console.cloud.google.com/")
        print("   📝 Sign in with your Google account")
        
        print("\n2. CREATE/SELECT PROJECT:")
        print("   📝 Create new project: 'Trading Sentiment Analysis'")
        print("   📝 Or select existing project")
        
        print("\n3. ENABLE YOUTUBE DATA API:")
        print("   📝 Go to: APIs & Services → Library")
        print("   📝 Search: 'YouTube Data API v3'")
        print("   📝 Click 'Enable'")
        print("   ⏱️ Wait 2-3 minutes for activation")
        
        print("\n4. VERIFY API KEY:")
        print("   📝 Go to: APIs & Services → Credentials")
        print("   📝 Check your API key exists")
        print("   📝 Verify it's not restricted too much")
        
        print("\n5. TEST AGAIN:")
        print("   📝 Run: python test_youtube_api_setup.py")
        
        print("\n🔍 COMMON ISSUES:")
        print("   ❌ API not enabled → Enable YouTube Data API v3")
        print("   ❌ Quota exceeded → Check usage or wait")
        print("   ❌ Key restricted → Check restrictions")
        print("   ❌ Wrong project → Select correct project")

async def main():
    """Main diagnostic function"""
    diagnostic = YouTubeAPIDiagnostic()
    
    # Run diagnostics
    await diagnostic.diagnose_api_key()
    
    # Provide setup instructions
    await diagnostic.provide_setup_instructions()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Follow the setup instructions above")
    print("2. Enable YouTube Data API v3")
    print("3. Test again with: python test_youtube_api_setup.py")
    print("4. Contact me if issues persist")

if __name__ == "__main__":
    asyncio.run(main())
