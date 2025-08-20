#!/usr/bin/env python3
"""
Debug YouTube API Call
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

async def debug_youtube_api():
    print("🔍 Debugging YouTube API Call")
    print("=" * 40)
    
    api_key = os.getenv('YOUTUBE_API_KEY')
    print(f"API Key: {api_key[:10]}..." if api_key else "None")
    
    # Test a simple search request
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': 'AAPL stock',
        'type': 'video',
        'maxResults': 1,
        'key': api_key
    }
    
    print(f"\n🔗 URL: {url}")
    print(f"📋 Params: {params}")
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"\n📡 Making request...")
            async with session.get(url, params=params) as response:
                print(f"📊 Status Code: {response.status}")
                print(f"📋 Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Success! Found {len(data.get('items', []))} items")
                    if data.get('items'):
                        item = data['items'][0]
                        print(f"📺 First video: {item['snippet']['title']}")
                elif response.status == 403:
                    error_data = await response.text()
                    print(f"❌ 403 Error - Response: {error_data}")
                    
                    # Check specific error details
                    try:
                        error_json = await response.json()
                        print(f"🔍 Error Details: {error_json}")
                    except:
                        print("Could not parse error as JSON")
                        
                else:
                    error_data = await response.text()
                    print(f"❌ Error {response.status}: {error_data}")
                    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    asyncio.run(debug_youtube_api())
