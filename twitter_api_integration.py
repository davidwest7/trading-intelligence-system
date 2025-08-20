#!/usr/bin/env python3
"""
Twitter/X API Integration
Real Twitter API integration to replace mock data and fix current issues
"""

import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

class TwitterAPIIntegration:
    """Real Twitter/X API integration"""
    
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        self.api_key = os.getenv('TWITTER_API_KEY', '')
        self.api_secret = os.getenv('TWITTER_API_SECRET', '')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN', '')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
        
        # API endpoints
        self.base_url = "https://api.twitter.com/2"
        self.v1_base_url = "https://api.twitter.com/1.1"
        
        # Rate limiting
        self.rate_limits = {
            'calls': 0,
            'limit': 300,  # Twitter API v2 limit per 15 minutes
            'reset_time': time.time() + 900  # 15 minutes
        }
    
    def _check_rate_limit(self):
        """Check if we're within rate limits"""
        if time.time() > self.rate_limits['reset_time']:
            self.rate_limits['calls'] = 0
            self.rate_limits['reset_time'] = time.time() + 900
        
        if self.rate_limits['calls'] >= self.rate_limits['limit']:
            return False
        
        self.rate_limits['calls'] += 1
        return True
    
    async def test_authentication(self):
        """Test Twitter API authentication"""
        print("🔑 Testing Twitter API authentication...")
        
        if not self.bearer_token:
            print("❌ Twitter Bearer Token not found!")
            print("📋 Please add TWITTER_BEARER_TOKEN to env_real_keys.env")
            return False
        
        # Test with a simple user lookup
        url = f"{self.base_url}/users/by/username/elonmusk"
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ Twitter API authentication successful!")
                        print(f"📊 Test user: {data.get('data', {}).get('name', 'Unknown')}")
                        return True
                    elif response.status == 401:
                        print("❌ Twitter Bearer Token is invalid")
                        return False
                    elif response.status == 403:
                        print("❌ Twitter API access forbidden - check permissions")
                        return False
                    else:
                        print(f"⚠️ Twitter API test failed: {response.status}")
                        return False
            except Exception as e:
                print(f"❌ Twitter API test error: {e}")
                return False
    
    async def search_tweets(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for tweets using Twitter API v2"""
        if not self._check_rate_limit():
            print("⚠️ Twitter API rate limit reached")
            return []
        
        if not self.bearer_token:
            print("❌ Twitter Bearer Token not found")
            return []
        
        url = f"{self.base_url}/tweets/search/recent"
        params = {
            'query': query,
            'max_results': min(max_results, 100),  # Twitter limit
            'tweet.fields': 'created_at,public_metrics,author_id,lang',
            'user.fields': 'username,name,verified',
            'expansions': 'author_id'
        }
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = []
                        
                        # Process tweets
                        for tweet in data.get('data', []):
                            # Get user info
                            user_info = {}
                            if 'includes' in data and 'users' in data['includes']:
                                for user in data['includes']['users']:
                                    if user['id'] == tweet.get('author_id'):
                                        user_info = user
                                        break
                            
                            tweets.append({
                                'id': tweet.get('id'),
                                'text': tweet.get('text'),
                                'author_id': tweet.get('author_id'),
                                'author_username': user_info.get('username'),
                                'author_name': user_info.get('name'),
                                'author_verified': user_info.get('verified', False),
                                'created_at': tweet.get('created_at'),
                                'lang': tweet.get('lang'),
                                'metrics': tweet.get('public_metrics', {}),
                                'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                                'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                                'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                                'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0)
                            })
                        
                        return tweets
                    else:
                        print(f"❌ Twitter search failed: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Twitter search error: {e}")
                return []
    
    async def get_user_tweets(self, username: str, max_results: int = 10) -> List[Dict]:
        """Get tweets from a specific user"""
        if not self._check_rate_limit():
            return []
        
        if not self.bearer_token:
            return []
        
        # First get user ID
        user_url = f"{self.base_url}/users/by/username/{username}"
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(user_url, headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        user_id = user_data.get('data', {}).get('id')
                        
                        if not user_id:
                            print(f"❌ User {username} not found")
                            return []
                        
                        # Get user's tweets
                        tweets_url = f"{self.base_url}/users/{user_id}/tweets"
                        params = {
                            'max_results': min(max_results, 100),
                            'tweet.fields': 'created_at,public_metrics,lang',
                            'exclude': 'retweets,replies'
                        }
                        
                        async with session.get(tweets_url, params=params, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                tweets = []
                                
                                for tweet in data.get('data', []):
                                    tweets.append({
                                        'id': tweet.get('id'),
                                        'text': tweet.get('text'),
                                        'author_username': username,
                                        'created_at': tweet.get('created_at'),
                                        'lang': tweet.get('lang'),
                                        'metrics': tweet.get('public_metrics', {}),
                                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                                        'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                                        'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                                        'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0)
                                    })
                                
                                return tweets
                            else:
                                print(f"❌ Failed to get tweets for {username}: {response.status}")
                                return []
                    else:
                        print(f"❌ User {username} lookup failed: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Error getting tweets for {username}: {e}")
                return []
    
    async def get_trending_topics(self) -> List[Dict]:
        """Get trending topics (using v1.1 API)"""
        if not self._check_rate_limit():
            return []
        
        # Note: This requires v1.1 API access
        url = f"{self.v1_base_url}/trends/place.json"
        params = {
            'id': 1  # Worldwide trends
        }
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        trends = []
                        
                        if data and len(data) > 0:
                            for trend in data[0].get('trends', [])[:10]:
                                trends.append({
                                    'name': trend.get('name'),
                                    'query': trend.get('query'),
                                    'tweet_volume': trend.get('tweet_volume'),
                                    'url': trend.get('url')
                                })
                        
                        return trends
                    else:
                        print(f"❌ Trending topics failed: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Trending topics error: {e}")
                return []
    
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment data from Twitter"""
        print(f"🐦 Getting Twitter sentiment for {symbol}...")
        
        if not await self.test_authentication():
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sources': {},
                'aggregated_sentiment': {},
                'tweets': [],
                'error': 'Authentication failed'
            }
        
        all_tweets = []
        sources_data = {}
        
        # Search queries for the symbol
        search_queries = [
            f'${symbol}',
            f'#{symbol}',
            f'{symbol} stock',
            f'{symbol} price',
            f'{symbol} trading'
        ]
        
        # Search for tweets
        for query in search_queries:
            try:
                tweets = await self.search_tweets(query, max_results=20)
                
                if tweets:
                    sources_data[query] = {
                        'count': len(tweets),
                        'tweets': tweets
                    }
                    all_tweets.extend(tweets)
                    print(f"✅ {query}: Found {len(tweets)} tweets")
                else:
                    print(f"⚠️ {query}: No tweets found")
                
                # Small delay to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Error searching for {query}: {e}")
                continue
        
        # Get tweets from financial influencers
        financial_users = [
            'CNBC', 'BloombergTV', 'MarketWatch', 'YahooFinance',
            'WSJmarkets', 'ReutersBiz', 'FTMarkets'
        ]
        
        for username in financial_users[:3]:  # Limit to avoid rate limits
            try:
                tweets = await self.get_user_tweets(username, max_results=5)
                
                if tweets:
                    # Filter for symbol mentions
                    symbol_tweets = [
                        tweet for tweet in tweets 
                        if symbol.lower() in tweet['text'].lower()
                    ]
                    
                    if symbol_tweets:
                        sources_data[f'user_{username}'] = {
                            'count': len(symbol_tweets),
                            'tweets': symbol_tweets
                        }
                        all_tweets.extend(symbol_tweets)
                        print(f"✅ @{username}: Found {len(symbol_tweets)} relevant tweets")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Error getting tweets from @{username}: {e}")
                continue
        
        # Create sentiment data structure
        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': sources_data,
            'aggregated_sentiment': {},
            'tweets': all_tweets
        }
        
        print(f"✅ Twitter sentiment: {len(all_tweets)} total tweets from {len(sources_data)} sources")
        return sentiment_data

async def main():
    """Test Twitter API integration"""
    print("🐦 Twitter/X API Integration Test")
    print("="*50)
    
    twitter = TwitterAPIIntegration()
    
    # Test authentication
    if not await twitter.test_authentication():
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Go to https://developer.twitter.com/en/portal/dashboard")
        print("2. Create a new app or use existing app")
        print("3. Go to 'Keys and Tokens' tab")
        print("4. Generate 'Bearer Token'")
        print("5. Add to env_real_keys.env:")
        print("   TWITTER_BEARER_TOKEN=your_bearer_token_here")
        print("\n📝 Note: Twitter API v2 requires Basic access (free) or higher")
        print("   Basic access allows 500,000 tweets/month")
        return
    
    # Test sentiment data collection
    sentiment_data = await twitter.get_sentiment_data('AAPL')
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Symbol: {sentiment_data['symbol']}")
    print(f"   Total Tweets: {len(sentiment_data['tweets'])}")
    print(f"   Sources: {len(sentiment_data['sources'])}")
    
    # Show sample tweets
    if sentiment_data['tweets']:
        print(f"\n📝 SAMPLE TWEETS:")
        for i, tweet in enumerate(sentiment_data['tweets'][:3]):
            author = tweet.get('author_username', 'Unknown')
            text = tweet.get('text', '')[:80]
            likes = tweet.get('like_count', 0)
            retweets = tweet.get('retweet_count', 0)
            print(f"   {i+1}. @{author}: {text}...")
            print(f"      ❤️ {likes} | 🔄 {retweets}")
    
    print(f"\n🎉 Twitter API integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())
