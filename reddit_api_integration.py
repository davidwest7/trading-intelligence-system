#!/usr/bin/env python3
"""
Reddit API Integration
Real Reddit API integration to replace mock data
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

class RedditAPIIntegration:
    """Real Reddit API integration"""
    
    def __init__(self):
        self.client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.user_agent = 'TradingSentimentBot/1.0'
        self.access_token = None
        self.token_expiry = None
        
        # Financial subreddits for trading sentiment
        self.financial_subreddits = [
            'investing', 'stocks', 'wallstreetbets', 'StockMarket',
            'ValueInvesting', 'CryptoCurrency', 'Bitcoin', 'Ethereum',
            'Options', 'DayTrading', 'SwingTrading'
        ]
        
        # Rate limiting
        self.rate_limits = {
            'calls': 0,
            'limit': 1000,  # Reddit API limit
            'reset_time': time.time() + 3600  # 1 hour
        }
    
    async def authenticate(self):
        """Authenticate with Reddit API"""
        if not self.client_id or not self.client_secret:
            print("❌ Reddit API credentials not found!")
            print("📋 Please add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to env_real_keys.env")
            return False
        
        # Check if we have a valid token
        if self.access_token and self.token_expiry and time.time() < self.token_expiry:
            return True
        
        # Get new access token
        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth_data = {
            'grant_type': 'client_credentials'
        }
        
        headers = {
            'User-Agent': self.user_agent
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    auth_url,
                    data=auth_data,
                    headers=headers,
                    auth=aiohttp.BasicAuth(self.client_id, self.client_secret)
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data.get('access_token')
                        self.token_expiry = time.time() + token_data.get('expires_in', 3600)
                        print("✅ Reddit API authenticated successfully!")
                        return True
                    else:
                        print(f"❌ Reddit authentication failed: {response.status}")
                        return False
            except Exception as e:
                print(f"❌ Reddit authentication error: {e}")
                return False
    
    def _check_rate_limit(self):
        """Check if we're within rate limits"""
        if time.time() > self.rate_limits['reset_time']:
            self.rate_limits['calls'] = 0
            self.rate_limits['reset_time'] = time.time() + 3600
        
        if self.rate_limits['calls'] >= self.rate_limits['limit']:
            return False
        
        self.rate_limits['calls'] += 1
        return True
    
    async def search_subreddit(self, subreddit: str, query: str, limit: int = 10) -> List[Dict]:
        """Search for posts in a specific subreddit"""
        if not self._check_rate_limit():
            print(f"⚠️ Rate limit reached for Reddit API")
            return []
        
        if not self.access_token:
            if not await self.authenticate():
                return []
        
        url = f"https://oauth.reddit.com/r/{subreddit}/search"
        params = {
            'q': query,
            'restrict_sr': 'on',
            'sort': 'new',
            't': 'day',
            'limit': limit
        }
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = []
                        
                        for post in data.get('data', {}).get('children', []):
                            post_data = post['data']
                            posts.append({
                                'id': post_data.get('id'),
                                'title': post_data.get('title'),
                                'content': post_data.get('selftext'),
                                'subreddit': subreddit,
                                'author': post_data.get('author'),
                                'score': post_data.get('score', 0),
                                'upvote_ratio': post_data.get('upvote_ratio', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'created_utc': post_data.get('created_utc'),
                                'url': post_data.get('url'),
                                'permalink': post_data.get('permalink')
                            })
                        
                        return posts
                    else:
                        print(f"❌ Reddit search failed for r/{subreddit}: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Reddit search error for r/{subreddit}: {e}")
                return []
    
    async def get_comments(self, post_id: str, limit: int = 10) -> List[Dict]:
        """Get comments for a specific post"""
        if not self._check_rate_limit():
            return []
        
        if not self.access_token:
            if not await self.authenticate():
                return []
        
        url = f"https://oauth.reddit.com/comments/{post_id}"
        params = {
            'limit': limit,
            'depth': 1
        }
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        comments = []
                        
                        if len(data) > 1:  # Comments are in the second element
                            comment_data = data[1]['data']['children']
                            for comment in comment_data:
                                if comment['kind'] == 't1':  # Regular comment
                                    comment_info = comment['data']
                                    comments.append({
                                        'id': comment_info.get('id'),
                                        'body': comment_info.get('body'),
                                        'author': comment_info.get('author'),
                                        'score': comment_info.get('score', 0),
                                        'created_utc': comment_info.get('created_utc')
                                    })
                        
                        return comments
                    else:
                        print(f"❌ Reddit comments failed for {post_id}: {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Reddit comments error for {post_id}: {e}")
                return []
    
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment data from Reddit"""
        print(f"📱 Getting Reddit sentiment for {symbol}...")
        
        if not await self.authenticate():
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sources': {},
                'aggregated_sentiment': {},
                'posts': [],
                'error': 'Authentication failed'
            }
        
        all_posts = []
        subreddit_data = {}
        
        # Search in each financial subreddit
        for subreddit in self.financial_subreddits:
            try:
                # Search for posts about the symbol
                posts = await self.search_subreddit(subreddit, symbol, limit=5)
                
                if posts:
                    # Get comments for each post
                    for post in posts:
                        comments = await self.get_comments(post['id'], limit=5)
                        post['comments'] = comments
                    
                    subreddit_data[subreddit] = {
                        'count': len(posts),
                        'posts': posts
                    }
                    all_posts.extend(posts)
                    
                    print(f"✅ r/{subreddit}: Found {len(posts)} posts")
                else:
                    print(f"⚠️ r/{subreddit}: No posts found")
                
                # Small delay to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing r/{subreddit}: {e}")
                continue
        
        # Create sentiment data structure
        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': subreddit_data,
            'aggregated_sentiment': {},
            'posts': all_posts
        }
        
        print(f"✅ Reddit sentiment: {len(all_posts)} total posts from {len(subreddit_data)} subreddits")
        return sentiment_data

async def main():
    """Test Reddit API integration"""
    print("📱 Reddit API Integration Test")
    print("="*50)
    
    reddit = RedditAPIIntegration()
    
    # Test authentication
    if not await reddit.authenticate():
        print("\n📋 SETUP INSTRUCTIONS:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Fill in the details:")
        print("   - Name: TradingSentimentBot")
        print("   - App type: script")
        print("   - Description: Trading sentiment analysis")
        print("   - About URL: (leave blank)")
        print("   - Redirect URI: http://localhost:8080")
        print("4. Copy the Client ID (under the app name)")
        print("5. Copy the Client Secret")
        print("6. Add to env_real_keys.env:")
        print("   REDDIT_CLIENT_ID=your_client_id_here")
        print("   REDDIT_CLIENT_SECRET=your_client_secret_here")
        return
    
    # Test sentiment data collection
    sentiment_data = await reddit.get_sentiment_data('AAPL')
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Symbol: {sentiment_data['symbol']}")
    print(f"   Total Posts: {len(sentiment_data['posts'])}")
    print(f"   Subreddits: {len(sentiment_data['sources'])}")
    
    # Show sample posts
    if sentiment_data['posts']:
        print(f"\n📝 SAMPLE POSTS:")
        for i, post in enumerate(sentiment_data['posts'][:3]):
            print(f"   {i+1}. r/{post['subreddit']}: {post['title'][:50]}...")
            print(f"      Score: {post['score']}, Comments: {post['num_comments']}")
    
    print(f"\n🎉 Reddit API integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())
