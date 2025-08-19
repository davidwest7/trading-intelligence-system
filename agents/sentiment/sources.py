"""
Data source implementations for sentiment analysis

Social media and news sources:
- Twitter/X API integration
- Reddit API integration  
- Financial news APIs
- Telegram/Discord monitoring
"""

import re
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

from .models import SentimentPost, SourceType


class BaseSentimentSource(ABC):
    """Base class for sentiment data sources"""
    
    def __init__(self, source_type: SourceType, config: Dict[str, Any]):
        self.source_type = source_type
        self.config = config
        self.rate_limiter = RateLimiter(
            calls_per_window=config.get('rate_limit', 100),
            window_seconds=config.get('rate_window', 900)  # 15 minutes
        )
    
    @abstractmethod
    async def collect_posts(self, query: str, since: datetime, 
                          max_posts: int = 100) -> List[SentimentPost]:
        """Collect posts for a given query"""
        pass
    
    @abstractmethod
    async def stream_posts(self, query: str) -> AsyncGenerator[SentimentPost, None]:
        """Stream real-time posts"""
        pass
    
    def preprocess_query(self, ticker: str) -> str:
        """Preprocess ticker for source-specific search"""
        return ticker


class TwitterSource(BaseSentimentSource):
    """
    Twitter/X sentiment data source
    
    Note: Requires Twitter API v2 access
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(SourceType.TWITTER, config)
        self.bearer_token = config.get('bearer_token')
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = "https://api.twitter.com/2"
        
        if not self.bearer_token:
            print("Warning: Twitter Bearer Token not provided. Using mock data.")
    
    async def collect_posts(self, query: str, since: datetime, 
                          max_posts: int = 100) -> List[SentimentPost]:
        """Collect Twitter posts for a ticker"""
        if not self.bearer_token:
            return self._generate_mock_posts(query, max_posts)
        
        await self.rate_limiter.wait()
        
        # Build Twitter API query
        search_query = self._build_twitter_query(query)
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        
        params = {
            'query': search_query,
            'max_results': min(max_posts, 100),  # Twitter API limit
            'start_time': since.isoformat() + 'Z',
            'tweet.fields': 'created_at,public_metrics,author_id,context_annotations',
            'user.fields': 'created_at,public_metrics,verified'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tweets/search/recent",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_twitter_response(data)
                    else:
                        print(f"Twitter API error: {response.status}")
                        return self._generate_mock_posts(query, max_posts)
        except Exception as e:
            print(f"Twitter API request failed: {e}")
            return self._generate_mock_posts(query, max_posts)
    
    async def stream_posts(self, query: str) -> AsyncGenerator[SentimentPost, None]:
        """Stream real-time Twitter posts"""
        if not self.bearer_token:
            # Mock streaming
            for i in range(5):
                await asyncio.sleep(1)
                posts = self._generate_mock_posts(query, 1)
                if posts:
                    yield posts[0]
            return
        
        # TODO: Implement Twitter streaming API v2
        # This would use the filtered stream endpoint
        yield SentimentPost(
            id="stream_placeholder",
            source=SourceType.TWITTER,
            text=f"Streaming placeholder for {query}",
            author="stream_user",
            timestamp=datetime.now(),
            sentiment_score=0.0,
            confidence=0.5,
            is_bot=False,
            reach=100
        )
    
    def _build_twitter_query(self, ticker: str) -> str:
        """Build Twitter search query"""
        # Include common variations
        variations = [
            f"${ticker}",      # $AAPL
            f"#{ticker}",      # #AAPL  
            ticker,            # AAPL
            f"{ticker.lower()}"  # aapl
        ]
        
        # Add context words for better filtering
        context_words = ["stock", "trading", "market", "price", "earnings"]
        
        query_parts = []
        for variation in variations[:2]:  # Limit to avoid too complex query
            query_parts.append(variation)
        
        # Combine with OR
        base_query = " OR ".join(query_parts)
        
        # Add filters
        filters = [
            "lang:en",           # English only
            "-is:retweet",       # Exclude retweets
            "-is:reply",         # Exclude replies
            "has:hashtags OR has:cashtags"  # Must have hashtags or cashtags
        ]
        
        return f"({base_query}) {' '.join(filters)}"
    
    def _parse_twitter_response(self, data: Dict[str, Any]) -> List[SentimentPost]:
        """Parse Twitter API response"""
        posts = []
        
        tweets = data.get('data', [])
        users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
        
        for tweet in tweets:
            user_id = tweet['author_id']
            user = users.get(user_id, {})
            
            post = SentimentPost(
                id=tweet['id'],
                source=SourceType.TWITTER,
                text=tweet['text'],
                author=user.get('username', f"user_{user_id}"),
                timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                sentiment_score=0.0,  # Will be calculated later
                confidence=0.8,
                is_bot=False,  # Will be determined by bot detector
                reach=tweet.get('public_metrics', {}).get('retweet_count', 0) + 
                      tweet.get('public_metrics', {}).get('like_count', 0),
                url=f"https://twitter.com/i/status/{tweet['id']}"
            )
            posts.append(post)
        
        return posts
    
    def _generate_mock_posts(self, query: str, count: int) -> List[SentimentPost]:
        """Generate mock Twitter posts for testing"""
        mock_texts = [
            f"${query} is looking bullish today! 📈 #stocks #trading",
            f"Thinking about buying more {query} on this dip 🤔",
            f"${query} earnings coming up next week, expecting good results!",
            f"Technical analysis shows {query} breaking resistance 🚀",
            f"Not sure about {query} right now, market seems uncertain 😕",
            f"${query} dividend announcement was better than expected 💰",
            f"Sold my {query} position today, taking profits 📊",
            f"${query} CEO interview was impressive, bullish long term 👍",
            f"Market volatility affecting {query} but fundamentals strong 💪",
            f"${query} vs competitors - who's your pick? #investing"
        ]
        
        posts = []
        for i in range(min(count, len(mock_texts))):
            post = SentimentPost(
                id=f"mock_twitter_{i}",
                source=SourceType.TWITTER,
                text=mock_texts[i],
                author=f"trader_{i+1}",
                timestamp=datetime.now() - timedelta(minutes=i*15),
                sentiment_score=0.0,
                confidence=0.7,
                is_bot=False,
                reach=50 + i * 20,
                url=f"https://twitter.com/mock/status/{i}"
            )
            posts.append(post)
        
        return posts


class RedditSource(BaseSentimentSource):
    """
    Reddit sentiment data source
    
    Note: Requires Reddit API credentials
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(SourceType.REDDIT, config)
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        self.user_agent = config.get('user_agent', 'TradingIntelligenceBot/1.0')
        
        # Target subreddits for financial content
        self.subreddits = [
            'wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis',
            'StockMarket', 'ValueInvesting', 'dividends', 'options'
        ]
        
        if not self.client_id:
            print("Warning: Reddit credentials not provided. Using mock data.")
    
    async def collect_posts(self, query: str, since: datetime, 
                          max_posts: int = 100) -> List[SentimentPost]:
        """Collect Reddit posts and comments"""
        if not self.client_id:
            return self._generate_mock_posts(query, max_posts)
        
        # TODO: Implement Reddit API integration using PRAW or direct API
        # This would search across relevant subreddits for ticker mentions
        
        return self._generate_mock_posts(query, max_posts)
    
    async def stream_posts(self, query: str) -> AsyncGenerator[SentimentPost, None]:
        """Stream Reddit posts (mock implementation)"""
        for i in range(3):
            await asyncio.sleep(2)
            posts = self._generate_mock_posts(query, 1)
            if posts:
                yield posts[0]
    
    def _generate_mock_posts(self, query: str, count: int) -> List[SentimentPost]:
        """Generate mock Reddit posts"""
        mock_texts = [
            f"DD: Why {query} is undervalued and my 10k YOLO position 🚀🚀🚀",
            f"Anyone else holding {query} through earnings? Diamond hands 💎🙌",
            f"PSA: {query} ex-dividend date is tomorrow, don't miss out!",
            f"Technical analysis of {query} - cup and handle formation?",
            f"Thoughts on {query} after today's news? Buying the dip or waiting?",
            f"My {query} position is up 200% this year. When to take profits?",
            f"Fundamental analysis: Why {query} is still cheap at current prices",
            f"Options play on {query} earnings - what are your strikes?",
            f"Long term outlook on {query} - still bullish after regulation news",
            f"Paper handed {query} last week, regretting it now 🤦‍♂️"
        ]
        
        posts = []
        for i in range(min(count, len(mock_texts))):
            post = SentimentPost(
                id=f"mock_reddit_{i}",
                source=SourceType.REDDIT,
                text=mock_texts[i],
                author=f"redditor_{i+1}",
                timestamp=datetime.now() - timedelta(hours=i*2),
                sentiment_score=0.0,
                confidence=0.6,
                is_bot=False,
                reach=100 + i * 50,  # Upvotes
                url=f"https://reddit.com/r/stocks/comments/mock_{i}"
            )
            posts.append(post)
        
        return posts


class NewsSource(BaseSentimentSource):
    """
    Financial news sentiment source
    
    Aggregates from multiple news APIs
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(SourceType.NEWS, config)
        self.news_api_key = config.get('news_api_key')
        self.alpha_vantage_key = config.get('alpha_vantage_key')
        
        # Financial news sources
        self.news_sources = [
            'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
            'cnbc.com', 'marketwatch.com', 'finance.yahoo.com',
            'seekingalpha.com', 'fool.com'
        ]
    
    async def collect_posts(self, query: str, since: datetime, 
                          max_posts: int = 100) -> List[SentimentPost]:
        """Collect financial news articles"""
        if not self.news_api_key:
            return self._generate_mock_posts(query, max_posts)
        
        # TODO: Implement news API integration
        # This would use NewsAPI, Alpha Vantage News, or similar services
        
        return self._generate_mock_posts(query, max_posts)
    
    async def stream_posts(self, query: str) -> AsyncGenerator[SentimentPost, None]:
        """Stream news articles (mock)"""
        for i in range(2):
            await asyncio.sleep(5)
            posts = self._generate_mock_posts(query, 1)
            if posts:
                yield posts[0]
    
    def _generate_mock_posts(self, query: str, count: int) -> List[SentimentPost]:
        """Generate mock news articles"""
        mock_headlines = [
            f"{query} Reports Strong Q3 Earnings, Beats Analyst Expectations",
            f"Analyst Upgrades {query} to Buy Rating on Growth Prospects",
            f"{query} Announces Strategic Partnership in AI Technology",
            f"Market Volatility Impacts {query} Shares in Early Trading",
            f"{query} CEO Discusses Company Vision in Exclusive Interview",
            f"Regulatory Changes Could Affect {query} Business Model",
            f"{query} Dividend Increase Signals Management Confidence",
            f"Technical Analysis: {query} Breaks Through Key Resistance Level",
            f"{query} Faces Headwinds from Supply Chain Disruptions",
            f"Wall Street Divided on {query} Outlook Following Recent News"
        ]
        
        posts = []
        for i in range(min(count, len(mock_headlines))):
            post = SentimentPost(
                id=f"mock_news_{i}",
                source=SourceType.NEWS,
                text=mock_headlines[i],
                author=f"Financial Reporter {i+1}",
                timestamp=datetime.now() - timedelta(hours=i*6),
                sentiment_score=0.0,
                confidence=0.9,  # News generally has higher confidence
                is_bot=False,
                reach=5000 + i * 1000,  # Estimated readership
                url=f"https://financialnews.com/article/mock_{i}"
            )
            posts.append(post)
        
        return posts


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_window: int, window_seconds: int):
        self.calls_per_window = calls_per_window
        self.window_seconds = window_seconds
        self.calls = []
    
    async def wait(self):
        """Wait if necessary to respect rate limits"""
        now = datetime.now()
        
        # Remove old calls outside the window
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.calls = [call_time for call_time in self.calls if call_time > cutoff]
        
        # Check if we need to wait
        if len(self.calls) >= self.calls_per_window:
            # Wait until the oldest call expires
            wait_until = self.calls[0] + timedelta(seconds=self.window_seconds)
            wait_time = (wait_until - now).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls.append(now)
