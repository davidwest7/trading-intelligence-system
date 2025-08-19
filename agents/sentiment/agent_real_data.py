"""
Real Data Sentiment Agent
Uses actual Twitter and Reddit APIs for sentiment analysis
"""

import asyncio
import os
import praw
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .models import SentimentPost, SentimentAnalysis, SentimentData, SourceType, MarketImpact
from common.models import BaseAgent

# Load environment variables
load_dotenv('env_real_keys.env')

class RealDataSentimentAgent(BaseAgent):
    """
    Enhanced Sentiment Agent using real Twitter and Reddit data
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataSentiment", config)
        self.metrics = {
            'total_posts_analyzed': 0,
            'sentiment_signals_generated': 0,
            'processing_time_avg': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Initialize API clients
        self._init_twitter_client()
        self._init_reddit_client()
        
        # Cache for API responses
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def _init_twitter_client(self):
        """Initialize Twitter API client"""
        try:
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if bearer_token:
                self.twitter_bearer_token = bearer_token
                self.twitter_headers = {
                    'Authorization': f'Bearer {bearer_token}',
                    'User-Agent': 'TradingIntelligenceBot/1.0'
                }
                print("✅ Twitter API client initialized")
            else:
                print("❌ Twitter Bearer Token not found")
                self.twitter_bearer_token = None
        except Exception as e:
            print(f"❌ Twitter client initialization failed: {e}")
            self.twitter_bearer_token = None
    
    def _init_reddit_client(self):
        """Initialize Reddit API client"""
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT')
            
            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                print("✅ Reddit API client initialized")
            else:
                print("❌ Reddit credentials not found")
                self.reddit = None
        except Exception as e:
            print(f"❌ Reddit client initialization failed: {e}")
            self.reddit = None
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        tickers = kwargs.get('tickers', args[0] if args else ['AAPL'])
        return await self.analyze_sentiment_optimized(tickers, **kwargs)
    
    async def analyze_sentiment_optimized(self, tickers: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze sentiment using real data sources"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Collect real data from multiple sources
            all_posts = []
            
            # Get Twitter data
            if self.twitter_bearer_token:
                twitter_posts = await self._get_twitter_sentiment(tickers)
                all_posts.extend(twitter_posts)
            
            # Get Reddit data
            if self.reddit:
                reddit_posts = await self._get_reddit_sentiment(tickers)
                all_posts.extend(reddit_posts)
            
            # Analyze sentiment
            sentiment_analysis = await self._analyze_sentiment_data(all_posts, tickers)
            
            # Generate signals
            sentiment_signals = self._generate_sentiment_signals(sentiment_analysis, tickers)
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics['total_posts_analyzed'] += len(all_posts)
            self.metrics['sentiment_signals_generated'] += len(sentiment_signals)
            self.metrics['processing_time_avg'] = processing_time
            
            return {
                'sentiment_analysis': sentiment_analysis.to_dict(),
                'sentiment_posts': [post.to_dict() for post in all_posts],
                'sentiment_signals': sentiment_signals,
                'summary': {
                    'total_posts_analyzed': len(all_posts),
                    'total_tickers': len(tickers),
                    'average_sentiment': sentiment_analysis.overall_score,
                    'sentiment_distribution': sentiment_analysis.sentiment_distribution,
                    'top_sentiment_sources': self._get_top_sources(all_posts)
                },
                'processing_info': {
                    'processing_time': processing_time,
                    'cache_hit_rate': self.metrics['cache_hit_rate'],
                    'data_sources': ['twitter', 'reddit'] if all_posts else []
                }
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self._create_empty_analysis(tickers)
    
    async def _get_twitter_sentiment(self, tickers: List[str]) -> List[SentimentPost]:
        """Get real Twitter sentiment data"""
        posts = []
        
        try:
            for ticker in tickers:
                cache_key = f"twitter_{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"
                
                # Check cache
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if datetime.now().timestamp() - timestamp < self.cache_ttl:
                        posts.extend(cached_data)
                        continue
                
                # Query Twitter API
                url = "https://api.twitter.com/2/tweets/search/recent"
                params = {
                    'query': f'${ticker} stock',
                    'max_results': 20,
                    'tweet.fields': 'created_at,public_metrics,author_id'
                }
                
                response = requests.get(url, headers=self.twitter_headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    tweets = data.get('data', [])
                    
                    for tweet in tweets:
                        # Simple sentiment analysis (in production, use more sophisticated NLP)
                        text = tweet.get('text', '').lower()
                        sentiment_score = self._calculate_text_sentiment(text)
                        
                        post = SentimentPost(
                            id=tweet.get('id'),
                            source=SourceType.TWITTER,
                            content=tweet.get('text', ''),
                            author=f"user_{tweet.get('author_id', 'unknown')}",
                            timestamp=datetime.fromisoformat(tweet.get('created_at', '').replace('Z', '+00:00')),
                            sentiment_score=sentiment_score,
                            confidence=0.7,
                            is_bot=False,
                            reach=tweet.get('public_metrics', {}).get('retweet_count', 0) + 
                                   tweet.get('public_metrics', {}).get('like_count', 0),
                            url=f"https://twitter.com/i/status/{tweet.get('id')}"
                        )
                        posts.append(post)
                
                # Cache results
                self.cache[cache_key] = (posts, datetime.now().timestamp())
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"Error fetching Twitter data: {e}")
        
        return posts
    
    async def _get_reddit_sentiment(self, tickers: List[str]) -> List[SentimentPost]:
        """Get real Reddit sentiment data"""
        posts = []
        
        try:
            # Target subreddits for financial content
            subreddits = ['wallstreetbets', 'investing', 'stocks', 'StockMarket']
            
            for ticker in tickers:
                for subreddit_name in subreddits:
                    try:
                        subreddit = self.reddit.subreddit(subreddit_name)
                        
                        # Search for posts mentioning the ticker
                        search_query = f"{ticker} stock"
                        search_results = subreddit.search(search_query, limit=10, sort='hot')
                        
                        for post in search_results:
                            # Calculate sentiment
                            text = f"{post.title} {post.selftext}".lower()
                            sentiment_score = self._calculate_text_sentiment(text)
                            
                            reddit_post = SentimentPost(
                                id=post.id,
                                source=SourceType.REDDIT,
                                content=f"{post.title}\n{post.selftext}",
                                author=post.author.name if post.author else "deleted",
                                timestamp=datetime.fromtimestamp(post.created_utc),
                                sentiment_score=sentiment_score,
                                confidence=0.6,
                                is_bot=False,
                                reach=post.score + post.num_comments,
                                url=f"https://reddit.com{post.permalink}"
                            )
                            posts.append(reddit_post)
                    
                    except Exception as e:
                        print(f"Error fetching from r/{subreddit_name}: {e}")
                        continue
                
                # Rate limiting
                await asyncio.sleep(2)
                
        except Exception as e:
            print(f"Error fetching Reddit data: {e}")
        
        return posts
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text"""
        # Simple keyword-based sentiment (in production, use advanced NLP)
        positive_words = ['bullish', 'buy', 'moon', 'rocket', 'gains', 'profit', 'up', 'good', 'great', 'positive']
        negative_words = ['bearish', 'sell', 'crash', 'dump', 'loss', 'down', 'bad', 'terrible', 'negative']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize sentiment score between -1 and 1
        sentiment = (positive_count - negative_count) / max(total_words, 10)
        return max(-1.0, min(1.0, sentiment))
    
    async def _analyze_sentiment_data(self, posts: List[SentimentPost], tickers: List[str]) -> SentimentAnalysis:
        """Analyze sentiment data and create analysis"""
        
        if not posts:
            return self._create_empty_sentiment_analysis(tickers)
        
        # Calculate overall sentiment
        sentiment_scores = [post.sentiment_score for post in posts]
        overall_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Sentiment distribution
        positive_count = sum(1 for score in sentiment_scores if score > 0.1)
        negative_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        sentiment_distribution = {
            'positive': positive_count / len(sentiment_scores) if sentiment_scores else 0.0,
            'negative': negative_count / len(sentiment_scores) if sentiment_scores else 0.0,
            'neutral': neutral_count / len(sentiment_scores) if sentiment_scores else 0.0
        }
        
        # Source breakdown
        source_breakdown = {}
        for post in posts:
            source = post.source.value
            if source not in source_breakdown:
                source_breakdown[source] = {'count': 0, 'avg_sentiment': 0.0}
            source_breakdown[source]['count'] += 1
            source_breakdown[source]['avg_sentiment'] += post.sentiment_score
        
        # Calculate averages
        for source in source_breakdown:
            count = source_breakdown[source]['count']
            source_breakdown[source]['avg_sentiment'] /= count
        
        return SentimentAnalysis(
            overall_score=overall_score,
            sentiment_distribution=sentiment_distribution,
            source_breakdown=source_breakdown,
            market_impact='positive' if overall_score > 0.1 else 'negative' if overall_score < -0.1 else 'neutral',
            confidence=min(0.9, len(posts) / 100.0),  # Higher confidence with more data
            timestamp=datetime.now()
        )
    
    def _generate_sentiment_signals(self, analysis: SentimentAnalysis, tickers: List[str]) -> List[Dict[str, Any]]:
        """Generate sentiment signals based on analysis"""
        signals = []
        
        try:
            # Strong positive sentiment signal
            if analysis.overall_score > 0.3 and analysis.confidence > 0.5:
                signals.append({
                    'signal_type': 'strong_positive_sentiment',
                    'tickers': tickers,
                    'strength': analysis.overall_score,
                    'confidence': analysis.confidence,
                    'description': f'Strong positive sentiment detected across {len(tickers)} tickers',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Strong negative sentiment signal
            elif analysis.overall_score < -0.3 and analysis.confidence > 0.5:
                signals.append({
                    'signal_type': 'strong_negative_sentiment',
                    'tickers': tickers,
                    'strength': abs(analysis.overall_score),
                    'confidence': analysis.confidence,
                    'description': f'Strong negative sentiment detected across {len(tickers)} tickers',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Sentiment shift signal
            if analysis.sentiment_distribution['positive'] > 0.6:
                signals.append({
                    'signal_type': 'sentiment_shift_positive',
                    'tickers': tickers,
                    'strength': analysis.sentiment_distribution['positive'],
                    'confidence': analysis.confidence,
                    'description': f'Positive sentiment shift detected ({analysis.sentiment_distribution["positive"]:.1%} positive)',
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            print(f"Error generating sentiment signals: {e}")
        
        return signals
    
    def _get_top_sources(self, posts: List[SentimentPost]) -> List[str]:
        """Get top sentiment sources by post count"""
        source_counts = {}
        for post in posts:
            source = post.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return sorted(source_counts.keys(), key=lambda x: source_counts[x], reverse=True)
    
    def _create_empty_analysis(self, tickers: List[str]) -> Dict[str, Any]:
        """Create empty analysis when no data is available"""
        return {
            'sentiment_analysis': self._create_empty_sentiment_analysis(tickers).to_dict(),
            'sentiment_posts': [],
            'sentiment_signals': [],
            'summary': {
                'total_posts_analyzed': 0,
                'total_tickers': len(tickers),
                'average_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0},
                'top_sentiment_sources': []
            },
            'processing_info': {
                'processing_time': 0.0,
                'cache_hit_rate': 0.0,
                'data_sources': []
            }
        }
    
    def _create_empty_sentiment_analysis(self, tickers: List[str]) -> SentimentAnalysis:
        """Create empty sentiment analysis"""
        return SentimentAnalysis(
            overall_score=0.0,
            sentiment_distribution={'positive': 0.0, 'negative': 0.0, 'neutral': 0.0},
            source_breakdown={},
            market_impact='neutral',
            confidence=0.0,
            timestamp=datetime.now()
        )
