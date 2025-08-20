#!/usr/bin/env python3
"""
Enhanced Sentiment Integration System
Focus: Free APIs + Advanced Extraction Techniques
"""

import asyncio
import aiohttp
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import feedparser
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from collections import defaultdict
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSentimentIntegration:
    """Enhanced sentiment integration with free APIs and advanced extraction"""
    
    def __init__(self):
        # API Keys
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.stocktwits_token = os.getenv('STOCKTWITS_TOKEN', '')
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        
        # Reddit credentials (from existing setup)
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Data storage
        self.sentiment_cache = {}
        self.entity_cache = {}
        
        # Rate limiting
        self.rate_limits = {
            'news_api': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 86400},
            'stocktwits': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 86400},
            'youtube': {'calls': 0, 'limit': 10000, 'reset_time': time.time() + 86400}
        }
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize sentiment analyzers
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Load spaCy model (if available)
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                
        except Exception as e:
            logger.error(f"Error initializing NLP: {e}")
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API call is within rate limits"""
        if api_name not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[api_name]
        
        # Reset if time has passed
        if time.time() > limit_info['reset_time']:
            limit_info['calls'] = 0
            limit_info['reset_time'] = time.time() + 86400
        
        if limit_info['calls'] >= limit_info['limit']:
            return False
            
        limit_info['calls'] += 1
        return True
    
    async def _make_api_request(self, session: aiohttp.ClientSession, url: str, 
                               params: dict = None, headers: dict = None) -> Optional[dict]:
        """Make API request with error handling"""
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning(f"Rate limited: {url}")
                    return None
                else:
                    logger.warning(f"API request failed: {url}, status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error making API request to {url}: {e}")
            return None
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment from multiple free sources"""
        logger.info(f"📰 Getting news sentiment for {symbol}")
        
        news_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_sentiment': {},
            'articles': []
        }
        
        async with aiohttp.ClientSession() as session:
            # 1. NewsAPI.org
            if self.news_api_key and self._check_rate_limit('news_api'):
                news_api_url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 50
                }
                
                data = await self._make_api_request(session, news_api_url, params=params)
                if data and 'articles' in data:
                    articles = []
                    for article in data['articles'][:20]:  # Limit to 20 articles
                        sentiment = self._analyze_text_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                        articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'sentiment': sentiment
                        })
                    
                    news_data['sources']['newsapi'] = {
                        'count': len(articles),
                        'articles': articles
                    }
            
            # 2. RSS Feeds
            rss_feeds = [
                f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US',
                f'https://seekingalpha.com/feed.xml?symbol={symbol}',
                f'https://www.marketwatch.com/rss/topstories',
                f'https://feeds.reuters.com/reuters/businessNews'
            ]
            
            rss_articles = []
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:  # Limit to 10 entries per feed
                        # Check if article mentions the symbol
                        content = entry.get('title', '') + ' ' + entry.get('summary', '')
                        if symbol.lower() in content.lower():
                            sentiment = self._analyze_text_sentiment(content)
                            rss_articles.append({
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'source': feed.feed.get('title', 'RSS'),
                                'sentiment': sentiment
                            })
                except Exception as e:
                    logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
            
            if rss_articles:
                news_data['sources']['rss_feeds'] = {
                    'count': len(rss_articles),
                    'articles': rss_articles
                }
        
        # Aggregate sentiment
        all_articles = []
        for source_data in news_data['sources'].values():
            all_articles.extend(source_data['articles'])
        
        if all_articles:
            news_data['aggregated_sentiment'] = self._aggregate_sentiment(all_articles)
        
        return news_data
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment from multiple platforms"""
        logger.info(f"📱 Getting social sentiment for {symbol}")
        
        social_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_sentiment': {},
            'posts': []
        }
        
        async with aiohttp.ClientSession() as session:
            # 1. StockTwits
            if self.stocktwits_token and self._check_rate_limit('stocktwits'):
                stocktwits_url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
                headers = {'Authorization': f'OAuth {self.stocktwits_token}'}
                
                data = await self._make_api_request(session, stocktwits_url, headers=headers)
                if data and 'messages' in data:
                    posts = []
                    for message in data['messages'][:50]:  # Limit to 50 posts
                        sentiment = self._analyze_text_sentiment(message.get('body', ''))
                        posts.append({
                            'id': message.get('id', ''),
                            'body': message.get('body', ''),
                            'user': message.get('user', {}).get('username', ''),
                            'created_at': message.get('created_at', ''),
                            'sentiment': sentiment
                        })
                    
                    social_data['sources']['stocktwits'] = {
                        'count': len(posts),
                        'posts': posts
                    }
            
            # 2. Enhanced Reddit (expanded subreddits)
            reddit_subreddits = [
                'investing', 'stocks', 'wallstreetbets', 'StockMarket',
                'ValueInvesting', 'CryptoCurrency', 'Bitcoin', 'Ethereum',
                'Options', 'DayTrading', 'SwingTrading'
            ]
            
            reddit_posts = []
            for subreddit in reddit_subreddits:
                try:
                    # Use Reddit API to get recent posts
                    reddit_url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': symbol,
                        'restrict_sr': 'on',
                        'sort': 'new',
                        't': 'day',
                        'limit': 10
                    }
                    
                    data = await self._make_api_request(session, reddit_url, params=params)
                    if data and 'data' in data and 'children' in data['data']:
                        for post in data['data']['children']:
                            post_data = post['data']
                            content = post_data.get('title', '') + ' ' + post_data.get('selftext', '')
                            if symbol.lower() in content.lower():
                                sentiment = self._analyze_text_sentiment(content)
                                reddit_posts.append({
                                    'id': post_data.get('id', ''),
                                    'title': post_data.get('title', ''),
                                    'content': post_data.get('selftext', ''),
                                    'subreddit': subreddit,
                                    'author': post_data.get('author', ''),
                                    'score': post_data.get('score', 0),
                                    'created_utc': post_data.get('created_utc', ''),
                                    'sentiment': sentiment
                                })
                except Exception as e:
                    logger.warning(f"Error getting Reddit posts from r/{subreddit}: {e}")
            
            if reddit_posts:
                social_data['sources']['reddit'] = {
                    'count': len(reddit_posts),
                    'posts': reddit_posts
                }
        
        # Aggregate sentiment
        all_posts = []
        for source_data in social_data['sources'].values():
            all_posts.extend(source_data['posts'])
        
        if all_posts:
            social_data['aggregated_sentiment'] = self._aggregate_sentiment(all_posts)
        
        return social_data
    
    async def get_youtube_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get YouTube sentiment from comments and video descriptions"""
        logger.info(f"📺 Getting YouTube sentiment for {symbol}")
        
        youtube_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_sentiment': {},
            'videos': []
        }
        
        if not self.youtube_api_key or not self._check_rate_limit('youtube'):
            return youtube_data
        
        async with aiohttp.ClientSession() as session:
            # Search for videos about the symbol
            search_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                'part': 'snippet',
                'q': f'{symbol} stock analysis',
                'type': 'video',
                'order': 'relevance',
                'maxResults': 10,
                'key': self.youtube_api_key
            }
            
            search_data = await self._make_api_request(session, search_url, params=params)
            if not search_data or 'items' not in search_data:
                return youtube_data
            
            videos = []
            for item in search_data['items']:
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                # Get video comments
                comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
                comment_params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'maxResults': 20,
                    'key': self.youtube_api_key
                }
                
                comments_data = await self._make_api_request(session, comments_url, params=comment_params)
                
                comments = []
                if comments_data and 'items' in comments_data:
                    for comment_item in comments_data['items']:
                        comment = comment_item['snippet']['topLevelComment']['snippet']
                        sentiment = self._analyze_text_sentiment(comment.get('textDisplay', ''))
                        comments.append({
                            'text': comment.get('textDisplay', ''),
                            'author': comment.get('authorDisplayName', ''),
                            'likeCount': comment.get('likeCount', 0),
                            'publishedAt': comment.get('publishedAt', ''),
                            'sentiment': sentiment
                        })
                
                # Analyze video description and title
                video_content = snippet.get('title', '') + ' ' + snippet.get('description', '')
                video_sentiment = self._analyze_text_sentiment(video_content)
                
                videos.append({
                    'id': video_id,
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'channelTitle': snippet.get('channelTitle', ''),
                    'publishedAt': snippet.get('publishedAt', ''),
                    'viewCount': snippet.get('viewCount', 0),
                    'sentiment': video_sentiment,
                    'comments': comments
                })
            
            if videos:
                youtube_data['sources']['youtube'] = {
                    'count': len(videos),
                    'videos': videos
                }
                
                # Aggregate sentiment from videos and comments
                all_content = []
                for video in videos:
                    all_content.append(video)
                    all_content.extend(video['comments'])
                
                youtube_data['aggregated_sentiment'] = self._aggregate_sentiment(all_content)
        
        return youtube_data
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis using multiple models"""
        if not text or len(text.strip()) < 10:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'compound': 0.0,
                'confidence': 0.0,
                'emotion': 'neutral'
            }
        
        # Clean text
        text = self._clean_text(text)
        
        # Multiple sentiment analysis approaches
        results = {}
        
        # 1. VADER
        try:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results['vader'] = vader_scores
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
        
        # 2. TextBlob
        try:
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
        
        # 3. Custom financial sentiment
        financial_sentiment = self._analyze_financial_sentiment(text)
        results['financial'] = financial_sentiment
        
        # Aggregate results
        return self._ensemble_sentiment(results)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """Custom financial sentiment analysis"""
        text_lower = text.lower()
        
        # Financial sentiment keywords
        bullish_keywords = [
            'bullish', 'buy', 'buying', 'long', 'moon', 'rocket', 'pump', 'rally',
            'breakout', 'uptrend', 'strong', 'growth', 'earnings beat', 'positive',
            'outperform', 'buy rating', 'price target raised', 'upgrade'
        ]
        
        bearish_keywords = [
            'bearish', 'sell', 'selling', 'short', 'dump', 'crash', 'dip', 'decline',
            'breakdown', 'downtrend', 'weak', 'loss', 'earnings miss', 'negative',
            'underperform', 'sell rating', 'price target cut', 'downgrade'
        ]
        
        # Count keyword occurrences
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        
        # Calculate financial sentiment score
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            financial_score = 0.0
        else:
            financial_score = (bullish_count - bearish_count) / total_keywords
        
        return {
            'financial_score': financial_score,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count,
            'total_keywords': total_keywords
        }
    
    def _ensemble_sentiment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Ensemble sentiment analysis combining multiple models"""
        if not results:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'compound': 0.0,
                'confidence': 0.0,
                'emotion': 'neutral'
            }
        
        # Weighted average of different models
        weights = {
            'vader': 0.4,
            'textblob': 0.3,
            'financial': 0.3
        }
        
        polarity = 0.0
        subjectivity = 0.0
        compound = 0.0
        total_weight = 0.0
        
        # VADER
        if 'vader' in results:
            vader = results['vader']
            weight = weights['vader']
            polarity += vader.get('compound', 0.0) * weight
            compound += vader.get('compound', 0.0) * weight
            total_weight += weight
        
        # TextBlob
        if 'textblob' in results:
            textblob = results['textblob']
            weight = weights['textblob']
            polarity += textblob.get('polarity', 0.0) * weight
            subjectivity += textblob.get('subjectivity', 0.0) * weight
            total_weight += weight
        
        # Financial
        if 'financial' in results:
            financial = results['financial']
            weight = weights['financial']
            polarity += financial.get('financial_score', 0.0) * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            polarity /= total_weight
            subjectivity /= total_weight
            compound /= total_weight
        
        # Determine emotion
        emotion = self._classify_emotion(compound)
        
        # Calculate confidence
        confidence = self._calculate_confidence(results)
        
        return {
            'polarity': round(polarity, 4),
            'subjectivity': round(subjectivity, 4),
            'compound': round(compound, 4),
            'confidence': round(confidence, 4),
            'emotion': emotion
        }
    
    def _classify_emotion(self, compound: float) -> str:
        """Classify emotion based on compound score"""
        if compound >= 0.5:
            return 'very_positive'
        elif compound >= 0.1:
            return 'positive'
        elif compound <= -0.5:
            return 'very_negative'
        elif compound <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score based on model agreement"""
        if len(results) < 2:
            return 0.5
        
        # Calculate agreement between models
        scores = []
        
        if 'vader' in results:
            scores.append(results['vader'].get('compound', 0.0))
        
        if 'textblob' in results:
            scores.append(results['textblob'].get('polarity', 0.0))
        
        if 'financial' in results:
            scores.append(results['financial'].get('financial_score', 0.0))
        
        if len(scores) < 2:
            return 0.5
        
        # Calculate standard deviation (lower = higher confidence)
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Convert to confidence (0-1, higher is better)
        confidence = max(0.0, 1.0 - std_dev)
        
        return confidence
    
    def _aggregate_sentiment(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple items"""
        if not items:
            return {
                'overall_polarity': 0.0,
                'overall_subjectivity': 0.0,
                'overall_compound': 0.0,
                'confidence': 0.0,
                'emotion_distribution': {},
                'total_items': 0
            }
        
        total_polarity = 0.0
        total_subjectivity = 0.0
        total_compound = 0.0
        total_confidence = 0.0
        emotion_counts = defaultdict(int)
        valid_items = 0
        
        for item in items:
            sentiment = item.get('sentiment', {})
            if sentiment:
                total_polarity += sentiment.get('polarity', 0.0)
                total_subjectivity += sentiment.get('subjectivity', 0.0)
                total_compound += sentiment.get('compound', 0.0)
                total_confidence += sentiment.get('confidence', 0.0)
                emotion_counts[sentiment.get('emotion', 'neutral')] += 1
                valid_items += 1
        
        if valid_items == 0:
            return {
                'overall_polarity': 0.0,
                'overall_subjectivity': 0.0,
                'overall_compound': 0.0,
                'confidence': 0.0,
                'emotion_distribution': {},
                'total_items': 0
            }
        
        # Calculate averages
        overall_polarity = total_polarity / valid_items
        overall_subjectivity = total_subjectivity / valid_items
        overall_compound = total_compound / valid_items
        overall_confidence = total_confidence / valid_items
        
        # Calculate emotion distribution
        emotion_distribution = {
            emotion: count / valid_items 
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'overall_polarity': round(overall_polarity, 4),
            'overall_subjectivity': round(overall_subjectivity, 4),
            'overall_compound': round(overall_compound, 4),
            'confidence': round(overall_confidence, 4),
            'emotion_distribution': emotion_distribution,
            'total_items': valid_items
        }
    
    async def get_comprehensive_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment from all sources"""
        logger.info(f"🎯 Getting comprehensive sentiment for {symbol}")
        
        start_time = time.time()
        
        # Collect sentiment from all sources
        tasks = [
            self.get_news_sentiment(symbol),
            self.get_social_sentiment(symbol),
            self.get_youtube_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'collection_time': round(time.time() - start_time, 2),
            'sources': {},
            'overall_sentiment': {},
            'summary': {}
        }
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error collecting sentiment: {result}")
                continue
            
            source_name = ['news', 'social', 'youtube'][i]
            comprehensive_data['sources'][source_name] = result
        
        # Calculate overall sentiment
        all_sentiments = []
        for source_data in comprehensive_data['sources'].values():
            if 'aggregated_sentiment' in source_data:
                all_sentiments.append(source_data['aggregated_sentiment'])
        
        if all_sentiments:
            comprehensive_data['overall_sentiment'] = self._aggregate_sentiment(all_sentiments)
        
        # Generate summary
        comprehensive_data['summary'] = self._generate_sentiment_summary(comprehensive_data)
        
        return comprehensive_data
    
    def _generate_sentiment_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sentiment summary and insights"""
        overall = data.get('overall_sentiment', {})
        
        # Determine sentiment trend
        compound = overall.get('overall_compound', 0.0)
        if compound >= 0.3:
            trend = 'bullish'
        elif compound <= -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Calculate source coverage
        source_counts = {}
        total_items = 0
        for source_name, source_data in data.get('sources', {}).items():
            count = 0
            if 'articles' in source_data:
                count = len(source_data['articles'])
            elif 'posts' in source_data:
                count = len(source_data['posts'])
            elif 'videos' in source_data:
                count = len(source_data['videos'])
            
            source_counts[source_name] = count
            total_items += count
        
        return {
            'trend': trend,
            'confidence': overall.get('confidence', 0.0),
            'total_items': total_items,
            'source_coverage': source_counts,
            'dominant_emotion': max(overall.get('emotion_distribution', {}).items(), 
                                  key=lambda x: x[1], default=('neutral', 0.0))[0]
        }

async def main():
    """Demo the enhanced sentiment integration"""
    print("🚀 Enhanced Sentiment Integration Demo")
    print("="*50)
    
    integration = EnhancedSentimentIntegration()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting comprehensive sentiment for {symbol}...")
    
    sentiment_data = await integration.get_comprehensive_sentiment(symbol)
    
    # Print results
    print(f"\n✅ Sentiment collection complete!")
    print(f"⏱️ Collection time: {sentiment_data['collection_time']} seconds")
    
    # Print summary
    summary = sentiment_data['summary']
    print(f"\n📈 SENTIMENT SUMMARY:")
    print(f"   Trend: {summary['trend'].upper()}")
    print(f"   Confidence: {summary['confidence']:.2%}")
    print(f"   Total Items: {summary['total_items']}")
    print(f"   Dominant Emotion: {summary['dominant_emotion']}")
    
    # Print source coverage
    print(f"\n📊 SOURCE COVERAGE:")
    for source, count in summary['source_coverage'].items():
        print(f"   {source.capitalize()}: {count} items")
    
    # Print overall sentiment
    overall = sentiment_data['overall_sentiment']
    print(f"\n🎯 OVERALL SENTIMENT:")
    print(f"   Polarity: {overall.get('overall_polarity', 0.0):.3f}")
    print(f"   Subjectivity: {overall.get('overall_subjectivity', 0.0):.3f}")
    print(f"   Compound: {overall.get('overall_compound', 0.0):.3f}")
    
    # Print emotion distribution
    emotions = overall.get('emotion_distribution', {})
    if emotions:
        print(f"\n😊 EMOTION DISTRIBUTION:")
        for emotion, percentage in emotions.items():
            print(f"   {emotion}: {percentage:.1%}")
    
    print(f"\n🎉 Enhanced sentiment integration demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
