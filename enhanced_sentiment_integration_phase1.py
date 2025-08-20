#!/usr/bin/env python3
"""
Enhanced Sentiment Integration - Phase 1
Combines working APIs with proper fallbacks
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

class EnhancedSentimentIntegrationPhase1:
    """Enhanced sentiment integration with working APIs and fallbacks"""
    
    def __init__(self):
        # API Keys
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Data storage
        self.sentiment_cache = {}
        
        # Rate limiting
        self.rate_limits = {
            'news_api': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 86400},
            'reddit': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 3600},
            'twitter': {'calls': 0, 'limit': 300, 'reset_time': time.time() + 900}
        }
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            from textblob import TextBlob
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            
            # Initialize sentiment analyzers
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.textblob_available = True
            
        except Exception as e:
            print(f"⚠️ NLP initialization warning: {e}")
            self.textblob_available = False
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API call is within rate limits"""
        if api_name not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[api_name]
        
        # Reset if time has passed
        if time.time() > limit_info['reset_time']:
            limit_info['calls'] = 0
            limit_info['reset_time'] = time.time() + (86400 if api_name == 'news_api' else 3600 if api_name == 'reddit' else 900)
        
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
                    print(f"⚠️ Rate limited: {url}")
                    return None
                else:
                    print(f"⚠️ API request failed: {url}, status: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error making API request to {url}: {e}")
            return None
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment from NewsAPI (WORKING)"""
        print(f"📰 Getting news sentiment for {symbol}...")
        
        news_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_sentiment': {},
            'articles': [],
            'status': 'WORKING'
        }
        
        if not self.news_api_key:
            print("❌ NewsAPI key not found")
            news_data['status'] = 'NO_API_KEY'
            return news_data
        
        if not self._check_rate_limit('news_api'):
            print("⚠️ NewsAPI rate limit reached")
            news_data['status'] = 'RATE_LIMITED'
            return news_data
        
        async with aiohttp.ClientSession() as session:
            # NewsAPI.org
            news_api_url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            
            data = await self._make_api_request(session, news_api_url, params=params)
            if data and 'articles' in data:
                articles = []
                for article in data['articles'][:10]:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    
                    if title or description:
                        text_to_analyze = f"{title} {description}".strip()
                        sentiment = self._analyze_text_sentiment(text_to_analyze)
                        
                        articles.append({
                            'title': title,
                            'description': description,
                            'url': article.get('url', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'sentiment': sentiment
                        })
                
                if articles:
                    news_data['sources']['newsapi'] = {
                        'count': len(articles),
                        'articles': articles
                    }
                    print(f"✅ NewsAPI: Found {len(articles)} articles")
                else:
                    print("⚠️ NewsAPI: No valid articles found")
            else:
                print("⚠️ NewsAPI: No data received")
        
        # Aggregate sentiment
        all_articles = []
        for source_data in news_data['sources'].values():
            all_articles.extend(source_data['articles'])
        
        if all_articles:
            news_data['aggregated_sentiment'] = self._aggregate_sentiment(all_articles)
            print(f"✅ News sentiment: {len(all_articles)} total articles")
        else:
            print("⚠️ No news articles found")
        
        return news_data
    
    async def get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Reddit sentiment (NEEDS CREDENTIALS)"""
        print(f"📱 Getting Reddit sentiment for {symbol}...")
        
        reddit_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_sentiment': {},
            'posts': [],
            'status': 'NEEDS_CREDENTIALS'
        }
        
        if not self.reddit_client_id or not self.reddit_client_secret:
            print("❌ Reddit API credentials not found!")
            print("📋 Please add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to env_real_keys.env")
            return reddit_data
        
        # For now, return mock data with status
        mock_posts = [
            {
                'id': '1',
                'title': f'Bullish on {symbol} - great earnings!',
                'content': f'{symbol} is looking strong with solid fundamentals',
                'subreddit': 'investing',
                'author': 'trader123',
                'score': 15,
                'sentiment': self._analyze_text_sentiment(f'Bullish on {symbol} - great earnings!')
            },
            {
                'id': '2',
                'title': f'{symbol} technical analysis',
                'content': f'{symbol} showing good support at current levels',
                'subreddit': 'stocks',
                'author': 'analyst456',
                'score': 8,
                'sentiment': self._analyze_text_sentiment(f'{symbol} showing good support at current levels')
            }
        ]
        
        reddit_data['sources']['reddit'] = {
            'count': len(mock_posts),
            'posts': mock_posts,
            'note': 'Mock data - real API needs credentials'
        }
        reddit_data['aggregated_sentiment'] = self._aggregate_sentiment(mock_posts)
        reddit_data['status'] = 'MOCK_DATA'
        
        print(f"⚠️ Reddit: Using mock data (needs real credentials)")
        return reddit_data
    
    async def get_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Twitter sentiment (NEEDS BEARER TOKEN)"""
        print(f"🐦 Getting Twitter sentiment for {symbol}...")
        
        twitter_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_sentiment': {},
            'tweets': [],
            'status': 'NEEDS_BEARER_TOKEN'
        }
        
        if not self.twitter_bearer_token or self.twitter_bearer_token == 'your_twitter_bearer_token_here':
            print("❌ Twitter Bearer Token not found!")
            print("📋 Please add TWITTER_BEARER_TOKEN to env_real_keys.env")
            return twitter_data
        
        # For now, return mock data with status
        mock_tweets = [
            {
                'id': '1',
                'text': f'${symbol} looking bullish today! Great earnings call.',
                'author_username': 'trader_pro',
                'like_count': 25,
                'retweet_count': 5,
                'sentiment': self._analyze_text_sentiment(f'${symbol} looking bullish today! Great earnings call.')
            },
            {
                'id': '2',
                'text': f'Technical analysis shows {symbol} has strong support at current levels',
                'author_username': 'tech_analyst',
                'like_count': 12,
                'retweet_count': 3,
                'sentiment': self._analyze_text_sentiment(f'Technical analysis shows {symbol} has strong support at current levels')
            }
        ]
        
        twitter_data['sources']['twitter'] = {
            'count': len(mock_tweets),
            'tweets': mock_tweets,
            'note': 'Mock data - real API needs bearer token'
        }
        twitter_data['aggregated_sentiment'] = self._aggregate_sentiment(mock_tweets)
        twitter_data['status'] = 'MOCK_DATA'
        
        print(f"⚠️ Twitter: Using mock data (needs real bearer token)")
        return twitter_data
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using multiple models"""
        if not text or len(text.strip()) < 5:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'compound': 0.0,
                'confidence': 0.0,
                'emotion': 'neutral'
            }
        
        # Clean text
        import re
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Multiple sentiment analysis approaches
        results = {}
        
        # 1. VADER
        try:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results['vader'] = vader_scores
        except Exception as e:
            print(f"⚠️ VADER analysis failed: {e}")
        
        # 2. TextBlob
        if self.textblob_available:
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                print(f"⚠️ TextBlob analysis failed: {e}")
        
        # 3. Custom financial sentiment
        financial_sentiment = self._analyze_financial_sentiment(text)
        results['financial'] = financial_sentiment
        
        # Aggregate results
        return self._ensemble_sentiment(results)
    
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
        emotion_counts = {}
        valid_items = 0
        
        for item in items:
            sentiment = item.get('sentiment', {})
            if sentiment:
                total_polarity += sentiment.get('polarity', 0.0)
                total_subjectivity += sentiment.get('subjectivity', 0.0)
                total_compound += sentiment.get('compound', 0.0)
                total_confidence += sentiment.get('confidence', 0.0)
                emotion = sentiment.get('emotion', 'neutral')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
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
        print(f"🎯 Getting comprehensive sentiment for {symbol}...")
        
        start_time = time.time()
        
        # Collect sentiment from all sources
        tasks = [
            self.get_news_sentiment(symbol),
            self.get_reddit_sentiment(symbol),
            self.get_twitter_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'collection_time': round(time.time() - start_time, 2),
            'sources': {},
            'overall_sentiment': {},
            'summary': {},
            'status_report': {}
        }
        
        # Process results
        source_names = ['news', 'reddit', 'twitter']
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Error collecting sentiment: {result}")
                comprehensive_data['sources'][source_names[i]] = {
                    'status': 'ERROR',
                    'error': str(result)
                }
                continue
            
            source_name = source_names[i]
            comprehensive_data['sources'][source_name] = result
            comprehensive_data['status_report'][source_name] = result.get('status', 'UNKNOWN')
        
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
            if 'articles' in source_data.get('sources', {}).get('newsapi', {}):
                count += len(source_data['sources']['newsapi']['articles'])
            if 'posts' in source_data.get('sources', {}).get('reddit', {}):
                count += len(source_data['sources']['reddit']['posts'])
            if 'tweets' in source_data.get('sources', {}).get('twitter', {}):
                count += len(source_data['sources']['twitter']['tweets'])
            
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
    print("🚀 Enhanced Sentiment Integration - Phase 1")
    print("="*60)
    
    integration = EnhancedSentimentIntegrationPhase1()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting comprehensive sentiment for {symbol}...")
    
    sentiment_data = await integration.get_comprehensive_sentiment(symbol)
    
    # Print results
    print(f"\n✅ Sentiment collection complete!")
    print(f"⏱️ Collection time: {sentiment_data['collection_time']} seconds")
    
    # Print status report
    print(f"\n📋 STATUS REPORT:")
    for source, status in sentiment_data['status_report'].items():
        status_emoji = {
            'WORKING': '✅',
            'MOCK_DATA': '⚠️',
            'NO_API_KEY': '❌',
            'NEEDS_CREDENTIALS': '🔑',
            'NEEDS_BEARER_TOKEN': '🔑',
            'RATE_LIMITED': '⏱️',
            'ERROR': '💥'
        }.get(status, '❓')
        print(f"   {status_emoji} {source.capitalize()}: {status}")
    
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
