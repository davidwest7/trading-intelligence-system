#!/usr/bin/env python3
"""
Comprehensive Data Integration - Phase 3
Includes YouTube Live News Monitoring and Defeat Beta API
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

class ComprehensiveDataIntegrationPhase3:
    """Comprehensive data integration with YouTube live news and Defeat Beta API"""
    
    def __init__(self):
        # API Keys
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.fmp_api_key = os.getenv('FMP_API_KEY', '')
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Data storage
        self.data_cache = {}
        
        # Rate limiting
        self.rate_limits = {
            'news_api': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 86400},
            'polygon': {'calls': 0, 'limit': 5000, 'reset_time': time.time() + 86400},
            'fmp': {'calls': 0, 'limit': 1000, 'reset_time': time.time() + 86400},
            'fred': {'calls': 0, 'limit': 1200, 'reset_time': time.time() + 86400},
            'youtube': {'calls': 0, 'limit': 10000, 'reset_time': time.time() + 86400}
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
    
    async def get_youtube_live_news(self, symbol: str) -> Dict[str, Any]:
        """Get YouTube live news coverage"""
        print(f"📺 Getting YouTube live news for {symbol}...")
        
        youtube_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'live_streams': [],
            'videos': [],
            'sentiment': {},
            'status': 'WORKING'
        }
        
        if not self.youtube_api_key:
            print("❌ YouTube API key not found")
            youtube_data['status'] = 'NO_API_KEY'
            return youtube_data
        
        if not self._check_rate_limit('youtube'):
            print("⚠️ YouTube rate limit reached")
            youtube_data['status'] = 'RATE_LIMITED'
            return youtube_data
        
        # Import YouTube integration
        try:
            from youtube_live_news_integration import YouTubeLiveNewsIntegration
            youtube = YouTubeLiveNewsIntegration()
            
            # Get earnings coverage
            coverage = await youtube.get_earnings_announcement_coverage(symbol)
            
            youtube_data['live_streams'] = coverage.get('live_streams', [])
            youtube_data['videos'] = coverage.get('videos', [])
            youtube_data['sentiment'] = coverage.get('sentiment', {})
            
            print(f"✅ YouTube: Found {len(youtube_data['videos'])} videos and {len(youtube_data['live_streams'])} live streams")
            
        except Exception as e:
            print(f"❌ YouTube integration error: {e}")
            youtube_data['status'] = 'ERROR'
            youtube_data['error'] = str(e)
        
        return youtube_data
    
    async def get_defeatbeta_data(self, symbol: str) -> Dict[str, Any]:
        """Get Defeat Beta data (corrected for actual capabilities)"""
        try:
            from defeatbeta_api_integration import DefeatBetaAPIIntegration
            defeatbeta = DefeatBetaAPIIntegration()
            
            # Get available data types
            data_sources = {}
            working_sources = 0
            total_sources = 0
            
            # Stock price data
            total_sources += 1
            price_data = await defeatbeta.get_stock_price_data(symbol)
            if price_data.get('status') == 'WORKING':
                data_sources['price'] = price_data
                working_sources += 1
            
            # News data
            total_sources += 1
            news_data = await defeatbeta.get_news_data(symbol)
            if news_data.get('status') == 'WORKING':
                data_sources['news'] = news_data
                working_sources += 1
            
            # Revenue data
            total_sources += 1
            revenue_data = await defeatbeta.get_revenue_data(symbol)
            if revenue_data.get('status') == 'WORKING':
                data_sources['revenue'] = revenue_data
                working_sources += 1
            
            # Financial statements (limited)
            total_sources += 1
            financial_data = await defeatbeta.get_financial_statements(symbol)
            if financial_data.get('status') == 'WORKING':
                data_sources['financial'] = financial_data
                working_sources += 1
            
            # Earnings data (basic)
            total_sources += 1
            earnings_data = await defeatbeta.get_earnings_data(symbol)
            if earnings_data.get('status') == 'WORKING':
                data_sources['earnings'] = earnings_data
                working_sources += 1
            
            success_rate = (working_sources / total_sources * 100) if total_sources > 0 else 0
            
            return {
                'status': 'WORKING' if working_sources > 0 else 'ERROR',
                'summary': f'Defeat Beta: {working_sources}/{total_sources} sources working',
                'data': {
                    'working_sources': working_sources,
                    'total_sources': total_sources,
                    'success_rate': success_rate,
                    'available_data': list(data_sources.keys()),
                    'note': 'Limited to basic stock data, news, and revenue. Use SEC/FMP for financial statements.'
                }
            }
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Defeat Beta error: {str(e)}'}
    
    async def get_sec_filings_data(self, symbol: str) -> Dict[str, Any]:
        """Get SEC filings data (WORKING)"""
        print(f"📋 Getting SEC filings data for {symbol}...")
        
        # Import SEC integration
        try:
            from sec_filings_integration import SECFilingsIntegration
            sec = SECFilingsIntegration()
            sec_data = await sec.get_comprehensive_sec_data(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': sec_data,
                'status': 'WORKING'
            }
        except Exception as e:
            print(f"❌ SEC integration error: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': {},
                'status': 'ERROR',
                'error': str(e)
            }
    
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
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data from all sources with improved error handling"""
        print(f"🎯 Getting comprehensive data for {symbol}...")
        
        # Validate symbol first
        validation_result = self._validate_symbol(symbol)
        if not validation_result['is_valid']:
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'validation': validation_result,
                'error': 'Invalid symbol detected',
                'status': 'INVALID_SYMBOL'
            }
        
        start_time = time.time()
        
        # Collect data from all sources
        tasks = [
            self.get_news_sentiment(symbol),
            self.get_youtube_live_news(symbol),
            self.get_defeatbeta_data(symbol),
            self.get_sec_filings_data(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'collection_time': round(time.time() - start_time, 2),
            'validation': validation_result,
            'sources': {},
            'summary': {},
            'status_report': {}
        }
        
        # Process results
        source_names = ['news', 'youtube', 'defeatbeta', 'sec']
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Error collecting data: {result}")
                comprehensive_data['sources'][source_names[i]] = {
                    'status': 'ERROR',
                    'error': str(result)
                }
                continue
            
            source_name = source_names[i]
            comprehensive_data['sources'][source_name] = result
            comprehensive_data['status_report'][source_name] = result.get('status', 'UNKNOWN')
        
        # Generate summary
        comprehensive_data['summary'] = self._generate_comprehensive_summary(comprehensive_data)
        
        return comprehensive_data
    
    def _validate_symbol(self, symbol: str) -> Dict[str, Any]:
        """Validate if a symbol is properly formatted"""
        import re
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'symbol': symbol
        }
        
        # Check if symbol is None or empty
        if not symbol or not isinstance(symbol, str):
            validation_result['is_valid'] = False
            validation_result['errors'].append('Symbol is None or empty')
            return validation_result
        
        # Check length - allow single letters (like "A" for Agilent)
        if len(symbol) < 1:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Symbol too short: {len(symbol)} characters (minimum 1)')
        
        if len(symbol) > 15:  # Increased limit for longer valid symbols
            validation_result['warnings'].append(f'Symbol unusually long: {len(symbol)} characters')
        
        # Check for invalid patterns - more permissive
        invalid_patterns = [
            r'^[0-9]+$',  # Pure numbers (but allow some numeric symbols)
            r'^[^A-Za-z0-9]+$',  # No alphanumeric characters
            r'^[A-Za-z]{20,}$',  # Too long (increased from 10)
            r'INVALID_',  # Invalid prefix
            r'TEST_',  # Test prefix
            r'DUMMY_',  # Dummy prefix
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, symbol):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'Symbol matches invalid pattern: {pattern}')
        
        # Check for known invalid symbols - more specific
        known_invalid_symbols = [
            'INVALID_SYMBOL_12345',
            'TEST_SYMBOL',
            'DUMMY_STOCK',
            'VERYLONGSYMBOLNAME123456789',  # Too long
        ]
        
        if symbol.upper() in [s.upper() for s in known_invalid_symbols]:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Symbol is in known invalid list')
        
        # Check for special characters - more permissive
        if re.search(r'[^A-Za-z0-9\.\-]', symbol):  # Allow dots and hyphens
            validation_result['is_valid'] = False
            validation_result['errors'].append('Symbol contains invalid special characters')
        
        # Check if it's all uppercase (common for stock symbols) - warning only
        if not symbol.isupper():
            validation_result['warnings'].append('Symbol should typically be uppercase')
        
        return validation_result
    
    def _generate_comprehensive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary and insights"""
        summary = {
            'data_coverage': {},
            'sentiment_analysis': {},
            'live_news_coverage': {},
            'financial_data': {},
            'institutional_insights': {},
            'overall_score': 0.0
        }
        
        # Data coverage analysis
        sources = data.get('sources', {})
        working_sources = sum(1 for source in sources.values() if source.get('status') == 'WORKING')
        total_sources = len(sources)
        
        summary['data_coverage'] = {
            'working_sources': working_sources,
            'total_sources': total_sources,
            'coverage_percentage': (working_sources / total_sources * 100) if total_sources > 0 else 0
        }
        
        # Sentiment analysis
        news_data = sources.get('news', {})
        if 'aggregated_sentiment' in news_data:
            sentiment = news_data['aggregated_sentiment']
            summary['sentiment_analysis'] = {
                'overall_compound': sentiment.get('overall_compound', 0.0),
                'confidence': sentiment.get('confidence', 0.0),
                'total_items': sentiment.get('total_items', 0),
                'trend': 'bullish' if sentiment.get('overall_compound', 0.0) >= 0.3 else 'bearish' if sentiment.get('overall_compound', 0.0) <= -0.3 else 'neutral'
            }
        
        # Live news coverage
        youtube_data = sources.get('youtube', {})
        if youtube_data:
            summary['live_news_coverage'] = {
                'live_streams': len(youtube_data.get('live_streams', [])),
                'videos': len(youtube_data.get('videos', [])),
                'sentiment_score': youtube_data.get('sentiment', {}).get('overall_sentiment', 0.0)
            }
        
        # Financial data
        defeatbeta_data = sources.get('defeatbeta', {})
        if defeatbeta_data and isinstance(defeatbeta_data, dict):
            # Handle the corrected Defeat Beta structure
            if 'data' in defeatbeta_data:
                defeatbeta_summary = defeatbeta_data['data']
                summary['financial_data'] = {
                    'working_sources': defeatbeta_summary.get('working_sources', 0),
                    'total_sources': defeatbeta_summary.get('total_sources', 0),
                    'success_rate': defeatbeta_summary.get('success_rate', 0.0),
                    'available_data': defeatbeta_summary.get('available_data', [])
                }
            elif 'summary' in defeatbeta_data:
                # Fallback for old structure
                defeatbeta_summary = defeatbeta_data['summary']
                summary['financial_data'] = {
                    'working_sources': defeatbeta_summary.get('working_sources', 0),
                    'data_points': defeatbeta_summary.get('data_points', 0),
                    'success_rate': defeatbeta_summary.get('success_rate', 0.0)
                }
            else:
                summary['financial_data'] = {
                    'working_sources': 0,
                    'total_sources': 0,
                    'success_rate': 0.0,
                    'available_data': []
                }
        
        # Institutional insights
        sec_data = sources.get('sec', {})
        if 'data' in sec_data and 'insights' in sec_data['data']:
            sec_insights = sec_data['data']['insights']
            summary['institutional_insights'] = {
                'insider_activity': sec_insights.get('insider_activity', 'unknown'),
                'institutional_interest': sec_insights.get('institutional_interest', 'unknown'),
                'recent_events': sec_insights.get('recent_events', 0),
                'filing_activity': sec_insights.get('filing_activity', 'unknown')
            }
        
        # Calculate overall score
        score = 0.0
        score += summary['data_coverage']['coverage_percentage'] * 0.25  # 25% weight
        score += abs(summary['sentiment_analysis'].get('overall_compound', 0.0)) * 0.2  # 20% weight
        score += summary['sentiment_analysis'].get('confidence', 0.0) * 0.2  # 20% weight
        score += (1.0 if summary['live_news_coverage'].get('videos', 0) > 0 or summary['live_news_coverage'].get('live_streams', 0) > 0 else 0.0) * 0.2  # 20% weight
        score += (summary['financial_data'].get('success_rate', 0.0) / 100) * 0.15  # 15% weight
        
        summary['overall_score'] = round(score, 2)
        
        return summary

async def main():
    """Demo the comprehensive data integration"""
    print("🚀 Comprehensive Data Integration - Phase 3")
    print("="*60)
    
    integration = ComprehensiveDataIntegrationPhase3()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting comprehensive data for {symbol}...")
    
    comprehensive_data = await integration.get_comprehensive_data(symbol)
    
    # Print results
    print(f"\n✅ Data collection complete!")
    print(f"⏱️ Collection time: {comprehensive_data['collection_time']} seconds")
    
    # Print status report
    print(f"\n📋 STATUS REPORT:")
    for source, status in comprehensive_data['status_report'].items():
        status_emoji = {
            'WORKING': '✅',
            'NO_API_KEY': '❌',
            'RATE_LIMITED': '⏱️',
            'ERROR': '💥',
            'NOT_INSTALLED': '📦'
        }.get(status, '❓')
        print(f"   {status_emoji} {source.capitalize()}: {status}")
    
    # Print comprehensive summary
    summary = comprehensive_data['summary']
    print(f"\n📈 COMPREHENSIVE SUMMARY:")
    print(f"   Overall Score: {summary['overall_score']}/1.0")
    print(f"   Data Coverage: {summary['data_coverage']['coverage_percentage']:.1f}%")
    print(f"   Working Sources: {summary['data_coverage']['working_sources']}/{summary['data_coverage']['total_sources']}")
    
    # Print sentiment analysis
    sentiment = summary['sentiment_analysis']
    if sentiment:
        print(f"\n🎯 SENTIMENT ANALYSIS:")
        print(f"   Trend: {sentiment['trend'].upper()}")
        print(f"   Compound Score: {sentiment['overall_compound']:.3f}")
        print(f"   Confidence: {sentiment['confidence']:.2%}")
        print(f"   Total Items: {sentiment['total_items']}")
    
    # Print live news coverage
    live_news = summary['live_news_coverage']
    if live_news:
        print(f"\n📺 LIVE NEWS COVERAGE:")
        print(f"   Live Streams: {live_news['live_streams']}")
        print(f"   Videos: {live_news['videos']}")
        print(f"   Sentiment Score: {live_news['sentiment_score']:.3f}")
    
    # Print financial data
    financial = summary['financial_data']
    if financial:
        print(f"\n📊 FINANCIAL DATA:")
        print(f"   Working Sources: {financial['working_sources']}")
        print(f"   Data Points: {financial['data_points']}")
        print(f"   Success Rate: {financial['success_rate']:.1f}%")
    
    # Print institutional insights
    institutional = summary['institutional_insights']
    if institutional:
        print(f"\n🏢 INSTITUTIONAL INSIGHTS:")
        print(f"   Insider Activity: {institutional['insider_activity']}")
        print(f"   Institutional Interest: {institutional['institutional_interest']}")
        print(f"   Recent Events: {institutional['recent_events']}")
        print(f"   Filing Activity: {institutional['filing_activity']}")
    
    print(f"\n🎉 Comprehensive data integration demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
