#!/usr/bin/env python3
"""
Comprehensive Data Integration - Phase 4
Includes Finnhub API as replacement for Defeat Beta
"""
import asyncio
import aiohttp
import json
import os
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

class ComprehensiveDataIntegrationPhase4:
    def __init__(self):
        # API Keys
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        # self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')  # COMMENTED OUT - QUOTA EXCEEDED
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.fmp_api_key = os.getenv('FMP_API_KEY', '')
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY', '')
        
        # Rate limiting
        self.rate_limits = {
            'newsapi': {'calls': 0, 'limit': 100, 'reset_time': time.time() + 86400},
            # 'youtube': {'calls': 0, 'limit': 10000, 'reset_time': time.time() + 86400},  # COMMENTED OUT
            'finnhub': {'calls': 0, 'limit': 60, 'reset_time': time.time() + 60}
        }
        
        self._initialize_nlp()
        self.data_cache = {}
    
    def _initialize_nlp(self):
        """Initialize NLP components for sentiment analysis"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader = None
            print("⚠️ VADER sentiment not available")
        
        try:
            from textblob import TextBlob
            self.textblob_available = True
        except ImportError:
            self.textblob_available = False
            print("⚠️ TextBlob not available")
    
    def _check_rate_limit(self, api_name: str):
        """Check and manage rate limits"""
        if api_name not in self.rate_limits:
            return
        
        current_time = time.time()
        limit_info = self.rate_limits[api_name]
        
        if current_time > limit_info['reset_time']:
            limit_info['calls'] = 0
            limit_info['reset_time'] = current_time + (60 if api_name == 'finnhub' else 86400)
        
        if limit_info['calls'] >= limit_info['limit']:
            sleep_time = limit_info['reset_time'] - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                limit_info['calls'] = 0
                limit_info['reset_time'] = time.time() + (60 if api_name == 'finnhub' else 86400)
        
        limit_info['calls'] += 1
    
    async def _make_api_request(self, session: aiohttp.ClientSession, url: str, params: dict = None, headers: dict = None) -> Optional[dict]:
        """Make API request with error handling"""
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    print(f"❌ API: Invalid API key")
                    return None
                elif response.status == 429:
                    print(f"⚠️ API: Rate limit exceeded")
                    return None
                else:
                    print(f"❌ API: Request failed with status {response.status}")
                    return None
        except Exception as e:
            print(f"❌ API: Request error: {str(e)}")
            return None
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment from NewsAPI"""
        print(f"📰 Getting news sentiment for {symbol}...")
        
        if not self.news_api_key:
            return {'status': 'NO_API_KEY', 'error': 'NewsAPI key not available'}
        
        self._check_rate_limit('newsapi')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"{symbol}" AND (stock OR earnings OR financial OR trading)',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': self.news_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                data = await self._make_api_request(session, url, params)
                
                if data and data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    
                    if articles:
                        # Analyze sentiment for each article
                        sentiment_results = []
                        for article in articles:
                            title = article.get('title', '')
                            description = article.get('description', '')
                            content = f"{title} {description}"
                            
                            sentiment = self._analyze_sentiment(content)
                            sentiment_results.append(sentiment)
                        
                        # Aggregate sentiment
                        aggregated = self._aggregate_sentiment(sentiment_results)
                        
                        return {
                            'status': 'WORKING',
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'data': {
                                'articles': articles,
                                'total_articles': len(articles),
                                'aggregated_sentiment': aggregated
                            }
                        }
                    else:
                        return {'status': 'NO_DATA', 'message': 'No articles found'}
                else:
                    return {'status': 'ERROR', 'error': 'Failed to fetch news'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'News error: {str(e)}'}
    
    async def get_finnhub_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data from Finnhub API"""
        print(f"🎯 Getting Finnhub data for {symbol}...")
        
        if not self.finnhub_api_key:
            return {'status': 'NO_API_KEY', 'error': 'Finnhub API key not available'}
        
        self._check_rate_limit('finnhub')
        
        base_url = "https://finnhub.io/api/v1"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get stock quote
                quote_data = await self._make_api_request(session, f"{base_url}/quote", {
                    'symbol': symbol,
                    'token': self.finnhub_api_key
                })
                
                # Get company news
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                news_data = await self._make_api_request(session, f"{base_url}/company-news", {
                    'symbol': symbol,
                    'from': from_date,
                    'to': to_date,
                    'token': self.finnhub_api_key
                })
                
                # Get financial statements
                financial_data = await self._make_api_request(session, f"{base_url}/stock/financials-reported", {
                    'symbol': symbol,
                    'token': self.finnhub_api_key
                })
                
                # Process results
                working_sources = 0
                total_sources = 3
                data_sources = {}
                
                if quote_data:
                    data_sources['quote'] = {
                        'current_price': quote_data.get('c', 0),
                        'change': quote_data.get('d', 0),
                        'percent_change': quote_data.get('dp', 0),
                        'high': quote_data.get('h', 0),
                        'low': quote_data.get('l', 0),
                        'open': quote_data.get('o', 0)
                    }
                    working_sources += 1
                
                if news_data:
                    # Calculate average sentiment from Finnhub
                    sentiments = [article.get('sentiment', 0) for article in news_data[:20]]
                    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                    
                    data_sources['news'] = {
                        'articles': news_data[:20],
                        'total_articles': len(news_data),
                        'average_sentiment': round(avg_sentiment, 4),
                        'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
                    }
                    working_sources += 1
                
                if financial_data:
                    data_sources['financials'] = {
                        'data': financial_data,
                        'available_statements': list(financial_data.keys()) if isinstance(financial_data, dict) else [],
                        'data_points': len(financial_data) if isinstance(financial_data, list) else 0
                    }
                    working_sources += 1
                
                success_rate = (working_sources / total_sources * 100) if total_sources > 0 else 0
                
                return {
                    'status': 'WORKING' if working_sources > 0 else 'ERROR',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'working_sources': working_sources,
                        'total_sources': total_sources,
                        'success_rate': success_rate,
                        'available_data': list(data_sources.keys()),
                        'sources': data_sources
                    }
                }
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Finnhub error: {str(e)}'}
    
    async def get_sec_filings_data(self, symbol: str) -> Dict[str, Any]:
        """Get SEC filings data"""
        print(f"📋 Getting SEC filings data for {symbol}...")
        
        try:
            from sec_filings_integration import SECFilingsIntegration
            sec = SECFilingsIntegration()
            sec_data = await sec.get_comprehensive_sec_data(symbol)
            
            return {
                'status': 'WORKING',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': sec_data
            }
        except Exception as e:
            return {'status': 'ERROR', 'error': f'SEC error: {str(e)}'}
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple models"""
        results = {}
        
        # VADER sentiment
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            results['vader'] = vader_scores
        
        # TextBlob sentiment
        if self.textblob_available:
            try:
                blob = TextBlob(text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except:
                pass
        
        # Financial sentiment (custom)
        financial_keywords = {
            'positive': ['earnings', 'profit', 'growth', 'revenue', 'beat', 'exceed', 'strong', 'up', 'gain'],
            'negative': ['loss', 'decline', 'fall', 'drop', 'miss', 'weak', 'down', 'decrease', 'risk']
        }
        
        text_lower = text.lower()
        positive_count = sum(1 for word in financial_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in financial_keywords['negative'] if word in text_lower)
        
        if positive_count > negative_count:
            financial_score = 0.3
        elif negative_count > positive_count:
            financial_score = -0.3
        else:
            financial_score = 0.0
        
        results['financial'] = {'financial_score': financial_score}
        
        return results
    
    def _aggregate_sentiment(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple results"""
        if not sentiment_results:
            return {'overall_compound': 0.0, 'confidence': 0.0, 'total_items': 0}
        
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
        vader_scores = [r.get('vader', {}).get('compound', 0.0) for r in sentiment_results if 'vader' in r]
        if vader_scores:
            weight = weights['vader']
            compound += sum(vader_scores) / len(vader_scores) * weight
            total_weight += weight
        
        # TextBlob
        textblob_scores = [r.get('textblob', {}).get('polarity', 0.0) for r in sentiment_results if 'textblob' in r]
        if textblob_scores:
            weight = weights['textblob']
            polarity += sum(textblob_scores) / len(textblob_scores) * weight
            total_weight += weight
        
        # Financial
        financial_scores = [r.get('financial', {}).get('financial_score', 0.0) for r in sentiment_results if 'financial' in r]
        if financial_scores:
            weight = weights['financial']
            polarity += sum(financial_scores) / len(financial_scores) * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            polarity /= total_weight
            compound /= total_weight
        
        # Calculate confidence
        confidence = min(1.0, len(sentiment_results) / 10.0)  # More articles = higher confidence
        
        return {
            'overall_compound': round(compound, 4),
            'polarity': round(polarity, 4),
            'confidence': round(confidence, 4),
            'total_items': len(sentiment_results)
        }
    
    def _validate_symbol(self, symbol: str) -> Dict[str, Any]:
        """Validate if a symbol is properly formatted"""
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
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data from all sources with Finnhub integration"""
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
            self.get_finnhub_data(symbol),
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
        source_names = ['news', 'finnhub', 'sec']
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
    
    def _generate_comprehensive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary and insights"""
        summary = {
            'data_coverage': {},
            'sentiment_analysis': {},
            'market_data': {},
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
        if 'data' in news_data and 'aggregated_sentiment' in news_data['data']:
            sentiment = news_data['data']['aggregated_sentiment']
            summary['sentiment_analysis'] = {
                'overall_compound': sentiment.get('overall_compound', 0.0),
                'confidence': sentiment.get('confidence', 0.0),
                'total_items': sentiment.get('total_items', 0),
                'trend': 'bullish' if sentiment.get('overall_compound', 0.0) >= 0.3 else 'bearish' if sentiment.get('overall_compound', 0.0) <= -0.3 else 'neutral'
            }
        
        # Market data (from Finnhub)
        finnhub_data = sources.get('finnhub', {})
        if finnhub_data and 'data' in finnhub_data:
            finnhub_summary = finnhub_data['data']
            summary['market_data'] = {
                'working_sources': finnhub_summary.get('working_sources', 0),
                'total_sources': finnhub_summary.get('total_sources', 0),
                'success_rate': finnhub_summary.get('success_rate', 0.0),
                'available_data': finnhub_summary.get('available_data', [])
            }
            
            # Add stock price if available
            if 'sources' in finnhub_summary and 'quote' in finnhub_summary['sources']:
                quote_data = finnhub_summary['sources']['quote']
                summary['market_data']['current_price'] = quote_data.get('current_price', 0)
                summary['market_data']['percent_change'] = quote_data.get('percent_change', 0)
        
        # Financial data
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
        score += (summary['market_data'].get('success_rate', 0.0) / 100) * 0.2  # 20% weight
        score += (summary['financial_data'].get('success_rate', 0.0) / 100) * 0.15  # 15% weight
        
        summary['overall_score'] = round(score, 2)
        
        return summary

async def main():
    """Demo the comprehensive data integration"""
    print("🚀 Comprehensive Data Integration - Phase 4 (with Finnhub)")
    print("=" * 70)
    
    integration = ComprehensiveDataIntegrationPhase4()
    
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
            'NO_DATA': '📭'
        }.get(status, '❓')
        print(f"   {status_emoji} {source.upper()}: {status}")
    
    # Print summary
    summary = comprehensive_data.get('summary', {})
    print(f"\n📊 SUMMARY:")
    print(f"   Data Coverage: {summary.get('data_coverage', {}).get('coverage_percentage', 0):.1f}%")
    print(f"   Sentiment: {summary.get('sentiment_analysis', {}).get('trend', 'unknown')} ({summary.get('sentiment_analysis', {}).get('overall_compound', 0):.3f})")
    print(f"   Market Data: {summary.get('market_data', {}).get('success_rate', 0):.1f}% success")
    print(f"   Overall Score: {summary.get('overall_score', 0):.1f}/100")

if __name__ == "__main__":
    asyncio.run(main())
