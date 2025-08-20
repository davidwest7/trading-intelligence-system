#!/usr/bin/env python3
"""
YouTube API & Google Trends Integration
Practical implementation with quota management and rate limiting
"""
import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

class YouTubeQuotaManager:
    """Manages YouTube API quota usage efficiently"""
    
    def __init__(self):
        self.daily_quota = 10000
        self.used_quota = 0
        self.reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Cost mapping for different operations
        self.cost_map = {
            'search': 100,
            'video_details': 1,
            'comments': 1,
            'live_streams': 100,
            'channel_details': 1
        }
    
    def can_make_request(self, operation: str) -> bool:
        """Check if we can make a request without exceeding quota"""
        self._check_reset()
        cost = self.cost_map.get(operation, 1)
        return (self.used_quota + cost) <= self.daily_quota
    
    def record_request(self, operation: str):
        """Record a request and its cost"""
        self._check_reset()
        cost = self.cost_map.get(operation, 1)
        self.used_quota += cost
        print(f"📊 YouTube API: Used {cost} units, Total: {self.used_quota}/{self.daily_quota}")
    
    def _check_reset(self):
        """Check if daily quota should reset"""
        if datetime.now() >= self.reset_time:
            self.used_quota = 0
            self.reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            print("🔄 YouTube API: Daily quota reset")

class YouTubeFinancialMonitor:
    """Monitors financial content on YouTube"""
    
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.quota_manager = YouTubeQuotaManager()
        
        # Financial channels to monitor
        self.financial_channels = {
            'CNBC': 'UCvJJ_dzjViJCoLf5uKUTwoxg',
            'Bloomberg TV': 'UCUMZ7gohGIzHESfsqNWgJzA',
            'Yahoo Finance': 'UCWJ8lQ7Hly9m8OOGfdQp1KA',
            'MarketWatch': 'UCJqFtip6NjHioctNqS8NqYA',
            'Reuters Business': 'UCZcHhYVQ1ioavJ0JkRgdHLQ'
        }
    
    async def get_financial_videos(self, symbol: str, max_results: int = 10) -> Dict[str, Any]:
        """Get recent financial videos for a symbol"""
        if not self.quota_manager.can_make_request('search'):
            return {'status': 'QUOTA_EXCEEDED', 'error': 'Daily quota exceeded'}
        
        if not self.api_key:
            return {'status': 'NO_API_KEY', 'error': 'YouTube API key not available'}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for financial videos about the symbol
                search_params = {
                    'part': 'snippet',
                    'q': f'{symbol} stock earnings financial',
                    'type': 'video',
                    'videoCategoryId': '25',  # News & Politics
                    'order': 'date',
                    'maxResults': max_results,
                    'key': self.api_key
                }
                
                url = f"{self.base_url}/search"
                async with session.get(url, params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.quota_manager.record_request('search')
                        
                        videos = []
                        for item in data.get('items', []):
                            video_data = {
                                'video_id': item['id']['videoId'],
                                'title': item['snippet']['title'],
                                'description': item['snippet']['description'],
                                'published_at': item['snippet']['publishedAt'],
                                'channel_title': item['snippet']['channelTitle'],
                                'thumbnails': item['snippet']['thumbnails']
                            }
                            videos.append(video_data)
                        
                        return {
                            'status': 'WORKING',
                            'symbol': symbol,
                            'videos_found': len(videos),
                            'videos': videos,
                            'quota_used': self.quota_manager.used_quota
                        }
                    else:
                        return {'status': 'ERROR', 'error': f'API request failed: {response.status}'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'YouTube error: {str(e)}'}
    
    async def get_video_sentiment(self, video_id: str) -> Dict[str, Any]:
        """Analyze video comments for sentiment"""
        if not self.quota_manager.can_make_request('comments'):
            return {'status': 'QUOTA_EXCEEDED', 'error': 'Daily quota exceeded'}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get video comments
                comment_params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'maxResults': 100,
                    'order': 'relevance',
                    'key': self.api_key
                }
                
                url = f"{self.base_url}/commentThreads"
                async with session.get(url, params=comment_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.quota_manager.record_request('comments')
                        
                        comments = []
                        total_sentiment = 0
                        
                        for item in data.get('items', []):
                            comment = item['snippet']['topLevelComment']['snippet']
                            comments.append({
                                'text': comment['textDisplay'],
                                'author': comment['authorDisplayName'],
                                'likes': comment['likeCount'],
                                'published_at': comment['publishedAt']
                            })
                        
                        # Simple sentiment analysis (can be enhanced)
                        sentiment_keywords = {
                            'positive': ['bullish', 'buy', 'strong', 'growth', 'profit', 'up', 'gain'],
                            'negative': ['bearish', 'sell', 'weak', 'loss', 'down', 'crash', 'drop']
                        }
                        
                        for comment in comments:
                            text_lower = comment['text'].lower()
                            positive_count = sum(1 for word in sentiment_keywords['positive'] if word in text_lower)
                            negative_count = sum(1 for word in sentiment_keywords['negative'] if word in text_lower)
                            
                            if positive_count > negative_count:
                                total_sentiment += 1
                            elif negative_count > positive_count:
                                total_sentiment -= 1
                        
                        avg_sentiment = total_sentiment / len(comments) if comments else 0
                        
                        return {
                            'status': 'WORKING',
                            'video_id': video_id,
                            'comments_analyzed': len(comments),
                            'average_sentiment': round(avg_sentiment, 4),
                            'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
                        }
                    else:
                        return {'status': 'ERROR', 'error': f'Comments request failed: {response.status}'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Comment analysis error: {str(e)}'}

class GoogleTrendsRateLimiter:
    """Manages Google Trends API rate limiting"""
    
    def __init__(self):
        self.requests_per_minute = 5
        self.request_times = []
    
    async def make_request(self, func, *args, **kwargs):
        """Make a rate-limited request"""
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we've made too many requests, wait
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                print(f"⏱️ Google Trends: Rate limited, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(time.time())
        
        # Make the actual request
        return await func(*args, **kwargs)

class GoogleTrendsAnalyzer:
    """Analyzes Google Trends data for financial insights"""
    
    def __init__(self):
        self.rate_limiter = GoogleTrendsRateLimiter()
        self.pytrends_available = False
        
        # Try to import pytrends
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.pytrends_available = True
            print("✅ Google Trends: pytrends library available")
        except ImportError:
            print("⚠️ Google Trends: pytrends library not available")
    
    async def get_stock_trends(self, symbol: str, timeframe: str = 'today 3-m') -> Dict[str, Any]:
        """Get search interest trends for a stock"""
        if not self.pytrends_available:
            return {'status': 'LIBRARY_MISSING', 'error': 'pytrends library not available'}
        
        try:
            # Build payload
            self.pytrends.build_payload([symbol], cat=0, timeframe=timeframe, geo='', gprop='')
            
            # Get interest over time
            interest_data = self.pytrends.interest_over_time()
            
            if interest_data.empty:
                return {'status': 'NO_DATA', 'error': 'No trend data available'}
            
            # Calculate trend metrics
            recent_data = interest_data.tail(7)  # Last 7 days
            avg_interest = recent_data[symbol].mean()
            trend_direction = 'up' if recent_data[symbol].iloc[-1] > recent_data[symbol].iloc[0] else 'down'
            volatility = recent_data[symbol].std()
            
            return {
                'status': 'WORKING',
                'symbol': symbol,
                'average_interest': round(avg_interest, 2),
                'trend_direction': trend_direction,
                'volatility': round(volatility, 2),
                'recent_data': recent_data[symbol].tolist(),
                'timeframe': timeframe
            }
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Trends error: {str(e)}'}
    
    async def get_related_topics(self, symbol: str) -> Dict[str, Any]:
        """Get related topics and queries for a stock"""
        if not self.pytrends_available:
            return {'status': 'LIBRARY_MISSING', 'error': 'pytrends library not available'}
        
        try:
            # Build payload
            self.pytrends.build_payload([symbol], cat=0, timeframe='today 3-m', geo='', gprop='')
            
            # Get related topics
            related_topics = self.pytrends.related_topics()
            related_queries = self.pytrends.related_queries()
            
            topics_data = {
                'top_topics': [],
                'rising_topics': [],
                'top_queries': [],
                'rising_queries': []
            }
            
            if symbol in related_topics and related_topics[symbol]['top'] is not None:
                topics_data['top_topics'] = related_topics[symbol]['top'].head(10).to_dict('records')
            
            if symbol in related_topics and related_topics[symbol]['rising'] is not None:
                topics_data['rising_topics'] = related_topics[symbol]['rising'].head(10).to_dict('records')
            
            if symbol in related_queries and related_queries[symbol]['top'] is not None:
                topics_data['top_queries'] = related_queries[symbol]['top'].head(10).to_dict('records')
            
            if symbol in related_queries and related_queries[symbol]['rising'] is not None:
                topics_data['rising_queries'] = related_queries[symbol]['rising'].head(10).to_dict('records')
            
            return {
                'status': 'WORKING',
                'symbol': symbol,
                'data': topics_data
            }
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Related topics error: {str(e)}'}
    
    async def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare search interest between multiple stocks"""
        if not self.pytrends_available:
            return {'status': 'LIBRARY_MISSING', 'error': 'pytrends library not available'}
        
        if len(symbols) > 5:
            return {'status': 'ERROR', 'error': 'Maximum 5 symbols allowed for comparison'}
        
        try:
            # Build payload for multiple symbols
            self.pytrends.build_payload(symbols, cat=0, timeframe='today 3-m', geo='', gprop='')
            
            # Get interest over time
            interest_data = self.pytrends.interest_over_time()
            
            if interest_data.empty:
                return {'status': 'NO_DATA', 'error': 'No comparison data available'}
            
            # Calculate relative interest
            recent_avg = interest_data.tail(7).mean()
            comparison_data = {}
            
            for symbol in symbols:
                if symbol in recent_avg:
                    comparison_data[symbol] = {
                        'average_interest': round(recent_avg[symbol], 2),
                        'rank': len([s for s in symbols if recent_avg.get(s, 0) > recent_avg[symbol]]) + 1
                    }
            
            return {
                'status': 'WORKING',
                'symbols': symbols,
                'comparison': comparison_data,
                'most_popular': max(comparison_data.items(), key=lambda x: x[1]['average_interest'])[0]
            }
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Comparison error: {str(e)}'}

class EnhancedTradingIntelligence:
    """Enhanced trading intelligence with YouTube and Google Trends"""
    
    def __init__(self):
        self.youtube = YouTubeFinancialMonitor()
        self.trends = GoogleTrendsAnalyzer()
    
    async def get_comprehensive_social_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive social and trending analysis"""
        print(f"🎯 Getting comprehensive social analysis for {symbol}...")
        
        start_time = time.time()
        
        # Collect data from both sources
        tasks = [
            self.youtube.get_financial_videos(symbol),
            self.trends.get_stock_trends(symbol),
            self.trends.get_related_topics(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'collection_time': round(time.time() - start_time, 2),
            'sources': {},
            'summary': {}
        }
        
        source_names = ['youtube', 'trends', 'related_topics']
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                analysis['sources'][source_names[i]] = {
                    'status': 'ERROR',
                    'error': str(result)
                }
            else:
                analysis['sources'][source_names[i]] = result
        
        # Generate summary
        analysis['summary'] = self._generate_social_summary(analysis['sources'])
        
        return analysis
    
    def _generate_social_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of social analysis"""
        summary = {
            'social_sentiment': {},
            'trending_analysis': {},
            'overall_score': 0.0
        }
        
        # YouTube analysis
        youtube_data = sources.get('youtube', {})
        if youtube_data.get('status') == 'WORKING':
            summary['social_sentiment']['youtube'] = {
                'videos_found': youtube_data.get('videos_found', 0),
                'quota_used': youtube_data.get('quota_used', 0),
                'status': 'active'
            }
        
        # Google Trends analysis
        trends_data = sources.get('trends', {})
        if trends_data.get('status') == 'WORKING':
            summary['trending_analysis']['search_interest'] = {
                'average_interest': trends_data.get('average_interest', 0),
                'trend_direction': trends_data.get('trend_direction', 'neutral'),
                'volatility': trends_data.get('volatility', 0)
            }
        
        # Related topics
        related_data = sources.get('related_topics', {})
        if related_data.get('status') == 'WORKING':
            data = related_data.get('data', {})
            summary['trending_analysis']['related_content'] = {
                'top_topics': len(data.get('top_topics', [])),
                'rising_topics': len(data.get('rising_topics', [])),
                'top_queries': len(data.get('top_queries', [])),
                'rising_queries': len(data.get('rising_queries', []))
            }
        
        # Calculate overall score
        score = 0.0
        
        # YouTube score (30% weight)
        if youtube_data.get('status') == 'WORKING':
            score += 30.0
        
        # Trends score (40% weight)
        if trends_data.get('status') == 'WORKING':
            score += 40.0
        
        # Related topics score (30% weight)
        if related_data.get('status') == 'WORKING':
            score += 30.0
        
        summary['overall_score'] = round(score, 2)
        
        return summary

async def main():
    """Demo the enhanced social analysis"""
    print("🚀 Enhanced Social Analysis Demo")
    print("=" * 50)
    
    intelligence = EnhancedTradingIntelligence()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting comprehensive social analysis for {symbol}...")
    
    analysis = await intelligence.get_comprehensive_social_analysis(symbol)
    
    # Print results
    print(f"\n✅ Analysis complete!")
    print(f"⏱️ Collection time: {analysis['collection_time']} seconds")
    print(f"📈 Overall score: {analysis['summary']['overall_score']}/100")
    
    # Print detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for source_name, source_data in analysis['sources'].items():
        status = source_data.get('status', 'UNKNOWN')
        emoji = '✅' if status == 'WORKING' else '❌'
        print(f"   {emoji} {source_name.upper()}: {status}")
        
        if status == 'WORKING':
            if source_name == 'youtube':
                print(f"      📺 Videos: {source_data.get('videos_found', 0)}")
                print(f"      📊 Quota: {source_data.get('quota_used', 0)}/10000")
            elif source_name == 'trends':
                print(f"      📈 Interest: {source_data.get('average_interest', 0)}")
                print(f"      📊 Trend: {source_data.get('trend_direction', 'neutral')}")
            elif source_name == 'related_topics':
                data = source_data.get('data', {})
                print(f"      🔍 Topics: {len(data.get('top_topics', []))}")
                print(f"      📈 Rising: {len(data.get('rising_topics', []))}")

if __name__ == "__main__":
    asyncio.run(main())
