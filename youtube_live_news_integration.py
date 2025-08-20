#!/usr/bin/env python3
"""
YouTube Live News Integration
Monitor live financial news during key events like earnings announcements
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

class YouTubeLiveNewsIntegration:
    """YouTube live news integration for financial monitoring"""
    
    def __init__(self):
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
        # Financial news channels to monitor
        self.financial_channels = {
            'CNBC': 'UCvJJ_dzjViJCoLf5uKUTwoxg',
            'Bloomberg TV': 'UCUMZ7gohGIzHESfsqNWgJzA',
            'Yahoo Finance': 'UCWJ8lQ7Hly9m8OOGfdQp1KA',
            'MarketWatch': 'UCJqFtip6NjHioctNqS8NqYA',
            'Reuters Business': 'UCZcHhYVQ1ioavJ0JkRgdHLQ',
            'Financial Times': 'UCJ-a0EZtxlE5Pq3jFwJkUzw',
            'Wall Street Journal': 'UC4sptDLKmqG497zai9R3U3g',
            'Fox Business': 'UCsT0YIqwnpJCM-mx7-gSA4Q'
        }
        
        # Key events to monitor
        self.key_events = [
            'earnings call',
            'earnings announcement',
            'quarterly results',
            'earnings report',
            'financial results',
            'revenue announcement',
            'profit announcement',
            'stock price',
            'market update',
            'trading session'
        ]
        
        # Rate limiting
        self.rate_limits = {
            'calls': 0,
            'limit': 10000,  # YouTube API daily limit
            'reset_time': time.time() + 86400
        }
    
    def _check_rate_limit(self):
        """Check YouTube API rate limits"""
        if time.time() > self.rate_limits['reset_time']:
            self.rate_limits['calls'] = 0
            self.rate_limits['reset_time'] = time.time() + 86400
        
        if self.rate_limits['calls'] >= self.rate_limits['limit']:
            return False
        
        self.rate_limits['calls'] += 1
        return True
    
    async def _make_youtube_request(self, session: aiohttp.ClientSession, url: str, 
                                   params: dict = None) -> Optional[dict]:
        """Make YouTube API request with error handling"""
        if not self._check_rate_limit():
            print("⚠️ YouTube API rate limit reached")
            return None
        
        if not self.youtube_api_key:
            print("❌ YouTube API key not found")
            return None
        
        default_params = {'key': self.youtube_api_key}
        if params:
            default_params.update(params)
        
        try:
            async with session.get(url, params=default_params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 403:
                    print("❌ YouTube API access forbidden")
                    return None
                else:
                    print(f"⚠️ YouTube API request failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error making YouTube request: {e}")
            return None
    
    async def get_live_streams(self, channel_id: str = None) -> List[Dict]:
        """Get current live streams from financial channels"""
        print(f"📺 Getting live streams...")
        
        live_streams = []
        
        async with aiohttp.ClientSession() as session:
            if channel_id:
                # Get live streams for specific channel
                channels_to_check = {k: v for k, v in self.financial_channels.items() if v == channel_id}
            else:
                # Get live streams from all financial channels
                channels_to_check = self.financial_channels
            
            for channel_name, channel_id in channels_to_check.items():
                try:
                    # Search for live streams
                    url = f"{self.base_url}/search"
                    params = {
                        'part': 'snippet',
                        'channelId': channel_id,
                        'eventType': 'live',
                        'type': 'video',
                        'maxResults': 5
                    }
                    
                    data = await self._make_youtube_request(session, url, params=params)
                    if data and 'items' in data:
                        for item in data['items']:
                            snippet = item['snippet']
                            live_streams.append({
                                'video_id': item['id']['videoId'],
                                'title': snippet.get('title', ''),
                                'description': snippet.get('description', ''),
                                'channel_title': snippet.get('channelTitle', ''),
                                'published_at': snippet.get('publishedAt', ''),
                                'live_broadcast_content': snippet.get('liveBroadcastContent', ''),
                                'channel_name': channel_name
                            })
                    
                    await asyncio.sleep(0.1)  # Small delay to respect rate limits
                    
                except Exception as e:
                    print(f"❌ Error getting live streams for {channel_name}: {e}")
                    continue
        
        return live_streams
    
    async def get_live_chat_messages(self, video_id: str, max_results: int = 100) -> List[Dict]:
        """Get live chat messages from a live stream"""
        print(f"💬 Getting live chat messages for {video_id}...")
        
        chat_messages = []
        
        async with aiohttp.ClientSession() as session:
            # Get live chat ID
            url = f"{self.base_url}/videos"
            params = {
                'part': 'liveStreamingDetails',
                'id': video_id
            }
            
            data = await self._make_youtube_request(session, url, params=params)
            if data and 'items' in data and data['items']:
                live_details = data['items'][0].get('liveStreamingDetails', {})
                live_chat_id = live_details.get('activeLiveChatId')
                
                if live_chat_id:
                    # Get live chat messages
                    chat_url = f"{self.base_url}/liveChat/messages"
                    chat_params = {
                        'part': 'snippet,authorDetails',
                        'liveChatId': live_chat_id,
                        'maxResults': min(max_results, 200)
                    }
                    
                    chat_data = await self._make_youtube_request(session, chat_url, params=chat_params)
                    if chat_data and 'items' in chat_data:
                        for item in chat_data['items']:
                            snippet = item['snippet']
                            author = item['authorDetails']
                            chat_messages.append({
                                'message_id': item['id'],
                                'message': snippet.get('displayMessage', ''),
                                'author_name': author.get('displayName', ''),
                                'author_channel_id': author.get('channelId', ''),
                                'published_at': snippet.get('publishedAt', ''),
                                'super_chat_details': snippet.get('superChatDetails', {}),
                                'user_badge': snippet.get('userBadges', [])
                            })
        
        return chat_messages
    
    async def search_financial_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for financial videos related to specific topics"""
        print(f"🔍 Searching for financial videos: {query}")
        
        videos = []
        
        async with aiohttp.ClientSession() as session:
            # Search across financial channels
            for channel_name, channel_id in self.financial_channels.items():
                try:
                    url = f"{self.base_url}/search"
                    params = {
                        'part': 'snippet',
                        'channelId': channel_id,
                        'q': query,
                        'type': 'video',
                        'order': 'relevance',
                        'maxResults': min(max_results // len(self.financial_channels), 5)
                    }
                    
                    data = await self._make_youtube_request(session, url, params=params)
                    if data and 'items' in data:
                        for item in data['items']:
                            snippet = item['snippet']
                            videos.append({
                                'video_id': item['id']['videoId'],
                                'title': snippet.get('title', ''),
                                'description': snippet.get('description', ''),
                                'channel_title': snippet.get('channelTitle', ''),
                                'published_at': snippet.get('publishedAt', ''),
                                'thumbnails': snippet.get('thumbnails', {}),
                                'channel_name': channel_name
                            })
                    
                    await asyncio.sleep(0.1)  # Small delay to respect rate limits
                    
                except Exception as e:
                    print(f"❌ Error searching videos for {channel_name}: {e}")
                    continue
        
        return videos
    
    async def get_earnings_announcement_coverage(self, symbol: str) -> Dict[str, Any]:
        """Get earnings announcement coverage from financial channels"""
        print(f"📊 Getting earnings coverage for {symbol}...")
        
        coverage_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'live_streams': [],
            'videos': [],
            'sentiment': {},
            'coverage_summary': {}
        }
        
        # Search for earnings-related content
        earnings_queries = [
            f'{symbol} earnings call',
            f'{symbol} earnings announcement',
            f'{symbol} quarterly results',
            f'{symbol} earnings report'
        ]
        
        all_videos = []
        for query in earnings_queries:
            videos = await self.search_financial_videos(query, max_results=5)
            all_videos.extend(videos)
        
        coverage_data['videos'] = all_videos
        
        # Get live streams
        live_streams = await self.get_live_streams()
        coverage_data['live_streams'] = live_streams
        
        # Analyze sentiment from video titles and descriptions
        sentiment_analysis = self._analyze_coverage_sentiment(all_videos)
        coverage_data['sentiment'] = sentiment_analysis
        
        # Generate coverage summary
        coverage_data['coverage_summary'] = {
            'total_videos': len(all_videos),
            'live_streams': len(live_streams),
            'channels_covered': len(set(video['channel_name'] for video in all_videos)),
            'earnings_mentions': sum(1 for video in all_videos if 'earnings' in video['title'].lower()),
            'sentiment_score': sentiment_analysis.get('overall_sentiment', 0.0)
        }
        
        return coverage_data
    
    def _analyze_coverage_sentiment(self, videos: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from video titles and descriptions"""
        if not videos:
            return {
                'overall_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_videos': 0
            }
        
        positive_keywords = [
            'beat', 'exceed', 'strong', 'growth', 'positive', 'up', 'rise', 'gain',
            'bullish', 'buy', 'upgrade', 'outperform', 'positive surprise'
        ]
        
        negative_keywords = [
            'miss', 'disappoint', 'weak', 'decline', 'negative', 'down', 'fall', 'drop',
            'bearish', 'sell', 'downgrade', 'underperform', 'negative surprise'
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for video in videos:
            title = video.get('title', '').lower()
            description = video.get('description', '').lower()
            text = f"{title} {description}"
            
            positive_matches = sum(1 for keyword in positive_keywords if keyword in text)
            negative_matches = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_matches > negative_matches:
                positive_count += 1
            elif negative_matches > positive_matches:
                negative_count += 1
            else:
                neutral_count += 1
        
        total_videos = len(videos)
        overall_sentiment = (positive_count - negative_count) / total_videos if total_videos > 0 else 0.0
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_videos': total_videos
        }
    
    async def monitor_key_events(self, symbols: List[str]) -> Dict[str, Any]:
        """Monitor key events for multiple symbols"""
        print(f"📺 Monitoring key events for {len(symbols)} symbols...")
        
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'live_streams': [],
            'key_events': []
        }
        
        # Get current live streams
        live_streams = await self.get_live_streams()
        monitoring_data['live_streams'] = live_streams
        
        # Monitor each symbol
        for symbol in symbols:
            coverage = await self.get_earnings_announcement_coverage(symbol)
            monitoring_data['symbols'][symbol] = coverage
        
        # Identify key events
        for stream in live_streams:
            title = stream.get('title', '').lower()
            for event in self.key_events:
                if event in title:
                    monitoring_data['key_events'].append({
                        'event_type': event,
                        'stream_title': stream.get('title', ''),
                        'channel': stream.get('channel_name', ''),
                        'video_id': stream.get('video_id', '')
                    })
        
        return monitoring_data

async def main():
    """Test YouTube live news integration"""
    print("📺 YouTube Live News Integration Test")
    print("="*50)
    
    youtube = YouTubeLiveNewsIntegration()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting earnings coverage for {symbol}...")
    
    coverage = await youtube.get_earnings_announcement_coverage(symbol)
    
    # Display results
    print(f"\n📋 RESULTS:")
    print(f"   Symbol: {coverage['symbol']}")
    print(f"   Total Videos: {coverage['coverage_summary']['total_videos']}")
    print(f"   Live Streams: {coverage['coverage_summary']['live_streams']}")
    print(f"   Channels Covered: {coverage['coverage_summary']['channels_covered']}")
    print(f"   Earnings Mentions: {coverage['coverage_summary']['earnings_mentions']}")
    
    # Display sentiment
    sentiment = coverage['sentiment']
    print(f"\n🎯 SENTIMENT ANALYSIS:")
    print(f"   Overall Sentiment: {sentiment['overall_sentiment']:.3f}")
    print(f"   Positive Videos: {sentiment['positive_count']}")
    print(f"   Negative Videos: {sentiment['negative_count']}")
    print(f"   Neutral Videos: {sentiment['neutral_count']}")
    
    # Display sample videos
    if coverage['videos']:
        print(f"\n📹 SAMPLE VIDEOS:")
        for i, video in enumerate(coverage['videos'][:3]):
            print(f"   {i+1}. {video['channel_name']}: {video['title'][:60]}...")
    
    # Display live streams
    if coverage['live_streams']:
        print(f"\n📺 LIVE STREAMS:")
        for i, stream in enumerate(coverage['live_streams'][:3]):
            print(f"   {i+1}. {stream['channel_name']}: {stream['title'][:60]}...")
    
    print(f"\n🎉 YouTube live news integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())
