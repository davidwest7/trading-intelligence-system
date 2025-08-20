"""
Real-Time Alternative Data Integration
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
from queue import Queue
import json
import requests
from bs4 import BeautifulSoup
import feedparser
import re

class RealTimeAlternativeData:
    """
    Real-Time Alternative Data Integration System
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'news_sources': [
                'reuters', 'bloomberg', 'financial_times', 'cnbc', 'yahoo_finance'
            ],
            'social_media_sources': [
                'twitter', 'reddit', 'stocktwits'
            ],
            'economic_indicators': [
                'gdp', 'inflation', 'employment', 'interest_rates', 'consumer_sentiment'
            ],
            'update_frequency': 60,  # 60 seconds
            'data_retention_hours': 24,
            'sentiment_threshold': 0.1
        }
        
        self.news_data = {}
        self.social_media_data = {}
        self.economic_data = {}
        self.geopolitical_events = []
        self.consumer_data = {}
        self.is_running = False
        self.data_queue = Queue(maxsize=10000)
        
    async def initialize(self):
        """Initialize alternative data sources"""
        try:
            print("🔬 Initializing Real-Time Alternative Data Integration...")
            
            # Initialize data storage
            self._initialize_data_storage()
            
            # Start data collection threads
            self._start_data_collection()
            
            print("✅ Alternative Data Integration initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Alternative Data: {e}")
            return False
    
    def _initialize_data_storage(self):
        """Initialize data storage structures"""
        self.news_data = {
            'articles': [],
            'sentiment_scores': [],
            'entities': {},
            'trending_topics': []
        }
        
        self.social_media_data = {
            'tweets': [],
            'reddit_posts': [],
            'stocktwits_posts': [],
            'sentiment_analysis': {},
            'trending_symbols': []
        }
        
        self.economic_data = {
            'indicators': {},
            'releases': [],
            'forecasts': {},
            'surprises': []
        }
        
        self.geopolitical_events = []
        self.consumer_data = {
            'spending_patterns': {},
            'retail_data': {},
            'credit_card_data': {}
        }
    
    def _start_data_collection(self):
        """Start data collection threads"""
        self.is_running = True
        
        # News collection thread
        self.news_thread = threading.Thread(target=self._collect_news_data, daemon=True)
        self.news_thread.start()
        
        # Social media collection thread
        self.social_thread = threading.Thread(target=self._collect_social_media_data, daemon=True)
        self.social_thread.start()
        
        # Economic data collection thread
        self.economic_thread = threading.Thread(target=self._collect_economic_data, daemon=True)
        self.economic_thread.start()
        
        # Geopolitical events thread
        self.geopolitical_thread = threading.Thread(target=self._collect_geopolitical_events, daemon=True)
        self.geopolitical_thread.start()
        
        # Consumer data thread
        self.consumer_thread = threading.Thread(target=self._collect_consumer_data, daemon=True)
        self.consumer_thread.start()
    
    def _collect_news_data(self):
        """Collect real-time news data"""
        while self.is_running:
            try:
                # Simulate news data collection
                articles = self._simulate_news_articles()
                
                for article in articles:
                    # Analyze sentiment
                    sentiment_score = self._analyze_news_sentiment(article['title'] + ' ' + article['content'])
                    
                    # Extract entities
                    entities = self._extract_entities(article['content'])
                    
                    # Store article
                    article_data = {
                        'title': article['title'],
                        'content': article['content'],
                        'source': article['source'],
                        'timestamp': datetime.now(),
                        'sentiment_score': sentiment_score,
                        'entities': entities,
                        'impact_score': self._calculate_impact_score(article, sentiment_score)
                    }
                    
                    self.news_data['articles'].append(article_data)
                    self.news_data['sentiment_scores'].append(sentiment_score)
                    
                    # Update entities
                    for entity in entities:
                        if entity not in self.news_data['entities']:
                            self.news_data['entities'][entity] = []
                        self.news_data['entities'][entity].append(article_data)
                
                # Keep only recent articles
                cutoff_time = datetime.now() - timedelta(hours=self.config['data_retention_hours'])
                self.news_data['articles'] = [
                    article for article in self.news_data['articles']
                    if article['timestamp'] > cutoff_time
                ]
                
                time.sleep(self.config['update_frequency'])
                
            except Exception as e:
                print(f"Error collecting news data: {e}")
                time.sleep(self.config['update_frequency'])
    
    def _collect_social_media_data(self):
        """Collect social media data"""
        while self.is_running:
            try:
                # Simulate social media data collection
                social_posts = self._simulate_social_media_posts()
                
                for post in social_posts:
                    # Analyze sentiment
                    sentiment_score = self._analyze_social_sentiment(post['content'])
                    
                    # Extract symbols
                    symbols = self._extract_symbols(post['content'])
                    
                    # Store post
                    post_data = {
                        'content': post['content'],
                        'platform': post['platform'],
                        'timestamp': datetime.now(),
                        'sentiment_score': sentiment_score,
                        'symbols': symbols,
                        'engagement': post.get('engagement', 0)
                    }
                    
                    if post['platform'] == 'twitter':
                        self.social_media_data['tweets'].append(post_data)
                    elif post['platform'] == 'reddit':
                        self.social_media_data['reddit_posts'].append(post_data)
                    elif post['platform'] == 'stocktwits':
                        self.social_media_data['stocktwits_posts'].append(post_data)
                    
                    # Update sentiment analysis
                    for symbol in symbols:
                        if symbol not in self.social_media_data['sentiment_analysis']:
                            self.social_media_data['sentiment_analysis'][symbol] = []
                        self.social_media_data['sentiment_analysis'][symbol].append(sentiment_score)
                
                # Keep only recent posts
                cutoff_time = datetime.now() - timedelta(hours=self.config['data_retention_hours'])
                for platform in ['tweets', 'reddit_posts', 'stocktwits_posts']:
                    self.social_media_data[platform] = [
                        post for post in self.social_media_data[platform]
                        if post['timestamp'] > cutoff_time
                    ]
                
                time.sleep(self.config['update_frequency'])
                
            except Exception as e:
                print(f"Error collecting social media data: {e}")
                time.sleep(self.config['update_frequency'])
    
    def _collect_economic_data(self):
        """Collect economic indicator data"""
        while self.is_running:
            try:
                # Simulate economic data collection
                economic_indicators = self._simulate_economic_indicators()
                
                for indicator in economic_indicators:
                    self.economic_data['indicators'][indicator['name']] = {
                        'value': indicator['value'],
                        'previous': indicator['previous'],
                        'forecast': indicator['forecast'],
                        'timestamp': datetime.now(),
                        'surprise': indicator['value'] - indicator['forecast'],
                        'impact': self._calculate_economic_impact(indicator)
                    }
                    
                    # Store release
                    release_data = {
                        'indicator': indicator['name'],
                        'value': indicator['value'],
                        'surprise': indicator['value'] - indicator['forecast'],
                        'timestamp': datetime.now()
                    }
                    self.economic_data['releases'].append(release_data)
                
                # Keep only recent releases
                cutoff_time = datetime.now() - timedelta(hours=self.config['data_retention_hours'])
                self.economic_data['releases'] = [
                    release for release in self.economic_data['releases']
                    if release['timestamp'] > cutoff_time
                ]
                
                time.sleep(self.config['update_frequency'] * 5)  # Less frequent updates
                
            except Exception as e:
                print(f"Error collecting economic data: {e}")
                time.sleep(self.config['update_frequency'] * 5)
    
    def _collect_geopolitical_events(self):
        """Collect geopolitical events"""
        while self.is_running:
            try:
                # Simulate geopolitical events
                events = self._simulate_geopolitical_events()
                
                for event in events:
                    event_data = {
                        'title': event['title'],
                        'description': event['description'],
                        'region': event['region'],
                        'severity': event['severity'],
                        'timestamp': datetime.now(),
                        'impact_score': self._calculate_geopolitical_impact(event)
                    }
                    
                    self.geopolitical_events.append(event_data)
                
                # Keep only recent events
                cutoff_time = datetime.now() - timedelta(hours=self.config['data_retention_hours'])
                self.geopolitical_events = [
                    event for event in self.geopolitical_events
                    if event['timestamp'] > cutoff_time
                ]
                
                time.sleep(self.config['update_frequency'] * 10)  # Less frequent updates
                
            except Exception as e:
                print(f"Error collecting geopolitical events: {e}")
                time.sleep(self.config['update_frequency'] * 10)
    
    def _collect_consumer_data(self):
        """Collect consumer behavior data"""
        while self.is_running:
            try:
                # Simulate consumer data collection
                consumer_data = self._simulate_consumer_data()
                
                for data_point in consumer_data:
                    category = data_point['category']
                    if category not in self.consumer_data['spending_patterns']:
                        self.consumer_data['spending_patterns'][category] = []
                    
                    self.consumer_data['spending_patterns'][category].append({
                        'value': data_point['value'],
                        'change': data_point['change'],
                        'timestamp': datetime.now()
                    })
                
                time.sleep(self.config['update_frequency'] * 15)  # Less frequent updates
                
            except Exception as e:
                print(f"Error collecting consumer data: {e}")
                time.sleep(self.config['update_frequency'] * 15)
    
    def _simulate_news_articles(self):
        """Simulate news articles"""
        articles = [
            {
                'title': 'Federal Reserve Signals Potential Rate Cut',
                'content': 'The Federal Reserve indicated today that it may consider cutting interest rates in response to economic data showing signs of slowing growth.',
                'source': 'reuters'
            },
            {
                'title': 'Tech Stocks Rally on Strong Earnings Reports',
                'content': 'Major technology companies reported better-than-expected earnings, driving a broad market rally in the tech sector.',
                'source': 'bloomberg'
            },
            {
                'title': 'Oil Prices Surge on Supply Concerns',
                'content': 'Crude oil prices jumped today amid concerns about supply disruptions in key producing regions.',
                'source': 'financial_times'
            }
        ]
        return articles
    
    def _simulate_social_media_posts(self):
        """Simulate social media posts"""
        posts = [
            {
                'content': 'AAPL looking bullish today! Strong earnings and great product pipeline #stocks #AAPL',
                'platform': 'twitter',
                'engagement': 150
            },
            {
                'content': 'What do you think about TSLA? The stock seems to be gaining momentum after the recent dip.',
                'platform': 'reddit',
                'engagement': 89
            },
            {
                'content': 'BTC breaking out! This could be the start of a major rally #crypto #bitcoin',
                'platform': 'stocktwits',
                'engagement': 234
            }
        ]
        return posts
    
    def _simulate_economic_indicators(self):
        """Simulate economic indicators"""
        indicators = [
            {
                'name': 'GDP_Growth',
                'value': 2.1,
                'previous': 2.0,
                'forecast': 2.2
            },
            {
                'name': 'Inflation_Rate',
                'value': 3.2,
                'previous': 3.1,
                'forecast': 3.0
            },
            {
                'name': 'Unemployment_Rate',
                'value': 3.8,
                'previous': 3.9,
                'forecast': 3.8
            }
        ]
        return indicators
    
    def _simulate_geopolitical_events(self):
        """Simulate geopolitical events"""
        events = [
            {
                'title': 'Trade Tensions Escalate Between US and China',
                'description': 'New tariffs announced on Chinese imports, raising concerns about global trade.',
                'region': 'Asia',
                'severity': 'high'
            },
            {
                'title': 'European Central Bank Policy Meeting',
                'description': 'ECB announces new monetary policy measures to support economic recovery.',
                'region': 'Europe',
                'severity': 'medium'
            }
        ]
        return events
    
    def _simulate_consumer_data(self):
        """Simulate consumer behavior data"""
        data = [
            {
                'category': 'retail_spending',
                'value': 450.2,
                'change': 0.05
            },
            {
                'category': 'online_shopping',
                'value': 125.8,
                'change': 0.12
            },
            {
                'category': 'travel_spending',
                'value': 89.3,
                'change': -0.03
            }
        ]
        return data
    
    def _analyze_news_sentiment(self, text):
        """Analyze news sentiment"""
        # Simple sentiment analysis
        positive_words = ['bullish', 'positive', 'growth', 'profit', 'gain', 'up', 'strong']
        negative_words = ['bearish', 'negative', 'loss', 'down', 'weak', 'decline']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return min(1.0, positive_count / 3.0)
        elif negative_count > positive_count:
            return max(-1.0, -negative_count / 3.0)
        else:
            return 0.0
    
    def _analyze_social_sentiment(self, text):
        """Analyze social media sentiment"""
        # Similar to news sentiment but with emoji support
        text_lower = text.lower()
        
        # Check for emojis
        emoji_sentiment = 0
        if '🚀' in text or '📈' in text or '💎' in text:
            emoji_sentiment = 0.3
        elif '📉' in text or '💸' in text or '🔥' in text:
            emoji_sentiment = -0.3
        
        # Text sentiment
        positive_words = ['bullish', 'positive', 'growth', 'profit', 'gain', 'up', 'strong', 'moon']
        negative_words = ['bearish', 'negative', 'loss', 'down', 'weak', 'decline', 'dump']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        text_sentiment = 0
        if positive_count > negative_count:
            text_sentiment = min(1.0, positive_count / 3.0)
        elif negative_count > positive_count:
            text_sentiment = max(-1.0, -negative_count / 3.0)
        
        return text_sentiment + emoji_sentiment
    
    def _extract_entities(self, text):
        """Extract entities from text"""
        # Simple entity extraction
        entities = []
        
        # Look for company names and symbols
        symbols = re.findall(r'\b[A-Z]{2,5}\b', text)
        entities.extend(symbols)
        
        # Look for country names
        countries = ['US', 'China', 'Europe', 'Japan', 'UK']
        for country in countries:
            if country.lower() in text.lower():
                entities.append(country)
        
        return list(set(entities))
    
    def _extract_symbols(self, text):
        """Extract stock symbols from text"""
        # Extract stock symbols (e.g., $AAPL, AAPL, #TSLA)
        symbols = re.findall(r'[\$#]?([A-Z]{2,5})\b', text)
        return list(set(symbols))
    
    def _calculate_impact_score(self, article, sentiment_score):
        """Calculate impact score for news article"""
        # Base impact on source credibility and sentiment
        source_credibility = {
            'reuters': 0.9,
            'bloomberg': 0.85,
            'financial_times': 0.8,
            'cnbc': 0.7,
            'yahoo_finance': 0.6
        }
        
        credibility = source_credibility.get(article['source'], 0.5)
        impact = credibility * abs(sentiment_score)
        
        return min(1.0, impact)
    
    def _calculate_economic_impact(self, indicator):
        """Calculate economic impact of indicator"""
        # Calculate impact based on surprise and importance
        surprise = abs(indicator['value'] - indicator['forecast'])
        
        importance_weights = {
            'GDP_Growth': 0.3,
            'Inflation_Rate': 0.25,
            'Unemployment_Rate': 0.2,
            'Interest_Rates': 0.25
        }
        
        importance = importance_weights.get(indicator['name'], 0.1)
        impact = surprise * importance
        
        return min(1.0, impact)
    
    def _calculate_geopolitical_impact(self, event):
        """Calculate geopolitical impact"""
        severity_weights = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }
        
        severity = severity_weights.get(event['severity'], 0.5)
        
        # Add regional importance
        regional_importance = {
            'US': 0.9,
            'China': 0.8,
            'Europe': 0.7,
            'Asia': 0.6
        }
        
        region_impact = regional_importance.get(event['region'], 0.5)
        
        return (severity + region_impact) / 2
    
    async def get_market_sentiment(self, symbol=None):
        """Get market sentiment from alternative data"""
        try:
            sentiment_data = {
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'economic_sentiment': 0.0,
                'geopolitical_sentiment': 0.0,
                'overall_sentiment': 0.0
            }
            
            # News sentiment
            if self.news_data['sentiment_scores']:
                sentiment_data['news_sentiment'] = np.mean(self.news_data['sentiment_scores'])
            
            # Social media sentiment
            if symbol and symbol in self.social_media_data['sentiment_analysis']:
                sentiment_data['social_sentiment'] = np.mean(
                    self.social_media_data['sentiment_analysis'][symbol]
                )
            else:
                # Overall social sentiment
                all_social_sentiments = []
                for platform in ['tweets', 'reddit_posts', 'stocktwits_posts']:
                    all_social_sentiments.extend([
                        post['sentiment_score'] for post in self.social_media_data[platform]
                    ])
                if all_social_sentiments:
                    sentiment_data['social_sentiment'] = np.mean(all_social_sentiments)
            
            # Economic sentiment
            if self.economic_data['releases']:
                economic_sentiments = []
                for release in self.economic_data['releases']:
                    if release['surprise'] > 0:
                        economic_sentiments.append(0.1)  # Positive surprise
                    elif release['surprise'] < 0:
                        economic_sentiments.append(-0.1)  # Negative surprise
                    else:
                        economic_sentiments.append(0.0)
                sentiment_data['economic_sentiment'] = np.mean(economic_sentiments)
            
            # Geopolitical sentiment
            if self.geopolitical_events:
                geopolitical_sentiments = []
                for event in self.geopolitical_events:
                    if event['severity'] == 'high':
                        geopolitical_sentiments.append(-0.2)  # Negative for high severity
                    elif event['severity'] == 'medium':
                        geopolitical_sentiments.append(-0.1)
                    else:
                        geopolitical_sentiments.append(0.0)
                sentiment_data['geopolitical_sentiment'] = np.mean(geopolitical_sentiments)
            
            # Overall sentiment (weighted average)
            weights = [0.3, 0.3, 0.2, 0.2]  # News, Social, Economic, Geopolitical
            sentiments = [
                sentiment_data['news_sentiment'],
                sentiment_data['social_sentiment'],
                sentiment_data['economic_sentiment'],
                sentiment_data['geopolitical_sentiment']
            ]
            
            sentiment_data['overall_sentiment'] = np.average(sentiments, weights=weights)
            
            return {
                'success': True,
                'sentiment_data': sentiment_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error getting market sentiment: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_consumer_insights(self):
        """Get consumer behavior insights"""
        try:
            insights = {}
            
            for category, data in self.consumer_data['spending_patterns'].items():
                if data:
                    recent_data = data[-5:]  # Last 5 data points
                    avg_change = np.mean([point['change'] for point in recent_data])
                    insights[category] = {
                        'trend': 'increasing' if avg_change > 0 else 'decreasing',
                        'change_rate': avg_change,
                        'current_value': recent_data[-1]['value'] if recent_data else 0
                    }
            
            return {
                'success': True,
                'insights': insights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error getting consumer insights: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_geopolitical_risk(self):
        """Get geopolitical risk assessment"""
        try:
            if not self.geopolitical_events:
                return {
                    'success': True,
                    'risk_level': 'low',
                    'risk_score': 0.1,
                    'recent_events': []
                }
            
            # Calculate risk score
            risk_scores = []
            for event in self.geopolitical_events:
                risk_scores.append(event['impact_score'])
            
            avg_risk_score = np.mean(risk_scores)
            
            # Determine risk level
            if avg_risk_score > 0.7:
                risk_level = 'high'
            elif avg_risk_score > 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'success': True,
                'risk_level': risk_level,
                'risk_score': avg_risk_score,
                'recent_events': self.geopolitical_events[-5:],  # Last 5 events
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error getting geopolitical risk: {e}")
            return {'success': False, 'error': str(e)}
    
    def stop(self):
        """Stop data collection"""
        self.is_running = False
        print("🛑 Alternative Data Integration stopped")
    
    def get_data_summary(self):
        """Get data collection summary"""
        return {
            'news_articles': len(self.news_data['articles']),
            'social_posts': len(self.social_media_data['tweets']) + len(self.social_media_data['reddit_posts']) + len(self.social_media_data['stocktwits_posts']),
            'economic_releases': len(self.economic_data['releases']),
            'geopolitical_events': len(self.geopolitical_events),
            'consumer_categories': len(self.consumer_data['spending_patterns']),
            'is_running': self.is_running
        }
    
    def get_available_sources(self):
        """Get available data sources"""
        return [
            'news_sources',
            'social_media_sources', 
            'economic_indicators',
            'geopolitical_events',
            'consumer_data'
        ]

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        self.stop()

class RealTimeDataIntegration:
    """
    Wrapper class for RealTimeAlternativeData with proper cleanup
    """
    
    def __init__(self, config=None):
        self.alt_data = RealTimeAlternativeData(config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the data integration"""
        if not self._initialized:
            success = await self.alt_data.initialize()
            self._initialized = success
            return success
        return True
    
    def get_available_sources(self):
        """Get available data sources"""
        return [
            'news_sources',
            'social_media_sources', 
            'economic_indicators',
            'geopolitical_events',
            'consumer_data'
        ]
    
    def stop(self):
        """Stop data collection"""
        if self._initialized:
            self.alt_data.stop()
            self._initialized = False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        self.stop()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.stop()
        except:
            pass
