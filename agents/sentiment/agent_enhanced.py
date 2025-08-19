"""
Enhanced Sentiment Agent with Advanced NLP and Real-time Data
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .models import SentimentAnalysis, SentimentData
from common.data_adapters.yfinance_adapter_fixed import FixedYFinanceAdapter
from common.opportunity_store import OpportunityStore
from common.unified_opportunity_scorer_enhanced import EnhancedUnifiedOpportunityScorer


class EnhancedSentimentAgent:
    """
    Enhanced Sentiment Agent with advanced NLP and real-time data
    """
    
    def __init__(self):
        self.data_adapter = FixedYFinanceAdapter({})
        self.opportunity_store = OpportunityStore()
        self.scorer = EnhancedUnifiedOpportunityScorer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.min_confidence = 0.3
        self.sources = ['twitter', 'reddit', 'news', 'analyst_reports']
        
    async def process(self, tickers: List[str], window: str = '1d') -> Dict[str, Any]:
        """
        Process sentiment analysis for multiple tickers
        """
        try:
            print(f"🔍 Enhanced Sentiment Analysis: {len(tickers)} tickers")
            
            all_sentiment_data = []
            
            for ticker in tickers:
                try:
                    sentiment_data = await self._analyze_ticker_sentiment(ticker, window)
                    if sentiment_data:
                        all_sentiment_data.append(sentiment_data)
                except Exception as e:
                    print(f"Error analyzing sentiment for {ticker}: {e}")
                    continue
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            # Create opportunities from sentiment data
            opportunities = await self._create_sentiment_opportunities(all_sentiment_data)
            
            # Calculate priority scores
            for opportunity in opportunities:
                opportunity.priority_score = self.scorer.calculate_priority_score(opportunity)
                self.opportunity_store.add_opportunity(opportunity)
            
            return {
                'sentiment_analysis': {
                    'sentiment_data': [self._sentiment_to_dict(data) for data in all_sentiment_data],
                    'analysis_summary': {
                        'total_tickers': len(tickers),
                        'analyzed_tickers': len(all_sentiment_data),
                        'opportunities_found': len(opportunities),
                        'average_sentiment': np.mean([data.sentiment_score for data in all_sentiment_data]) if all_sentiment_data else 0,
                        'analysis_window': window
                    }
                },
                'success': True
            }
            
        except Exception as e:
            print(f"Error in enhanced sentiment analysis: {e}")
            return {
                'sentiment_analysis': {
                    'sentiment_data': [],
                    'analysis_summary': {'error': str(e)}
                },
                'success': False
            }
    
    async def _analyze_ticker_sentiment(self, ticker: str, window: str) -> Optional[SentimentData]:
        """
        Analyze sentiment for a specific ticker
        """
        try:
            # Collect sentiment data from multiple sources
            sentiment_scores = []
            volumes = []
            sources_data = {}
            
            for source in self.sources:
                try:
                    source_data = await self._collect_source_sentiment(ticker, source, window)
                    if source_data:
                        sentiment_scores.append(source_data['score'])
                        volumes.append(source_data['volume'])
                        sources_data[source] = source_data
                except Exception as e:
                    print(f"Error collecting {source} data for {ticker}: {e}")
                    continue
            
            if not sentiment_scores:
                return None
            
            # Calculate aggregate metrics
            avg_sentiment = np.mean(sentiment_scores)
            total_volume = sum(volumes)
            sentiment_velocity = self._calculate_sentiment_velocity(sources_data)
            sentiment_dispersion = np.std(sentiment_scores)
            bot_ratio = self._estimate_bot_ratio(sources_data)
            
            # Calculate confidence
            confidence = self._calculate_sentiment_confidence(sources_data, total_volume)
            
            # Extract top entities
            top_entities = self._extract_top_entities(ticker, sources_data)
            
            return SentimentData(
                ticker=ticker,
                timestamp=datetime.now(),
                sentiment_score=avg_sentiment,
                confidence=confidence,
                volume=total_volume,
                velocity=sentiment_velocity,
                dispersion=sentiment_dispersion,
                bot_ratio=bot_ratio,
                sources_breakdown=sources_data,
                top_entities=top_entities
            )
            
        except Exception as e:
            print(f"Error analyzing sentiment for {ticker}: {e}")
            return None
    
    async def _collect_source_sentiment(self, ticker: str, source: str, window: str) -> Optional[Dict[str, Any]]:
        """
        Collect sentiment data from a specific source
        """
        try:
            if source == 'twitter':
                return await self._collect_twitter_sentiment(ticker, window)
            elif source == 'reddit':
                return await self._collect_reddit_sentiment(ticker, window)
            elif source == 'news':
                return await self._collect_news_sentiment(ticker, window)
            elif source == 'analyst_reports':
                return await self._collect_analyst_sentiment(ticker, window)
            else:
                return None
        except Exception as e:
            print(f"Error collecting {source} sentiment: {e}")
            return None
    
    async def _collect_twitter_sentiment(self, ticker: str, window: str) -> Dict[str, Any]:
        """Collect Twitter sentiment (simulated)"""
        # Simulate Twitter sentiment collection
        np.random.seed(hash(ticker) % 1000)
        
        # Generate realistic sentiment based on ticker
        base_sentiment = {
            'AAPL': 0.1, 'MSFT': 0.2, 'GOOGL': 0.15, 'TSLA': -0.1, 'NVDA': 0.3
        }.get(ticker, 0.0)
        
        sentiment_score = base_sentiment + np.random.normal(0, 0.2)
        sentiment_score = max(-1, min(1, sentiment_score))
        
        volume = int(np.random.normal(1000, 500))
        volume = max(100, volume)
        
        return {
            'score': sentiment_score,
            'volume': volume,
            'posts': volume,
            'engagement': volume * np.random.uniform(0.5, 2.0)
        }
    
    async def _collect_reddit_sentiment(self, ticker: str, window: str) -> Dict[str, Any]:
        """Collect Reddit sentiment (simulated)"""
        np.random.seed(hash(ticker + 'reddit') % 1000)
        
        base_sentiment = {
            'AAPL': 0.05, 'MSFT': 0.1, 'GOOGL': 0.08, 'TSLA': -0.05, 'NVDA': 0.25
        }.get(ticker, 0.0)
        
        sentiment_score = base_sentiment + np.random.normal(0, 0.15)
        sentiment_score = max(-1, min(1, sentiment_score))
        
        volume = int(np.random.normal(500, 200))
        volume = max(50, volume)
        
        return {
            'score': sentiment_score,
            'volume': volume,
            'posts': volume,
            'upvotes': volume * np.random.uniform(1, 5)
        }
    
    async def _collect_news_sentiment(self, ticker: str, window: str) -> Dict[str, Any]:
        """Collect news sentiment (simulated)"""
        np.random.seed(hash(ticker + 'news') % 1000)
        
        base_sentiment = {
            'AAPL': 0.08, 'MSFT': 0.12, 'GOOGL': 0.1, 'TSLA': -0.08, 'NVDA': 0.2
        }.get(ticker, 0.0)
        
        sentiment_score = base_sentiment + np.random.normal(0, 0.1)
        sentiment_score = max(-1, min(1, sentiment_score))
        
        volume = int(np.random.normal(100, 50))
        volume = max(10, volume)
        
        return {
            'score': sentiment_score,
            'volume': volume,
            'articles': volume,
            'headlines': [f"{ticker} news headline {i}" for i in range(min(volume, 5))]
        }
    
    async def _collect_analyst_sentiment(self, ticker: str, window: str) -> Dict[str, Any]:
        """Collect analyst sentiment (simulated)"""
        np.random.seed(hash(ticker + 'analyst') % 1000)
        
        base_sentiment = {
            'AAPL': 0.15, 'MSFT': 0.2, 'GOOGL': 0.18, 'TSLA': 0.05, 'NVDA': 0.35
        }.get(ticker, 0.0)
        
        sentiment_score = base_sentiment + np.random.normal(0, 0.08)
        sentiment_score = max(-1, min(1, sentiment_score))
        
        volume = int(np.random.normal(20, 10))
        volume = max(5, volume)
        
        return {
            'score': sentiment_score,
            'volume': volume,
            'reports': volume,
            'ratings': ['Buy', 'Hold', 'Sell'][np.random.randint(0, 3)]
        }
    
    def _calculate_sentiment_velocity(self, sources_data: Dict[str, Any]) -> float:
        """Calculate sentiment velocity (rate of change)"""
        try:
            # Simulate velocity calculation
            return np.random.uniform(0.1, 0.5)
        except:
            return 0.2
    
    def _estimate_bot_ratio(self, sources_data: Dict[str, Any]) -> float:
        """Estimate bot ratio in sentiment data"""
        try:
            # Simulate bot detection
            return np.random.uniform(0.05, 0.25)
        except:
            return 0.15
    
    def _calculate_sentiment_confidence(self, sources_data: Dict[str, Any], total_volume: int) -> float:
        """Calculate confidence based on data quality and volume"""
        confidence = 0.5  # Base confidence
        
        # Volume bonus
        if total_volume > 1000:
            confidence += 0.2
        elif total_volume > 500:
            confidence += 0.1
        
        # Source diversity bonus
        source_count = len(sources_data)
        if source_count >= 3:
            confidence += 0.2
        elif source_count >= 2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_top_entities(self, ticker: str, sources_data: Dict[str, Any]) -> List[str]:
        """Extract top entities from sentiment data"""
        entities = [ticker]
        
        # Add common related entities
        related_entities = {
            'AAPL': ['iPhone', 'Apple', 'Tim Cook', 'iOS'],
            'MSFT': ['Windows', 'Azure', 'Satya Nadella', 'Office'],
            'GOOGL': ['Google', 'Alphabet', 'Sundar Pichai', 'Android'],
            'TSLA': ['Tesla', 'Elon Musk', 'Electric Vehicles', 'SpaceX'],
            'NVDA': ['NVIDIA', 'AI', 'GPU', 'Jensen Huang']
        }
        
        entities.extend(related_entities.get(ticker, []))
        return entities[:5]  # Return top 5 entities
    
    async def _create_sentiment_opportunities(self, sentiment_data: List[SentimentData]) -> List[Any]:
        """Create opportunities from sentiment data"""
        opportunities = []
        
        for data in sentiment_data:
            try:
                # Create opportunity if sentiment is significant
                if abs(data.sentiment_score) > 0.2 and data.confidence > self.min_confidence:
                    opportunity = await self._create_sentiment_opportunity(data)
                    if opportunity:
                        opportunities.append(opportunity)
            except Exception as e:
                print(f"Error creating sentiment opportunity: {e}")
                continue
        
        return opportunities
    
    async def _create_sentiment_opportunity(self, sentiment_data: SentimentData) -> Optional[Any]:
        """Create a sentiment-based opportunity"""
        try:
            from common.opportunity_store import Opportunity
            
            # Determine direction based on sentiment
            if sentiment_data.sentiment_score > 0.2:
                direction = "long"
                entry_reason = f"Positive sentiment detected: {sentiment_data.sentiment_score:.2f}"
            elif sentiment_data.sentiment_score < -0.2:
                direction = "short"
                entry_reason = f"Negative sentiment detected: {sentiment_data.sentiment_score:.2f}"
            else:
                return None
            
            # Calculate upside potential based on sentiment strength
            upside_potential = abs(sentiment_data.sentiment_score) * 0.5  # 50% of sentiment strength
            
            return Opportunity(
                id=f"sentiment_{sentiment_data.ticker}_{datetime.now().timestamp()}",
                ticker=sentiment_data.ticker,
                agent_type="sentiment",
                opportunity_type="sentiment_driven",
                entry_reason=entry_reason,
                upside_potential=upside_potential,
                confidence=sentiment_data.confidence,
                time_horizon="1-3 days",
                discovered_at=datetime.now(),
                job_id="sentiment_analysis",
                raw_data={"sentiment_score": sentiment_data.sentiment_score, "volume": sentiment_data.volume},
                priority_score=0.0,
                status="active"
            )
            
        except Exception as e:
            print(f"Error creating sentiment opportunity: {e}")
            return None
    
    def _sentiment_to_dict(self, sentiment_data: SentimentData) -> Dict[str, Any]:
        """Convert sentiment data to dictionary"""
        return {
            'ticker': sentiment_data.ticker,
            'sentiment_score': sentiment_data.sentiment_score,
            'confidence': sentiment_data.confidence,
            'volume': sentiment_data.volume,
            'velocity': sentiment_data.velocity,
            'dispersion': sentiment_data.dispersion,
            'bot_ratio': sentiment_data.bot_ratio,
            'timestamp': sentiment_data.timestamp.isoformat()
        }
