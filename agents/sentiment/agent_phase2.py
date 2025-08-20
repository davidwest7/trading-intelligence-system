"""
Sentiment Agent - Phase 2 Standardized

Sentiment analysis agent with uncertainty quantification (μ, σ, horizon).
Analyzes social media, news, and market sentiment to emit standardized signals.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import re

from common.models import BaseAgent
from schemas.contracts import Signal, SignalType, RegimeType, HorizonType, DirectionType


logger = logging.getLogger(__name__)


class SentimentAgentPhase2(BaseAgent):
    """
    Sentiment Analysis Agent with Uncertainty Quantification
    
    Features:
    - Social media sentiment analysis (Twitter, Reddit)
    - News sentiment analysis
    - Market sentiment indicators (VIX, Put/Call ratio)
    - Multi-source sentiment fusion
    - Uncertainty quantification based on sentiment confidence
    - Real-time sentiment tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("sentiment_agent_v2", SignalType.SENTIMENT, config)
        
        # Sentiment analysis parameters
        self.sentiment_sources = config.get('sentiment_sources', ['social', 'news', 'market']) if config else ['social', 'news', 'market']
        self.min_confidence = config.get('min_confidence', 0.4) if config else 0.4
        self.sentiment_window = config.get('sentiment_window', 24) if config else 24  # hours
        
        # Social media parameters
        self.social_platforms = ['twitter', 'reddit', 'stocktwits']
        self.min_social_volume = 10  # Minimum mentions for signal
        
        # News parameters
        self.news_sources = ['reuters', 'bloomberg', 'cnbc', 'marketwatch']
        self.news_relevance_threshold = 0.7
        
        # Market sentiment parameters
        self.vix_threshold_low = 15
        self.vix_threshold_high = 25
        self.put_call_threshold = 1.2
        
        # Sentiment scoring weights
        self.sentiment_weights = {
            'social': 0.4,
            'news': 0.4,
            'market': 0.2
        }
        
        # Performance tracking
        self.sentiment_history = {}
        self.accuracy_metrics = {}
        
    async def generate_signals(self, symbols: List[str], **kwargs) -> List[Signal]:
        """
        Generate sentiment signals with uncertainty quantification
        
        Args:
            symbols: List of symbols to analyze
            **kwargs: Additional parameters (sentiment_data, trace_id, etc.)
            
        Returns:
            List of standardized Signal objects
        """
        try:
            signals = []
            sentiment_data = kwargs.get('sentiment_data', {})
            trace_id = kwargs.get('trace_id')
            
            for symbol in symbols:
                symbol_signals = await self._analyze_symbol_sentiment(
                    symbol, sentiment_data, trace_id
                )
                if symbol_signals:
                    signals.extend(symbol_signals)
            
            logger.info(f"Generated {len(signals)} sentiment signals for {len(symbols)} symbols")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating sentiment signals: {e}")
            return []
    
    async def _analyze_symbol_sentiment(self, symbol: str, sentiment_data: Dict[str, Any],
                                      trace_id: Optional[str] = None) -> List[Signal]:
        """Analyze sentiment for a single symbol"""
        try:
            # Get sentiment data for symbol
            symbol_sentiment = sentiment_data.get(symbol, {})
            if not symbol_sentiment:
                # Generate synthetic sentiment data for demo
                symbol_sentiment = self._generate_synthetic_sentiment(symbol)
            
            # Analyze different sentiment sources
            sentiment_scores = {}
            confidence_scores = {}
            
            if 'social' in self.sentiment_sources:
                social_score, social_conf = await self._analyze_social_sentiment(
                    symbol, symbol_sentiment.get('social', {})
                )
                sentiment_scores['social'] = social_score
                confidence_scores['social'] = social_conf
            
            if 'news' in self.sentiment_sources:
                news_score, news_conf = await self._analyze_news_sentiment(
                    symbol, symbol_sentiment.get('news', {})
                )
                sentiment_scores['news'] = news_score
                confidence_scores['news'] = news_conf
            
            if 'market' in self.sentiment_sources:
                market_score, market_conf = await self._analyze_market_sentiment(
                    symbol, symbol_sentiment.get('market', {})
                )
                sentiment_scores['market'] = market_score
                confidence_scores['market'] = market_conf
            
            # Fuse sentiment scores
            fused_score, fused_confidence = self._fuse_sentiment_scores(
                sentiment_scores, confidence_scores
            )
            
            if abs(fused_score) < 0.01 or fused_confidence < self.min_confidence:
                return []
            
            # Determine market conditions for uncertainty calculation
            market_conditions = self._assess_sentiment_conditions(
                symbol_sentiment, sentiment_scores, confidence_scores
            )
            
            # Create standardized signal
            signal = self.create_signal(
                symbol=symbol,
                mu=fused_score,
                confidence=fused_confidence,
                market_conditions=market_conditions,
                trace_id=trace_id,
                metadata={
                    'sentiment_scores': sentiment_scores,
                    'confidence_scores': confidence_scores,
                    'sentiment_sources': list(sentiment_scores.keys()),
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return [signal]
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return []
    
    def _generate_synthetic_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic sentiment data for demo"""
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate social sentiment
        social_mentions = np.random.randint(20, 200)
        social_sentiment = np.random.normal(0.1, 0.3)  # Slightly positive bias
        social_confidence = np.random.uniform(0.5, 0.9)
        
        # Generate news sentiment
        news_articles = np.random.randint(5, 30)
        news_sentiment = np.random.normal(0.05, 0.25)
        news_relevance = np.random.uniform(0.6, 0.95)
        
        # Generate market sentiment
        vix_level = np.random.uniform(12, 30)
        put_call_ratio = np.random.uniform(0.8, 1.5)
        fear_greed_index = np.random.uniform(20, 80)
        
        return {
            'social': {
                'mentions': social_mentions,
                'sentiment_score': social_sentiment,
                'confidence': social_confidence,
                'platforms': {
                    'twitter': np.random.randint(5, social_mentions//2),
                    'reddit': np.random.randint(2, social_mentions//3),
                    'stocktwits': np.random.randint(1, social_mentions//4)
                }
            },
            'news': {
                'articles': news_articles,
                'sentiment_score': news_sentiment,
                'relevance': news_relevance,
                'sources': {
                    'reuters': np.random.randint(0, news_articles//3),
                    'bloomberg': np.random.randint(0, news_articles//3),
                    'cnbc': np.random.randint(0, news_articles//3)
                }
            },
            'market': {
                'vix': vix_level,
                'put_call_ratio': put_call_ratio,
                'fear_greed_index': fear_greed_index,
                'market_sentiment': np.random.normal(0, 0.2)
            }
        }
    
    async def _analyze_social_sentiment(self, symbol: str, 
                                      social_data: Dict[str, Any]) -> tuple[float, float]:
        """Analyze social media sentiment"""
        try:
            if not social_data:
                return 0.0, 0.0
            
            mentions = social_data.get('mentions', 0)
            sentiment_score = social_data.get('sentiment_score', 0.0)
            base_confidence = social_data.get('confidence', 0.5)
            
            # Volume threshold
            if mentions < self.min_social_volume:
                return 0.0, 0.0
            
            # Adjust sentiment based on volume
            volume_factor = min(mentions / 100, 2.0)  # Cap at 2x
            adjusted_sentiment = sentiment_score * np.log(volume_factor + 1)
            
            # Calculate confidence based on volume and platform diversity
            platforms = social_data.get('platforms', {})
            platform_diversity = len([p for p in platforms.values() if p > 0])
            
            volume_confidence = min(mentions / 50, 1.0)
            diversity_confidence = platform_diversity / len(self.social_platforms)
            
            final_confidence = base_confidence * volume_confidence * diversity_confidence
            
            # Convert to expected return (social sentiment is short-term)
            expected_return = adjusted_sentiment * 0.03  # Max 3% impact
            
            return expected_return, final_confidence
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return 0.0, 0.0
    
    async def _analyze_news_sentiment(self, symbol: str, 
                                    news_data: Dict[str, Any]) -> tuple[float, float]:
        """Analyze news sentiment"""
        try:
            if not news_data:
                return 0.0, 0.0
            
            articles = news_data.get('articles', 0)
            sentiment_score = news_data.get('sentiment_score', 0.0)
            relevance = news_data.get('relevance', 0.5)
            
            # Relevance threshold
            if relevance < self.news_relevance_threshold:
                return 0.0, 0.0
            
            # Volume threshold
            if articles < 3:
                return 0.0, 0.0
            
            # Adjust sentiment based on relevance and volume
            volume_factor = min(articles / 10, 1.5)  # Cap at 1.5x
            relevance_factor = relevance
            
            adjusted_sentiment = sentiment_score * volume_factor * relevance_factor
            
            # Calculate confidence based on article count and source diversity
            sources = news_data.get('sources', {})
            source_diversity = len([s for s in sources.values() if s > 0])
            
            volume_confidence = min(articles / 15, 1.0)
            diversity_confidence = source_diversity / len(self.news_sources)
            relevance_confidence = relevance
            
            final_confidence = volume_confidence * diversity_confidence * relevance_confidence
            
            # Convert to expected return (news sentiment is medium-term)
            expected_return = adjusted_sentiment * 0.04  # Max 4% impact
            
            return expected_return, final_confidence
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return 0.0, 0.0
    
    async def _analyze_market_sentiment(self, symbol: str, 
                                      market_data: Dict[str, Any]) -> tuple[float, float]:
        """Analyze market sentiment indicators"""
        try:
            if not market_data:
                return 0.0, 0.0
            
            vix = market_data.get('vix', 20)
            put_call_ratio = market_data.get('put_call_ratio', 1.0)
            fear_greed_index = market_data.get('fear_greed_index', 50)
            market_sentiment = market_data.get('market_sentiment', 0.0)
            
            # VIX analysis (inverse relationship with sentiment)
            if vix < self.vix_threshold_low:
                vix_sentiment = 0.2  # Low volatility = positive sentiment
            elif vix > self.vix_threshold_high:
                vix_sentiment = -0.3  # High volatility = negative sentiment
            else:
                vix_sentiment = (self.vix_threshold_low + self.vix_threshold_high) / 2 - vix
                vix_sentiment = vix_sentiment / 20  # Normalize
            
            # Put/Call ratio analysis
            if put_call_ratio > self.put_call_threshold:
                pc_sentiment = -0.2  # High put/call = bearish
            else:
                pc_sentiment = (1.0 - put_call_ratio) * 0.3
            
            # Fear & Greed index
            fg_sentiment = (fear_greed_index - 50) / 100  # Normalize to [-0.5, 0.5]
            
            # Combine market sentiment indicators
            combined_sentiment = (vix_sentiment + pc_sentiment + fg_sentiment + market_sentiment) / 4
            
            # Confidence based on indicator agreement
            indicators = [vix_sentiment, pc_sentiment, fg_sentiment, market_sentiment]
            sentiment_std = np.std(indicators)
            confidence = max(0.3, 1.0 - sentiment_std)  # Lower std = higher confidence
            
            # Convert to expected return (market sentiment is broad)
            expected_return = combined_sentiment * 0.02  # Max 2% impact
            
            return expected_return, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return 0.0, 0.0
    
    def _fuse_sentiment_scores(self, sentiment_scores: Dict[str, float], 
                             confidence_scores: Dict[str, float]) -> tuple[float, float]:
        """Fuse sentiment scores from multiple sources"""
        try:
            if not sentiment_scores:
                return 0.0, 0.0
            
            # Weighted average of sentiment scores
            weighted_sentiment = 0.0
            total_weight = 0.0
            
            for source, score in sentiment_scores.items():
                weight = self.sentiment_weights.get(source, 1.0)
                confidence = confidence_scores.get(source, 0.5)
                
                # Weight by source importance and confidence
                effective_weight = weight * confidence
                
                weighted_sentiment += score * effective_weight
                total_weight += effective_weight
            
            if total_weight == 0:
                return 0.0, 0.0
            
            fused_sentiment = weighted_sentiment / total_weight
            
            # Calculate fused confidence
            # Higher confidence when sources agree
            source_sentiments = list(sentiment_scores.values())
            source_confidences = list(confidence_scores.values())
            
            if len(source_sentiments) > 1:
                # Agreement factor based on standard deviation
                sentiment_std = np.std(source_sentiments)
                agreement_factor = max(0.3, 1.0 - sentiment_std * 5)  # Penalty for disagreement
            else:
                agreement_factor = 1.0
            
            # Average confidence weighted by agreement
            avg_confidence = np.mean(source_confidences)
            fused_confidence = avg_confidence * agreement_factor
            
            # Ensure reasonable ranges
            fused_sentiment = np.clip(fused_sentiment, -0.08, 0.08)
            fused_confidence = np.clip(fused_confidence, 0.0, 1.0)
            
            return fused_sentiment, fused_confidence
            
        except Exception as e:
            logger.error(f"Error fusing sentiment scores: {e}")
            return 0.0, 0.0
    
    def _assess_sentiment_conditions(self, symbol_sentiment: Dict[str, Any],
                                   sentiment_scores: Dict[str, float],
                                   confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess sentiment conditions for uncertainty calculation"""
        try:
            # Calculate sentiment volatility
            if len(sentiment_scores) > 1:
                sentiment_volatility = np.std(list(sentiment_scores.values()))
            else:
                sentiment_volatility = 0.1
            
            # Calculate confidence spread
            if len(confidence_scores) > 1:
                confidence_spread = max(confidence_scores.values()) - min(confidence_scores.values())
            else:
                confidence_spread = 0.0
            
            # Social volume factor
            social_data = symbol_sentiment.get('social', {})
            social_volume = social_data.get('mentions', 0)
            volume_factor = min(social_volume / 100, 2.0)
            
            # News recency factor (more recent = less uncertain)
            news_data = symbol_sentiment.get('news', {})
            news_recency = news_data.get('avg_age_hours', 24)
            recency_factor = max(0.5, 1.0 - (news_recency / 48))  # Decay over 48 hours
            
            return {
                'volatility': sentiment_volatility,
                'liquidity': volume_factor,  # Higher volume = better "liquidity" of sentiment
                'confidence_spread': confidence_spread,
                'recency_factor': recency_factor,
                'source_count': len(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Error assessing sentiment conditions: {e}")
            return {
                'volatility': 0.15,
                'liquidity': 1.0,
                'confidence_spread': 0.2,
                'recency_factor': 0.8,
                'source_count': 1
            }
    
    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeType:
        """Detect market regime based on sentiment indicators"""
        try:
            volatility = market_data.get('volatility', 0.15)
            confidence_spread = market_data.get('confidence_spread', 0.2)
            source_count = market_data.get('source_count', 1)
            
            # High uncertainty regime
            if confidence_spread > 0.3 or volatility > 0.3:
                return RegimeType.HIGH_VOL
            
            # Low information regime
            if source_count < 2:
                return RegimeType.ILLIQUID
            
            # Stable sentiment regime
            if volatility < 0.1 and confidence_spread < 0.1:
                return RegimeType.LOW_VOL
            
            # Default to risk-on
            return RegimeType.RISK_ON
            
        except Exception as e:
            logger.error(f"Error detecting sentiment regime: {e}")
            return RegimeType.RISK_ON
