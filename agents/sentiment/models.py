"""
Data models for Sentiment Analysis Agent
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish" 
    NEUTRAL = "neutral"


class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    TELEGRAM = "telegram"
    DISCORD = "discord"


class MarketImpact(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    TICKER = "TICKER"


@dataclass
class Entity:
    """Named entity with sentiment context"""
    text: str
    entity_type: EntityType
    confidence: float
    sentiment: float  # -1 to 1
    mentions: int = 1
    symbol: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.text,
            "type": self.entity_type.value,
            "sentiment": self.sentiment,
            "mentions": self.mentions,
            "confidence": self.confidence,
            "symbol": self.symbol
        }


@dataclass
class SentimentPost:
    """Individual social media post/article with sentiment"""
    id: str
    source: str
    content: str
    author: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    is_bot: bool
    reach: int  # followers, upvotes, etc.
    entities: List[Entity] = field(default_factory=list)
    url: Optional[str] = None
    
    @property
    def text(self) -> str:
        return self.content
    
    @property
    def sentiment(self):
        return SentimentLabel(
            "bullish" if self.sentiment_score > 0.2 else "bearish" if self.sentiment_score < -0.2 else "neutral"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "is_bot": self.is_bot,
            "reach": self.reach,
            "entities": [e.to_dict() for e in self.entities],
            "url": self.url
        }


@dataclass
class SourceBreakdown:
    """Sentiment breakdown for a specific source"""
    score: float  # -1 to 1
    volume: int
    confidence: float
    bot_ratio: float
    top_posts: List[SentimentPost] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "volume": self.volume,
            "confidence": self.confidence,
            "bot_ratio": self.bot_ratio,
            "top_posts": [post.to_dict() for post in self.top_posts[:3]]
        }


@dataclass
class SentimentAnalysis:
    """Complete sentiment analysis for a ticker"""
    overall_score: float  # -1 to 1
    sentiment_distribution: Dict[str, float]
    source_breakdown: Dict[str, Dict[str, Any]]
    market_impact: str
    confidence: float  # 0 to 1
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "sentiment_distribution": self.sentiment_distribution,
            "source_breakdown": self.source_breakdown,
            "market_impact": self.market_impact,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BotDetectionFeatures:
    """Features used for bot detection"""
    account_age_days: int
    posts_per_day: float
    followers_count: int
    following_count: int
    profile_completeness: float  # 0 to 1
    posting_pattern_score: float  # 0 to 1 (1 = very regular/bot-like)
    content_similarity_score: float  # 0 to 1 (1 = very similar to other posts)
    network_centrality: float  # 0 to 1
    verified: bool
    
    def calculate_bot_probability(self) -> float:
        """Calculate probability that account is a bot"""
        # Weighted scoring based on bot indicators
        weights = {
            'new_account': 0.2 if self.account_age_days < 30 else 0.0,
            'high_frequency': 0.3 if self.posts_per_day > 50 else 0.0,
            'low_followers': 0.1 if self.followers_count < 10 else 0.0,
            'incomplete_profile': 0.2 if self.profile_completeness < 0.3 else 0.0,
            'regular_pattern': 0.3 * self.posting_pattern_score,
            'similar_content': 0.4 * self.content_similarity_score,
            'verified_reduction': -0.5 if self.verified else 0.0
        }
        
        bot_score = sum(weights.values())
        return max(0.0, min(1.0, bot_score))


@dataclass
class SentimentRequest:
    """Request for sentiment analysis"""
    tickers: List[str]
    window: str  # "1m", "5m", "15m", "1h", "4h", "1d"
    sources: List[str] = field(default_factory=lambda: ["twitter", "reddit", "news"])
    min_confidence: float = 0.7
    include_bot_posts: bool = False
    max_posts_per_source: int = 1000
    
    def validate(self) -> bool:
        """Validate request parameters"""
        valid_windows = ["1m", "5m", "15m", "1h", "4h", "1d"]
        valid_sources = ["twitter", "reddit", "news", "telegram", "discord"]
        
        return (
            len(self.tickers) > 0 and
            self.window in valid_windows and
            all(source in valid_sources for source in self.sources) and
            0.0 <= self.min_confidence <= 1.0
        )


@dataclass
class SentimentData:
    """Sentiment data for a ticker"""
    ticker: str
    sentiment_score: float
    confidence: float
    timestamp: datetime
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


@dataclass
class SentimentMetrics:
    """Performance metrics for sentiment analysis"""
    total_posts_processed: int
    bot_posts_filtered: int
    duplicate_posts_filtered: int
    entities_resolved: int
    sentiment_accuracy: float
    processing_time_avg: float
    cache_hit_rate: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_posts_processed": self.total_posts_processed,
            "bot_posts_filtered": self.bot_posts_filtered,
            "duplicate_posts_filtered": self.duplicate_posts_filtered,
            "entities_resolved": self.entities_resolved,
            "sentiment_accuracy": self.sentiment_accuracy,
            "processing_time_avg": self.processing_time_avg,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate
        }



