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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.text,
            "type": self.entity_type.value,
            "sentiment": self.sentiment,
            "mentions": self.mentions,
            "confidence": self.confidence
        }


@dataclass
class SentimentPost:
    """Individual social media post/article with sentiment"""
    id: str
    source: SourceType
    text: str
    author: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    is_bot: bool
    reach: int  # followers, upvotes, etc.
    entities: List[Entity] = field(default_factory=list)
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
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
    ticker: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    volume: int  # Total mentions
    velocity: float  # Rate of change in mentions
    dispersion: float  # Variance across sources
    bot_ratio: float  # Estimated bot activity
    sources_breakdown: Dict[str, SourceBreakdown]
    top_entities: List[Entity]
    sentiment_label: SentimentLabel
    momentum: float  # Sentiment momentum (change over time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "volume": self.volume,
            "velocity": self.velocity,
            "dispersion": self.dispersion,
            "bot_ratio": self.bot_ratio,
            "sources_breakdown": {
                source: breakdown.to_dict() 
                for source, breakdown in self.sources_breakdown.items()
            },
            "top_entities": [entity.to_dict() for entity in self.top_entities],
            "sentiment_label": self.sentiment_label.value,
            "momentum": self.momentum
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
