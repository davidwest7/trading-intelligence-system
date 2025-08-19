"""
Bot Detection System for Social Media Posts

Advanced bot detection using multiple signals:
- Account metadata analysis
- Posting pattern analysis  
- Content similarity detection
- Network analysis
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from .models import BotDetectionFeatures, SentimentPost


class BotDetector:
    """
    Advanced bot detection system for social media analysis
    
    Uses multiple signals to identify automated accounts:
    1. Account age and metadata
    2. Posting frequency and patterns
    3. Content similarity and repetition
    4. Network behavior analysis
    """
    
    def __init__(self):
        self.known_bots: Set[str] = set()
        self.known_humans: Set[str] = set()
        self.content_hashes: Dict[str, List[str]] = defaultdict(list)
        self.posting_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Bot detection thresholds
        self.thresholds = {
            'min_account_age_days': 7,
            'max_posts_per_day': 100,
            'min_profile_completeness': 0.3,
            'max_posting_regularity': 0.8,
            'max_content_similarity': 0.7,
            'confidence_threshold': 0.6
        }
    
    def detect_bots(self, posts: List[SentimentPost]) -> Dict[str, float]:
        """
        Detect bots in a list of posts
        
        Args:
            posts: List of social media posts to analyze
            
        Returns:
            Dictionary mapping author to bot probability (0-1)
        """
        author_posts = defaultdict(list)
        for post in posts:
            author_posts[post.author].append(post)
        
        bot_probabilities = {}
        
        for author, author_post_list in author_posts.items():
            # Skip if already classified
            if author in self.known_bots:
                bot_probabilities[author] = 1.0
                continue
            elif author in self.known_humans:
                bot_probabilities[author] = 0.0
                continue
            
            # Extract features for this author
            features = self._extract_bot_features(author, author_post_list)
            
            # Calculate bot probability
            bot_prob = features.calculate_bot_probability()
            
            # Apply additional heuristics
            bot_prob = self._apply_heuristics(author, author_post_list, bot_prob)
            
            bot_probabilities[author] = bot_prob
            
            # Update knowledge base
            if bot_prob > 0.8:
                self.known_bots.add(author)
            elif bot_prob < 0.2:
                self.known_humans.add(author)
        
        return bot_probabilities
    
    def _extract_bot_features(self, author: str, posts: List[SentimentPost]) -> BotDetectionFeatures:
        """Extract bot detection features for an author"""
        
        # Account age (mock - would get from API in real implementation)
        account_age = self._estimate_account_age(author, posts)
        
        # Posting frequency
        if len(posts) > 1:
            time_span = (posts[-1].timestamp - posts[0].timestamp).total_seconds() / 86400  # days
            posts_per_day = len(posts) / max(time_span, 1)
        else:
            posts_per_day = 1
        
        # Profile analysis (mock data)
        followers_count = self._estimate_followers(author)
        following_count = self._estimate_following(author)
        profile_completeness = self._calculate_profile_completeness(author)
        verified = self._is_verified(author)
        
        # Posting pattern analysis
        posting_pattern_score = self._analyze_posting_patterns(posts)
        
        # Content similarity
        content_similarity_score = self._analyze_content_similarity(posts)
        
        # Network centrality (simplified)
        network_centrality = self._calculate_network_centrality(author)
        
        return BotDetectionFeatures(
            account_age_days=account_age,
            posts_per_day=posts_per_day,
            followers_count=followers_count,
            following_count=following_count,
            profile_completeness=profile_completeness,
            posting_pattern_score=posting_pattern_score,
            content_similarity_score=content_similarity_score,
            network_centrality=network_centrality,
            verified=verified
        )
    
    def _estimate_account_age(self, author: str, posts: List[SentimentPost]) -> int:
        """Estimate account age based on available data"""
        # In real implementation, would get from API
        # For now, estimate based on author name patterns
        if re.match(r'^[a-zA-Z]+\d{4,}$', author):  # Name + many numbers = likely newer
            return np.random.randint(1, 100)
        else:
            return np.random.randint(100, 2000)
    
    def _estimate_followers(self, author: str) -> int:
        """Estimate follower count (mock)"""
        # In real implementation, would get from API
        return max(1, int(np.random.exponential(1000)))
    
    def _estimate_following(self, author: str) -> int:
        """Estimate following count (mock)"""
        return max(1, int(np.random.exponential(500)))
    
    def _calculate_profile_completeness(self, author: str) -> float:
        """Calculate profile completeness score"""
        # Mock calculation based on username patterns
        score = 0.5  # Base score
        
        if len(author) > 3:
            score += 0.2
        if not re.search(r'\d{3,}', author):  # No long number sequences
            score += 0.2
        if '_' not in author and '-' not in author:  # No underscores/dashes
            score += 0.1
            
        return min(1.0, score)
    
    def _is_verified(self, author: str) -> bool:
        """Check if account is verified (mock)"""
        # Mock - assume 5% of accounts are verified
        return hash(author) % 20 == 0
    
    def _analyze_posting_patterns(self, posts: List[SentimentPost]) -> float:
        """Analyze posting patterns for bot-like regularity"""
        if len(posts) < 3:
            return 0.0
        
        # Calculate time intervals between posts
        intervals = []
        sorted_posts = sorted(posts, key=lambda p: p.timestamp)
        
        for i in range(1, len(sorted_posts)):
            interval = (sorted_posts[i].timestamp - sorted_posts[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # High regularity suggests bot behavior
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 1.0
        
        coefficient_of_variation = std_interval / mean_interval
        
        # Low variation = high regularity = more bot-like
        regularity_score = max(0.0, 1.0 - coefficient_of_variation)
        
        return min(1.0, regularity_score)
    
    def _analyze_content_similarity(self, posts: List[SentimentPost]) -> float:
        """Analyze content similarity between posts"""
        if len(posts) < 2:
            return 0.0
        
        # Simple content hashing for similarity detection
        content_hashes = []
        for post in posts:
            # Normalize text
            normalized = re.sub(r'[^\w\s]', '', post.text.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Create hash
            content_hash = hashlib.md5(normalized.encode()).hexdigest()
            content_hashes.append(content_hash)
        
        # Count unique hashes
        unique_hashes = len(set(content_hashes))
        total_hashes = len(content_hashes)
        
        # High similarity = low uniqueness = more bot-like
        similarity_score = 1.0 - (unique_hashes / total_hashes)
        
        return similarity_score
    
    def _calculate_network_centrality(self, author: str) -> float:
        """Calculate network centrality (simplified)"""
        # In real implementation, would analyze interaction networks
        # Mock calculation based on author characteristics
        return np.random.uniform(0, 0.5)  # Most accounts have low centrality
    
    def _apply_heuristics(self, author: str, posts: List[SentimentPost], 
                         base_prob: float) -> float:
        """Apply additional heuristic rules"""
        adjusted_prob = base_prob
        
        # Username patterns that suggest bots
        if re.match(r'^[a-zA-Z]+\d{8,}$', author):  # Name + 8+ digits
            adjusted_prob += 0.2
        
        if re.match(r'^[a-zA-Z]+_[a-zA-Z]+\d+$', author):  # firstname_lastname123
            adjusted_prob += 0.1
        
        # Content patterns
        for post in posts[:5]:  # Check first 5 posts
            # Very short posts
            if len(post.text) < 10:
                adjusted_prob += 0.05
            
            # Posts with many hashtags
            hashtag_count = post.text.count('#')
            if hashtag_count > 5:
                adjusted_prob += 0.1
            
            # Posts with many mentions
            mention_count = post.text.count('@')
            if mention_count > 3:
                adjusted_prob += 0.05
        
        # Temporal patterns
        if len(posts) > 10:
            # Check for posts at exact hour intervals
            hour_posts = defaultdict(int)
            for post in posts:
                hour_posts[post.timestamp.hour] += 1
            
            # If most posts are at exact hours, likely bot
            exact_hour_ratio = sum(1 for count in hour_posts.values() if count > 2) / len(hour_posts)
            if exact_hour_ratio > 0.7:
                adjusted_prob += 0.2
        
        return min(1.0, adjusted_prob)
    
    def get_bot_statistics(self) -> Dict[str, Any]:
        """Get bot detection statistics"""
        return {
            "known_bots": len(self.known_bots),
            "known_humans": len(self.known_humans),
            "thresholds": self.thresholds,
            "total_analyzed": len(self.known_bots) + len(self.known_humans)
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update bot detection thresholds"""
        self.thresholds.update(new_thresholds)
    
    def add_known_bot(self, author: str):
        """Manually add a known bot"""
        self.known_bots.add(author)
        self.known_humans.discard(author)
    
    def add_known_human(self, author: str):
        """Manually add a known human"""
        self.known_humans.add(author)
        self.known_bots.discard(author)


class ContentDeduplicator:
    """Remove duplicate and near-duplicate content"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
    
    def deduplicate_posts(self, posts: List[SentimentPost]) -> List[SentimentPost]:
        """Remove duplicate posts"""
        unique_posts = []
        
        for post in posts:
            content_hash = self._get_content_hash(post.text)
            
            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                unique_posts.append(post)
        
        return unique_posts
    
    def _get_content_hash(self, text: str) -> str:
        """Get normalized content hash"""
        # Remove URLs, mentions, hashtags for better deduplication
        normalized = re.sub(r'http\S+', '', text)
        normalized = re.sub(r'@\w+', '', normalized)
        normalized = re.sub(r'#\w+', '', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
