"""
Financial Sentiment Analysis Engine

Advanced sentiment analysis specifically tuned for financial content:
- Financial terminology awareness
- Context-dependent sentiment scoring
- Multi-model ensemble approach
- Confidence calibration
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


class FinancialSentimentAnalyzer:
    """
    Advanced financial sentiment analyzer
    
    Features:
    - Financial terminology-aware sentiment scoring
    - Context-dependent analysis (earnings, guidance, etc.)
    - Multi-model ensemble (VADER + TextBlob + custom rules)
    - Confidence calibration based on signal strength
    - Ticker-specific sentiment extraction
    """
    
    def __init__(self):
        # Initialize sentiment models
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        
        # Financial sentiment lexicon
        self.financial_lexicon = {
            # Very Positive (0.8 - 1.0)
            "beat": 0.9, "exceed": 0.8, "outperform": 0.9, "surge": 0.9,
            "soar": 1.0, "rally": 0.8, "breakout": 0.8, "upgrade": 0.8,
            "bullish": 0.9, "moonshot": 1.0, "rocket": 0.9, "lambo": 0.8,
            
            # Positive (0.3 - 0.7)
            "gain": 0.6, "rise": 0.5, "increase": 0.4, "growth": 0.6,
            "profit": 0.5, "strong": 0.6, "solid": 0.5, "good": 0.4,
            "buy": 0.7, "long": 0.6, "hold": 0.3, "diamond hands": 0.8,
            
            # Negative (-0.3 to -0.7)
            "fall": -0.5, "decline": -0.5, "decrease": -0.4, "drop": -0.6,
            "loss": -0.6, "weak": -0.6, "poor": -0.7, "bad": -0.5,
            "sell": -0.7, "short": -0.6, "bearish": -0.8, "paper hands": -0.6,
            
            # Very Negative (-0.8 to -1.0)
            "miss": -0.9, "disappoint": -0.8, "underperform": -0.9, "plunge": -0.9,
            "crash": -1.0, "collapse": -1.0, "tank": -0.9, "dump": -0.8,
            "downgrade": -0.8, "rug pull": -1.0, "bagholding": -0.8,
            
            # Context-dependent terms
            "guidance": 0.0,  # Depends on direction
            "earnings": 0.0,  # Depends on beat/miss
            "revenue": 0.0,   # Depends on context
            "forecast": 0.0,  # Depends on direction
        }
        
        # Financial negation terms
        self.negation_terms = [
            "not", "no", "never", "none", "nobody", "nothing", "neither",
            "nowhere", "hardly", "barely", "scarcely", "seldom", "rarely"
        ]
        
        # Intensity modifiers
        self.intensity_modifiers = {
            "very": 1.3, "extremely": 1.5, "highly": 1.3, "super": 1.4,
            "really": 1.2, "quite": 1.1, "pretty": 1.1, "somewhat": 0.8,
            "slightly": 0.7, "barely": 0.6, "hardly": 0.5, "little": 0.7
        }
        
        # Ticker-specific context patterns
        self.ticker_patterns = {
            "earnings_beat": r"(beat|exceed).{0,20}(earnings|estimates|expectations)",
            "earnings_miss": r"(miss|disappoint).{0,20}(earnings|estimates|expectations)",
            "price_target": r"price.{0,10}target.{0,10}(\$?\d+)",
            "analyst_upgrade": r"(upgrade|raise|increase).{0,20}rating",
            "analyst_downgrade": r"(downgrade|lower|cut).{0,20}rating"
        }
    
    def analyze_sentiment(self, text: str, ticker: str = None) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis for financial text
        
        Args:
            text: Text to analyze
            ticker: Optional ticker symbol for context
            
        Returns:
            Dictionary with sentiment_score, confidence, and breakdown
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Get sentiment from multiple models
        sentiment_scores = []
        confidences = []
        
        # VADER sentiment
        if self.vader:
            vader_result = self._analyze_with_vader(processed_text)
            sentiment_scores.append(vader_result['sentiment'])
            confidences.append(vader_result['confidence'])
        
        # TextBlob sentiment
        if TextBlob:
            textblob_result = self._analyze_with_textblob(processed_text)
            sentiment_scores.append(textblob_result['sentiment'])
            confidences.append(textblob_result['confidence'])
        
        # Financial lexicon sentiment
        lexicon_result = self._analyze_with_financial_lexicon(processed_text)
        sentiment_scores.append(lexicon_result['sentiment'])
        confidences.append(lexicon_result['confidence'])
        
        # Ticker-specific sentiment
        if ticker:
            ticker_result = self._analyze_ticker_context(processed_text, ticker)
            sentiment_scores.append(ticker_result['sentiment'])
            confidences.append(ticker_result['confidence'])
        
        # Ensemble the results
        final_sentiment, final_confidence = self._ensemble_results(
            sentiment_scores, confidences
        )
        
        # Apply post-processing adjustments
        final_sentiment = self._apply_adjustments(
            final_sentiment, processed_text, ticker
        )
        
        return {
            "sentiment_score": final_sentiment,
            "confidence": final_confidence,
            "breakdown": {
                "vader": sentiment_scores[0] if len(sentiment_scores) > 0 else 0.0,
                "textblob": sentiment_scores[1] if len(sentiment_scores) > 1 else 0.0,
                "financial_lexicon": sentiment_scores[2] if len(sentiment_scores) > 2 else 0.0,
                "ticker_context": sentiment_scores[3] if len(sentiment_scores) > 3 else 0.0,
            },
            "text_length": len(text),
            "processed_length": len(processed_text)
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        processed = text.lower()
        
        # Handle emoji-to-text conversion for financial emojis
        emoji_replacements = {
            "🚀": " rocket moon ",
            "📈": " up bullish ",
            "📉": " down bearish ",
            "💎": " diamond hands hold ",
            "🌙": " moon rocket ",
            "🐻": " bearish down ",
            "🐂": " bullish up ",
            "💰": " money profit ",
            "💸": " loss money down ",
            "🔥": " hot amazing ",
            "❄️": " cold bad ",
            "🎯": " target good ",
            "⚠️": " warning bad ",
            "✅": " good positive ",
            "❌": " bad negative "
        }
        
        for emoji, replacement in emoji_replacements.items():
            processed = processed.replace(emoji, replacement)
        
        # Handle financial abbreviations
        abbreviations = {
            "ath": "all time high",
            "atl": "all time low", 
            "hodl": "hold",
            "btfd": "buy the fucking dip",
            "dd": "due diligence",
            "yolo": "high risk bet",
            "fomo": "fear of missing out",
            "fud": "fear uncertainty doubt",
            "rekt": "destroyed loss",
            "lambo": "lamborghini rich",
            "tendies": "profits money"
        }
        
        for abbr, full in abbreviations.items():
            processed = re.sub(r'\b' + abbr + r'\b', full, processed)
        
        return processed
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        
        # VADER returns compound score from -1 to 1
        sentiment = scores['compound']
        
        # Calculate confidence based on the magnitude and distribution
        pos, neu, neg = scores['pos'], scores['neu'], scores['neg']
        
        # Higher confidence when one sentiment dominates
        max_component = max(pos, neu, neg)
        confidence = max_component if max_component > 0.6 else 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_scores": scores
        }
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        
        # TextBlob returns polarity from -1 to 1
        sentiment = blob.sentiment.polarity
        
        # Use subjectivity as confidence proxy (more subjective = higher confidence)
        confidence = blob.sentiment.subjectivity
        
        return {
            "sentiment": sentiment,
            "confidence": confidence
        }
    
    def _analyze_with_financial_lexicon(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using financial-specific lexicon"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        sentiment_scores = []
        for i, word in enumerate(words):
            if word in self.financial_lexicon:
                base_score = self.financial_lexicon[word]
                
                # Check for negation in previous 3 words
                negated = any(
                    neg_word in words[max(0, i-3):i] 
                    for neg_word in self.negation_terms
                )
                
                # Check for intensity modifiers
                intensity = 1.0
                for j in range(max(0, i-2), i):
                    if j < len(words) and words[j] in self.intensity_modifiers:
                        intensity = self.intensity_modifiers[words[j]]
                        break
                
                # Apply modifications
                score = base_score * intensity
                if negated:
                    score *= -1
                
                sentiment_scores.append(score)
        
        if not sentiment_scores:
            return {"sentiment": 0.0, "confidence": 0.0}
        
        # Calculate weighted average
        avg_sentiment = np.mean(sentiment_scores)
        
        # Confidence based on number of financial terms and score consistency
        confidence = min(1.0, len(sentiment_scores) / 10)  # More terms = higher confidence
        if len(sentiment_scores) > 1:
            # Boost confidence if scores are consistent
            score_std = np.std(sentiment_scores)
            if score_std < 0.3:  # Consistent scores
                confidence *= 1.2
        
        return {
            "sentiment": avg_sentiment,
            "confidence": confidence,
            "terms_found": len(sentiment_scores)
        }
    
    def _analyze_ticker_context(self, text: str, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment in ticker-specific context"""
        sentiment_adjustments = []
        confidence_boost = 0.0
        
        # Check for ticker-specific patterns
        for pattern_name, pattern in self.ticker_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence_boost += 0.2
                
                if "beat" in pattern_name or "upgrade" in pattern_name:
                    sentiment_adjustments.append(0.3)
                elif "miss" in pattern_name or "downgrade" in pattern_name:
                    sentiment_adjustments.append(-0.3)
                elif "target" in pattern_name:
                    # Try to extract the target and compare context
                    sentiment_adjustments.append(0.1)  # Neutral positive
        
        # Check for ticker mentions with directional terms
        ticker_contexts = [
            (r'\$?' + ticker + r'.{0,20}(up|rising|gaining)', 0.2),
            (r'\$?' + ticker + r'.{0,20}(down|falling|dropping)', -0.2),
            (r'(buying|bullish on).{0,20}\$?' + ticker, 0.3),
            (r'(selling|bearish on).{0,20}\$?' + ticker, -0.3),
        ]
        
        for pattern, sentiment_value in ticker_contexts:
            if re.search(pattern, text, re.IGNORECASE):
                sentiment_adjustments.append(sentiment_value)
                confidence_boost += 0.1
        
        # Calculate final values
        if sentiment_adjustments:
            avg_sentiment = np.mean(sentiment_adjustments)
            confidence = min(1.0, 0.5 + confidence_boost)
        else:
            avg_sentiment = 0.0
            confidence = 0.0
        
        return {
            "sentiment": avg_sentiment,
            "confidence": confidence,
            "adjustments": len(sentiment_adjustments)
        }
    
    def _ensemble_results(self, sentiments: List[float], 
                         confidences: List[float]) -> Tuple[float, float]:
        """Ensemble multiple sentiment results"""
        if not sentiments:
            return 0.0, 0.0
        
        # Weight by confidence
        total_weight = sum(confidences)
        if total_weight == 0:
            return np.mean(sentiments), np.mean(confidences)
        
        weighted_sentiment = sum(
            sent * conf for sent, conf in zip(sentiments, confidences)
        ) / total_weight
        
        # Average confidence, boosted if models agree
        avg_confidence = np.mean(confidences)
        
        # Agreement bonus: if models agree on direction, boost confidence
        if len(sentiments) > 1:
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            total_count = len(sentiments)
            
            # If majority agrees on direction
            if positive_count > total_count * 0.6 or negative_count > total_count * 0.6:
                avg_confidence *= 1.2
        
        return weighted_sentiment, min(1.0, avg_confidence)
    
    def _apply_adjustments(self, sentiment: float, text: str, ticker: str) -> float:
        """Apply final adjustments to sentiment score"""
        adjusted = sentiment
        
        # Penalize very short texts
        if len(text.split()) < 5:
            adjusted *= 0.8
        
        # Boost confidence for longer, more detailed texts
        elif len(text.split()) > 20:
            adjusted *= 1.1
        
        # Check for uncertainty indicators
        uncertainty_terms = ["maybe", "might", "could", "possibly", "uncertain", "unsure"]
        if any(term in text.lower() for term in uncertainty_terms):
            adjusted *= 0.8
        
        # Check for strong conviction terms
        conviction_terms = ["definitely", "certainly", "absolutely", "guaranteed", "sure"]
        if any(term in text.lower() for term in conviction_terms):
            adjusted *= 1.2
        
        # Clamp to valid range
        return max(-1.0, min(1.0, adjusted))
    
    def batch_analyze(self, texts: List[str], ticker: str = None) -> List[Dict[str, Any]]:
        """Analyze multiple texts efficiently"""
        return [self.analyze_sentiment(text, ticker) for text in texts]
    
    def get_lexicon_coverage(self, text: str) -> Dict[str, Any]:
        """Get statistics about lexicon coverage in text"""
        words = re.findall(r'\b\w+\b', text.lower())
        financial_words = [word for word in words if word in self.financial_lexicon]
        
        return {
            "total_words": len(words),
            "financial_words": len(financial_words),
            "coverage_ratio": len(financial_words) / max(len(words), 1),
            "financial_terms": financial_words
        }
