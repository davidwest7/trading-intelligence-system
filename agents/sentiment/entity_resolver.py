"""
Entity Resolution and Named Entity Recognition for Financial Content

Extracts and resolves financial entities from text:
- Company names → Ticker symbols
- Executive names → Company associations  
- Financial metrics → Standardized values
- Geographic locations → Market relevance
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import json

try:
    import spacy
except ImportError:
    spacy = None

from .models import Entity, EntityType


class EntityResolver:
    """
    Financial entity resolution and normalization
    
    Capabilities:
    - Extract named entities from financial text
    - Map company names to ticker symbols
    - Resolve executive/person names to companies
    - Extract and normalize financial metrics
    - Determine entity sentiment context
    """
    
    def __init__(self):
        # Load spaCy model (would need to install: python -m spacy download en_core_web_sm)
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Using fallback entity extraction.")
                self.nlp = None
        else:
            print("Warning: spaCy not installed. Using fallback entity extraction.")
            self.nlp = None
        
        # Ticker symbol mappings (subset for demo)
        self.company_to_ticker = {
            # Major tech companies
            "apple": "AAPL",
            "apple inc": "AAPL", 
            "microsoft": "MSFT",
            "microsoft corp": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "amazon.com": "AMZN",
            "tesla": "TSLA",
            "tesla inc": "TSLA",
            "netflix": "NFLX",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "intel": "INTC",
            "oracle": "ORCL",
            "salesforce": "CRM",
            
            # Financial institutions
            "goldman sachs": "GS",
            "jp morgan": "JPM",
            "jpmorgan": "JPM",
            "bank of america": "BAC",
            "wells fargo": "WFC",
            "morgan stanley": "MS",
            "citigroup": "C",
            "american express": "AXP",
            
            # Other major companies
            "berkshire hathaway": "BRK.B",
            "exxon": "XOM",
            "johnson & johnson": "JNJ",
            "procter & gamble": "PG",
            "coca cola": "KO",
            "walmart": "WMT",
            "visa": "V",
            "mastercard": "MA",
            "boeing": "BA",
            "caterpillar": "CAT",
            "general electric": "GE",
            "general motors": "GM",
            "ford": "F",
            "disney": "DIS"
        }
        
        # Executive to company mappings
        self.executive_to_company = {
            "tim cook": "AAPL",
            "satya nadella": "MSFT", 
            "sundar pichai": "GOOGL",
            "andy jassy": "AMZN",
            "elon musk": "TSLA",
            "mark zuckerberg": "META",
            "jensen huang": "NVDA",
            "warren buffett": "BRK.B",
            "jamie dimon": "JPM"
        }
        
        # Common financial terms and their normalized forms
        self.financial_terms = {
            "earnings": "EARNINGS",
            "revenue": "REVENUE", 
            "profit": "PROFIT",
            "loss": "LOSS",
            "dividend": "DIVIDEND",
            "buyback": "BUYBACK",
            "merger": "MERGER",
            "acquisition": "ACQUISITION",
            "ipo": "IPO",
            "bankruptcy": "BANKRUPTCY",
            "guidance": "GUIDANCE",
            "forecast": "FORECAST"
        }
        
        # Sentiment-bearing financial terms
        self.sentiment_terms = {
            # Positive
            "beat": 1.0, "exceed": 0.8, "outperform": 0.9, "surge": 0.9,
            "rally": 0.8, "gain": 0.6, "rise": 0.5, "increase": 0.4,
            "growth": 0.6, "profit": 0.5, "success": 0.7, "strong": 0.6,
            "bullish": 0.9, "buy": 0.8, "upgrade": 0.7, "positive": 0.6,
            
            # Negative  
            "miss": -1.0, "disappoint": -0.8, "underperform": -0.9, "plunge": -0.9,
            "crash": -1.0, "fall": -0.6, "decline": -0.5, "decrease": -0.4,
            "loss": -0.6, "failure": -0.8, "weak": -0.6, "poor": -0.7,
            "bearish": -0.9, "sell": -0.8, "downgrade": -0.7, "negative": -0.6,
            
            # Neutral but important
            "neutral": 0.0, "hold": 0.0, "maintain": 0.0, "unchanged": 0.0
        }
    
    def extract_entities(self, text: str, target_tickers: List[str] = None) -> List[Entity]:
        """
        Extract and resolve entities from financial text
        
        Args:
            text: Input text to analyze
            target_tickers: Optional list of tickers to focus on
            
        Returns:
            List of resolved entities with sentiment
        """
        entities = []
        
        # Extract using spaCy if available
        if self.nlp:
            entities.extend(self._extract_with_spacy(text))
        
        # Extract using regex patterns (fallback or supplement)
        entities.extend(self._extract_with_regex(text))
        
        # Filter and enhance entities
        if target_tickers:
            entities = self._filter_relevant_entities(entities, target_tickers)
        
        # Add sentiment context
        entities = self._add_sentiment_context(entities, text)
        
        # Deduplicate and merge similar entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                # Resolve to ticker if it's a company
                resolved_text = self._resolve_company_name(ent.text.lower())
                
                entity = Entity(
                    text=resolved_text or ent.text,
                    entity_type=entity_type,
                    confidence=0.8,  # Base confidence for spaCy
                    sentiment=0.0    # Will be filled later
                )
                entities.append(entity)
        
        return entities
    
    def _extract_with_regex(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Ticker symbols (e.g., $AAPL, TSLA)
        ticker_pattern = r'\$?([A-Z]{1,5})\b'
        for match in re.finditer(ticker_pattern, text):
            ticker = match.group(1)
            if len(ticker) >= 2:  # Filter out single letters
                entity = Entity(
                    text=ticker,
                    entity_type=EntityType.TICKER,
                    confidence=0.9,
                    sentiment=0.0
                )
                entities.append(entity)
        
        # Money amounts (e.g., $1.2B, $500M)
        money_pattern = r'\$(\d+(?:\.\d+)?)\s*([BMK])\b'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            amount = match.group(0)
            entity = Entity(
                text=amount,
                entity_type=EntityType.MONEY,
                confidence=0.9,
                sentiment=0.0
            )
            entities.append(entity)
        
        # Percentages (e.g., 15%, -3.2%)
        percent_pattern = r'[-+]?\d+(?:\.\d+)?%'
        for match in re.finditer(percent_pattern, text):
            percentage = match.group(0)
            entity = Entity(
                text=percentage,
                entity_type=EntityType.PERCENT,
                confidence=0.9,
                sentiment=0.0
            )
            entities.append(entity)
        
        # Company names (check against our mapping)
        for company_name, ticker in self.company_to_ticker.items():
            if company_name in text.lower():
                entity = Entity(
                    text=ticker,
                    entity_type=EntityType.ORGANIZATION,
                    confidence=0.7,
                    sentiment=0.0
                )
                entities.append(entity)
        
        # Executive names
        for exec_name, ticker in self.executive_to_company.items():
            if exec_name in text.lower():
                entity = Entity(
                    text=f"{exec_name} ({ticker})",
                    entity_type=EntityType.PERSON,
                    confidence=0.8,
                    sentiment=0.0
                )
                entities.append(entity)
        
        return entities
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our EntityType enum"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT
        }
        return mapping.get(spacy_label)
    
    def _resolve_company_name(self, company_name: str) -> Optional[str]:
        """Resolve company name to ticker symbol"""
        # Clean the company name
        cleaned = re.sub(r'\b(inc|corp|corporation|ltd|llc|company|co)\b\.?', '', 
                        company_name.lower()).strip()
        
        # Direct lookup
        if cleaned in self.company_to_ticker:
            return self.company_to_ticker[cleaned]
        
        # Fuzzy matching for partial names
        for company, ticker in self.company_to_ticker.items():
            if cleaned in company or company in cleaned:
                return ticker
        
        return None
    
    def _filter_relevant_entities(self, entities: List[Entity], 
                                target_tickers: List[str]) -> List[Entity]:
        """Filter entities relevant to target tickers"""
        relevant = []
        target_set = set(ticker.upper() for ticker in target_tickers)
        
        for entity in entities:
            # Always include if it's a target ticker
            if entity.entity_type == EntityType.TICKER and entity.text in target_set:
                relevant.append(entity)
            # Include if it's a resolved company that matches target
            elif entity.entity_type == EntityType.ORGANIZATION and entity.text in target_set:
                relevant.append(entity)
            # Include financial terms and metrics
            elif entity.entity_type in [EntityType.MONEY, EntityType.PERCENT]:
                relevant.append(entity)
            # Include executives of target companies
            elif entity.entity_type == EntityType.PERSON:
                for ticker in target_set:
                    if ticker in entity.text:
                        relevant.append(entity)
                        break
        
        return relevant
    
    def _add_sentiment_context(self, entities: List[Entity], text: str) -> List[Entity]:
        """Add sentiment context to entities based on surrounding text"""
        text_lower = text.lower()
        
        for entity in entities:
            # Find entity position in text
            entity_pos = text_lower.find(entity.text.lower())
            if entity_pos == -1:
                continue
            
            # Extract context window (50 chars before/after)
            start = max(0, entity_pos - 50)
            end = min(len(text), entity_pos + len(entity.text) + 50)
            context = text_lower[start:end]
            
            # Calculate sentiment based on surrounding terms
            sentiment_score = 0.0
            sentiment_count = 0
            
            for term, score in self.sentiment_terms.items():
                if term in context:
                    # Weight by proximity to entity
                    term_pos = context.find(term)
                    entity_pos_in_context = entity_pos - start
                    distance = abs(term_pos - entity_pos_in_context)
                    
                    # Closer terms have more influence
                    weight = max(0.1, 1.0 - (distance / 50.0))
                    sentiment_score += score * weight
                    sentiment_count += weight
            
            if sentiment_count > 0:
                entity.sentiment = sentiment_score / sentiment_count
                entity.sentiment = max(-1.0, min(1.0, entity.sentiment))  # Clamp
            
            # Boost confidence if we found sentiment context
            if sentiment_count > 0:
                entity.confidence = min(1.0, entity.confidence + 0.1)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate and merge similar entities"""
        seen = {}
        
        for entity in entities:
            key = (entity.text.upper(), entity.entity_type)
            
            if key in seen:
                # Merge entities
                existing = seen[key]
                existing.mentions += 1
                existing.confidence = max(existing.confidence, entity.confidence)
                # Average sentiment weighted by confidence
                total_weight = existing.confidence + entity.confidence
                existing.sentiment = (
                    (existing.sentiment * existing.confidence + 
                     entity.sentiment * entity.confidence) / total_weight
                )
            else:
                seen[key] = entity
        
        return list(seen.values())
    
    def resolve_ticker_synonyms(self, text: str) -> str:
        """Replace company names with ticker symbols in text"""
        result = text
        
        for company, ticker in self.company_to_ticker.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(company), re.IGNORECASE)
            result = pattern.sub(ticker, result)
        
        return result
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information for a ticker"""
        # Reverse lookup
        company_name = None
        for company, tick in self.company_to_ticker.items():
            if tick == ticker.upper():
                company_name = company
                break
        
        # Find executives
        executives = []
        for exec_name, exec_ticker in self.executive_to_company.items():
            if exec_ticker == ticker.upper():
                executives.append(exec_name)
        
        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "executives": executives
        }
    
    def update_mappings(self, company_mappings: Dict[str, str] = None,
                       executive_mappings: Dict[str, str] = None):
        """Update entity mappings"""
        if company_mappings:
            self.company_to_ticker.update(company_mappings)
        
        if executive_mappings:
            self.executive_to_company.update(executive_mappings)
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about loaded mappings"""
        return {
            "companies": len(self.company_to_ticker),
            "executives": len(self.executive_to_company),
            "financial_terms": len(self.financial_terms),
            "sentiment_terms": len(self.sentiment_terms)
        }
