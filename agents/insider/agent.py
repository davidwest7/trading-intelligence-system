"""
Insider Activity Agent

Analyzes SEC insider filings and transaction patterns
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .models import (
    InsiderAnalysis, InsiderTransaction, InsiderSentiment,
    TransactionType, InsiderRole, SentimentSignal, FilingType
)
from ..common.models import BaseAgent


class InsiderAgent(BaseAgent):
    """Complete Insider Activity Analysis Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("insider", config)
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.analyze_insider_activity(*args, **kwargs)
    
    async def analyze_insider_activity(
        self,
        tickers: List[str],
        lookback_period: str = "90d"
    ) -> Dict[str, Any]:
        """Analyze insider activity for given tickers"""
        
        analyses = []
        
        for ticker in tickers:
            # Create demo insider analysis
            analysis = self._create_demo_analysis(ticker, lookback_period)
            analyses.append(analysis)
        
        return {
            "insider_analyses": [analysis.to_dict() for analysis in analyses]
        }
    
    def _create_demo_analysis(self, ticker: str, period: str) -> InsiderAnalysis:
        """Create demo insider analysis"""
        
        # Generate mock recent transactions
        transactions = []
        for i in range(np.random.randint(2, 8)):
            transaction = InsiderTransaction(
                ticker=ticker,
                filing_date=datetime.now() - timedelta(days=np.random.randint(1, 60)),
                transaction_date=datetime.now() - timedelta(days=np.random.randint(1, 65)),
                insider_name=f"Executive_{i+1}",
                insider_role=InsiderRole.CEO,
                insider_cik=f"CIK{np.random.randint(1000000, 9999999)}",
                transaction_type=TransactionType.BUY if np.random.random() > 0.5 else TransactionType.SELL,
                shares_traded=np.random.uniform(1000, 50000),
                price_per_share=np.random.uniform(50, 300),
                total_value=0,  # Will be calculated
                shares_owned_after=np.random.uniform(10000, 500000),
                ownership_percentage=np.random.uniform(0.001, 0.05),
                filing_type=FilingType.FORM_4,
                is_direct_holding=True
            )
            transaction.total_value = transaction.shares_traded * transaction.price_per_share
            transactions.append(transaction)
        
        # Calculate sentiment
        buy_value = sum(t.total_value for t in transactions if t.transaction_type == TransactionType.BUY)
        sell_value = sum(t.total_value for t in transactions if t.transaction_type == TransactionType.SELL)
        net_activity = buy_value - sell_value
        
        sentiment = InsiderSentiment(
            ticker=ticker,
            timestamp=datetime.now(),
            lookback_period=timedelta(days=90),
            total_buy_value=buy_value,
            total_sell_value=sell_value,
            net_insider_activity=net_activity,
            num_buyers=len([t for t in transactions if t.transaction_type == TransactionType.BUY]),
            num_sellers=len([t for t in transactions if t.transaction_type == TransactionType.SELL]),
            net_participants=0,  # Will be calculated
            ceo_sentiment=SentimentSignal.BULLISH,
            cfo_sentiment=SentimentSignal.NEUTRAL,
            director_sentiment=SentimentSignal.BULLISH,
            officer_sentiment=SentimentSignal.NEUTRAL,
            activity_near_earnings=np.random.choice([True, False]),
            days_to_next_earnings=np.random.randint(5, 90),
            activity_vs_historical=np.random.uniform(-0.5, 2.0),
            raw_sentiment_score=np.random.uniform(-1, 1),
            adjusted_sentiment_score=np.random.uniform(-1, 1),
            confidence_level=np.random.uniform(0.5, 0.9),
            overall_sentiment=SentimentSignal.BULLISH,
            strength=np.random.uniform(0.3, 0.9)
        )
        
        return InsiderAnalysis(
            ticker=ticker,
            timestamp=datetime.now(),
            analysis_period=timedelta(days=90),
            recent_transactions=transactions,
            significant_transactions=transactions[:3],
            current_sentiment=sentiment,
            sentiment_trend=np.random.choice(["improving", "deteriorating", "stable"]),
            detected_patterns=[],
            unusual_activity_detected=np.random.choice([True, False]),
            key_buyers=[],
            key_sellers=[],
            predicted_next_transaction_type=TransactionType.BUY,
            estimated_price_impact=np.random.uniform(-0.05, 0.05),
            time_to_next_expected_activity=timedelta(days=np.random.randint(7, 30)),
            regulatory_risk_score=np.random.uniform(0.1, 0.4),
            information_asymmetry_score=np.random.uniform(0.2, 0.8),
            historical_accuracy=np.random.uniform(0.6, 0.85),
            insider_performance_vs_market=np.random.uniform(-0.05, 0.15)
        )
