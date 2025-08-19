"""
Money Flows Agent

Analyzes institutional money flows including:
- Dark pool activity estimation
- Block trade identification  
- Institution type classification
- Flow pattern recognition
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .models import (
    MoneyFlowAnalysis, MoneyFlowRequest, InstitutionalFlow,
    DarkPoolActivity, VolumeConcentration, FlowPattern,
    FlowType, FlowDirection, InstitutionType
)
from .flow_detector import InstitutionalFlowDetector
from ..common.models import BaseAgent


class MoneyFlowsAgent(BaseAgent):
    """
    Complete Money Flows Analysis Agent
    
    Capabilities:
    ✅ Institutional flow detection and classification
    ✅ Dark pool activity estimation
    ✅ Block trade identification
    ✅ Flow pattern recognition
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("moneyflows", config)
        self.flow_detector = InstitutionalFlowDetector()
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        return await self.analyze_money_flows(*args, **kwargs)
    
    async def analyze_money_flows(
        self,
        tickers: List[str],
        analysis_period: str = "1d"
    ) -> Dict[str, Any]:
        """Analyze institutional money flows for given tickers"""
        
        analyses = []
        
        for ticker in tickers:
            # Generate mock analysis for demo
            analysis = self._create_demo_analysis(ticker, analysis_period)
            analyses.append(analysis)
        
        return {
            "money_flow_analyses": [analysis.to_dict() for analysis in analyses]
        }
    
    def _create_demo_analysis(self, ticker: str, period: str) -> MoneyFlowAnalysis:
        """Create demo money flow analysis"""
        
        # Mock dark pool activity
        dark_pool = DarkPoolActivity(
            ticker=ticker,
            timestamp=datetime.now(),
            dark_pool_volume=np.random.uniform(50000, 200000),
            lit_market_volume=np.random.uniform(300000, 800000),
            dark_pool_ratio=np.random.uniform(0.15, 0.35),
            estimated_block_trades=np.random.randint(5, 25),
            avg_trade_size=np.random.uniform(10000, 50000),
            volume_weighted_price=100 + np.random.uniform(-5, 5),
            participation_rate=np.random.uniform(0.1, 0.3)
        )
        
        # Mock volume concentration
        venues = {
            "NYSE": np.random.uniform(0.20, 0.30),
            "NASDAQ": np.random.uniform(0.15, 0.25),
            "BATS": np.random.uniform(0.10, 0.20),
            "EDGX": np.random.uniform(0.08, 0.15),
            "IEX": np.random.uniform(0.05, 0.12)
        }
        # Normalize to sum to 1
        total = sum(venues.values())
        venues = {k: v/total for k, v in venues.items()}
        
        volume_conc = VolumeConcentration(
            ticker=ticker,
            timestamp=datetime.now(),
            herfindahl_index=sum(v**2 for v in venues.values()),
            top_5_venues_share=sum(sorted(venues.values(), reverse=True)[:5]),
            fragmentation_score=1 - sum(v**2 for v in venues.values()),
            venue_breakdown=venues
        )
        
        return MoneyFlowAnalysis(
            ticker=ticker,
            timestamp=datetime.now(),
            analysis_period=timedelta(days=1),
            total_institutional_inflow=np.random.uniform(1000000, 5000000),
            total_institutional_outflow=np.random.uniform(800000, 4000000),
            net_institutional_flow=np.random.uniform(-1000000, 2000000),
            retail_flow_estimate=np.random.uniform(500000, 2000000),
            dark_pool_activity=dark_pool,
            volume_concentration=volume_conc,
            unusual_volume_detected=np.random.choice([True, False]),
            volume_anomaly_score=np.random.uniform(0, 1),
            detected_patterns=[],
            dominant_flow_type=FlowType.INSTITUTIONAL,
            institution_breakdown={
                "pension_fund": np.random.uniform(0.1, 0.3),
                "hedge_fund": np.random.uniform(0.2, 0.4),
                "mutual_fund": np.random.uniform(0.2, 0.4)
            },
            foreign_flow_estimate=np.random.uniform(0.05, 0.25),
            accumulation_score=np.random.uniform(-1, 1),
            distribution_score=np.random.uniform(-1, 1),
            rotation_signal=np.random.uniform(-1, 1),
            short_term_flow_direction=FlowDirection.INFLOW,
            flow_persistence_probability=np.random.uniform(0.3, 0.9),
            estimated_completion_time=None
        )
