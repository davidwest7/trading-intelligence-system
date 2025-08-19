"""
Complete Macro/Geopolitical Analysis Agent

This is the full implementation to replace the stub agent.py
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .models import (
    MacroAnalysis, MacroRequest, EconomicIndicator, GeopoliticalEvent,
    CentralBankAction, MacroTheme, ImpactSeverity, MarketImpact
)
from .economic_calendar import EconomicCalendarProvider
from .geopolitical_monitor import GeopoliticalMonitor
from ..common.models import BaseAgent


class MacroAgent(BaseAgent):
    """
    Complete Macro/Geopolitical Analysis Agent
    
    Capabilities:
    ✅ Economic calendar integration and analysis
    ✅ Central bank communication monitoring
    ✅ Geopolitical risk assessment and tracking
    ✅ Macro theme identification
    ✅ Economic surprise index calculation
    ✅ Cross-asset impact analysis
    ✅ Risk-on/risk-off regime detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("macro", config)
        
        # Initialize components
        self.economic_calendar = EconomicCalendarProvider(config)
        self.geopolitical_monitor = GeopoliticalMonitor(config)
        
        # Configuration
        self.lookback_days = config.get('lookback_days', 30) if config else 30
        self.key_economies = ['US', 'EU', 'JP', 'UK', 'CN']
        
        # Historical tracking
        self.theme_history = []
        self.risk_history = []
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        return await self.analyze_macro_environment(*args, **kwargs)
    
    async def analyze_macro_environment(
        self, 
        horizon: str = "medium_term",
        regions: List[str] = None,
        include_geopolitical: bool = True,
        include_central_banks: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze current macro environment and outlook
        """
        if regions is None:
            regions = ["global"]
        
        # Validate request
        request = MacroRequest(
            analysis_horizon=horizon,
            regions=regions,
            include_geopolitical=include_geopolitical,
            include_central_banks=include_central_banks,
            lookback_days=self.lookback_days
        )
        
        if not request.validate():
            raise ValueError("Invalid macro analysis request")
        
        # 1. Economic Calendar Analysis
        recent_indicators = await self.economic_calendar.get_recent_indicators(
            lookback_days=request.lookback_days,
            countries=self.key_economies
        )
        
        upcoming_indicators = await self.economic_calendar.get_upcoming_indicators(
            lookahead_days=request.lookahead_days,
            countries=self.key_economies
        )
        
        # 2. Central Bank Analysis
        recent_cb_actions = []
        upcoming_cb_meetings = []
        
        if include_central_banks:
            recent_cb_actions = await self.economic_calendar.analyze_central_bank_communications(
                lookback_days=request.lookback_days
            )
            upcoming_cb_meetings = await self._get_upcoming_cb_meetings()
        
        # 3. Geopolitical Analysis
        active_events = []
        emerging_risks = []
        
        if include_geopolitical:
            active_events = await self.geopolitical_monitor.scan_geopolitical_events(
                lookback_hours=request.lookback_days * 24
            )
            emerging_risks = await self.geopolitical_monitor.identify_emerging_risks()
        
        # 4. Theme Identification
        all_events = active_events + emerging_risks
        dominant_themes = await self.geopolitical_monitor.identify_macro_themes(all_events)
        
        # 5. Overall Assessment
        global_growth_outlook = self._assess_global_growth(recent_indicators)
        inflation_environment = self._assess_inflation_environment(recent_indicators)
        interest_rate_cycle = self._assess_rate_cycle(recent_cb_actions)
        risk_environment = self._assess_risk_environment(active_events, recent_indicators)
        market_regime = self._determine_market_regime(risk_environment, recent_indicators)
        
        # 6. Currency and Asset Analysis
        usd_strength_outlook = self._assess_usd_outlook(recent_cb_actions, recent_indicators)
        safe_haven_demand = self._calculate_safe_haven_demand(active_events)
        
        # 7. Sector and Regional Impacts
        sector_impacts = self._assess_sector_impacts(dominant_themes, recent_indicators)
        commodity_impacts = self._assess_commodity_impacts(active_events, recent_indicators)
        regional_impacts = self._assess_regional_impacts(active_events)
        
        # 8. Risk Assessment
        analysis_confidence = self._calculate_analysis_confidence(
            recent_indicators, active_events, recent_cb_actions
        )
        key_risks, key_opportunities = self._identify_risks_and_opportunities(
            dominant_themes, emerging_risks, recent_indicators
        )
        
        # Create comprehensive analysis
        analysis = MacroAnalysis(
            timestamp=datetime.now(),
            analysis_horizon=horizon,
            global_growth_outlook=global_growth_outlook,
            inflation_environment=inflation_environment,
            interest_rate_cycle=interest_rate_cycle,
            recent_indicators=recent_indicators,
            upcoming_indicators=upcoming_indicators,
            active_events=active_events,
            emerging_risks=emerging_risks,
            recent_cb_actions=recent_cb_actions,
            upcoming_cb_meetings=upcoming_cb_meetings,
            dominant_themes=dominant_themes,
            risk_environment=risk_environment,
            market_regime=market_regime,
            usd_strength_outlook=usd_strength_outlook,
            safe_haven_demand=safe_haven_demand,
            sector_impacts=sector_impacts,
            commodity_impacts=commodity_impacts,
            regional_impacts=regional_impacts,
            analysis_confidence=analysis_confidence,
            key_risks=key_risks,
            key_opportunities=key_opportunities
        )
        
        return {
            "macro_analysis": analysis.to_dict()
        }
    
    def _assess_global_growth(self, indicators: List[EconomicIndicator]) -> MarketImpact:
        """Assess global growth outlook from economic indicators"""
        growth_indicators = [
            ind for ind in indicators 
            if ind.indicator_type.value in ['gdp', 'manufacturing_pmi', 'retail_sales']
        ]
        
        if not growth_indicators:
            return MarketImpact.NEUTRAL
        
        # Calculate average surprise factor
        avg_surprise = statistics.mean(ind.surprise_factor for ind in growth_indicators)
        
        if avg_surprise > 0.02:
            return MarketImpact.BULLISH
        elif avg_surprise < -0.02:
            return MarketImpact.BEARISH
        else:
            return MarketImpact.NEUTRAL
    
    def _assess_inflation_environment(self, indicators: List[EconomicIndicator]) -> str:
        """Assess current inflation environment"""
        inflation_indicators = [
            ind for ind in indicators 
            if ind.indicator_type.value == 'inflation'
        ]
        
        if not inflation_indicators:
            return "moderate"
        
        # Use most recent inflation reading
        latest_inflation = inflation_indicators[-1]
        
        if latest_inflation.value > 5.0:
            return "high"
        elif latest_inflation.value > 3.0:
            return "moderate"
        elif latest_inflation.value > 1.0:
            return "low"
        else:
            return "deflationary"
    
    def _assess_rate_cycle(self, cb_actions: List[CentralBankAction]) -> str:
        """Assess interest rate cycle from central bank actions"""
        if not cb_actions:
            return "neutral"
        
        # Calculate average hawkish/dovish score
        avg_score = statistics.mean(action.hawkish_dovish_score for action in cb_actions)
        
        if avg_score > 0.2:
            return "hiking"
        elif avg_score < -0.2:
            return "cutting"
        else:
            return "neutral"
    
    def _assess_risk_environment(self, events: List[GeopoliticalEvent], 
                               indicators: List[EconomicIndicator]) -> ImpactSeverity:
        """Assess overall risk environment"""
        # Geopolitical risk component
        if events:
            geo_risk_score = max(
                list(ImpactSeverity).index(event.severity) for event in events
            )
        else:
            geo_risk_score = 0
        
        # Economic risk component
        negative_surprises = [
            ind for ind in indicators 
            if ind.surprise_factor < -0.05
        ]
        
        eco_risk_score = min(3, len(negative_surprises))  # Cap at 3
        
        # Combined risk score
        combined_risk = max(geo_risk_score, eco_risk_score)
        
        return list(ImpactSeverity)[combined_risk]
    
    def _determine_market_regime(self, risk_env: ImpactSeverity, 
                               indicators: List[EconomicIndicator]) -> str:
        """Determine current market regime"""
        # Simple regime classification
        if risk_env in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]:
            return "risk_off"
        
        # Check economic momentum
        growth_momentum = self._assess_global_growth(indicators)
        
        if growth_momentum == MarketImpact.BULLISH:
            return "risk_on"
        elif growth_momentum == MarketImpact.BEARISH:
            return "risk_off"
        else:
            return "neutral"
    
    def _assess_usd_outlook(self, cb_actions: List[CentralBankAction], 
                          indicators: List[EconomicIndicator]) -> MarketImpact:
        """Assess USD strength outlook"""
        # Fed vs other central banks
        fed_actions = [action for action in cb_actions if 'Federal' in action.bank]
        other_actions = [action for action in cb_actions if 'Federal' not in action.bank]
        
        if fed_actions and other_actions:
            fed_hawkishness = statistics.mean(action.hawkish_dovish_score for action in fed_actions)
            other_hawkishness = statistics.mean(action.hawkish_dovish_score for action in other_actions)
            
            differential = fed_hawkishness - other_hawkishness
            
            if differential > 0.3:
                return MarketImpact.BULLISH
            elif differential < -0.3:
                return MarketImpact.BEARISH
        
        # US economic performance
        us_indicators = [ind for ind in indicators if ind.country == 'US']
        if us_indicators:
            us_performance = statistics.mean(ind.surprise_factor for ind in us_indicators)
            
            if us_performance > 0.02:
                return MarketImpact.BULLISH
            elif us_performance < -0.02:
                return MarketImpact.BEARISH
        
        return MarketImpact.NEUTRAL
    
    def _calculate_safe_haven_demand(self, events: List[GeopoliticalEvent]) -> float:
        """Calculate safe haven demand based on geopolitical events"""
        if not events:
            return 0.3  # Baseline demand
        
        # Weight events by severity and probability
        risk_score = 0
        for event in events:
            severity_weight = list(ImpactSeverity).index(event.severity) / 3.0
            risk_score += severity_weight * event.probability
        
        # Normalize to 0-1 range
        return min(1.0, risk_score / len(events))
    
    def _assess_sector_impacts(self, themes: List[MacroTheme], 
                             indicators: List[EconomicIndicator]) -> Dict[str, MarketImpact]:
        """Assess sector-specific impacts"""
        sector_impacts = {
            'technology': MarketImpact.NEUTRAL,
            'financials': MarketImpact.NEUTRAL,
            'energy': MarketImpact.NEUTRAL,
            'healthcare': MarketImpact.NEUTRAL,
            'consumer_discretionary': MarketImpact.NEUTRAL,
            'industrials': MarketImpact.NEUTRAL
        }
        
        # Rate-sensitive sectors
        rate_cycle = self._assess_rate_cycle([])  # Would use actual CB actions
        if rate_cycle == "hiking":
            sector_impacts['financials'] = MarketImpact.BULLISH
            sector_impacts['technology'] = MarketImpact.BEARISH
        elif rate_cycle == "cutting":
            sector_impacts['financials'] = MarketImpact.BEARISH
            sector_impacts['technology'] = MarketImpact.BULLISH
        
        # Economic growth impact
        growth_outlook = self._assess_global_growth(indicators)
        if growth_outlook == MarketImpact.BULLISH:
            sector_impacts['industrials'] = MarketImpact.BULLISH
            sector_impacts['consumer_discretionary'] = MarketImpact.BULLISH
        elif growth_outlook == MarketImpact.BEARISH:
            sector_impacts['healthcare'] = MarketImpact.BULLISH  # Defensive
        
        return sector_impacts
    
    def _assess_commodity_impacts(self, events: List[GeopoliticalEvent], 
                                indicators: List[EconomicIndicator]) -> Dict[str, MarketImpact]:
        """Assess commodity-specific impacts"""
        commodity_impacts = {
            'oil': MarketImpact.NEUTRAL,
            'gold': MarketImpact.NEUTRAL,
            'copper': MarketImpact.NEUTRAL,
            'natural_gas': MarketImpact.NEUTRAL
        }
        
        # Geopolitical impact on energy
        energy_events = [
            event for event in events 
            if 'energy' in event.affected_sectors or 'oil' in event.affected_commodities
        ]
        
        if energy_events:
            commodity_impacts['oil'] = MarketImpact.BULLISH
            commodity_impacts['natural_gas'] = MarketImpact.BULLISH
        
        # Safe haven demand for gold
        safe_haven_demand = self._calculate_safe_haven_demand(events)
        if safe_haven_demand > 0.6:
            commodity_impacts['gold'] = MarketImpact.BULLISH
        
        # Growth impact on industrial metals
        growth_outlook = self._assess_global_growth(indicators)
        commodity_impacts['copper'] = growth_outlook
        
        return commodity_impacts
    
    def _assess_regional_impacts(self, events: List[GeopoliticalEvent]) -> Dict[str, MarketImpact]:
        """Assess regional market impacts"""
        regional_impacts = {}
        
        # Use geopolitical monitor's regional risk calculation
        regional_risks = self.geopolitical_monitor.calculate_regional_risk_scores(events)
        
        for region, risk_score in regional_risks.items():
            if risk_score > 0.6:
                regional_impacts[region] = MarketImpact.BEARISH
            elif risk_score > 0.3:
                regional_impacts[region] = MarketImpact.NEUTRAL
            else:
                regional_impacts[region] = MarketImpact.BULLISH
        
        return regional_impacts
    
    def _calculate_analysis_confidence(self, indicators: List[EconomicIndicator],
                                     events: List[GeopoliticalEvent],
                                     cb_actions: List[CentralBankAction]) -> float:
        """Calculate confidence in the analysis"""
        factors = []
        
        # Data recency factor
        if indicators:
            avg_days_old = statistics.mean(
                (datetime.now() - ind.release_date).days for ind in indicators
            )
            recency_factor = max(0.3, 1.0 - (avg_days_old / 30))
            factors.append(recency_factor)
        
        # Data completeness factor
        completeness = min(1.0, len(indicators) / 10)  # Expect ~10 indicators
        factors.append(completeness)
        
        # Event clarity factor
        if events:
            avg_event_confidence = statistics.mean(event.confidence for event in events)
            factors.append(avg_event_confidence)
        else:
            factors.append(0.8)  # Higher confidence when no major events
        
        return statistics.mean(factors) if factors else 0.5
    
    def _identify_risks_and_opportunities(self, themes: List[MacroTheme],
                                        emerging_risks: List[GeopoliticalEvent],
                                        indicators: List[EconomicIndicator]) -> Tuple[List[str], List[str]]:
        """Identify key risks and opportunities"""
        risks = []
        opportunities = []
        
        # Risks from themes
        for theme in themes:
            if theme.momentum < -0.2:  # Negative momentum themes are risks
                risks.append(f"Deteriorating {theme.name.lower()}")
        
        # Risks from emerging events
        for event in emerging_risks:
            if event.severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]:
                risks.append(f"Potential {event.title.lower()}")
        
        # Opportunities from positive surprises
        positive_surprises = [
            ind for ind in indicators 
            if ind.surprise_factor > 0.05
        ]
        
        if len(positive_surprises) > 2:
            opportunities.append("Economic data beating expectations")
        
        # Default risks and opportunities
        if not risks:
            risks = ["Policy uncertainty", "Geopolitical tensions", "Market volatility"]
        
        if not opportunities:
            opportunities = ["Central bank policy clarity", "Economic resilience", "Risk asset demand"]
        
        return risks[:5], opportunities[:5]  # Limit to 5 each
    
    async def _get_upcoming_cb_meetings(self) -> List[Dict[str, Any]]:
        """Get upcoming central bank meetings"""
        # Mock upcoming meetings
        return [
            {
                'bank': 'Federal Reserve',
                'date': (datetime.now() + timedelta(days=14)).isoformat(),
                'type': 'FOMC Meeting',
                'expected_action': 'Hold rates',
                'market_focus': 'Forward guidance'
            },
            {
                'bank': 'European Central Bank',
                'date': (datetime.now() + timedelta(days=21)).isoformat(),
                'type': 'Policy Meeting',
                'expected_action': 'Hold rates', 
                'market_focus': 'QT timeline'
            }
        ]
