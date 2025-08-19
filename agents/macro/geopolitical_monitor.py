"""
Geopolitical Risk Monitor

Monitors and analyzes geopolitical events and their market impacts:
- News sentiment analysis for geopolitical events
- Risk assessment and probability estimation
- Market impact modeling
- Event correlation and theme identification
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
import hashlib
from dataclasses import dataclass

from .models import (
    GeopoliticalEvent, GeopoliticalEventType, Region, ImpactSeverity,
    MarketImpact, MacroTheme
)


class GeopoliticalMonitor:
    """
    Monitor for geopolitical events and risk assessment
    
    Features:
    - News scraping and event detection
    - Risk probability assessment
    - Market impact analysis
    - Theme identification and tracking
    - Regional risk assessment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # News sources and APIs
        self.news_sources = [
            'reuters', 'bloomberg', 'ap', 'bbc', 'cnn',
            'ft', 'wsj', 'economist', 'stratfor'
        ]
        
        # Event severity keywords
        self.severity_keywords = {
            ImpactSeverity.CRITICAL: [
                'war', 'invasion', 'nuclear', 'terrorism', 'coup',
                'collapse', 'crisis', 'pandemic', 'catastrophe'
            ],
            ImpactSeverity.HIGH: [
                'conflict', 'sanction', 'embargo', 'strike', 'protest',
                'election', 'default', 'recession', 'inflation surge'
            ],
            ImpactSeverity.MEDIUM: [
                'tension', 'dispute', 'negotiation', 'policy change',
                'regulation', 'trade war', 'cyber attack'
            ],
            ImpactSeverity.LOW: [
                'meeting', 'statement', 'announcement', 'visit',
                'agreement', 'cooperation'
            ]
        }
        
        # Market impact keywords
        self.market_impact_keywords = {
            MarketImpact.VERY_BULLISH: ['peace', 'resolution', 'agreement', 'cooperation'],
            MarketImpact.BULLISH: ['stability', 'calm', 'positive', 'optimism'],
            MarketImpact.NEUTRAL: ['unchanged', 'stable', 'monitoring'],
            MarketImpact.BEARISH: ['tension', 'uncertainty', 'concern', 'risk'],
            MarketImpact.VERY_BEARISH: ['war', 'crisis', 'collapse', 'catastrophe']
        }
        
        # Active events tracking
        self.active_events: Dict[str, GeopoliticalEvent] = {}
        self.event_themes: Dict[str, MacroTheme] = {}
    
    async def scan_geopolitical_events(self, lookback_hours: int = 24) -> List[GeopoliticalEvent]:
        """
        Scan for new geopolitical events in the specified time window
        
        Args:
            lookback_hours: Hours to look back for events
            
        Returns:
            List of detected geopolitical events
        """
        events = []
        
        # In real implementation, would scrape news sources
        # For demo, generate realistic mock events
        mock_events = await self._generate_mock_events(lookback_hours)
        events.extend(mock_events)
        
        # Update active events
        for event in events:
            self.active_events[event.event_id] = event
        
        return events
    
    async def _generate_mock_events(self, lookback_hours: int) -> List[GeopoliticalEvent]:
        """Generate realistic mock geopolitical events"""
        events = []
        
        # Event templates
        event_templates = [
            {
                'type': GeopoliticalEventType.TRADE_WAR,
                'title': 'US-China Trade Tensions Escalate',
                'description': 'New tariffs announced affecting technology sector',
                'region': Region.GLOBAL,
                'countries': ['US', 'CN'],
                'severity': ImpactSeverity.HIGH,
                'sectors': ['technology', 'manufacturing'],
                'currencies': ['USD', 'CNY'],
                'commodities': ['steel', 'aluminum']
            },
            {
                'type': GeopoliticalEventType.CENTRAL_BANK_ACTION,
                'title': 'ECB Signals Potential Rate Cuts',
                'description': 'European Central Bank hints at dovish policy shift',
                'region': Region.EUROPE,
                'countries': ['EU'],
                'severity': ImpactSeverity.MEDIUM,
                'sectors': ['banking', 'real_estate'],
                'currencies': ['EUR'],
                'commodities': ['gold']
            },
            {
                'type': GeopoliticalEventType.ELECTIONS,
                'title': 'Election Uncertainty in Key Emerging Market',
                'description': 'Polling suggests tight race with policy implications',
                'region': Region.LATIN_AMERICA,
                'countries': ['BR'],
                'severity': ImpactSeverity.MEDIUM,
                'sectors': ['emerging_markets'],
                'currencies': ['BRL'],
                'commodities': ['oil', 'copper']
            },
            {
                'type': GeopoliticalEventType.SANCTIONS,
                'title': 'New Sanctions Target Energy Sector',
                'description': 'Additional sanctions imposed on oil exports',
                'region': Region.MIDDLE_EAST,
                'countries': ['RU', 'IR'],
                'severity': ImpactSeverity.HIGH,
                'sectors': ['energy', 'oil_gas'],
                'currencies': ['RUB'],
                'commodities': ['oil', 'natural_gas']
            },
            {
                'type': GeopoliticalEventType.CYBER_ATTACK,
                'title': 'Major Infrastructure Cyber Attack',
                'description': 'Critical infrastructure targeted by state actors',
                'region': Region.NORTH_AMERICA,
                'countries': ['US'],
                'severity': ImpactSeverity.HIGH,
                'sectors': ['technology', 'utilities'],
                'currencies': ['USD'],
                'commodities': []
            }
        ]
        
        # Generate 1-3 events for the lookback period
        num_events = np.random.randint(1, 4)
        selected_templates = np.random.choice(event_templates, size=min(num_events, len(event_templates)), replace=False)
        
        for i, template in enumerate(selected_templates):
            event_time = datetime.now() - timedelta(hours=np.random.randint(1, lookback_hours))
            
            # Generate unique event ID
            event_id = hashlib.md5(f"{template['title']}{event_time}".encode()).hexdigest()[:12]
            
            # Assess probability and market impacts
            probability = self._assess_event_probability(template)
            equity_impact, currency_impact, commodity_impact, bond_impact = self._assess_market_impacts(template)
            
            event = GeopoliticalEvent(
                event_id=event_id,
                event_type=template['type'],
                title=template['title'],
                description=template['description'],
                region=template['region'],
                countries_involved=template['countries'],
                start_date=event_time,
                estimated_duration=self._estimate_duration(template['type']),
                severity=template['severity'],
                probability=probability,
                equity_impact=equity_impact,
                currency_impact=currency_impact,
                commodity_impact=commodity_impact,
                bond_impact=bond_impact,
                affected_sectors=template['sectors'],
                affected_currencies=template['currencies'],
                affected_commodities=template['commodities'],
                confidence=np.random.uniform(0.6, 0.9),
                sources=['reuters', 'bloomberg'],
                last_updated=datetime.now()
            )
            
            events.append(event)
        
        return events
    
    def _assess_event_probability(self, template: Dict[str, Any]) -> float:
        """Assess probability of event occurrence/escalation"""
        base_probabilities = {
            GeopoliticalEventType.TRADE_WAR: 0.3,
            GeopoliticalEventType.CENTRAL_BANK_ACTION: 0.7,
            GeopoliticalEventType.ELECTIONS: 0.9,  # If announced, very likely
            GeopoliticalEventType.SANCTIONS: 0.6,
            GeopoliticalEventType.CYBER_ATTACK: 0.4,
            GeopoliticalEventType.MILITARY_CONFLICT: 0.2,
            GeopoliticalEventType.POLICY_CHANGE: 0.5
        }
        
        base_prob = base_probabilities.get(template['type'], 0.5)
        
        # Adjust based on severity
        severity_adjustments = {
            ImpactSeverity.CRITICAL: 0.1,
            ImpactSeverity.HIGH: 0.0,
            ImpactSeverity.MEDIUM: -0.1,
            ImpactSeverity.LOW: -0.2
        }
        
        adjustment = severity_adjustments.get(template['severity'], 0.0)
        final_prob = base_prob + adjustment + np.random.normal(0, 0.1)
        
        return max(0.1, min(0.9, final_prob))
    
    def _assess_market_impacts(self, template: Dict[str, Any]) -> Tuple[MarketImpact, MarketImpact, MarketImpact, MarketImpact]:
        """Assess market impacts for different asset classes"""
        
        # Default impacts based on event type
        impact_mapping = {
            GeopoliticalEventType.TRADE_WAR: {
                'equity': MarketImpact.BEARISH,
                'currency': MarketImpact.NEUTRAL,  # Depends on country
                'commodity': MarketImpact.BULLISH,  # Supply disruptions
                'bond': MarketImpact.BULLISH       # Flight to safety
            },
            GeopoliticalEventType.CENTRAL_BANK_ACTION: {
                'equity': MarketImpact.NEUTRAL,
                'currency': MarketImpact.BULLISH,
                'commodity': MarketImpact.NEUTRAL,
                'bond': MarketImpact.BEARISH
            },
            GeopoliticalEventType.ELECTIONS: {
                'equity': MarketImpact.BEARISH,    # Uncertainty
                'currency': MarketImpact.BEARISH,  # Uncertainty
                'commodity': MarketImpact.NEUTRAL,
                'bond': MarketImpact.BULLISH      # Flight to safety
            },
            GeopoliticalEventType.SANCTIONS: {
                'equity': MarketImpact.BEARISH,
                'currency': MarketImpact.BEARISH,
                'commodity': MarketImpact.VERY_BULLISH,  # Supply disruption
                'bond': MarketImpact.BULLISH
            },
            GeopoliticalEventType.MILITARY_CONFLICT: {
                'equity': MarketImpact.VERY_BEARISH,
                'currency': MarketImpact.VERY_BEARISH,
                'commodity': MarketImpact.VERY_BULLISH,
                'bond': MarketImpact.VERY_BULLISH
            }
        }
        
        default_impacts = impact_mapping.get(template['type'], {
            'equity': MarketImpact.NEUTRAL,
            'currency': MarketImpact.NEUTRAL,
            'commodity': MarketImpact.NEUTRAL,
            'bond': MarketImpact.NEUTRAL
        })
        
        # Add some randomness
        def add_noise(impact: MarketImpact) -> MarketImpact:
            impacts = list(MarketImpact)
            current_index = impacts.index(impact)
            
            # 70% chance to stay same, 30% chance to move +/- 1
            if np.random.random() < 0.7:
                return impact
            else:
                change = np.random.choice([-1, 1])
                new_index = max(0, min(len(impacts) - 1, current_index + change))
                return impacts[new_index]
        
        return (
            add_noise(default_impacts['equity']),
            add_noise(default_impacts['currency']),
            add_noise(default_impacts['commodity']),
            add_noise(default_impacts['bond'])
        )
    
    def _estimate_duration(self, event_type: GeopoliticalEventType) -> Optional[timedelta]:
        """Estimate event duration based on type"""
        duration_estimates = {
            GeopoliticalEventType.TRADE_WAR: timedelta(days=365),
            GeopoliticalEventType.CENTRAL_BANK_ACTION: timedelta(days=90),
            GeopoliticalEventType.ELECTIONS: timedelta(days=30),
            GeopoliticalEventType.SANCTIONS: timedelta(days=180),
            GeopoliticalEventType.CYBER_ATTACK: timedelta(days=7),
            GeopoliticalEventType.MILITARY_CONFLICT: timedelta(days=120),
            GeopoliticalEventType.POLICY_CHANGE: timedelta(days=90)
        }
        
        base_duration = duration_estimates.get(event_type, timedelta(days=30))
        
        # Add random variation
        variation = np.random.uniform(0.5, 2.0)
        return base_duration * variation
    
    async def identify_emerging_risks(self, confidence_threshold: float = 0.6) -> List[GeopoliticalEvent]:
        """
        Identify emerging geopolitical risks that could impact markets
        
        Args:
            confidence_threshold: Minimum confidence for risk identification
            
        Returns:
            List of emerging risks with high impact probability
        """
        emerging_risks = []
        
        # Scan recent events for escalation patterns
        recent_events = [
            event for event in self.active_events.values()
            if (datetime.now() - event.start_date).days <= 7
        ]
        
        # Look for escalation patterns
        for event in recent_events:
            if event.confidence >= confidence_threshold:
                # Check if event could escalate
                escalation_probability = self._assess_escalation_risk(event)
                
                if escalation_probability > 0.4:  # 40% escalation risk
                    # Create escalated version of event
                    escalated_event = self._create_escalated_event(event, escalation_probability)
                    emerging_risks.append(escalated_event)
        
        # Generate new potential risks
        potential_risks = await self._generate_potential_risks()
        emerging_risks.extend(potential_risks)
        
        return emerging_risks
    
    def _assess_escalation_risk(self, event: GeopoliticalEvent) -> float:
        """Assess risk of event escalation"""
        base_escalation_rates = {
            GeopoliticalEventType.TRADE_WAR: 0.3,
            GeopoliticalEventType.MILITARY_CONFLICT: 0.6,
            GeopoliticalEventType.SANCTIONS: 0.4,
            GeopoliticalEventType.CYBER_ATTACK: 0.5,
            GeopoliticalEventType.ELECTIONS: 0.2,
            GeopoliticalEventType.POLICY_CHANGE: 0.3
        }
        
        base_rate = base_escalation_rates.get(event.event_type, 0.2)
        
        # Adjust for severity and time
        severity_multiplier = {
            ImpactSeverity.CRITICAL: 1.5,
            ImpactSeverity.HIGH: 1.2,
            ImpactSeverity.MEDIUM: 1.0,
            ImpactSeverity.LOW: 0.8
        }
        
        multiplier = severity_multiplier.get(event.severity, 1.0)
        
        # Time decay - events get less likely to escalate over time
        days_since_start = (datetime.now() - event.start_date).days
        time_decay = max(0.5, 1.0 - (days_since_start / 30))  # Decay over 30 days
        
        return min(0.8, base_rate * multiplier * time_decay)
    
    def _create_escalated_event(self, base_event: GeopoliticalEvent, 
                              escalation_probability: float) -> GeopoliticalEvent:
        """Create an escalated version of an existing event"""
        escalated_id = f"{base_event.event_id}_escalated"
        
        # Increase severity
        severity_upgrade = {
            ImpactSeverity.LOW: ImpactSeverity.MEDIUM,
            ImpactSeverity.MEDIUM: ImpactSeverity.HIGH,
            ImpactSeverity.HIGH: ImpactSeverity.CRITICAL,
            ImpactSeverity.CRITICAL: ImpactSeverity.CRITICAL
        }
        
        new_severity = severity_upgrade.get(base_event.severity, base_event.severity)
        
        # Worsen market impacts
        impact_downgrade = {
            MarketImpact.VERY_BULLISH: MarketImpact.BULLISH,
            MarketImpact.BULLISH: MarketImpact.NEUTRAL,
            MarketImpact.NEUTRAL: MarketImpact.BEARISH,
            MarketImpact.BEARISH: MarketImpact.VERY_BEARISH,
            MarketImpact.VERY_BEARISH: MarketImpact.VERY_BEARISH
        }
        
        return GeopoliticalEvent(
            event_id=escalated_id,
            event_type=base_event.event_type,
            title=f"Escalated: {base_event.title}",
            description=f"Escalation of {base_event.description}",
            region=base_event.region,
            countries_involved=base_event.countries_involved,
            start_date=datetime.now(),
            estimated_duration=base_event.estimated_duration,
            severity=new_severity,
            probability=escalation_probability,
            equity_impact=impact_downgrade.get(base_event.equity_impact, base_event.equity_impact),
            currency_impact=impact_downgrade.get(base_event.currency_impact, base_event.currency_impact),
            commodity_impact=base_event.commodity_impact,  # Commodities often benefit from risk
            bond_impact=MarketImpact.BULLISH,  # Flight to safety
            affected_sectors=base_event.affected_sectors,
            affected_currencies=base_event.affected_currencies,
            affected_commodities=base_event.affected_commodities,
            confidence=base_event.confidence * 0.8,  # Lower confidence for predictions
            sources=base_event.sources,
            last_updated=datetime.now()
        )
    
    async def _generate_potential_risks(self) -> List[GeopoliticalEvent]:
        """Generate potential new risks based on current global situation"""
        potential_risks = []
        
        # Template for potential risks
        risk_templates = [
            {
                'type': GeopoliticalEventType.COMMODITY_DISRUPTION,
                'title': 'Potential Oil Supply Disruption',
                'description': 'Rising tensions could affect major shipping routes',
                'region': Region.MIDDLE_EAST,
                'probability': 0.25
            },
            {
                'type': GeopoliticalEventType.CYBER_ATTACK,
                'title': 'Financial Infrastructure Cyber Risk',
                'description': 'Increased cyber threat against banking systems',
                'region': Region.GLOBAL,
                'probability': 0.35
            }
        ]
        
        for template in risk_templates:
            if np.random.random() < 0.3:  # 30% chance to include each risk
                event_id = hashlib.md5(f"{template['title']}_potential".encode()).hexdigest()[:12]
                
                risk_event = GeopoliticalEvent(
                    event_id=event_id,
                    event_type=template['type'],
                    title=template['title'],
                    description=template['description'],
                    region=template['region'],
                    countries_involved=['Multiple'],
                    start_date=datetime.now() + timedelta(days=np.random.randint(1, 30)),
                    estimated_duration=timedelta(days=30),
                    severity=ImpactSeverity.MEDIUM,
                    probability=template['probability'],
                    equity_impact=MarketImpact.BEARISH,
                    currency_impact=MarketImpact.NEUTRAL,
                    commodity_impact=MarketImpact.BULLISH,
                    bond_impact=MarketImpact.BULLISH,
                    affected_sectors=['energy', 'technology'],
                    affected_currencies=['USD'],
                    affected_commodities=['oil'],
                    confidence=0.4,  # Lower confidence for potential risks
                    sources=['analysis'],
                    last_updated=datetime.now()
                )
                
                potential_risks.append(risk_event)
        
        return potential_risks
    
    async def identify_macro_themes(self, events: List[GeopoliticalEvent]) -> List[MacroTheme]:
        """
        Identify overarching macro themes from geopolitical events
        
        Args:
            events: List of geopolitical events to analyze
            
        Returns:
            List of identified macro themes
        """
        themes = []
        
        # Group events by similarity
        event_clusters = self._cluster_events_by_theme(events)
        
        for cluster_name, cluster_events in event_clusters.items():
            if len(cluster_events) >= 2:  # At least 2 events to form a theme
                theme = self._create_theme_from_cluster(cluster_name, cluster_events)
                themes.append(theme)
        
        return themes
    
    def _cluster_events_by_theme(self, events: List[GeopoliticalEvent]) -> Dict[str, List[GeopoliticalEvent]]:
        """Cluster events into thematic groups"""
        clusters = {
            'trade_tensions': [],
            'monetary_policy': [],
            'geopolitical_conflict': [],
            'cyber_security': [],
            'energy_security': [],
            'political_instability': []
        }
        
        for event in events:
            # Simple rule-based clustering
            if event.event_type in [GeopoliticalEventType.TRADE_WAR, GeopoliticalEventType.SANCTIONS]:
                clusters['trade_tensions'].append(event)
            elif event.event_type == GeopoliticalEventType.CENTRAL_BANK_ACTION:
                clusters['monetary_policy'].append(event)
            elif event.event_type in [GeopoliticalEventType.MILITARY_CONFLICT, GeopoliticalEventType.CYBER_ATTACK]:
                clusters['geopolitical_conflict'].append(event)
            elif event.event_type == GeopoliticalEventType.CYBER_ATTACK:
                clusters['cyber_security'].append(event)
            elif 'energy' in event.affected_sectors or 'oil' in event.affected_commodities:
                clusters['energy_security'].append(event)
            elif event.event_type in [GeopoliticalEventType.ELECTIONS, GeopoliticalEventType.POLICY_CHANGE]:
                clusters['political_instability'].append(event)
        
        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}
    
    def _create_theme_from_cluster(self, cluster_name: str, 
                                 events: List[GeopoliticalEvent]) -> MacroTheme:
        """Create a macro theme from a cluster of events"""
        theme_id = hashlib.md5(f"{cluster_name}_{datetime.now()}".encode()).hexdigest()[:12]
        
        # Calculate theme characteristics
        avg_severity = np.mean([
            list(ImpactSeverity).index(event.severity) for event in events
        ])
        
        strength = min(1.0, len(events) / 5.0)  # Stronger with more events
        momentum = self._calculate_theme_momentum(events)
        
        # Determine market relevance
        equity_relevance = np.mean([
            self._impact_to_score(event.equity_impact) for event in events
        ])
        
        theme_descriptions = {
            'trade_tensions': 'Escalating global trade disputes affecting international commerce',
            'monetary_policy': 'Central bank policy divergence creating currency volatility',
            'geopolitical_conflict': 'Rising geopolitical tensions threatening global stability',
            'cyber_security': 'Increasing cyber threats to critical infrastructure',
            'energy_security': 'Energy supply disruptions affecting global markets',
            'political_instability': 'Political uncertainty in key regions'
        }
        
        return MacroTheme(
            theme_id=theme_id,
            name=cluster_name.replace('_', ' ').title(),
            description=theme_descriptions.get(cluster_name, f"Theme related to {cluster_name}"),
            start_date=min(event.start_date for event in events),
            current_phase='developing',
            strength=strength,
            momentum=momentum,
            equity_relevance=abs(equity_relevance),
            fx_relevance=0.8,  # Geopolitical events usually affect FX
            commodity_relevance=0.6,
            bond_relevance=0.7,
            related_events=[event.event_id for event in events],
            key_indicators=['vix', 'safe_haven_flows'],
            expected_duration=timedelta(days=180),
            key_catalysts=[event.title for event in events[:3]]
        )
    
    def _calculate_theme_momentum(self, events: List[GeopoliticalEvent]) -> float:
        """Calculate momentum of a theme based on recent events"""
        now = datetime.now()
        
        # Weight events by recency
        weighted_impact = 0
        total_weight = 0
        
        for event in events:
            days_ago = (now - event.start_date).days
            weight = max(0.1, 1.0 - (days_ago / 30))  # Decay over 30 days
            
            impact_score = list(ImpactSeverity).index(event.severity)
            weighted_impact += impact_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize to -1 to 1 range
        avg_impact = weighted_impact / total_weight
        max_impact = len(ImpactSeverity) - 1
        
        return (avg_impact / max_impact) * 2 - 1
    
    def _impact_to_score(self, impact: MarketImpact) -> float:
        """Convert market impact to numerical score"""
        impact_scores = {
            MarketImpact.VERY_BEARISH: -1.0,
            MarketImpact.BEARISH: -0.5,
            MarketImpact.NEUTRAL: 0.0,
            MarketImpact.BULLISH: 0.5,
            MarketImpact.VERY_BULLISH: 1.0
        }
        return impact_scores.get(impact, 0.0)
    
    def calculate_regional_risk_scores(self, events: List[GeopoliticalEvent]) -> Dict[str, float]:
        """Calculate risk scores for different regions"""
        regional_risks = {region.value: 0.0 for region in Region}
        regional_counts = {region.value: 0 for region in Region}
        
        for event in events:
            risk_score = list(ImpactSeverity).index(event.severity) * event.probability
            regional_risks[event.region.value] += risk_score
            regional_counts[event.region.value] += 1
        
        # Average and normalize
        for region in regional_risks:
            if regional_counts[region] > 0:
                regional_risks[region] = regional_risks[region] / regional_counts[region]
                regional_risks[region] = regional_risks[region] / 3.0  # Normalize to 0-1
        
        return regional_risks
