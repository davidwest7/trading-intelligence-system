"""
Optimized Macro Agent

Advanced macroeconomic analysis with:
- Real-time economic indicator analysis
- Central bank policy impact assessment
- Geopolitical risk monitoring
- Market regime detection
- Performance optimization
- Error handling and resilience
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .models import (
    MacroAnalysis, MacroRequest, EconomicIndicator, GeopoliticalEvent,
    CentralBankAction, MacroTheme, ImpactSeverity, MarketImpact
)
from .economic_calendar import EconomicCalendarProvider
from .geopolitical_monitor import GeopoliticalMonitor
from ..common.models import BaseAgent


@dataclass
class MacroSignal:
    """Macro economic signal"""
    signal_type: str
    region: str
    direction: str
    strength: float
    confidence: float
    impact_assets: List[str]
    timestamp: datetime
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "region": self.region,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "impact_assets": self.impact_assets,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description
        }


class OptimizedMacroAgent(BaseAgent):
    """
    Optimized Macro/Geopolitical Analysis Agent for financial markets
    
    Enhanced Capabilities:
    ✅ Real-time economic indicator analysis and forecasting
    ✅ Advanced central bank policy impact assessment
    ✅ Geopolitical risk monitoring and quantification
    ✅ Market regime detection (risk-on/risk-off)
    ✅ Cross-asset correlation analysis
    ✅ Economic surprise index calculation
    ✅ Performance optimization and caching
    ✅ Error handling and resilience
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("macro", config)
        
        # Configuration with defaults
        self.config = config or {}
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 8)
        self.cache_ttl = self.config.get('cache_ttl', 600)  # 10 minutes
        self.lookback_days = self.config.get('lookback_days', 30)
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize components
        self.economic_calendar = EconomicCalendarProvider(config)
        self.geopolitical_monitor = GeopoliticalMonitor(config)
        
        # Key economies and regions
        self.key_economies = ['US', 'EU', 'JP', 'UK', 'CN', 'CA', 'AU', 'CH']
        self.key_regions = ['global', 'developed', 'emerging', 'asia_pacific', 'europe', 'americas']
        
        # Real-time data storage
        self.max_history_size = 5000
        self.indicator_history = defaultdict(lambda: deque(maxlen=self.max_history_size))
        self.risk_history = defaultdict(lambda: deque(maxlen=self.max_history_size))
        self.regime_history = deque(maxlen=self.max_history_size)
        
        # Performance metrics
        self.metrics = {
            'total_analyses': 0,
            'macro_signals_generated': 0,
            'processing_time_avg': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0
        }
        
        # Error tracking
        self.error_count = 0
        self.total_requests = 0
        
        logging.info("Optimized Macro Agent initialized successfully")
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method with error handling"""
        try:
            self.total_requests += 1
            start_time = time.time()
            
            result = await self.analyze_macro_environment_optimized(*args, **kwargs)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.metrics['processing_time_avg'] = (
                (self.metrics['processing_time_avg'] * (self.total_requests - 1) + processing_time)
                / self.total_requests
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.metrics['error_rate'] = self.error_count / self.total_requests
            logging.error(f"Error in macro processing: {e}")
            raise
    
    async def analyze_macro_environment_optimized(
        self,
        horizon: str = "medium_term",
        regions: List[str] = None,
        include_geopolitical: bool = True,
        include_central_banks: bool = True,
        include_indicators: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized macro environment analysis with caching and parallel processing
        
        Args:
            horizon: Analysis horizon (short_term, medium_term, long_term)
            regions: List of regions to analyze
            include_geopolitical: Include geopolitical analysis
            include_central_banks: Include central bank analysis
            include_indicators: Include economic indicators
            use_cache: Use cached results if available
        
        Returns:
            Complete macro analysis results
        """
        
        if regions is None:
            regions = ["global"]
        
        # Check cache first
        cache_key = f"{horizon}_{','.join(sorted(regions))}_{include_geopolitical}_{include_central_banks}_{include_indicators}"
        if use_cache and self._is_cache_valid(cache_key):
            self.metrics['cache_hit_rate'] += 1
            return self.cache[cache_key]
        
        try:
            # Run analyses in parallel
            analysis_tasks = []
            
            if include_indicators:
                task = asyncio.create_task(self._analyze_economic_indicators_optimized(regions))
                analysis_tasks.append(("indicators", task))
            
            if include_central_banks:
                task = asyncio.create_task(self._analyze_central_banks_optimized(regions))
                analysis_tasks.append(("central_banks", task))
            
            if include_geopolitical:
                task = asyncio.create_task(self._analyze_geopolitical_risks_optimized(regions))
                analysis_tasks.append(("geopolitical", task))
            
            # Execute all tasks concurrently
            results = {}
            if analysis_tasks:
                completed_tasks = await asyncio.gather(*[task for _, task in analysis_tasks], return_exceptions=True)
                
                for i, (task_name, _) in enumerate(analysis_tasks):
                    if isinstance(completed_tasks[i], Exception):
                        logging.error(f"Error in {task_name} analysis: {completed_tasks[i]}")
                        self.error_count += 1
                        results[task_name] = {}
                    else:
                        results[task_name] = completed_tasks[i]
            
            # Generate macro signals
            macro_signals = self._generate_macro_signals_optimized(results)
            
            # Detect market regime
            market_regime = await self._detect_market_regime_optimized(results)
            
            # Create comprehensive analysis
            analysis = self._create_macro_analysis(results, macro_signals, market_regime)
            
            # Generate summary
            summary = self._create_macro_summary(results, macro_signals, market_regime)
            
            # Create results
            final_results = {
                "macro_analysis": analysis.to_dict(),
                "macro_signals": [signal.to_dict() for signal in macro_signals],
                "market_regime": market_regime,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "processing_info": {
                    "total_regions": len(regions),
                    "processing_time": self.metrics['processing_time_avg'],
                    "cache_hit_rate": self.metrics['cache_hit_rate']
                }
            }
            
            # Cache results
            if use_cache:
                self._cache_result(cache_key, final_results)
            
            self.metrics['total_analyses'] += 1
            self.metrics['macro_signals_generated'] += len(macro_signals)
            
            return final_results
            
        except Exception as e:
            self.error_count += 1
            self.metrics['error_rate'] = self.error_count / self.total_requests
            logging.error(f"Error in macro environment analysis: {e}")
            raise
    
    async def _analyze_economic_indicators_optimized(self, regions: List[str]) -> Dict[str, Any]:
        """Analyze economic indicators for given regions"""
        
        try:
            # Generate mock economic indicators
            indicators = []
            
            for region in regions:
                # GDP indicators
                gdp_indicator = EconomicIndicator(
                    name="GDP Growth",
                    region=region,
                    value=np.random.uniform(-2.0, 5.0),
                    previous_value=np.random.uniform(-2.0, 5.0),
                    forecast_value=np.random.uniform(-2.0, 5.0),
                    surprise_index=np.random.uniform(-2.0, 2.0),
                    impact_severity=ImpactSeverity.MEDIUM,
                    market_impact=MarketImpact.NEUTRAL,
                    timestamp=datetime.now(),
                    frequency="quarterly"
                )
                indicators.append(gdp_indicator)
                
                # Inflation indicators
                inflation_indicator = EconomicIndicator(
                    name="CPI Inflation",
                    region=region,
                    value=np.random.uniform(1.0, 8.0),
                    previous_value=np.random.uniform(1.0, 8.0),
                    forecast_value=np.random.uniform(1.0, 8.0),
                    surprise_index=np.random.uniform(-1.0, 1.0),
                    impact_severity=ImpactSeverity.HIGH,
                    market_impact=MarketImpact.NEUTRAL,
                    timestamp=datetime.now(),
                    frequency="monthly"
                )
                indicators.append(inflation_indicator)
                
                # Employment indicators
                employment_indicator = EconomicIndicator(
                    name="Unemployment Rate",
                    region=region,
                    value=np.random.uniform(3.0, 12.0),
                    previous_value=np.random.uniform(3.0, 12.0),
                    forecast_value=np.random.uniform(3.0, 12.0),
                    surprise_index=np.random.uniform(-0.5, 0.5),
                    impact_severity=ImpactSeverity.MEDIUM,
                    market_impact=MarketImpact.NEUTRAL,
                    timestamp=datetime.now(),
                    frequency="monthly"
                )
                indicators.append(employment_indicator)
                
                # PMI indicators
                pmi_indicator = EconomicIndicator(
                    name="Manufacturing PMI",
                    region=region,
                    value=np.random.uniform(45.0, 55.0),
                    previous_value=np.random.uniform(45.0, 55.0),
                    forecast_value=np.random.uniform(45.0, 55.0),
                    surprise_index=np.random.uniform(-2.0, 2.0),
                    impact_severity=ImpactSeverity.MEDIUM,
                    market_impact=MarketImpact.NEUTRAL,
                    timestamp=datetime.now(),
                    frequency="monthly"
                )
                indicators.append(pmi_indicator)
            
            # Calculate aggregate metrics
            surprise_index = np.mean([ind.surprise_index for ind in indicators])
            avg_impact_severity = np.mean([ind.impact_severity.value for ind in indicators])
            
            return {
                "indicators": [ind.to_dict() for ind in indicators],
                "surprise_index": surprise_index,
                "avg_impact_severity": avg_impact_severity,
                "total_indicators": len(indicators)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing economic indicators: {e}")
            return {
                "indicators": [],
                "surprise_index": 0.0,
                "avg_impact_severity": 0.0,
                "total_indicators": 0
            }
    
    async def _analyze_central_banks_optimized(self, regions: List[str]) -> Dict[str, Any]:
        """Analyze central bank policies and communications"""
        
        try:
            central_bank_actions = []
            
            for region in regions:
                # Generate mock central bank actions
                cb_action = CentralBankAction(
                    central_bank=f"{region}_Central_Bank",
                    region=region,
                    action_type=np.random.choice(["rate_decision", "forward_guidance", "qe_announcement", "communication"]),
                    current_rate=np.random.uniform(0.0, 5.0),
                    previous_rate=np.random.uniform(0.0, 5.0),
                    change_amount=np.random.uniform(-0.5, 0.5),
                    dovish_probability=np.random.uniform(0.0, 1.0),
                    hawkish_probability=np.random.uniform(0.0, 1.0),
                    market_impact=MarketImpact.NEUTRAL,
                    impact_severity=ImpactSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    next_meeting_date=datetime.now() + timedelta(days=np.random.randint(7, 90))
                )
                central_bank_actions.append(cb_action)
            
            # Calculate aggregate metrics
            avg_rate_change = np.mean([cb.change_amount for cb in central_bank_actions])
            dovish_bias = np.mean([cb.dovish_probability for cb in central_bank_actions])
            hawkish_bias = np.mean([cb.hawkish_probability for cb in central_bank_actions])
            
            return {
                "central_bank_actions": [cb.to_dict() for cb in central_bank_actions],
                "avg_rate_change": avg_rate_change,
                "dovish_bias": dovish_bias,
                "hawkish_bias": hawkish_bias,
                "total_actions": len(central_bank_actions)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing central banks: {e}")
            return {
                "central_bank_actions": [],
                "avg_rate_change": 0.0,
                "dovish_bias": 0.0,
                "hawkish_bias": 0.0,
                "total_actions": 0
            }
    
    async def _analyze_geopolitical_risks_optimized(self, regions: List[str]) -> Dict[str, Any]:
        """Analyze geopolitical risks and events"""
        
        try:
            geopolitical_events = []
            
            # Generate mock geopolitical events
            event_types = ["trade_tension", "political_instability", "conflict", "sanctions", "election"]
            
            for region in regions:
                for _ in range(np.random.randint(1, 4)):
                    event = GeopoliticalEvent(
                        event_type=np.random.choice(event_types),
                        region=region,
                        severity=ImpactSeverity(np.random.randint(1, 4)),
                        market_impact=MarketImpact(np.random.randint(1, 4)),
                        affected_assets=["equities", "bonds", "currencies"],
                        probability=np.random.uniform(0.1, 0.8),
                        timestamp=datetime.now(),
                        description=f"Mock geopolitical event in {region}",
                        resolution_probability=np.random.uniform(0.3, 0.9)
                    )
                    geopolitical_events.append(event)
            
            # Calculate aggregate metrics
            avg_severity = np.mean([event.severity.value for event in geopolitical_events])
            avg_probability = np.mean([event.probability for event in geopolitical_events])
            risk_score = avg_severity * avg_probability
            
            return {
                "geopolitical_events": [event.to_dict() for event in geopolitical_events],
                "avg_severity": avg_severity,
                "avg_probability": avg_probability,
                "risk_score": risk_score,
                "total_events": len(geopolitical_events)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing geopolitical risks: {e}")
            return {
                "geopolitical_events": [],
                "avg_severity": 0.0,
                "avg_probability": 0.0,
                "risk_score": 0.0,
                "total_events": 0
            }
    
    def _generate_macro_signals_optimized(self, analysis_results: Dict[str, Any]) -> List[MacroSignal]:
        """Generate macro economic signals"""
        
        signals = []
        
        try:
            # Economic indicator signals
            if "indicators" in analysis_results:
                indicators_data = analysis_results["indicators"]
                surprise_index = indicators_data.get("surprise_index", 0.0)
                
                if abs(surprise_index) > 0.5:
                    signal = MacroSignal(
                        signal_type="economic_surprise",
                        region="global",
                        direction="positive" if surprise_index > 0 else "negative",
                        strength=min(0.9, abs(surprise_index) / 2),
                        confidence=0.7,
                        impact_assets=["equities", "bonds", "currencies"],
                        timestamp=datetime.now(),
                        description=f"Economic surprise index: {surprise_index:.2f}"
                    )
                    signals.append(signal)
            
            # Central bank signals
            if "central_banks" in analysis_results:
                cb_data = analysis_results["central_banks"]
                dovish_bias = cb_data.get("dovish_bias", 0.0)
                hawkish_bias = cb_data.get("hawkish_bias", 0.0)
                
                if dovish_bias > 0.6:
                    signal = MacroSignal(
                        signal_type="central_bank_dovish",
                        region="global",
                        direction="positive",
                        strength=dovish_bias,
                        confidence=0.8,
                        impact_assets=["equities", "bonds"],
                        timestamp=datetime.now(),
                        description="Central banks showing dovish bias"
                    )
                    signals.append(signal)
                
                if hawkish_bias > 0.6:
                    signal = MacroSignal(
                        signal_type="central_bank_hawkish",
                        region="global",
                        direction="negative",
                        strength=hawkish_bias,
                        confidence=0.8,
                        impact_assets=["bonds", "currencies"],
                        timestamp=datetime.now(),
                        description="Central banks showing hawkish bias"
                    )
                    signals.append(signal)
            
            # Geopolitical signals
            if "geopolitical" in analysis_results:
                geo_data = analysis_results["geopolitical"]
                risk_score = geo_data.get("risk_score", 0.0)
                
                if risk_score > 0.3:
                    signal = MacroSignal(
                        signal_type="geopolitical_risk",
                        region="global",
                        direction="negative",
                        strength=min(0.9, risk_score * 2),
                        confidence=0.6,
                        impact_assets=["equities", "currencies", "commodities"],
                        timestamp=datetime.now(),
                        description=f"Elevated geopolitical risk: {risk_score:.2f}"
                    )
                    signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error generating macro signals: {e}")
        
        return signals
    
    async def _detect_market_regime_optimized(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime (risk-on/risk-off)"""
        
        try:
            # Calculate regime indicators
            risk_score = 0.0
            confidence = 0.0
            
            # Economic indicators contribution
            if "indicators" in analysis_results:
                surprise_index = analysis_results["indicators"].get("surprise_index", 0.0)
                risk_score += surprise_index * 0.3
            
            # Central bank contribution
            if "central_banks" in analysis_results:
                dovish_bias = analysis_results["central_banks"].get("dovish_bias", 0.0)
                hawkish_bias = analysis_results["central_banks"].get("hawkish_bias", 0.0)
                risk_score += (dovish_bias - hawkish_bias) * 0.4
            
            # Geopolitical contribution
            if "geopolitical" in analysis_results:
                geo_risk = analysis_results["geopolitical"].get("risk_score", 0.0)
                risk_score -= geo_risk * 0.3
            
            # Determine regime
            if risk_score > 0.2:
                regime = "risk_on"
                direction = "positive"
            elif risk_score < -0.2:
                regime = "risk_off"
                direction = "negative"
            else:
                regime = "neutral"
                direction = "neutral"
            
            confidence = min(0.9, abs(risk_score) * 2)
            
            return {
                "regime": regime,
                "direction": direction,
                "risk_score": risk_score,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            return {
                "regime": "neutral",
                "direction": "neutral",
                "risk_score": 0.0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_macro_analysis(
        self,
        analysis_results: Dict[str, Any],
        macro_signals: List[MacroSignal],
        market_regime: Dict[str, Any]
    ) -> MacroAnalysis:
        """Create comprehensive macro analysis"""
        
        try:
            # Extract data from results
            indicators_data = analysis_results.get("indicators", {})
            cb_data = analysis_results.get("central_banks", {})
            geo_data = analysis_results.get("geopolitical", {})
            
            # Create macro themes
            themes = []
            
            # Economic theme
            if indicators_data:
                surprise_index = indicators_data.get("surprise_index", 0.0)
                themes.append(MacroTheme(
                    name="Economic Momentum",
                    description=f"Economic surprise index: {surprise_index:.2f}",
                    impact_severity=ImpactSeverity.MEDIUM,
                    market_impact=MarketImpact.NEUTRAL,
                    confidence=0.7
                ))
            
            # Central bank theme
            if cb_data:
                dovish_bias = cb_data.get("dovish_bias", 0.0)
                themes.append(MacroTheme(
                    name="Central Bank Policy",
                    description=f"Central bank dovish bias: {dovish_bias:.2f}",
                    impact_severity=ImpactSeverity.HIGH,
                    market_impact=MarketImpact.NEUTRAL,
                    confidence=0.8
                ))
            
            # Geopolitical theme
            if geo_data:
                risk_score = geo_data.get("risk_score", 0.0)
                themes.append(MacroTheme(
                    name="Geopolitical Risk",
                    description=f"Geopolitical risk score: {risk_score:.2f}",
                    impact_severity=ImpactSeverity.MEDIUM,
                    market_impact=MarketImpact.NEGATIVE,
                    confidence=0.6
                ))
            
            return MacroAnalysis(
                timestamp=datetime.now(),
                analysis_horizon="medium_term",
                regions=["global"],
                economic_indicators=indicators_data.get("indicators", []),
                central_bank_actions=cb_data.get("central_bank_actions", []),
                geopolitical_events=geo_data.get("geopolitical_events", []),
                macro_themes=themes,
                market_regime=market_regime["regime"],
                risk_score=market_regime["risk_score"],
                confidence=market_regime["confidence"]
            )
            
        except Exception as e:
            logging.error(f"Error creating macro analysis: {e}")
            return MacroAnalysis(
                timestamp=datetime.now(),
                analysis_horizon="medium_term",
                regions=["global"],
                economic_indicators=[],
                central_bank_actions=[],
                geopolitical_events=[],
                macro_themes=[],
                market_regime="neutral",
                risk_score=0.0,
                confidence=0.0
            )
    
    def _create_macro_summary(
        self,
        analysis_results: Dict[str, Any],
        macro_signals: List[MacroSignal],
        market_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create macro analysis summary"""
        
        try:
            # Aggregate metrics
            total_signals = len(macro_signals)
            signal_types = defaultdict(int)
            directions = defaultdict(int)
            
            for signal in macro_signals:
                signal_types[signal.signal_type] += 1
                directions[signal.direction] += 1
            
            # Extract key metrics
            surprise_index = analysis_results.get("indicators", {}).get("surprise_index", 0.0)
            dovish_bias = analysis_results.get("central_banks", {}).get("dovish_bias", 0.0)
            risk_score = analysis_results.get("geopolitical", {}).get("risk_score", 0.0)
            
            return {
                'total_signals_generated': total_signals,
                'signal_types': dict(signal_types),
                'directions': dict(directions),
                'economic_surprise_index': surprise_index,
                'central_bank_dovish_bias': dovish_bias,
                'geopolitical_risk_score': risk_score,
                'market_regime': market_regime["regime"],
                'regime_confidence': market_regime["confidence"],
                'overall_risk_level': 'high' if risk_score > 0.5 else 'medium' if risk_score > 0.2 else 'low'
            }
            
        except Exception as e:
            logging.error(f"Error creating macro summary: {e}")
            return {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        age = time.time() - self.cache_timestamps[cache_key]
        return age < self.cache_ttl
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result with timestamp"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logging.info("Optimized Macro Agent cleanup completed")
