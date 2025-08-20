"""
Macro Agent - Phase 2 Standardized

Macroeconomic analysis agent with uncertainty quantification (μ, σ, horizon).
Analyzes economic indicators, central bank policies, and geopolitical events.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from common.models import BaseAgent
from schemas.contracts import Signal, SignalType, RegimeType, HorizonType, DirectionType


logger = logging.getLogger(__name__)


class MacroAgentPhase2(BaseAgent):
    """
    Macroeconomic Analysis Agent with Uncertainty Quantification
    
    Features:
    - Economic indicator analysis (GDP, inflation, employment)
    - Central bank policy tracking (interest rates, QE)
    - Geopolitical risk assessment
    - Cross-asset correlation analysis
    - Regime change detection
    - Long-term trend analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("macro_agent_v2", SignalType.MACRO, config)
        
        # Macro analysis parameters
        self.economic_indicators = config.get('economic_indicators', [
            'gdp_growth', 'inflation_rate', 'unemployment_rate', 
            'interest_rates', 'money_supply'
        ]) if config else ['gdp_growth', 'inflation_rate', 'unemployment_rate', 'interest_rates', 'money_supply']
        
        self.min_confidence = config.get('min_confidence', 0.4) if config else 0.4
        self.forecast_horizon = config.get('forecast_horizon', 90) if config else 90  # days
        
        # Economic thresholds
        self.inflation_target = 2.0  # Central bank target
        self.unemployment_threshold = 5.0  # Natural rate approximation
        self.gdp_recession_threshold = -0.5  # Quarterly decline
        
        # Central bank parameters
        self.fed_rate_ranges = {
            'accommodative': (0.0, 2.0),
            'neutral': (2.0, 4.0),
            'restrictive': (4.0, 8.0)
        }
        
        # Geopolitical factors
        self.geopolitical_factors = [
            'trade_tensions', 'military_conflicts', 'sanctions',
            'elections', 'policy_uncertainty'
        ]
        
        # Cross-asset indicators
        self.cross_asset_indicators = [
            'yield_curve', 'credit_spreads', 'currency_strength',
            'commodity_prices', 'volatility_index'
        ]
        
        # Performance tracking
        self.forecast_accuracy = {}
        self.regime_detection_history = []
        
    async def generate_signals(self, symbols: List[str], **kwargs) -> List[Signal]:
        """
        Generate macro signals with uncertainty quantification
        
        Args:
            symbols: List of symbols to analyze
            **kwargs: Additional parameters (macro_data, trace_id, etc.)
            
        Returns:
            List of standardized Signal objects
        """
        try:
            signals = []
            macro_data = kwargs.get('macro_data', {})
            trace_id = kwargs.get('trace_id')
            
            # Analyze global macro environment
            global_analysis = await self._analyze_global_macro(macro_data)
            
            # Generate signals for each symbol based on macro environment
            for symbol in symbols:
                symbol_signals = await self._analyze_symbol_macro(
                    symbol, macro_data, global_analysis, trace_id
                )
                if symbol_signals:
                    signals.extend(symbol_signals)
            
            logger.info(f"Generated {len(signals)} macro signals for {len(symbols)} symbols")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating macro signals: {e}")
            return []
    
    async def _analyze_global_macro(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze global macroeconomic environment"""
        try:
            if not macro_data:
                macro_data = self._generate_synthetic_macro()
            
            analysis = {}
            
            # Economic indicators analysis
            econ_analysis = await self._analyze_economic_indicators(
                macro_data.get('economic', {})
            )
            analysis['economic'] = econ_analysis
            
            # Central bank policy analysis
            cb_analysis = await self._analyze_central_bank_policy(
                macro_data.get('central_bank', {})
            )
            analysis['central_bank'] = cb_analysis
            
            # Geopolitical risk analysis
            geo_analysis = await self._analyze_geopolitical_risks(
                macro_data.get('geopolitical', {})
            )
            analysis['geopolitical'] = geo_analysis
            
            # Cross-asset analysis
            cross_asset_analysis = await self._analyze_cross_asset_signals(
                macro_data.get('cross_asset', {})
            )
            analysis['cross_asset'] = cross_asset_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing global macro: {e}")
            return {}
    
    def _generate_synthetic_macro(self) -> Dict[str, Any]:
        """Generate synthetic macro data for demo"""
        np.random.seed(int(datetime.now().timestamp()) % 2**32)
        
        return {
            'economic': {
                'gdp_growth': np.random.normal(2.5, 1.0),  # Annual %
                'inflation_rate': np.random.normal(2.8, 0.8),  # Annual %
                'unemployment_rate': np.random.normal(4.2, 0.5),  # %
                'interest_rates': np.random.normal(4.5, 1.0),  # %
                'money_supply_growth': np.random.normal(6.0, 2.0),  # Annual %
                'consumer_confidence': np.random.normal(60, 15),  # Index
                'manufacturing_pmi': np.random.normal(52, 5)  # Index
            },
            'central_bank': {
                'fed_rate': np.random.normal(4.5, 0.5),
                'rate_change_expectation': np.random.normal(0.0, 0.25),
                'qe_status': np.random.choice(['expanding', 'stable', 'tapering']),
                'balance_sheet_change': np.random.normal(0, 5),  # %
                'hawkish_dovish_score': np.random.normal(0, 1)  # -2 to 2
            },
            'geopolitical': {
                'trade_tension_score': np.random.uniform(0, 10),
                'military_conflict_risk': np.random.uniform(0, 5),
                'sanctions_impact': np.random.uniform(0, 3),
                'election_uncertainty': np.random.uniform(0, 8),
                'policy_uncertainty_index': np.random.uniform(50, 200)
            },
            'cross_asset': {
                'yield_curve_slope': np.random.normal(0.8, 0.5),  # 10Y-2Y spread
                'credit_spreads': np.random.normal(150, 50),  # bps
                'dxy_strength': np.random.normal(0, 2),  # % change
                'oil_price_change': np.random.normal(0, 15),  # %
                'gold_price_change': np.random.normal(0, 10),  # %
                'vix_level': np.random.uniform(12, 35)
            }
        }
    
    async def _analyze_economic_indicators(self, econ_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze economic indicators"""
        try:
            if not econ_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            signals = []
            
            # GDP Growth analysis
            gdp_growth = econ_data.get('gdp_growth', 2.0)
            if gdp_growth < self.gdp_recession_threshold:
                signals.append(('gdp_recession', -0.8, 0.9))
            elif gdp_growth > 4.0:
                signals.append(('gdp_strong', 0.6, 0.8))
            else:
                gdp_signal = (gdp_growth - 2.5) / 5.0  # Normalize around 2.5% trend
                signals.append(('gdp_trend', gdp_signal, 0.6))
            
            # Inflation analysis
            inflation = econ_data.get('inflation_rate', 2.0)
            inflation_deviation = inflation - self.inflation_target
            if abs(inflation_deviation) > 1.0:  # Significant deviation
                inflation_signal = -inflation_deviation / 3.0  # Inverse relationship with assets
                signals.append(('inflation', inflation_signal, 0.8))
            
            # Unemployment analysis
            unemployment = econ_data.get('unemployment_rate', 4.5)
            if unemployment > self.unemployment_threshold + 1.0:
                signals.append(('unemployment_high', -0.5, 0.7))
            elif unemployment < 3.5:  # Very low unemployment
                signals.append(('unemployment_low', 0.3, 0.6))
            
            # Interest rates analysis
            interest_rates = econ_data.get('interest_rates', 4.0)
            if interest_rates > 6.0:
                signals.append(('rates_restrictive', -0.6, 0.8))
            elif interest_rates < 1.0:
                signals.append(('rates_accommodative', 0.4, 0.7))
            
            # PMI analysis
            pmi = econ_data.get('manufacturing_pmi', 50)
            pmi_signal = (pmi - 50) / 50  # Normalize around 50
            if abs(pmi_signal) > 0.1:
                signals.append(('pmi', pmi_signal, 0.6))
            
            # Combine signals
            if signals:
                weights = [s[2] for s in signals]
                weighted_signals = [s[1] * s[2] for s in signals]
                
                combined_signal = sum(weighted_signals) / sum(weights)
                combined_confidence = np.mean(weights)
            else:
                combined_signal = 0.0
                combined_confidence = 0.0
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'components': dict([(s[0], {'signal': s[1], 'confidence': s[2]}) for s in signals])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing economic indicators: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_central_bank_policy(self, cb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze central bank policy stance"""
        try:
            if not cb_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            fed_rate = cb_data.get('fed_rate', 4.0)
            rate_expectation = cb_data.get('rate_change_expectation', 0.0)
            qe_status = cb_data.get('qe_status', 'stable')
            hawkish_dovish = cb_data.get('hawkish_dovish_score', 0.0)
            
            signals = []
            
            # Rate level analysis
            if fed_rate < self.fed_rate_ranges['accommodative'][1]:
                signals.append(('accommodative_rates', 0.5, 0.8))
            elif fed_rate > self.fed_rate_ranges['restrictive'][0]:
                signals.append(('restrictive_rates', -0.4, 0.8))
            
            # Rate expectation analysis
            if abs(rate_expectation) > 0.1:  # Significant expected change
                rate_signal = -rate_expectation * 2.0  # Inverse relationship
                signals.append(('rate_expectation', rate_signal, 0.7))
            
            # QE policy analysis
            qe_signals = {
                'expanding': 0.4,
                'stable': 0.0,
                'tapering': -0.3
            }
            if qe_status in qe_signals:
                signals.append(('qe_policy', qe_signals[qe_status], 0.6))
            
            # Hawkish/Dovish stance
            if abs(hawkish_dovish) > 0.5:
                hd_signal = -hawkish_dovish / 4.0  # Hawkish = negative for assets
                signals.append(('policy_stance', hd_signal, 0.7))
            
            # Combine signals
            if signals:
                weights = [s[2] for s in signals]
                weighted_signals = [s[1] * s[2] for s in signals]
                
                combined_signal = sum(weighted_signals) / sum(weights)
                combined_confidence = np.mean(weights)
            else:
                combined_signal = 0.0
                combined_confidence = 0.0
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'policy_stance': qe_status,
                'rate_direction': 'up' if rate_expectation > 0.1 else 'down' if rate_expectation < -0.1 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing central bank policy: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_geopolitical_risks(self, geo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze geopolitical risks"""
        try:
            if not geo_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            trade_tension = geo_data.get('trade_tension_score', 0)
            military_risk = geo_data.get('military_conflict_risk', 0)
            sanctions = geo_data.get('sanctions_impact', 0)
            election_uncertainty = geo_data.get('election_uncertainty', 0)
            policy_uncertainty = geo_data.get('policy_uncertainty_index', 100)
            
            # Calculate risk scores (higher = more negative for markets)
            risk_scores = []
            
            if trade_tension > 5:
                risk_scores.append(trade_tension / 10)
            
            if military_risk > 2:
                risk_scores.append(military_risk / 5)
            
            if sanctions > 1:
                risk_scores.append(sanctions / 3)
            
            if election_uncertainty > 5:
                risk_scores.append(election_uncertainty / 10)
            
            if policy_uncertainty > 150:
                risk_scores.append((policy_uncertainty - 100) / 200)
            
            # Combine risk scores
            if risk_scores:
                avg_risk = np.mean(risk_scores)
                risk_signal = -avg_risk * 0.3  # Negative impact on markets
                confidence = min(len(risk_scores) / 5, 1.0)  # More factors = higher confidence
            else:
                risk_signal = 0.0
                confidence = 0.2  # Low confidence when no significant risks
            
            return {
                'signal': risk_signal,
                'confidence': confidence,
                'risk_factors': len(risk_scores),
                'avg_risk_score': np.mean(risk_scores) if risk_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing geopolitical risks: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_cross_asset_signals(self, cross_asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-asset market signals"""
        try:
            if not cross_asset_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            yield_curve = cross_asset_data.get('yield_curve_slope', 0.8)
            credit_spreads = cross_asset_data.get('credit_spreads', 150)
            dxy_strength = cross_asset_data.get('dxy_strength', 0)
            oil_change = cross_asset_data.get('oil_price_change', 0)
            vix_level = cross_asset_data.get('vix_level', 20)
            
            signals = []
            
            # Yield curve analysis
            if yield_curve < 0:  # Inverted yield curve
                signals.append(('yield_curve_inversion', -0.6, 0.9))
            elif yield_curve < 0.5:  # Flattening curve
                signals.append(('yield_curve_flattening', -0.3, 0.7))
            
            # Credit spreads analysis
            if credit_spreads > 300:  # Widening spreads
                signals.append(('credit_stress', -0.5, 0.8))
            elif credit_spreads < 100:  # Tight spreads
                signals.append(('credit_optimism', 0.3, 0.6))
            
            # Dollar strength analysis
            if abs(dxy_strength) > 3:  # Significant dollar move
                dxy_signal = -dxy_strength / 10  # Strong dollar negative for commodities/EM
                signals.append(('dollar_strength', dxy_signal, 0.7))
            
            # VIX analysis
            if vix_level > 30:  # High volatility
                signals.append(('high_volatility', -0.4, 0.8))
            elif vix_level < 15:  # Low volatility (complacency)
                signals.append(('low_volatility', 0.2, 0.6))
            
            # Oil price analysis (inflation proxy)
            if abs(oil_change) > 10:  # Significant oil move
                oil_signal = oil_change / 50  # Normalize
                signals.append(('oil_inflation', oil_signal, 0.6))
            
            # Combine signals
            if signals:
                weights = [s[2] for s in signals]
                weighted_signals = [s[1] * s[2] for s in signals]
                
                combined_signal = sum(weighted_signals) / sum(weights)
                combined_confidence = np.mean(weights)
            else:
                combined_signal = 0.0
                combined_confidence = 0.0
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'yield_curve_status': 'inverted' if yield_curve < 0 else 'normal',
                'volatility_regime': 'high' if vix_level > 25 else 'low' if vix_level < 15 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset signals: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_symbol_macro(self, symbol: str, macro_data: Dict[str, Any],
                                  global_analysis: Dict[str, Any], 
                                  trace_id: Optional[str] = None) -> List[Signal]:
        """Analyze macro impact on specific symbol"""
        try:
            # Combine global macro signals
            macro_signals = []
            confidences = []
            
            for component, analysis in global_analysis.items():
                if analysis and 'signal' in analysis and 'confidence' in analysis:
                    macro_signals.append(analysis['signal'])
                    confidences.append(analysis['confidence'])
            
            if not macro_signals:
                return []
            
            # Weight components based on relevance to equities
            component_weights = {
                'economic': 0.4,
                'central_bank': 0.3,
                'cross_asset': 0.2,
                'geopolitical': 0.1
            }
            
            # Calculate weighted macro signal
            weighted_signal = 0.0
            total_weight = 0.0
            
            for i, (component, analysis) in enumerate(global_analysis.items()):
                if analysis and 'signal' in analysis:
                    weight = component_weights.get(component, 0.25)
                    confidence = analysis.get('confidence', 0.5)
                    
                    weighted_signal += analysis['signal'] * weight * confidence
                    total_weight += weight * confidence
            
            if total_weight == 0:
                return []
            
            final_signal = weighted_signal / total_weight
            final_confidence = np.mean(confidences)
            
            # Apply symbol-specific adjustments
            final_signal *= self._get_symbol_macro_sensitivity(symbol)
            
            if abs(final_signal) < 0.005 or final_confidence < self.min_confidence:
                return []
            
            # Determine market conditions for uncertainty calculation
            market_conditions = self._assess_macro_conditions(global_analysis)
            
            # Create standardized signal
            signal = self.create_signal(
                symbol=symbol,
                mu=final_signal,
                confidence=final_confidence,
                market_conditions=market_conditions,
                trace_id=trace_id,
                metadata={
                    'global_analysis': global_analysis,
                    'component_weights': component_weights,
                    'macro_sensitivity': self._get_symbol_macro_sensitivity(symbol),
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return [signal]
            
        except Exception as e:
            logger.error(f"Error analyzing macro for {symbol}: {e}")
            return []
    
    def _get_symbol_macro_sensitivity(self, symbol: str) -> float:
        """Get symbol's sensitivity to macro factors"""
        # This would typically use sector/industry mappings
        # For demo, use simple heuristics
        if symbol in ['XLF', 'JPM', 'BAC', 'GS']:  # Financials
            return 1.2  # High sensitivity to rates
        elif symbol in ['XLE', 'XOM', 'CVX']:  # Energy
            return 0.8  # Moderate sensitivity
        elif symbol in ['XLU', 'NEE', 'DUK']:  # Utilities
            return 1.1  # High sensitivity to rates
        elif symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Large cap tech
            return 0.9  # Moderate sensitivity
        else:
            return 1.0  # Default sensitivity
    
    def _assess_macro_conditions(self, global_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess macro conditions for uncertainty calculation"""
        try:
            # Calculate macro uncertainty based on signal dispersion
            signals = []
            for analysis in global_analysis.values():
                if analysis and 'signal' in analysis:
                    signals.append(analysis['signal'])
            
            macro_volatility = np.std(signals) if len(signals) > 1 else 0.1
            
            # Policy uncertainty
            cb_analysis = global_analysis.get('central_bank', {})
            policy_uncertainty = 1.0 - cb_analysis.get('confidence', 0.5)
            
            # Geopolitical risk
            geo_analysis = global_analysis.get('geopolitical', {})
            geo_risk = abs(geo_analysis.get('signal', 0.0)) * 2
            
            # Cross-asset stress
            cross_analysis = global_analysis.get('cross_asset', {})
            market_stress = abs(cross_analysis.get('signal', 0.0)) * 3
            
            return {
                'volatility': macro_volatility,
                'liquidity': max(0.5, 1.0 - market_stress),  # Market stress reduces liquidity
                'policy_uncertainty': policy_uncertainty,
                'geopolitical_risk': geo_risk,
                'signal_count': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error assessing macro conditions: {e}")
            return {
                'volatility': 0.15,
                'liquidity': 1.0,
                'policy_uncertainty': 0.3,
                'geopolitical_risk': 0.2,
                'signal_count': 1
            }
    
    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeType:
        """Detect macro regime"""
        try:
            volatility = market_data.get('volatility', 0.15)
            policy_uncertainty = market_data.get('policy_uncertainty', 0.3)
            geopolitical_risk = market_data.get('geopolitical_risk', 0.2)
            liquidity = market_data.get('liquidity', 1.0)
            
            # High uncertainty regime
            if policy_uncertainty > 0.6 or geopolitical_risk > 0.4:
                return RegimeType.HIGH_VOL
            
            # Low liquidity regime
            if liquidity < 0.6:
                return RegimeType.ILLIQUID
            
            # Stable macro regime
            if volatility < 0.1 and policy_uncertainty < 0.2:
                return RegimeType.LOW_VOL
            
            # Default to risk-on
            return RegimeType.RISK_ON
            
        except Exception as e:
            logger.error(f"Error detecting macro regime: {e}")
            return RegimeType.RISK_ON
