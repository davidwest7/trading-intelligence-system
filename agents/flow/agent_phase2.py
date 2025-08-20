"""
Flow Agent - Phase 2 Standardized

Money flow analysis agent with uncertainty quantification (μ, σ, horizon).
Analyzes institutional flows, dark pools, and order flow to emit standardized signals.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from common.models import BaseAgent
from schemas.contracts import Signal, SignalType, RegimeType, HorizonType, DirectionType


logger = logging.getLogger(__name__)


class FlowAgentPhase2(BaseAgent):
    """
    Money Flow Analysis Agent with Uncertainty Quantification
    
    Features:
    - Institutional flow detection (dark pools, block trades)
    - Order flow imbalance analysis
    - Volume profile analysis
    - Smart money vs retail money classification
    - Real-time flow tracking
    - Uncertainty quantification based on flow quality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("flow_agent_v2", SignalType.FLOW, config)
        
        # Flow analysis parameters
        self.min_block_size = config.get('min_block_size', 10000) if config else 10000  # USD
        self.flow_window = config.get('flow_window', 4) if config else 4  # hours
        self.min_confidence = config.get('min_confidence', 0.35) if config else 0.35
        
        # Dark pool parameters
        self.dark_pool_threshold = 0.3  # 30% of volume
        self.institutional_size_threshold = 50000  # USD
        
        # Order flow parameters
        self.imbalance_threshold = 0.2  # 20% imbalance
        self.tick_analysis_periods = [5, 15, 30]  # minutes
        
        # Volume profile parameters
        self.vwap_periods = [20, 50, 200]
        self.volume_spike_threshold = 2.0  # 2x average volume
        
        # Smart money indicators
        self.smart_money_indicators = [
            'large_block_trades',
            'dark_pool_activity',
            'unusual_options_activity',
            'insider_activity'
        ]
        
        # Performance tracking
        self.flow_accuracy = {}
        self.prediction_history = []
        
    async def generate_signals(self, symbols: List[str], **kwargs) -> List[Signal]:
        """
        Generate flow signals with uncertainty quantification
        
        Args:
            symbols: List of symbols to analyze
            **kwargs: Additional parameters (flow_data, trace_id, etc.)
            
        Returns:
            List of standardized Signal objects
        """
        try:
            signals = []
            flow_data = kwargs.get('flow_data', {})
            trace_id = kwargs.get('trace_id')
            
            for symbol in symbols:
                symbol_signals = await self._analyze_symbol_flow(
                    symbol, flow_data, trace_id
                )
                if symbol_signals:
                    signals.extend(symbol_signals)
            
            logger.info(f"Generated {len(signals)} flow signals for {len(symbols)} symbols")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating flow signals: {e}")
            return []
    
    async def _analyze_symbol_flow(self, symbol: str, flow_data: Dict[str, Any],
                                 trace_id: Optional[str] = None) -> List[Signal]:
        """Analyze flow for a single symbol"""
        try:
            # Get flow data for symbol
            symbol_flow = flow_data.get(symbol, {})
            if not symbol_flow:
                # Generate synthetic flow data for demo
                symbol_flow = self._generate_synthetic_flow(symbol)
            
            # Analyze different flow components
            flow_analysis = {}
            
            # Institutional flow analysis
            institutional_flow = await self._analyze_institutional_flow(
                symbol, symbol_flow.get('institutional', {})
            )
            flow_analysis['institutional'] = institutional_flow
            
            # Order flow imbalance analysis
            order_flow = await self._analyze_order_flow(
                symbol, symbol_flow.get('orders', {})
            )
            flow_analysis['order_flow'] = order_flow
            
            # Volume profile analysis
            volume_profile = await self._analyze_volume_profile(
                symbol, symbol_flow.get('volume', {})
            )
            flow_analysis['volume_profile'] = volume_profile
            
            # Dark pool analysis
            dark_pool = await self._analyze_dark_pool_activity(
                symbol, symbol_flow.get('dark_pools', {})
            )
            flow_analysis['dark_pool'] = dark_pool
            
            # Combine flow signals
            combined_signal, confidence = self._combine_flow_signals(flow_analysis)
            
            if abs(combined_signal) < 0.005 or confidence < self.min_confidence:
                return []
            
            # Determine market conditions for uncertainty calculation
            market_conditions = self._assess_flow_conditions(symbol_flow, flow_analysis)
            
            # Create standardized signal
            signal = self.create_signal(
                symbol=symbol,
                mu=combined_signal,
                confidence=confidence,
                market_conditions=market_conditions,
                trace_id=trace_id,
                metadata={
                    'flow_analysis': flow_analysis,
                    'flow_type': 'multi_component',
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return [signal]
            
        except Exception as e:
            logger.error(f"Error analyzing flow for {symbol}: {e}")
            return []
    
    def _generate_synthetic_flow(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic flow data for demo"""
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate institutional flow data
        block_trades = np.random.randint(5, 25)
        avg_block_size = np.random.uniform(20000, 200000)
        institutional_bias = np.random.normal(0.05, 0.2)
        
        # Generate order flow data
        buy_volume = np.random.uniform(40, 60)  # Percentage
        sell_volume = 100 - buy_volume
        order_imbalance = (buy_volume - sell_volume) / 100
        
        # Generate volume profile data
        total_volume = np.random.lognormal(12, 1)
        volume_spike = np.random.uniform(0.5, 3.0)
        vwap_deviation = np.random.normal(0, 0.02)
        
        # Generate dark pool data
        dark_pool_volume = np.random.uniform(0.1, 0.5)
        dark_pool_price_improvement = np.random.uniform(-0.001, 0.002)
        
        return {
            'institutional': {
                'block_trades': block_trades,
                'avg_block_size': avg_block_size,
                'net_flow': institutional_bias,
                'smart_money_ratio': np.random.uniform(0.3, 0.8)
            },
            'orders': {
                'buy_volume_pct': buy_volume,
                'sell_volume_pct': sell_volume,
                'order_imbalance': order_imbalance,
                'tick_direction': np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            },
            'volume': {
                'total_volume': total_volume,
                'volume_spike': volume_spike,
                'vwap_deviation': vwap_deviation,
                'volume_profile_poc': np.random.uniform(-0.02, 0.02)  # Point of Control
            },
            'dark_pools': {
                'dark_volume_pct': dark_pool_volume,
                'price_improvement': dark_pool_price_improvement,
                'block_participation': np.random.uniform(0.2, 0.7)
            }
        }
    
    async def _analyze_institutional_flow(self, symbol: str, 
                                        institutional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze institutional flow patterns"""
        try:
            if not institutional_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            block_trades = institutional_data.get('block_trades', 0)
            avg_block_size = institutional_data.get('avg_block_size', 0)
            net_flow = institutional_data.get('net_flow', 0.0)
            smart_money_ratio = institutional_data.get('smart_money_ratio', 0.5)
            
            # Block trade significance
            if avg_block_size < self.min_block_size:
                return {'signal': 0.0, 'confidence': 0.0}
            
            # Calculate institutional signal strength
            size_factor = min(avg_block_size / self.institutional_size_threshold, 2.0)
            volume_factor = min(block_trades / 10, 1.5)
            smart_money_factor = smart_money_ratio
            
            signal_strength = net_flow * size_factor * volume_factor * smart_money_factor
            
            # Calculate confidence
            confidence = min(
                block_trades / 15,  # More trades = higher confidence
                avg_block_size / (2 * self.institutional_size_threshold),  # Larger size = higher confidence
                smart_money_ratio  # Smart money ratio as confidence
            )
            confidence = min(confidence, 1.0)
            
            return {
                'signal': signal_strength,
                'confidence': confidence,
                'block_trades': block_trades,
                'avg_size': avg_block_size,
                'smart_money_ratio': smart_money_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional flow: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_order_flow(self, symbol: str, 
                                order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze order flow imbalance"""
        try:
            if not order_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            buy_volume_pct = order_data.get('buy_volume_pct', 50)
            sell_volume_pct = order_data.get('sell_volume_pct', 50)
            order_imbalance = order_data.get('order_imbalance', 0.0)
            tick_direction = order_data.get('tick_direction', 0)
            
            # Calculate order flow signal
            # Normalize imbalance to [-1, 1] range
            normalized_imbalance = np.clip(order_imbalance * 2, -1, 1)
            
            # Weight by tick direction
            tick_weight = {-1: -0.3, 0: 0.0, 1: 0.3}.get(tick_direction, 0.0)
            
            # Combined signal
            flow_signal = (normalized_imbalance * 0.7 + tick_weight * 0.3) * 0.02  # Max 2% impact
            
            # Calculate confidence based on imbalance magnitude
            confidence = min(abs(order_imbalance) / self.imbalance_threshold, 1.0)
            
            return {
                'signal': flow_signal,
                'confidence': confidence,
                'imbalance': order_imbalance,
                'buy_pct': buy_volume_pct,
                'sell_pct': sell_volume_pct
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_volume_profile(self, symbol: str, 
                                    volume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume profile patterns"""
        try:
            if not volume_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            total_volume = volume_data.get('total_volume', 0)
            volume_spike = volume_data.get('volume_spike', 1.0)
            vwap_deviation = volume_data.get('vwap_deviation', 0.0)
            volume_profile_poc = volume_data.get('volume_profile_poc', 0.0)
            
            # Volume spike analysis
            spike_signal = 0.0
            if volume_spike > self.volume_spike_threshold:
                # Volume spike suggests strong momentum
                spike_signal = min((volume_spike - 1.0) / 2.0, 0.5) * np.sign(vwap_deviation)
            
            # VWAP deviation analysis
            vwap_signal = vwap_deviation * 10  # Amplify small deviations
            vwap_signal = np.clip(vwap_signal, -0.03, 0.03)
            
            # Point of Control analysis
            poc_signal = volume_profile_poc * 5  # Point of Control deviation
            poc_signal = np.clip(poc_signal, -0.02, 0.02)
            
            # Combined volume signal
            combined_signal = spike_signal + vwap_signal + poc_signal
            
            # Calculate confidence
            volume_confidence = min(volume_spike / 2.0, 1.0)
            deviation_confidence = min(abs(vwap_deviation) * 20, 1.0)
            
            confidence = (volume_confidence + deviation_confidence) / 2
            
            return {
                'signal': combined_signal,
                'confidence': confidence,
                'volume_spike': volume_spike,
                'vwap_deviation': vwap_deviation,
                'poc_signal': poc_signal
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    async def _analyze_dark_pool_activity(self, symbol: str, 
                                        dark_pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dark pool trading activity"""
        try:
            if not dark_pool_data:
                return {'signal': 0.0, 'confidence': 0.0}
            
            dark_volume_pct = dark_pool_data.get('dark_volume_pct', 0.0)
            price_improvement = dark_pool_data.get('price_improvement', 0.0)
            block_participation = dark_pool_data.get('block_participation', 0.0)
            
            # Dark pool threshold check
            if dark_volume_pct < self.dark_pool_threshold:
                return {'signal': 0.0, 'confidence': 0.2}
            
            # Dark pool signal analysis
            # High dark pool activity suggests institutional interest
            volume_signal = (dark_volume_pct - self.dark_pool_threshold) * 2.0
            
            # Price improvement indicates buying/selling pressure
            price_signal = price_improvement * 20  # Amplify small improvements
            
            # Block participation indicates institutional nature
            block_signal = block_participation * 0.02
            
            # Combined dark pool signal
            combined_signal = (volume_signal + price_signal + block_signal) * 0.01
            combined_signal = np.clip(combined_signal, -0.03, 0.03)
            
            # Calculate confidence
            activity_confidence = min(dark_volume_pct / 0.5, 1.0)  # Higher % = higher confidence
            block_confidence = block_participation
            
            confidence = (activity_confidence + block_confidence) / 2
            
            return {
                'signal': combined_signal,
                'confidence': confidence,
                'dark_volume_pct': dark_volume_pct,
                'price_improvement': price_improvement,
                'institutional_nature': block_participation
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dark pool activity: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    def _combine_flow_signals(self, flow_analysis: Dict[str, Any]) -> Tuple[float, float]:
        """Combine signals from different flow components"""
        try:
            signals = []
            confidences = []
            weights = {
                'institutional': 0.4,
                'order_flow': 0.3,
                'volume_profile': 0.2,
                'dark_pool': 0.1
            }
            
            # Extract signals and confidences
            for component, analysis in flow_analysis.items():
                if analysis and 'signal' in analysis and 'confidence' in analysis:
                    signal = analysis['signal']
                    confidence = analysis['confidence']
                    weight = weights.get(component, 0.25)
                    
                    if confidence > 0.1:  # Only include confident signals
                        signals.append(signal * weight * confidence)
                        confidences.append(confidence * weight)
            
            if not signals:
                return 0.0, 0.0
            
            # Weighted combination
            combined_signal = sum(signals) / max(sum(weights.values()), 1.0)
            combined_confidence = sum(confidences) / max(sum(weights.values()), 1.0)
            
            # Signal agreement bonus
            if len(signals) > 1:
                signal_agreement = 1.0 - np.std([s / max(abs(s), 0.001) for s in signals if abs(s) > 0.001])
                combined_confidence *= max(0.5, signal_agreement)
            
            # Ensure reasonable ranges
            combined_signal = np.clip(combined_signal, -0.06, 0.06)
            combined_confidence = np.clip(combined_confidence, 0.0, 1.0)
            
            return combined_signal, combined_confidence
            
        except Exception as e:
            logger.error(f"Error combining flow signals: {e}")
            return 0.0, 0.0
    
    def _assess_flow_conditions(self, symbol_flow: Dict[str, Any],
                              flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess flow conditions for uncertainty calculation"""
        try:
            # Calculate flow volatility
            signals = []
            for analysis in flow_analysis.values():
                if analysis and 'signal' in analysis:
                    signals.append(analysis['signal'])
            
            flow_volatility = np.std(signals) if len(signals) > 1 else 0.1
            
            # Volume-based liquidity measure
            volume_data = symbol_flow.get('volume', {})
            volume_spike = volume_data.get('volume_spike', 1.0)
            liquidity_factor = min(volume_spike / 2.0, 2.0)
            
            # Institutional activity level
            institutional_data = symbol_flow.get('institutional', {})
            institutional_activity = institutional_data.get('smart_money_ratio', 0.5)
            
            # Dark pool activity
            dark_pool_data = symbol_flow.get('dark_pools', {})
            dark_pool_activity = dark_pool_data.get('dark_volume_pct', 0.2)
            
            return {
                'volatility': flow_volatility,
                'liquidity': liquidity_factor,
                'institutional_activity': institutional_activity,
                'dark_pool_activity': dark_pool_activity,
                'signal_count': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error assessing flow conditions: {e}")
            return {
                'volatility': 0.15,
                'liquidity': 1.0,
                'institutional_activity': 0.5,
                'dark_pool_activity': 0.2,
                'signal_count': 1
            }
    
    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeType:
        """Detect market regime based on flow characteristics"""
        try:
            volatility = market_data.get('volatility', 0.15)
            liquidity = market_data.get('liquidity', 1.0)
            institutional_activity = market_data.get('institutional_activity', 0.5)
            dark_pool_activity = market_data.get('dark_pool_activity', 0.2)
            
            # High institutional activity regime
            if institutional_activity > 0.7 and dark_pool_activity > 0.4:
                return RegimeType.LIQUID
            
            # Low liquidity regime
            if liquidity < 0.5:
                return RegimeType.ILLIQUID
            
            # High volatility regime
            if volatility > 0.25:
                return RegimeType.HIGH_VOL
            
            # Stable flow regime
            if volatility < 0.1 and liquidity > 1.5:
                return RegimeType.LOW_VOL
            
            # Default to risk-on
            return RegimeType.RISK_ON
            
        except Exception as e:
            logger.error(f"Error detecting flow regime: {e}")
            return RegimeType.RISK_ON
