"""
Institutional Flow Detection Engine

Detects and analyzes institutional money flows through:
- Block trade identification
- Dark pool activity estimation
- Institution type classification
- Flow pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .models import (
    InstitutionalFlow, DarkPoolActivity, VolumeConcentration,
    FlowPattern, FlowType, FlowDirection, InstitutionType
)


class InstitutionalFlowDetector:
    """
    Advanced institutional flow detection system
    
    Features:
    - Block trade identification using statistical methods
    - Dark pool activity estimation
    - Institution type classification based on trade characteristics
    - Flow pattern recognition (accumulation, distribution, rotation)
    """
    
    def __init__(self):
        # Detection thresholds
        self.block_trade_threshold = 10000  # Minimum shares for block trade
        self.institution_confidence_threshold = 0.7
        
        # Pattern detection parameters
        self.accumulation_window = 20  # bars for pattern detection
        self.distribution_window = 20
        
        # Institution classification features
        self.institution_signatures = {
            InstitutionType.PENSION_FUND: {
                'avg_trade_size': 50000,
                'time_of_day_preference': [9.5, 15.5],  # Hours
                'volatility_sensitivity': 0.3,
                'momentum_following': 0.2
            },
            InstitutionType.HEDGE_FUND: {
                'avg_trade_size': 25000,
                'time_of_day_preference': [10, 16],
                'volatility_sensitivity': 0.8,
                'momentum_following': 0.9
            },
            InstitutionType.MUTUAL_FUND: {
                'avg_trade_size': 75000,
                'time_of_day_preference': [11, 14],
                'volatility_sensitivity': 0.4,
                'momentum_following': 0.3
            }
        }
    
    def detect_institutional_flows(self, trade_data: pd.DataFrame,
                                 ticker: str) -> List[InstitutionalFlow]:
        """
        Detect institutional flows from trade data
        
        Args:
            trade_data: DataFrame with columns: timestamp, price, volume, trade_type
            ticker: Stock ticker
            
        Returns:
            List of detected institutional flows
        """
        flows = []
        
        if len(trade_data) == 0:
            return flows
        
        # Identify block trades
        block_trades = self._identify_block_trades(trade_data)
        
        # Classify institution types
        for trade in block_trades:
            institution_type = self._classify_institution_type(trade, trade_data)
            flow_type = self._determine_flow_type(trade, trade_data)
            direction = self._determine_flow_direction(trade, trade_data)
            
            flow = InstitutionalFlow(
                timestamp=trade['timestamp'],
                ticker=ticker,
                flow_type=flow_type,
                direction=direction,
                volume=trade['volume'],
                notional_value=trade['price'] * trade['volume'],
                price_impact=self._calculate_price_impact(trade, trade_data),
                confidence=self._calculate_confidence(trade, trade_data),
                estimated_institution_type=institution_type
            )
            
            flows.append(flow)
        
        return flows
    
    def _identify_block_trades(self, trade_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify block trades using statistical methods"""
        if len(trade_data) == 0:
            return []
        
        # Calculate volume statistics
        volume_mean = trade_data['volume'].mean()
        volume_std = trade_data['volume'].std()
        
        # Block trade threshold: either absolute threshold or statistical outlier
        statistical_threshold = volume_mean + 2 * volume_std
        threshold = max(self.block_trade_threshold, statistical_threshold)
        
        # Identify block trades
        block_trades = trade_data[trade_data['volume'] >= threshold].copy()
        
        # Convert to list of dictionaries
        block_trade_list = []
        for _, trade in block_trades.iterrows():
            block_trade_list.append({
                'timestamp': trade['timestamp'],
                'price': trade['price'],
                'volume': trade['volume'],
                'trade_type': trade.get('trade_type', 'unknown')
            })
        
        return block_trade_list
    
    def _classify_institution_type(self, trade: Dict[str, Any], 
                                 context_data: pd.DataFrame) -> Optional[InstitutionType]:
        """Classify the likely institution type based on trade characteristics"""
        
        trade_hour = trade['timestamp'].hour + trade['timestamp'].minute / 60.0
        trade_size = trade['volume']
        
        # Calculate features for classification
        features = {
            'trade_size': trade_size,
            'time_of_day': trade_hour,
            'volatility_context': self._calculate_volatility_context(trade, context_data),
            'momentum_context': self._calculate_momentum_context(trade, context_data)
        }
        
        # Score against each institution type
        scores = {}
        for inst_type, signature in self.institution_signatures.items():
            score = self._calculate_institution_score(features, signature)
            scores[inst_type] = score
        
        # Return best match if above threshold
        best_type = max(scores.keys(), key=lambda k: scores[k])
        if scores[best_type] > self.institution_confidence_threshold:
            return best_type
        
        return None
    
    def _calculate_institution_score(self, features: Dict[str, float],
                                   signature: Dict[str, Any]) -> float:
        """Calculate similarity score to institution signature"""
        
        # Trade size similarity
        size_diff = abs(features['trade_size'] - signature['avg_trade_size']) / signature['avg_trade_size']
        size_score = max(0, 1 - size_diff)
        
        # Time preference similarity
        time_pref = signature['time_of_day_preference']
        if time_pref[0] <= features['time_of_day'] <= time_pref[1]:
            time_score = 1.0
        else:
            time_score = 0.5
        
        # Volatility sensitivity match
        vol_diff = abs(features['volatility_context'] - signature['volatility_sensitivity'])
        vol_score = max(0, 1 - vol_diff)
        
        # Momentum following match
        momentum_diff = abs(features['momentum_context'] - signature['momentum_following'])
        momentum_score = max(0, 1 - momentum_diff)
        
        # Weighted average
        total_score = (
            size_score * 0.4 +
            time_score * 0.2 +
            vol_score * 0.2 +
            momentum_score * 0.2
        )
        
        return total_score
    
    def _calculate_volatility_context(self, trade: Dict[str, Any],
                                    context_data: pd.DataFrame) -> float:
        """Calculate volatility context at time of trade"""
        # Get recent volatility
        recent_returns = context_data['price'].pct_change().dropna().tail(20)
        if len(recent_returns) > 1:
            volatility = recent_returns.std()
            # Normalize to 0-1 range (assuming 0-5% daily vol range)
            return min(1.0, volatility / 0.05)
        return 0.5
    
    def _calculate_momentum_context(self, trade: Dict[str, Any],
                                  context_data: pd.DataFrame) -> float:
        """Calculate momentum context at time of trade"""
        # Calculate recent price momentum
        if len(context_data) > 10:
            recent_prices = context_data['price'].tail(10)
            momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            # Normalize to 0-1 range
            return max(0, min(1, (momentum + 0.1) / 0.2))  # Assume -10% to +10% range
        return 0.5
    
    def _determine_flow_type(self, trade: Dict[str, Any],
                           context_data: pd.DataFrame) -> FlowType:
        """Determine the type of flow based on trade characteristics"""
        
        # Large trades are likely institutional
        if trade['volume'] > 50000:
            return FlowType.INSTITUTIONAL
        
        # Check for algorithmic patterns
        if self._is_algorithmic_trade(trade, context_data):
            return FlowType.ALGORITHMIC
        
        # Check for dark pool characteristics
        if self._is_dark_pool_trade(trade, context_data):
            return FlowType.DARK_POOL
        
        # Default to retail for smaller trades
        return FlowType.RETAIL
    
    def _is_algorithmic_trade(self, trade: Dict[str, Any],
                           context_data: pd.DataFrame) -> bool:
        """Detect if trade has algorithmic characteristics"""
        # Simple heuristic: regular timing patterns or round lot sizes
        volume = trade['volume']
        
        # Check for round lots (multiples of 100)
        if volume % 100 == 0 and volume <= 10000:
            return True
        
        # Check for very precise timing (would need more data)
        return False
    
    def _is_dark_pool_trade(self, trade: Dict[str, Any],
                          context_data: pd.DataFrame) -> bool:
        """Detect if trade likely came from dark pool"""
        # Heuristic: large trade with minimal price impact
        price_impact = self._calculate_price_impact(trade, context_data)
        
        # Large volume but small price impact suggests dark pool
        if trade['volume'] > 25000 and abs(price_impact) < 0.001:
            return True
        
        return False
    
    def _determine_flow_direction(self, trade: Dict[str, Any],
                                context_data: pd.DataFrame) -> FlowDirection:
        """Determine if flow is inflow or outflow"""
        # Use Lee-Ready algorithm or similar
        # For simplicity, use trade type if available
        trade_type = trade.get('trade_type', 'unknown')
        
        if trade_type == 'buy' or trade_type == 'market_buy':
            return FlowDirection.INFLOW
        elif trade_type == 'sell' or trade_type == 'market_sell':
            return FlowDirection.OUTFLOW
        else:
            # Use price relative to recent midpoint
            if len(context_data) > 0:
                recent_price = context_data['price'].iloc[-1]
                if trade['price'] > recent_price:
                    return FlowDirection.INFLOW
                elif trade['price'] < recent_price:
                    return FlowDirection.OUTFLOW
        
        return FlowDirection.NEUTRAL
    
    def _calculate_price_impact(self, trade: Dict[str, Any],
                              context_data: pd.DataFrame) -> float:
        """Calculate price impact of the trade"""
        if len(context_data) < 2:
            return 0.0
        
        # Get price before and after trade
        trade_time = trade['timestamp']
        
        # Find closest price data points
        before_price = None
        after_price = None
        
        for i, row in context_data.iterrows():
            if row['timestamp'] <= trade_time:
                before_price = row['price']
            elif row['timestamp'] > trade_time and after_price is None:
                after_price = row['price']
                break
        
        if before_price and after_price:
            return (after_price - before_price) / before_price
        
        return 0.0
    
    def _calculate_confidence(self, trade: Dict[str, Any],
                            context_data: pd.DataFrame) -> float:
        """Calculate confidence in flow classification"""
        base_confidence = 0.5
        
        # Higher confidence for larger trades
        if trade['volume'] > 100000:
            base_confidence += 0.3
        elif trade['volume'] > 50000:
            base_confidence += 0.2
        
        # Higher confidence if we have more context data
        if len(context_data) > 100:
            base_confidence += 0.1
        
        # Higher confidence if trade has clear directional signal
        price_impact = abs(self._calculate_price_impact(trade, context_data))
        if price_impact > 0.005:  # 0.5% impact
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def detect_flow_patterns(self, flows: List[InstitutionalFlow],
                           ticker: str) -> List[FlowPattern]:
        """Detect flow patterns from institutional flows"""
        patterns = []
        
        if len(flows) < 5:  # Need minimum flows for pattern detection
            return patterns
        
        # Sort flows by timestamp
        sorted_flows = sorted(flows, key=lambda f: f.timestamp)
        
        # Detect accumulation patterns
        accumulation_pattern = self._detect_accumulation_pattern(sorted_flows, ticker)
        if accumulation_pattern:
            patterns.append(accumulation_pattern)
        
        # Detect distribution patterns
        distribution_pattern = self._detect_distribution_pattern(sorted_flows, ticker)
        if distribution_pattern:
            patterns.append(distribution_pattern)
        
        # Detect rotation patterns
        rotation_pattern = self._detect_rotation_pattern(sorted_flows, ticker)
        if rotation_pattern:
            patterns.append(rotation_pattern)
        
        return patterns
    
    def _detect_accumulation_pattern(self, flows: List[InstitutionalFlow],
                                   ticker: str) -> Optional[FlowPattern]:
        """Detect accumulation pattern"""
        # Look for sustained inflows over time
        inflows = [f for f in flows if f.direction == FlowDirection.INFLOW]
        
        if len(inflows) < 3:
            return None
        
        # Check for clustering in time
        time_span = (inflows[-1].timestamp - inflows[0].timestamp).total_seconds()
        if time_span > 86400 * 7:  # More than a week
            return None
        
        # Calculate pattern strength
        total_inflow = sum(f.notional_value for f in inflows)
        total_outflow = sum(f.notional_value for f in flows if f.direction == FlowDirection.OUTFLOW)
        
        if total_inflow > total_outflow * 2:  # Strong accumulation
            pattern_id = f"acc_{ticker}_{int(inflows[0].timestamp.timestamp())}"
            
            return FlowPattern(
                pattern_id=pattern_id,
                pattern_type="accumulation",
                ticker=ticker,
                start_time=inflows[0].timestamp,
                duration=timedelta(seconds=time_span),
                strength=min(1.0, total_inflow / (total_inflow + total_outflow)),
                confidence=0.7,
                associated_flows=inflows,
                key_characteristics={
                    "total_inflow": total_inflow,
                    "inflow_count": len(inflows),
                    "avg_trade_size": total_inflow / len(inflows)
                }
            )
        
        return None
    
    def _detect_distribution_pattern(self, flows: List[InstitutionalFlow],
                                   ticker: str) -> Optional[FlowPattern]:
        """Detect distribution pattern"""
        # Look for sustained outflows over time
        outflows = [f for f in flows if f.direction == FlowDirection.OUTFLOW]
        
        if len(outflows) < 3:
            return None
        
        # Similar logic to accumulation but for outflows
        time_span = (outflows[-1].timestamp - outflows[0].timestamp).total_seconds()
        if time_span > 86400 * 7:  # More than a week
            return None
        
        total_outflow = sum(f.notional_value for f in outflows)
        total_inflow = sum(f.notional_value for f in flows if f.direction == FlowDirection.INFLOW)
        
        if total_outflow > total_inflow * 2:  # Strong distribution
            pattern_id = f"dist_{ticker}_{int(outflows[0].timestamp.timestamp())}"
            
            return FlowPattern(
                pattern_id=pattern_id,
                pattern_type="distribution",
                ticker=ticker,
                start_time=outflows[0].timestamp,
                duration=timedelta(seconds=time_span),
                strength=min(1.0, total_outflow / (total_inflow + total_outflow)),
                confidence=0.7,
                associated_flows=outflows,
                key_characteristics={
                    "total_outflow": total_outflow,
                    "outflow_count": len(outflows),
                    "avg_trade_size": total_outflow / len(outflows)
                }
            )
        
        return None
    
    def _detect_rotation_pattern(self, flows: List[InstitutionalFlow],
                               ticker: str) -> Optional[FlowPattern]:
        """Detect sector/style rotation pattern"""
        # Look for alternating inflows and outflows suggesting rotation
        if len(flows) < 4:
            return None
        
        # Check for alternating pattern
        direction_changes = 0
        for i in range(1, len(flows)):
            if flows[i].direction != flows[i-1].direction:
                direction_changes += 1
        
        # High frequency of direction changes suggests rotation
        if direction_changes / len(flows) > 0.5:
            pattern_id = f"rot_{ticker}_{int(flows[0].timestamp.timestamp())}"
            
            return FlowPattern(
                pattern_id=pattern_id,
                pattern_type="rotation",
                ticker=ticker,
                start_time=flows[0].timestamp,
                duration=flows[-1].timestamp - flows[0].timestamp,
                strength=direction_changes / len(flows),
                confidence=0.6,
                associated_flows=flows,
                key_characteristics={
                    "direction_changes": direction_changes,
                    "alternation_frequency": direction_changes / len(flows)
                }
            )
        
        return None
