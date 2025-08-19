"""
LOB (Limit Order Book) and Microstructure Features
Advanced order book analysis and market microstructure features
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import math


class OrderSide(str, Enum):
    """Order side enumeration"""
    BID = "bid"
    ASK = "ask"


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    size: int
    side: OrderSide
    timestamp: datetime
    venue: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "venue": self.venue
        }


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_trade_price: float
    last_trade_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bids": [bid.to_dict() for bid in self.bids],
            "asks": [ask.to_dict() for ask in self.asks],
            "last_trade_price": self.last_trade_price,
            "last_trade_size": self.last_trade_size
        }


class LOBFeatureExtractor:
    """
    Advanced LOB feature extraction for market microstructure analysis
    
    Features:
    - Order book imbalance metrics
    - Price impact estimation
    - Liquidity measures
    - Order flow analysis
    - Market impact modeling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration
        self.max_levels = self.config.get('max_levels', 10)
        self.impact_horizon = self.config.get('impact_horizon', 5)  # minutes
        self.liquidity_threshold = self.config.get('liquidity_threshold', 1000)
        
        # Storage
        self.order_book_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.trade_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def extract_lob_features(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract comprehensive LOB features"""
        try:
            features = {}
            
            # Basic order book metrics
            features.update(await self._extract_basic_metrics(order_book))
            
            # Imbalance features
            features.update(await self._extract_imbalance_features(order_book))
            
            # Liquidity features
            features.update(await self._extract_liquidity_features(order_book))
            
            # Price impact features
            features.update(await self._extract_price_impact_features(order_book))
            
            # Order flow features
            features.update(await self._extract_order_flow_features(order_book))
            
            # Microstructure features
            features.update(await self._extract_microstructure_features(order_book))
            
            # Store for history
            self.order_book_history[order_book.symbol].append(order_book)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting LOB features: {e}")
            return {}
    
    async def _extract_basic_metrics(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract basic order book metrics"""
        try:
            if not order_book.bids or not order_book.asks:
                return {}
            
            best_bid = order_book.bids[0].price
            best_ask = order_book.asks[0].price
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            
            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread": spread,
                "spread_bps": spread_bps,
                "bid_size": order_book.bids[0].size,
                "ask_size": order_book.asks[0].size,
                "bid_ask_ratio": order_book.bids[0].size / order_book.asks[0].size if order_book.asks[0].size > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting basic metrics: {e}")
            return {}
    
    async def _extract_imbalance_features(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract order book imbalance features"""
        try:
            features = {}
            
            # Aggregate imbalance across levels
            total_bid_size = sum(bid.size for bid in order_book.bids[:self.max_levels])
            total_ask_size = sum(ask.size for ask in order_book.asks[:self.max_levels])
            
            if total_bid_size + total_ask_size > 0:
                imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
                features["order_imbalance"] = imbalance
                features["order_imbalance_abs"] = abs(imbalance)
            
            # Level-by-level imbalance
            level_imbalances = []
            for i in range(min(len(order_book.bids), len(order_book.asks), self.max_levels)):
                bid_size = order_book.bids[i].size
                ask_size = order_book.asks[i].size
                total_size = bid_size + ask_size
                
                if total_size > 0:
                    level_imb = (bid_size - ask_size) / total_size
                    level_imbalances.append(level_imb)
            
            if level_imbalances:
                features["level_imbalance_mean"] = np.mean(level_imbalances)
                features["level_imbalance_std"] = np.std(level_imbalances)
                features["level_imbalance_max"] = max(level_imbalances)
                features["level_imbalance_min"] = min(level_imbalances)
            
            # Price-weighted imbalance
            bid_value = sum(bid.price * bid.size for bid in order_book.bids[:self.max_levels])
            ask_value = sum(ask.price * ask.size for ask in order_book.asks[:self.max_levels])
            
            if bid_value + ask_value > 0:
                value_imbalance = (bid_value - ask_value) / (bid_value + ask_value)
                features["value_imbalance"] = value_imbalance
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting imbalance features: {e}")
            return {}
    
    async def _extract_liquidity_features(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract liquidity measures"""
        try:
            features = {}
            
            # Depth at different levels
            for level in range(1, min(self.max_levels + 1, 6)):
                if len(order_book.bids) >= level and len(order_book.asks) >= level:
                    bid_depth = sum(bid.size for bid in order_book.bids[:level])
                    ask_depth = sum(ask.size for ask in order_book.asks[:level])
                    features[f"bid_depth_{level}"] = bid_depth
                    features[f"ask_depth_{level}"] = ask_depth
                    features[f"total_depth_{level}"] = bid_depth + ask_depth
            
            # Liquidity concentration
            if order_book.bids and order_book.asks:
                bid_concentration = order_book.bids[0].size / sum(bid.size for bid in order_book.bids[:self.max_levels])
                ask_concentration = order_book.asks[0].size / sum(ask.size for ask in order_book.asks[:self.max_levels])
                features["bid_concentration"] = bid_concentration
                features["ask_concentration"] = ask_concentration
            
            # Large order detection
            large_bid_orders = sum(1 for bid in order_book.bids[:self.max_levels] 
                                 if bid.size > self.liquidity_threshold)
            large_ask_orders = sum(1 for ask in order_book.asks[:self.max_levels] 
                                 if ask.size > self.liquidity_threshold)
            features["large_bid_orders"] = large_bid_orders
            features["large_ask_orders"] = large_ask_orders
            features["large_orders_total"] = large_bid_orders + large_ask_orders
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting liquidity features: {e}")
            return {}
    
    async def _extract_price_impact_features(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract price impact estimation features"""
        try:
            features = {}
            
            if not order_book.bids or not order_book.asks:
                return features
            
            mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
            
            # Kyle's lambda estimation
            total_bid_size = sum(bid.size for bid in order_book.bids[:self.max_levels])
            total_ask_size = sum(ask.size for ask in order_book.asks[:self.max_levels])
            
            if total_bid_size > 0 and total_ask_size > 0:
                # Simplified Kyle's lambda
                kyle_lambda = (order_book.asks[0].price - order_book.bids[0].price) / (total_bid_size + total_ask_size)
                features["kyle_lambda"] = kyle_lambda
            
            # Price impact for different order sizes
            for size in [1000, 5000, 10000, 50000]:
                bid_impact = await self._estimate_buy_impact(order_book, size)
                ask_impact = await self._estimate_sell_impact(order_book, size)
                
                features[f"buy_impact_{size}"] = bid_impact
                features[f"sell_impact_{size}"] = ask_impact
                features[f"impact_skew_{size}"] = bid_impact - ask_impact
            
            # Market impact curve slope
            impacts = []
            sizes = [1000, 5000, 10000, 50000]
            for size in sizes:
                impact = await self._estimate_buy_impact(order_book, size)
                impacts.append(impact)
            
            if len(impacts) > 1:
                # Linear regression slope
                slope = np.polyfit(sizes, impacts, 1)[0]
                features["impact_curve_slope"] = slope
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting price impact features: {e}")
            return {}
    
    async def _extract_order_flow_features(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract order flow analysis features"""
        try:
            features = {}
            
            # Order flow imbalance
            if len(self.order_book_history[order_book.symbol]) > 1:
                prev_snapshot = self.order_book_history[order_book.symbol][-2]
                
                # Size changes
                bid_size_change = order_book.bids[0].size - prev_snapshot.bids[0].size
                ask_size_change = order_book.asks[0].size - prev_snapshot.asks[0].size
                
                features["bid_size_change"] = bid_size_change
                features["ask_size_change"] = ask_size_change
                features["size_change_imbalance"] = bid_size_change - ask_size_change
                
                # Price changes
                bid_price_change = order_book.bids[0].price - prev_snapshot.bids[0].price
                ask_price_change = order_book.asks[0].price - prev_snapshot.asks[0].price
                
                features["bid_price_change"] = bid_price_change
                features["ask_price_change"] = ask_price_change
                features["price_change_imbalance"] = bid_price_change - ask_price_change
            
            # Order book resilience
            if len(self.order_book_history[order_book.symbol]) > 5:
                recent_snapshots = list(self.order_book_history[order_book.symbol])[-5:]
                spread_changes = []
                
                for i in range(1, len(recent_snapshots)):
                    prev_spread = recent_snapshots[i-1].asks[0].price - recent_snapshots[i-1].bids[0].price
                    curr_spread = recent_snapshots[i].asks[0].price - recent_snapshots[i].bids[0].price
                    spread_changes.append(curr_spread - prev_spread)
                
                if spread_changes:
                    features["spread_volatility"] = np.std(spread_changes)
                    features["spread_trend"] = np.mean(spread_changes)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting order flow features: {e}")
            return {}
    
    async def _extract_microstructure_features(self, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Extract advanced microstructure features"""
        try:
            features = {}
            
            # Order book shape
            if len(order_book.bids) >= 5 and len(order_book.asks) >= 5:
                bid_prices = [bid.price for bid in order_book.bids[:5]]
                ask_prices = [ask.price for ask in order_book.asks[:5]]
                
                # Price curvature
                bid_curvature = await self._calculate_curvature(bid_prices)
                ask_curvature = await self._calculate_curvature(ask_prices)
                features["bid_curvature"] = bid_curvature
                features["ask_curvature"] = ask_curvature
                features["curvature_imbalance"] = bid_curvature - ask_curvature
            
            # Order book toxicity
            if len(self.order_book_history[order_book.symbol]) > 10:
                recent_snapshots = list(self.order_book_history[order_book.symbol])[-10:]
                price_reversals = 0
                
                for i in range(2, len(recent_snapshots)):
                    # Check for price reversals
                    prev_mid = (recent_snapshots[i-2].bids[0].price + recent_snapshots[i-2].asks[0].price) / 2
                    curr_mid = (recent_snapshots[i].bids[0].price + recent_snapshots[i].asks[0].price) / 2
                    
                    if (curr_mid - prev_mid) * (recent_snapshots[i-1].bids[0].price - prev_mid) < 0:
                        price_reversals += 1
                
                features["price_reversals"] = price_reversals
                features["toxicity_score"] = price_reversals / len(recent_snapshots)
            
            # Market efficiency
            if len(self.order_book_history[order_book.symbol]) > 20:
                recent_snapshots = list(self.order_book_history[order_book.symbol])[-20:]
                mid_prices = [(snap.bids[0].price + snap.asks[0].price) / 2 for snap in recent_snapshots]
                
                # Hurst exponent for mean reversion
                hurst = await self._calculate_hurst_exponent(mid_prices)
                features["hurst_exponent"] = hurst
                features["mean_reversion_strength"] = 1 - hurst if hurst < 0.5 else 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting microstructure features: {e}")
            return {}
    
    async def _estimate_buy_impact(self, order_book: OrderBookSnapshot, size: int) -> float:
        """Estimate price impact for a buy order of given size"""
        try:
            remaining_size = size
            total_cost = 0
            levels_used = 0
            
            for ask in order_book.asks[:self.max_levels]:
                if remaining_size <= 0:
                    break
                
                level_size = min(remaining_size, ask.size)
                total_cost += level_size * ask.price
                remaining_size -= level_size
                levels_used += 1
            
            if size - remaining_size > 0:
                avg_price = total_cost / (size - remaining_size)
                mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
                impact = (avg_price - mid_price) / mid_price
                return impact
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error estimating buy impact: {e}")
            return 0.0
    
    async def _estimate_sell_impact(self, order_book: OrderBookSnapshot, size: int) -> float:
        """Estimate price impact for a sell order of given size"""
        try:
            remaining_size = size
            total_cost = 0
            levels_used = 0
            
            for bid in order_book.bids[:self.max_levels]:
                if remaining_size <= 0:
                    break
                
                level_size = min(remaining_size, bid.size)
                total_cost += level_size * bid.price
                remaining_size -= level_size
                levels_used += 1
            
            if size - remaining_size > 0:
                avg_price = total_cost / (size - remaining_size)
                mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
                impact = (mid_price - avg_price) / mid_price
                return impact
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error estimating sell impact: {e}")
            return 0.0
    
    async def _calculate_curvature(self, prices: List[float]) -> float:
        """Calculate price curvature"""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Second derivative approximation
            curvature = prices[0] - 2 * prices[1] + prices[2]
            return curvature
            
        except Exception as e:
            self.logger.error(f"Error calculating curvature: {e}")
            return 0.0
    
    async def _calculate_hurst_exponent(self, prices: List[float]) -> float:
        """Calculate Hurst exponent for mean reversion detection"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Simplified Hurst calculation
            returns = np.diff(np.log(prices))
            if len(returns) < 2:
                return 0.5
            
            # Variance ratio test
            var_1 = np.var(returns)
            var_2 = np.var(returns[::2])
            
            if var_1 > 0 and var_2 > 0:
                hurst = 0.5 * np.log2(var_2 / var_1)
                return max(0.1, min(0.9, hurst))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5


# Factory function for easy integration
async def create_lob_extractor(config: Optional[Dict[str, Any]] = None) -> LOBFeatureExtractor:
    """Create and initialize LOB feature extractor"""
    return LOBFeatureExtractor(config)
