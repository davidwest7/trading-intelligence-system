"""
Market Microstructure Analysis for HFT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

class MarketMicrostructure:
    """
    Market microstructure analysis for high-frequency trading
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'tick_size': 0.01,
            'min_spread': 0.001,
            'max_depth_levels': 10,
            'volume_threshold': 1000
        }
        
        self.order_book_cache = {}
        self.microstructure_metrics = {}
        
    def calculate_spread(self, orderbook: Dict[str, Any]) -> float:
        """Calculate bid-ask spread"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            best_bid = max(bid[0] for bid in bids)
            best_ask = min(ask[0] for ask in asks)
            
            spread = best_ask - best_bid
            spread_pct = spread / best_bid if best_bid > 0 else 0.0
            
            return spread_pct
            
        except Exception as e:
            print(f"Error calculating spread: {e}")
            return 0.0
    
    def calculate_depth(self, orderbook: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market depth at different levels"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            # Calculate bid depth
            bid_depth = {}
            for i, (price, size) in enumerate(bids[:self.config['max_depth_levels']]):
                level = f"bid_{i+1}"
                bid_depth[level] = {
                    'price': price,
                    'size': size,
                    'cumulative_size': sum(bid[1] for bid in bids[:i+1])
                }
            
            # Calculate ask depth
            ask_depth = {}
            for i, (price, size) in enumerate(asks[:self.config['max_depth_levels']]):
                level = f"ask_{i+1}"
                ask_depth[level] = {
                    'price': price,
                    'size': size,
                    'cumulative_size': sum(ask[1] for ask in asks[:i+1])
                }
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_bid_depth': sum(bid[1] for bid in bids),
                'total_ask_depth': sum(ask[1] for ask in asks)
            }
            
        except Exception as e:
            print(f"Error calculating depth: {e}")
            return {}
    
    def calculate_imbalance(self, orderbook: Dict[str, Any]) -> float:
        """Calculate order book imbalance"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            # Calculate volume-weighted imbalance
            bid_volume = sum(bid[1] for bid in bids[:5])  # Top 5 levels
            ask_volume = sum(ask[1] for ask in asks[:5])  # Top 5 levels
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            
            imbalance = (bid_volume - ask_volume) / total_volume
            return imbalance
            
        except Exception as e:
            print(f"Error calculating imbalance: {e}")
            return 0.0
    
    def calculate_market_impact(self, order_size: int, orderbook: Dict[str, Any]) -> float:
        """Calculate expected market impact of an order"""
        try:
            depth = self.calculate_depth(orderbook)
            
            # Calculate how much of the orderbook would be consumed
            ask_depth = depth.get('ask_depth', {})
            cumulative_ask_size = 0
            impact_levels = 0
            
            for level, data in ask_depth.items():
                cumulative_ask_size += data['size']
                impact_levels += 1
                
                if cumulative_ask_size >= order_size:
                    break
            
            # Calculate average price impact
            if impact_levels > 0:
                avg_impact = impact_levels * self.config['tick_size']
                return avg_impact
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating market impact: {e}")
            return 0.0
    
    def analyze_microstructure(self, orderbook: Dict[str, Any], trades: List[Dict] = None) -> Dict[str, Any]:
        """Comprehensive microstructure analysis"""
        try:
            # Calculate basic metrics
            spread = self.calculate_spread(orderbook)
            depth = self.calculate_depth(orderbook)
            imbalance = self.calculate_imbalance(orderbook)
            
            # Calculate additional metrics
            metrics = {
                'spread': spread,
                'spread_bps': spread * 10000,  # Basis points
                'depth': depth,
                'imbalance': imbalance,
                'timestamp': datetime.now()
            }
            
            # Add trade analysis if available
            if trades:
                trade_metrics = self._analyze_trades(trades)
                metrics.update(trade_metrics)
            
            # Calculate market quality indicators
            metrics['market_quality'] = self._calculate_market_quality(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing microstructure: {e}")
            return {}
    
    def _analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze trade patterns"""
        try:
            if not trades:
                return {}
            
            # Calculate trade statistics
            trade_sizes = [trade.get('size', 0) for trade in trades]
            trade_prices = [trade.get('price', 0) for trade in trades]
            
            metrics = {
                'avg_trade_size': np.mean(trade_sizes) if trade_sizes else 0,
                'median_trade_size': np.median(trade_sizes) if trade_sizes else 0,
                'trade_size_std': np.std(trade_sizes) if trade_sizes else 0,
                'price_volatility': np.std(trade_prices) if trade_prices else 0,
                'trade_count': len(trades),
                'large_trades': sum(1 for size in trade_sizes if size > self.config['volume_threshold'])
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing trades: {e}")
            return {}
    
    def _calculate_market_quality(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market quality indicators"""
        try:
            spread = metrics.get('spread', 0)
            imbalance = abs(metrics.get('imbalance', 0))
            
            # Market quality score (lower is better)
            quality_score = spread * 100 + imbalance * 50
            
            # Liquidity score
            total_depth = metrics.get('depth', {}).get('total_bid_depth', 0) + \
                         metrics.get('depth', {}).get('total_ask_depth', 0)
            liquidity_score = min(1.0, total_depth / 1000000)  # Normalize to 1M shares
            
            return {
                'quality_score': quality_score,
                'liquidity_score': liquidity_score,
                'efficiency_score': max(0, 1 - quality_score),
                'stability_score': max(0, 1 - imbalance)
            }
            
        except Exception as e:
            print(f"Error calculating market quality: {e}")
            return {}
    
    def get_microstructure_summary(self, symbol: str) -> Dict[str, Any]:
        """Get microstructure summary for a symbol"""
        try:
            metrics = self.microstructure_metrics.get(symbol, {})
            
            return {
                'symbol': symbol,
                'avg_spread': metrics.get('avg_spread', 0),
                'avg_depth': metrics.get('avg_depth', 0),
                'avg_imbalance': metrics.get('avg_imbalance', 0),
                'market_quality': metrics.get('market_quality', {}),
                'last_updated': metrics.get('timestamp', datetime.now())
            }
            
        except Exception as e:
            print(f"Error getting microstructure summary: {e}")
            return {}
