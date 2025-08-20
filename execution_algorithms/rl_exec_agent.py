#!/usr/bin/env python3
"""
Reinforcement Learning Execution Agent
Microstructure RL with LOB state, venue selection, queue modeling, short-term alpha co-optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from enum import Enum
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    HIDDEN = "hidden"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class VenueType(Enum):
    """Venue types"""
    PRIMARY = "primary"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    CROSSING_NETWORK = "crossing_network"


@dataclass
class OrderBookLevel:
    """Order book level"""
    price: float
    size: int
    venue: str
    timestamp: datetime
    side: OrderSide


@dataclass
class LOBState:
    """Limit Order Book state representation"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_trade_price: float
    last_trade_size: int
    bid_ask_spread: float
    mid_price: float
    imbalance: float
    depth_ratio: float
    volatility: float
    momentum: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert LOB state to feature vector for RL"""
        features = []
        
        # Price features
        features.extend([
            self.last_trade_price,
            self.mid_price,
            self.bid_ask_spread,
            self.bid_ask_spread / self.mid_price if self.mid_price > 0 else 0
        ])
        
        # Depth features (top 5 levels)
        bid_prices = [level.price for level in self.bids[:5]]
        bid_sizes = [level.size for level in self.bids[:5]]
        ask_prices = [level.price for level in self.asks[:5]]
        ask_sizes = [level.size for level in self.asks[:5]]
        
        # Pad to 5 levels if necessary
        while len(bid_prices) < 5:
            bid_prices.append(0)
            bid_sizes.append(0)
        while len(ask_prices) < 5:
            ask_prices.append(0)
            ask_sizes.append(0)
        
        features.extend(bid_prices + bid_sizes + ask_prices + ask_sizes)
        
        # Market microstructure features
        features.extend([
            self.imbalance,
            self.depth_ratio,
            self.volatility,
            self.momentum
        ])
        
        return np.array(features, dtype=float)


@dataclass
class VenueState:
    """Venue-specific state"""
    venue_id: str
    venue_type: VenueType
    market_share: float
    average_fill_rate: float
    latency_ms: float
    fee_structure: Dict[str, float]
    queue_position: int
    estimated_queue_length: int
    recent_fill_probability: float
    adverse_selection_cost: float


@dataclass
class ExecutionAction:
    """Execution action taken by RL agent"""
    action_id: str
    order_type: OrderType
    venue: str
    quantity: int
    price: Optional[float]
    time_in_force: str
    hidden_quantity: int = 0
    min_quantity: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionReward:
    """Reward signal for RL training"""
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    fee_cost: float
    opportunity_cost: float
    total_reward: float
    fill_rate: float
    adverse_selection: float


@dataclass
class ExecutionOrder:
    """Order to be executed"""
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: int
    remaining_quantity: int
    urgency: float  # 0 = no urgency, 1 = immediate
    alpha_signal: float  # Short-term alpha signal
    max_participation: float  # Maximum market participation rate
    start_time: datetime
    end_time: datetime
    benchmark_price: float
    risk_aversion: float = 0.5


class QueueModel:
    """Model queue positions and fill probabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.queue_history = {}
        self.fill_models = {}
    
    def estimate_queue_position(self, venue: str, price: float, 
                              lob_state: LOBState, order_size: int) -> int:
        """Estimate queue position for a limit order"""
        try:
            # Find the relevant price level
            if price <= lob_state.mid_price:  # Buy order
                relevant_levels = [level for level in lob_state.bids if level.price == price]
            else:  # Sell order
                relevant_levels = [level for level in lob_state.asks if level.price == price]
            
            if relevant_levels:
                # Estimate position based on existing size at this level
                total_size_at_level = sum(level.size for level in relevant_levels)
                # Assume FIFO queue - our position is at the end
                estimated_position = total_size_at_level + order_size // 2
                return max(1, estimated_position)
            else:
                # New price level - we're first in queue
                return 1
                
        except Exception as e:
            self.logger.error(f"Error estimating queue position: {e}")
            return 1
    
    def estimate_fill_probability(self, venue: str, queue_position: int,
                                order_size: int, time_horizon: int,
                                lob_state: LOBState) -> float:
        """Estimate probability of fill within time horizon"""
        try:
            # Simple model based on queue position and market activity
            
            # Base fill rate depends on market activity (proxy: recent volume)
            base_fill_rate = min(0.8, lob_state.last_trade_size / 10000)  # Normalize
            
            # Adjust for queue position (exponential decay)
            position_adjustment = np.exp(-queue_position / 1000)
            
            # Adjust for order size (larger orders harder to fill)
            size_adjustment = np.exp(-order_size / 5000)
            
            # Adjust for time horizon
            time_adjustment = min(1.0, time_horizon / 60)  # 60 seconds for full probability
            
            fill_probability = base_fill_rate * position_adjustment * size_adjustment * time_adjustment
            
            return np.clip(fill_probability, 0.01, 0.99)
            
        except Exception as e:
            self.logger.error(f"Error estimating fill probability: {e}")
            return 0.5
    
    def estimate_adverse_selection(self, price: float, lob_state: LOBState,
                                 alpha_signal: float) -> float:
        """Estimate adverse selection cost"""
        try:
            # Distance from mid price
            price_distance = abs(price - lob_state.mid_price) / lob_state.mid_price
            
            # Alpha signal strength (negative alpha means we're picking off stale quotes)
            alpha_component = max(0, -alpha_signal) * 0.1
            
            # Market volatility component
            volatility_component = lob_state.volatility * 0.05
            
            adverse_selection = price_distance + alpha_component + volatility_component
            
            return np.clip(adverse_selection, 0, 0.02)  # Max 2% adverse selection
            
        except Exception as e:
            self.logger.error(f"Error estimating adverse selection: {e}")
            return 0.001


class VenueSelector:
    """Select optimal venue for order execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.venue_performance = {}
        
        # Initialize venue characteristics
        self.venues = {
            'NYSE': VenueState('NYSE', VenueType.PRIMARY, 0.25, 0.85, 2.0, 
                             {'taker': 0.003, 'maker': -0.001}, 0, 100, 0.8, 0.001),
            'NASDAQ': VenueState('NASDAQ', VenueType.PRIMARY, 0.23, 0.82, 1.8,
                               {'taker': 0.003, 'maker': -0.001}, 0, 120, 0.78, 0.0012),
            'BATS': VenueState('BATS', VenueType.ECN, 0.12, 0.75, 1.5,
                             {'taker': 0.002, 'maker': -0.002}, 0, 80, 0.72, 0.0008),
            'EDGX': VenueState('EDGX', VenueType.ECN, 0.08, 0.70, 1.7,
                             {'taker': 0.0025, 'maker': -0.0015}, 0, 60, 0.68, 0.0009),
            'DARK1': VenueState('DARK1', VenueType.DARK_POOL, 0.05, 0.60, 3.0,
                              {'taker': 0.001, 'maker': 0.001}, 0, 200, 0.55, 0.0005)
        }
    
    def select_venue(self, execution_order: ExecutionOrder, lob_state: LOBState,
                    action: ExecutionAction) -> str:
        """Select optimal venue for execution"""
        try:
            venue_scores = {}
            
            for venue_id, venue_state in self.venues.items():
                score = self._calculate_venue_score(
                    venue_state, execution_order, lob_state, action
                )
                venue_scores[venue_id] = score
            
            # Select venue with highest score
            best_venue = max(venue_scores, key=venue_scores.get)
            
            self.logger.debug(f"Selected venue {best_venue} with score {venue_scores[best_venue]:.3f}")
            return best_venue
            
        except Exception as e:
            self.logger.error(f"Error selecting venue: {e}")
            return 'NYSE'  # Default fallback
    
    def _calculate_venue_score(self, venue_state: VenueState, 
                             execution_order: ExecutionOrder,
                             lob_state: LOBState, action: ExecutionAction) -> float:
        """Calculate score for a venue"""
        score = 0.0
        
        # Fill probability component
        fill_weight = 0.4
        score += fill_weight * venue_state.recent_fill_probability
        
        # Cost component (lower is better)
        cost_weight = 0.3
        if action.order_type == OrderType.MARKET:
            cost = venue_state.fee_structure.get('taker', 0.003)
        else:
            cost = venue_state.fee_structure.get('maker', -0.001)
        score -= cost_weight * cost * 1000  # Scale to reasonable range
        
        # Latency component (lower is better)
        latency_weight = 0.2
        score -= latency_weight * venue_state.latency_ms / 10.0
        
        # Market impact component
        impact_weight = 0.1
        market_share_bonus = venue_state.market_share * 2  # Higher market share = lower impact
        score += impact_weight * market_share_bonus
        
        return score


class RLExecutionAgent:
    """Reinforcement Learning execution agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # RL components
        self.q_table = {}  # Simple Q-learning table (would use neural network in production)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.epsilon = self.config.get('epsilon', 0.1)  # Exploration rate
        
        # Execution components
        self.queue_model = QueueModel()
        self.venue_selector = VenueSelector()
        
        # State and action spaces
        self.state_dim = 34  # LOB state feature vector size
        self.action_space = self._define_action_space()
        
        # Performance tracking
        self.execution_history = []
        self.performance_metrics = {}
        
        self.logger.info("Initialized RL Execution Agent")
    
    def _define_action_space(self) -> List[Dict[str, Any]]:
        """Define discrete action space for RL agent"""
        actions = []
        
        # Market orders
        actions.append({'type': OrderType.MARKET, 'aggression': 1.0})
        
        # Limit orders at different price levels
        for offset in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:  # bps from mid
            actions.append({
                'type': OrderType.LIMIT,
                'price_offset': offset,
                'aggression': max(0.1, 1.0 - abs(offset) * 10)
            })
        
        # Hidden orders
        for offset in [-0.01, 0, 0.01]:
            actions.append({
                'type': OrderType.HIDDEN,
                'price_offset': offset,
                'aggression': 0.5
            })
        
        # Iceberg orders
        for offset in [-0.005, 0, 0.005]:
            actions.append({
                'type': OrderType.ICEBERG,
                'price_offset': offset,
                'aggression': 0.3,
                'display_size': 0.1  # 10% of total size
            })
        
        return actions
    
    async def execute_order(self, execution_order: ExecutionOrder,
                          lob_state: LOBState) -> ExecutionAction:
        """Execute order using RL policy"""
        try:
            # Convert state to feature vector
            state_vector = self._create_state_vector(execution_order, lob_state)
            
            # Select action using epsilon-greedy policy
            action_idx = self._select_action(state_vector)
            action_spec = self.action_space[action_idx]
            
            # Create execution action
            action = self._create_execution_action(
                action_spec, execution_order, lob_state
            )
            
            # Select venue
            action.venue = self.venue_selector.select_venue(
                execution_order, lob_state, action
            )
            
            self.logger.info(f"RL agent selected action: {action.order_type} on {action.venue}")
            return action
            
        except Exception as e:
            self.logger.error(f"Error in RL execution: {e}")
            # Fallback to simple market order
            return ExecutionAction(
                action_id=f"fallback_{datetime.now().strftime('%H%M%S')}",
                order_type=OrderType.MARKET,
                venue='NYSE',
                quantity=min(1000, execution_order.remaining_quantity),
                price=None,
                time_in_force='IOC'
            )
    
    def _create_state_vector(self, execution_order: ExecutionOrder,
                           lob_state: LOBState) -> np.ndarray:
        """Create state vector for RL agent"""
        # LOB features
        lob_features = lob_state.to_feature_vector()
        
        # Order features
        order_features = np.array([
            execution_order.remaining_quantity / 10000,  # Normalize
            execution_order.urgency,
            execution_order.alpha_signal,
            execution_order.max_participation,
            execution_order.risk_aversion,
            (datetime.now() - execution_order.start_time).total_seconds() / 3600  # Hours elapsed
        ])
        
        # Combine features
        state_vector = np.concatenate([lob_features, order_features])
        
        # Ensure fixed size
        if len(state_vector) < self.state_dim:
            state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]
        
        return state_vector
    
    def _select_action(self, state_vector: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        # Discretize state for Q-table lookup (simple approach)
        state_key = tuple(np.round(state_vector, 2))
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(len(self.action_space))
        else:
            # Exploit: best known action
            if state_key in self.q_table:
                q_values = self.q_table[state_key]
                return np.argmax(q_values)
            else:
                # Initialize Q-values for new state
                self.q_table[state_key] = np.zeros(len(self.action_space))
                return np.random.randint(len(self.action_space))
    
    def _create_execution_action(self, action_spec: Dict[str, Any],
                               execution_order: ExecutionOrder,
                               lob_state: LOBState) -> ExecutionAction:
        """Create execution action from action specification"""
        # Calculate order quantity based on urgency and participation limits
        max_size = min(
            execution_order.remaining_quantity,
            int(lob_state.last_trade_size * execution_order.max_participation),
            10000  # Hard limit
        )
        
        # Adjust size based on action aggression
        aggression = action_spec.get('aggression', 0.5)
        order_quantity = max(100, int(max_size * aggression))
        
        # Calculate price for limit orders
        price = None
        if action_spec['type'] in [OrderType.LIMIT, OrderType.HIDDEN, OrderType.ICEBERG]:
            price_offset = action_spec.get('price_offset', 0)
            if execution_order.side == OrderSide.BUY:
                price = lob_state.mid_price * (1 + price_offset)
            else:
                price = lob_state.mid_price * (1 - price_offset)
            
            # Round to tick size (assuming $0.01)
            price = round(price, 2)
        
        # Handle special order types
        hidden_quantity = 0
        if action_spec['type'] == OrderType.HIDDEN:
            hidden_quantity = order_quantity
        elif action_spec['type'] == OrderType.ICEBERG:
            display_size = action_spec.get('display_size', 0.1)
            hidden_quantity = int(order_quantity * (1 - display_size))
            order_quantity = int(order_quantity * display_size)
        
        return ExecutionAction(
            action_id=f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            order_type=action_spec['type'],
            venue='',  # Will be set by venue selector
            quantity=order_quantity,
            price=price,
            time_in_force='GTC' if action_spec['type'] != OrderType.MARKET else 'IOC',
            hidden_quantity=hidden_quantity
        )
    
    async def update_policy(self, state_vector: np.ndarray, action_idx: int,
                          reward: float, next_state_vector: np.ndarray) -> None:
        """Update RL policy with experience"""
        try:
            # Discretize states
            state_key = tuple(np.round(state_vector, 2))
            next_state_key = tuple(np.round(next_state_vector, 2))
            
            # Initialize Q-values if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(len(self.action_space))
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(len(self.action_space))
            
            # Q-learning update
            current_q = self.q_table[state_key][action_idx]
            max_next_q = np.max(self.q_table[next_state_key])
            
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            
            self.q_table[state_key][action_idx] = new_q
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.9995)
            
            self.logger.debug(f"Updated Q-value: {current_q:.4f} -> {new_q:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error updating RL policy: {e}")
    
    def calculate_reward(self, execution_order: ExecutionOrder,
                        action: ExecutionAction, fill_price: float,
                        fill_quantity: int, lob_state: LOBState) -> ExecutionReward:
        """Calculate reward signal for RL training"""
        try:
            # Implementation shortfall
            if execution_order.side == OrderSide.BUY:
                implementation_shortfall = (fill_price - execution_order.benchmark_price) / execution_order.benchmark_price
            else:
                implementation_shortfall = (execution_order.benchmark_price - fill_price) / execution_order.benchmark_price
            
            # Market impact (simplified)
            market_impact = abs(fill_price - lob_state.mid_price) / lob_state.mid_price
            
            # Timing cost (based on alpha signal)
            timing_cost = -execution_order.alpha_signal * implementation_shortfall
            
            # Fee cost
            venue_state = self.venue_selector.venues.get(action.venue)
            if venue_state:
                if action.order_type == OrderType.MARKET:
                    fee_rate = venue_state.fee_structure.get('taker', 0.003)
                else:
                    fee_rate = venue_state.fee_structure.get('maker', -0.001)
                fee_cost = fee_rate * fill_quantity * fill_price / 10000  # bps
            else:
                fee_cost = 0.003  # Default taker fee
            
            # Opportunity cost (unfilled quantity)
            unfilled_ratio = (execution_order.remaining_quantity - fill_quantity) / execution_order.remaining_quantity
            opportunity_cost = unfilled_ratio * abs(execution_order.alpha_signal) * 0.01
            
            # Adverse selection cost
            adverse_selection = self.queue_model.estimate_adverse_selection(
                fill_price, lob_state, execution_order.alpha_signal
            )
            
            # Total reward (negative of total cost)
            total_cost = (
                abs(implementation_shortfall) +
                market_impact +
                abs(timing_cost) +
                abs(fee_cost) +
                opportunity_cost +
                adverse_selection
            )
            
            total_reward = -total_cost  # Negative cost becomes positive reward
            
            # Fill rate bonus
            fill_rate = fill_quantity / action.quantity if action.quantity > 0 else 0
            fill_bonus = fill_rate * 0.1  # Small bonus for filling orders
            total_reward += fill_bonus
            
            return ExecutionReward(
                implementation_shortfall=implementation_shortfall,
                market_impact=market_impact,
                timing_cost=timing_cost,
                fee_cost=fee_cost,
                opportunity_cost=opportunity_cost,
                total_reward=total_reward,
                fill_rate=fill_rate,
                adverse_selection=adverse_selection
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return ExecutionReward(
                implementation_shortfall=0.0,
                market_impact=0.0,
                timing_cost=0.0,
                fee_cost=0.0,
                opportunity_cost=0.0,
                total_reward=-0.01,  # Small penalty for errors
                fill_rate=0.0,
                adverse_selection=0.0
            )
    
    async def optimize_execution_schedule(self, execution_order: ExecutionOrder,
                                        historical_lob_data: pd.DataFrame) -> List[ExecutionAction]:
        """Optimize execution schedule using RL"""
        try:
            execution_schedule = []
            remaining_quantity = execution_order.remaining_quantity
            current_time = execution_order.start_time
            
            while remaining_quantity > 0 and current_time < execution_order.end_time:
                # Create current LOB state (would be real-time in production)
                lob_state = self._create_synthetic_lob_state(
                    execution_order.symbol, current_time, historical_lob_data
                )
                
                # Update remaining quantity in order
                temp_order = ExecutionOrder(
                    order_id=execution_order.order_id,
                    symbol=execution_order.symbol,
                    side=execution_order.side,
                    total_quantity=execution_order.total_quantity,
                    remaining_quantity=remaining_quantity,
                    urgency=execution_order.urgency,
                    alpha_signal=execution_order.alpha_signal,
                    max_participation=execution_order.max_participation,
                    start_time=execution_order.start_time,
                    end_time=execution_order.end_time,
                    benchmark_price=execution_order.benchmark_price,
                    risk_aversion=execution_order.risk_aversion
                )
                
                # Get next action
                action = await self.execute_order(temp_order, lob_state)
                execution_schedule.append(action)
                
                # Simulate execution (would be real fills in production)
                fill_quantity = min(action.quantity, remaining_quantity)
                remaining_quantity -= fill_quantity
                
                # Advance time
                current_time += timedelta(minutes=5)  # 5-minute intervals
            
            return execution_schedule
            
        except Exception as e:
            self.logger.error(f"Error optimizing execution schedule: {e}")
            return []
    
    def _create_synthetic_lob_state(self, symbol: str, timestamp: datetime,
                                  historical_data: pd.DataFrame) -> LOBState:
        """Create synthetic LOB state for backtesting"""
        # Simplified synthetic LOB state
        mid_price = 100.0 + np.random.randn() * 2  # Random walk around $100
        spread = 0.01 + np.random.exponential(0.005)  # Random spread
        
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Create order book levels
        bids = []
        asks = []
        
        for i in range(10):
            bid_level = OrderBookLevel(
                price=bid_price - i * 0.01,
                size=np.random.randint(100, 2000),
                venue="NYSE",
                timestamp=timestamp,
                side=OrderSide.BUY
            )
            bids.append(bid_level)
            
            ask_level = OrderBookLevel(
                price=ask_price + i * 0.01,
                size=np.random.randint(100, 2000),
                venue="NYSE",
                timestamp=timestamp,
                side=OrderSide.SELL
            )
            asks.append(ask_level)
        
        # Calculate microstructure features
        total_bid_size = sum(level.size for level in bids[:3])
        total_ask_size = sum(level.size for level in asks[:3])
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        return LOBState(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            last_trade_price=mid_price + np.random.uniform(-0.005, 0.005),
            last_trade_size=np.random.randint(100, 1000),
            bid_ask_spread=spread,
            mid_price=mid_price,
            imbalance=imbalance,
            depth_ratio=total_bid_size / max(1, total_ask_size),
            volatility=np.random.uniform(0.1, 0.3),
            momentum=np.random.uniform(-0.02, 0.02)
        )
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get execution performance summary"""
        if not self.execution_history:
            return {"error": "No execution history available"}
        
        # Calculate performance metrics
        total_executions = len(self.execution_history)
        
        # Q-table statistics
        q_table_size = len(self.q_table)
        avg_q_values = np.mean([np.mean(q_vals) for q_vals in self.q_table.values()]) if self.q_table else 0
        
        return {
            "rl_agent": {
                "total_executions": total_executions,
                "q_table_size": q_table_size,
                "average_q_value": avg_q_values,
                "exploration_rate": self.epsilon,
                "learning_rate": self.learning_rate
            },
            "venues": {
                venue_id: {
                    "market_share": venue.market_share,
                    "avg_fill_rate": venue.average_fill_rate,
                    "latency_ms": venue.latency_ms
                }
                for venue_id, venue in self.venue_selector.venues.items()
            },
            "action_space": {
                "total_actions": len(self.action_space),
                "order_types": list(set(action['type'].value for action in self.action_space))
            }
        }


# Factory function
async def create_rl_execution_agent(config: Optional[Dict[str, Any]] = None) -> RLExecutionAgent:
    """Create and initialize RL execution agent"""
    return RLExecutionAgent(config)


# Example usage
async def main():
    """Example usage of RL execution agent"""
    # Create execution order
    execution_order = ExecutionOrder(
        order_id="RL_ORDER_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=10000,
        remaining_quantity=10000,
        urgency=0.3,
        alpha_signal=0.02,  # 2% positive alpha
        max_participation=0.15,  # Max 15% of volume
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2),
        benchmark_price=150.00,
        risk_aversion=0.5
    )
    
    # Create RL agent
    rl_agent = await create_rl_execution_agent({
        'learning_rate': 0.01,
        'epsilon': 0.1,
        'discount_factor': 0.95
    })
    
    # Create sample LOB state
    bids = [
        OrderBookLevel(149.98, 1000, "NYSE", datetime.now(), OrderSide.BUY),
        OrderBookLevel(149.97, 1500, "NYSE", datetime.now(), OrderSide.BUY),
        OrderBookLevel(149.96, 800, "NYSE", datetime.now(), OrderSide.BUY)
    ]
    
    asks = [
        OrderBookLevel(150.02, 1200, "NYSE", datetime.now(), OrderSide.SELL),
        OrderBookLevel(150.03, 900, "NYSE", datetime.now(), OrderSide.SELL),
        OrderBookLevel(150.04, 1100, "NYSE", datetime.now(), OrderSide.SELL)
    ]
    
    lob_state = LOBState(
        symbol="AAPL",
        timestamp=datetime.now(),
        bids=bids,
        asks=asks,
        last_trade_price=150.00,
        last_trade_size=500,
        bid_ask_spread=0.04,
        mid_price=150.00,
        imbalance=0.1,
        depth_ratio=1.2,
        volatility=0.2,
        momentum=0.005
    )
    
    # Execute order
    action = await rl_agent.execute_order(execution_order, lob_state)
    
    print("RL Execution Agent Results:")
    print(f"Order Type: {action.order_type.value}")
    print(f"Venue: {action.venue}")
    print(f"Quantity: {action.quantity}")
    print(f"Price: ${action.price:.2f}" if action.price else "Market Price")
    
    # Simulate execution and calculate reward
    fill_price = 150.01
    fill_quantity = action.quantity
    
    reward = rl_agent.calculate_reward(execution_order, action, fill_price, fill_quantity, lob_state)
    print(f"\nExecution Reward: {reward.total_reward:.4f}")
    print(f"Implementation Shortfall: {reward.implementation_shortfall:.4f}")
    print(f"Fill Rate: {reward.fill_rate:.2%}")
    
    # Get performance summary
    summary = await rl_agent.get_performance_summary()
    print(f"\nRL Agent Summary:")
    print(f"Q-table size: {summary['rl_agent']['q_table_size']}")
    print(f"Exploration rate: {summary['rl_agent']['exploration_rate']:.3f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
