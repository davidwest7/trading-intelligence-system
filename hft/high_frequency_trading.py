"""
High-Frequency Trading Infrastructure
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
from queue import Queue
import json

class HighFrequencyTradingEngine:
    """
    High-Frequency Trading Engine with microsecond latency
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'max_latency_ms': 1.0,  # 1ms maximum latency
            'order_queue_size': 10000,
            'risk_limits': {
                'max_position_size': 0.02,  # 2% max position
                'max_daily_loss': 0.05,     # 5% max daily loss
                'max_drawdown': 0.10        # 10% max drawdown
            },
            'arbitrage_threshold': 0.001,   # 0.1% minimum spread
            'market_making_spread': 0.002   # 0.2% bid-ask spread
        }
        
        self.order_queue = Queue(maxsize=self.config['order_queue_size'])
        self.execution_engine = None
        self.risk_manager = None
        self.arbitrage_detector = None
        self.market_maker = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize HFT components"""
        try:
            print("🚀 Initializing High-Frequency Trading Engine...")
            
            # Initialize components
            self.execution_engine = ExecutionEngine(self.config)
            self.risk_manager = RiskManager(self.config['risk_limits'])
            self.arbitrage_detector = ArbitrageDetector(self.config['arbitrage_threshold'])
            self.market_maker = MarketMaker(self.config['market_making_spread'])
            
            # Start execution thread
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()
            
            print("✅ HFT Engine initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing HFT Engine: {e}")
            return False
    
    def _execution_loop(self):
        """High-frequency execution loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Process orders with microsecond precision
                if not self.order_queue.empty():
                    order = self.order_queue.get_nowait()
                    self._process_order(order)
                
                # Check latency
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                if execution_time > self.config['max_latency_ms']:
                    print(f"⚠️ High latency detected: {execution_time:.3f}ms")
                
                # Microsecond sleep
                time.sleep(0.000001)  # 1 microsecond
                
            except Exception as e:
                print(f"Error in execution loop: {e}")
    
    def _process_order(self, order):
        """Process order with minimal latency"""
        try:
            # Risk check
            if not self.risk_manager.check_order(order):
                print(f"❌ Order rejected by risk manager: {order['symbol']}")
                return
            
            # Execute order
            execution_result = self.execution_engine.execute_order(order)
            
            # Update risk metrics
            self.risk_manager.update_metrics(execution_result)
            
        except Exception as e:
            print(f"Error processing order: {e}")
    
    async def submit_order(self, order):
        """Submit order to HFT engine"""
        try:
            # Add timestamp for latency tracking
            order['timestamp'] = time.time()
            order['order_id'] = self._generate_order_id()
            
            # Add to queue
            self.order_queue.put_nowait(order)
            
            return {
                'success': True,
                'order_id': order['order_id'],
                'status': 'queued'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_order_id(self):
        """Generate unique order ID"""
        return f"ORD_{int(time.time() * 1000000)}"
    
    async def start(self):
        """Start HFT engine"""
        self.is_running = True
        print("🚀 HFT Engine started")
    
    async def stop(self):
        """Stop HFT engine"""
        self.is_running = False
        print("🛑 HFT Engine stopped")


class ExecutionEngine:
    """Ultra-fast execution engine"""
    
    def __init__(self, config):
        self.config = config
        self.execution_history = []
        self.latency_stats = []
    
    def execute_order(self, order):
        """Execute order with minimal latency"""
        start_time = time.time()
        
        try:
            # Simulate execution
            execution_result = {
                'order_id': order['order_id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['quantity'],
                'price': order['price'],
                'execution_time': time.time(),
                'status': 'filled'
            }
            
            # Record latency
            latency = (time.time() - start_time) * 1000000  # Microseconds
            self.latency_stats.append(latency)
            
            # Keep only recent stats
            if len(self.latency_stats) > 1000:
                self.latency_stats = self.latency_stats[-1000:]
            
            self.execution_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            return {
                'order_id': order['order_id'],
                'status': 'failed',
                'error': str(e)
            }
    
    def get_latency_stats(self):
        """Get latency statistics"""
        if not self.latency_stats:
            return {'avg_latency_us': 0, 'max_latency_us': 0, 'min_latency_us': 0}
        
        return {
            'avg_latency_us': np.mean(self.latency_stats),
            'max_latency_us': np.max(self.latency_stats),
            'min_latency_us': np.min(self.latency_stats),
            'p95_latency_us': np.percentile(self.latency_stats, 95),
            'p99_latency_us': np.percentile(self.latency_stats, 99)
        }


class RiskManager:
    """Real-time risk management"""
    
    def __init__(self, risk_limits):
        self.risk_limits = risk_limits
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0
    
    def check_order(self, order):
        """Check if order meets risk limits"""
        try:
            symbol = order['symbol']
            quantity = order['quantity']
            price = order['price']
            
            # Position size check
            position_value = abs(quantity * price)
            if position_value > self.risk_limits['max_position_size']:
                return False
            
            # Daily loss check
            if self.daily_pnl < -self.risk_limits['max_daily_loss']:
                return False
            
            # Drawdown check
            if self.max_drawdown > self.risk_limits['max_drawdown']:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in risk check: {e}")
            return False
    
    def update_metrics(self, execution_result):
        """Update risk metrics after execution"""
        try:
            if execution_result['status'] == 'filled':
                symbol = execution_result['symbol']
                quantity = execution_result['quantity']
                price = execution_result['price']
                
                # Update position
                if symbol not in self.positions:
                    self.positions[symbol] = 0
                
                if execution_result['side'] == 'buy':
                    self.positions[symbol] += quantity
                else:
                    self.positions[symbol] -= quantity
                
                # Update P&L (simplified)
                # In real implementation, this would track realized/unrealized P&L
                
        except Exception as e:
            print(f"Error updating risk metrics: {e}")


class ArbitrageDetector:
    """Cross-exchange arbitrage detection"""
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.arbitrage_opportunities = []
    
    async def detect_arbitrage(self, market_data):
        """Detect arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            # Group by symbol
            symbols = set(data['symbol'] for data in market_data)
            
            for symbol in symbols:
                symbol_data = [d for d in market_data if d['symbol'] == symbol]
                
                if len(symbol_data) < 2:
                    continue
                
                # Find best bid and ask across exchanges
                best_bid = max(d['bid'] for d in symbol_data)
                best_ask = min(d['ask'] for d in symbol_data)
                
                # Calculate spread
                spread = (best_ask - best_bid) / best_bid
                
                if spread > self.threshold:
                    opportunity = {
                        'symbol': symbol,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'spread': spread,
                        'potential_profit': spread * best_bid,
                        'timestamp': time.time()
                    }
                    opportunities.append(opportunity)
            
            self.arbitrage_opportunities = opportunities
            return opportunities
            
        except Exception as e:
            print(f"Error detecting arbitrage: {e}")
            return []


class MarketMaker:
    """Automated market making"""
    
    def __init__(self, spread):
        self.spread = spread
        self.active_quotes = {}
    
    async def generate_quotes(self, market_data):
        """Generate bid-ask quotes"""
        try:
            quotes = []
            
            for data in market_data:
                mid_price = (data['bid'] + data['ask']) / 2
                half_spread = mid_price * self.spread / 2
                
                quote = {
                    'symbol': data['symbol'],
                    'bid': mid_price - half_spread,
                    'ask': mid_price + half_spread,
                    'bid_size': 1000,  # Default size
                    'ask_size': 1000,
                    'timestamp': time.time()
                }
                quotes.append(quote)
                
                # Store active quote
                self.active_quotes[data['symbol']] = quote
            
            return quotes
            
        except Exception as e:
            print(f"Error generating quotes: {e}")
            return []
    
    async def update_quotes(self, market_data):
        """Update quotes based on market conditions"""
        try:
            updated_quotes = []
            
            for data in market_data:
                if data['symbol'] in self.active_quotes:
                    current_quote = self.active_quotes[data['symbol']]
                    
                    # Adjust spread based on volatility
                    volatility = self._calculate_volatility(data)
                    adjusted_spread = self.spread * (1 + volatility)
                    
                    mid_price = (data['bid'] + data['ask']) / 2
                    half_spread = mid_price * adjusted_spread / 2
                    
                    updated_quote = {
                        'symbol': data['symbol'],
                        'bid': mid_price - half_spread,
                        'ask': mid_price + half_spread,
                        'bid_size': current_quote['bid_size'],
                        'ask_size': current_quote['ask_size'],
                'timestamp': time.time()
                    }
                    
                    updated_quotes.append(updated_quote)
                    self.active_quotes[data['symbol']] = updated_quote
            
            return updated_quotes
            
        except Exception as e:
            print(f"Error updating quotes: {e}")
            return []
    
    def _calculate_volatility(self, market_data):
        """Calculate market volatility"""
        # Simplified volatility calculation
        return 0.1  # 10% volatility


class SmartOrderRouter:
    """Smart order routing across venues"""
    
    def __init__(self):
        self.venue_latency = {}
        self.venue_liquidity = {}
        self.venue_costs = {}
    
    async def route_order(self, order, available_venues):
        """Route order to optimal venue"""
        try:
            best_venue = None
            best_score = -1
            
            for venue in available_venues:
                # Calculate venue score based on latency, liquidity, and costs
                latency_score = self._calculate_latency_score(venue)
                liquidity_score = self._calculate_liquidity_score(venue, order)
                cost_score = self._calculate_cost_score(venue)
                
                total_score = latency_score * 0.4 + liquidity_score * 0.4 + cost_score * 0.2
                
                if total_score > best_score:
                    best_score = total_score
                    best_venue = venue
            
            return {
                'venue': best_venue,
                'score': best_score,
                'order': order
            }
            
        except Exception as e:
            print(f"Error routing order: {e}")
            return None
    
    def _calculate_latency_score(self, venue):
        """Calculate latency score for venue"""
        if venue in self.venue_latency:
            latency = self.venue_latency[venue]
            return max(0, 1 - latency / 100)  # Normalize to 0-1
        return 0.5  # Default score
    
    def _calculate_liquidity_score(self, venue, order):
        """Calculate liquidity score for venue"""
        if venue in self.venue_liquidity:
            available_liquidity = self.venue_liquidity[venue].get(order['symbol'], 0)
            return min(1, available_liquidity / order['quantity'])
        return 0.5  # Default score
    
    def _calculate_cost_score(self, venue):
        """Calculate cost score for venue"""
        if venue in self.venue_costs:
            cost = self.venue_costs[venue]
            return max(0, 1 - cost)  # Lower cost = higher score
        return 0.5  # Default score
