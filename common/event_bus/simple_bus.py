#!/usr/bin/env python3
"""
Simple Event Bus for Testing

A simplified event bus that works without external dependencies
for testing the complete architecture.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
import uuid
from dataclasses import asdict
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimpleEventBus:
    """
    Simple event bus for testing without external dependencies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Event storage
        self.events: List[Dict[str, Any]] = []
        self.max_events = config.get('max_queue_size', 10000)
        
        # Event handlers
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            'messages_published': 0,
            'messages_consumed': 0,
            'errors': 0,
            'latency_avg_ms': 0.0,
            'throughput_msgs_per_sec': 0.0,
        }
        
        # Processing queue
        self.processing_queue = asyncio.Queue(maxsize=self.max_events)
        
        # Health status
        self.is_healthy = False
        self.last_heartbeat = datetime.utcnow()
        self._running = False
        
    async def start(self):
        """Start the event bus"""
        logger.info("🚀 Starting Simple Event Bus...")
        
        try:
            self._running = True
            asyncio.create_task(self._process_events())
            self.is_healthy = True
            logger.info("✅ Simple Event Bus started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start event bus: {e}")
            raise
    
    async def stop(self):
        """Stop the event bus"""
        logger.info("🛑 Stopping Simple Event Bus...")
        
        self._running = False
        self.is_healthy = False
        
        logger.info("✅ Simple Event Bus stopped")
    
    async def publish_market_tick(self, source: str, symbol: str, price: float, volume: float):
        """Publish market tick event"""
        event = {
            'type': 'market_tick',
            'source': source,
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': datetime.utcnow().isoformat(),
            'id': str(uuid.uuid4())
        }
        
        await self._publish_event(event)
    
    async def publish_agent_signal(self, source: str, agent_name: str, signal_type: str, 
                                 confidence: float, additional_data: Dict[str, Any] = None):
        """Publish agent signal event"""
        event = {
            'type': 'agent_signal',
            'source': source,
            'agent_name': agent_name,
            'signal_type': signal_type,
            'confidence': confidence,
            'additional_data': additional_data or {},
            'timestamp': datetime.utcnow().isoformat(),
            'id': str(uuid.uuid4())
        }
        
        await self._publish_event(event)
    
    async def _publish_event(self, event: Dict[str, Any]):
        """Publish an event"""
        try:
            # Store event
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)
            
            # Add to processing queue
            await self.processing_queue.put(event)
            
            # Update metrics
            self.metrics['messages_published'] += 1
            
            logger.debug(f"Published event: {event['type']} from {event.get('source', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            self.metrics['errors'] += 1
    
    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Notify handlers
                await self._notify_handlers(event)
                
                # Update metrics
                self.metrics['messages_consumed'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self.metrics['errors'] += 1
    
    async def _notify_handlers(self, event: Dict[str, Any]):
        """Notify all handlers of an event"""
        event_type = event['type']
        
        if event_type in self.handlers:
            # Run all handlers concurrently
            tasks = []
            for handler in self.handlers[event_type]:
                task = asyncio.create_task(self._safe_handler_call(handler, event))
                tasks.append(task)
                
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_handler_call(self, handler: Callable, event: Dict[str, Any]):
        """Safely call a handler"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Error in event handler: {e}")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to events of a specific type"""
        self.handlers[event_type].append(handler)
        logger.info(f"Subscribed to {event_type} events")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from events"""
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)
            logger.info(f"Unsubscribed from {event_type} events")
    
    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history, optionally filtered by type"""
        events = self.events
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
            
        return events[-limit:] if events else []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
