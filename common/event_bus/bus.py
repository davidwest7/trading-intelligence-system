"""
Event Bus implementation for inter-agent communication
"""

import asyncio
import json
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(str, Enum):
    MARKET_TICK = "market_tick"
    NEWS_EVENT = "news_event" 
    AGENT_SIGNAL = "agent_signal"
    RANKED_OPPORTUNITY = "ranked_opportunity"
    SYSTEM_STATUS = "system_status"


@dataclass
class Event:
    """Base event class"""
    type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'correlation_id': self.correlation_id
        }


class MarketTickEvent(Event):
    """Market data tick event"""
    
    def __init__(self, source: str, symbol: str, price: float, volume: float, **kwargs):
        super().__init__(
            type=EventType.MARKET_TICK,
            source=source,
            timestamp=datetime.now(),
            data={'symbol': symbol, 'price': price, 'volume': volume},
            **kwargs
        )
        self.symbol = symbol
        self.price = price
        self.volume = volume


class AgentSignalEvent(Event):
    """Agent signal event"""
    
    def __init__(self, source: str, agent_name: str, signal_type: str, 
                 confidence: float, **kwargs):
        super().__init__(
            type=EventType.AGENT_SIGNAL,
            source=source,
            timestamp=datetime.now(),
            data={
                'agent_name': agent_name,
                'signal_type': signal_type, 
                'confidence': confidence
            },
            **kwargs
        )
        self.agent_name = agent_name
        self.signal_type = signal_type
        self.confidence = confidence


class EventBus:
    """
    Async event bus for agent communication
    
    Features:
    - Topic-based subscription
    - Event filtering
    - Event persistence (optional)
    - Rate limiting
    - Event replay
    
    TODO Items:
    1. Implement Redis backend for persistence
    2. Add event replay functionality
    3. Implement event batching for performance
    4. Add dead letter queue for failed events
    5. Implement event schema validation
    6. Add metrics and monitoring
    7. Implement event sourcing patterns
    8. Add circuit breaker for subscribers
    9. Implement event transformation pipelines
    10. Add distributed event bus support
    """
    
    def __init__(self, persist_events: bool = False, max_history: int = 10000):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.persist_events = persist_events
        self.max_history = max_history
        self._running = False
        self._event_queue = asyncio.Queue()
        
    async def start(self):
        """Start the event bus"""
        if self._running:
            return
            
        self._running = True
        # Start event processing task
        asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop the event bus"""
        self._running = False
        
    async def publish(self, event: Event):
        """
        Publish an event to the bus
        
        Args:
            event: Event to publish
        """
        if not self._running:
            await self.start()
            
        await self._event_queue.put(event)
        
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """
        Subscribe to events of a specific type
        
        Args:
            event_type: Type of events to subscribe to
            handler: Async function to handle events
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
            
        self.subscribers[event_type].append(handler)
        
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Unsubscribe from events"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
            
    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Store event in history
                if self.persist_events:
                    self.event_history.append(event)
                    if len(self.event_history) > self.max_history:
                        self.event_history.pop(0)
                
                # Notify subscribers
                await self._notify_subscribers(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
                
    async def _notify_subscribers(self, event: Event):
        """Notify all subscribers of an event"""
        if event.type in self.subscribers:
            # Run all handlers concurrently
            tasks = []
            for handler in self.subscribers[event.type]:
                task = asyncio.create_task(self._safe_handler_call(handler, event))
                tasks.append(task)
                
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
    async def _safe_handler_call(self, handler: Callable, event: Event):
        """Safely call event handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            print(f"Error in event handler {handler.__name__}: {e}")
            
    def get_event_history(self, event_type: EventType = None, 
                         limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type"""
        if not self.persist_events:
            return []
            
        events = self.event_history
        if event_type:
            events = [e for e in events if e.type == event_type]
            
        return events[-limit:] if events else []
        
    async def publish_market_tick(self, source: str, symbol: str, 
                                 price: float, volume: float):
        """Convenience method to publish market tick"""
        event = MarketTickEvent(source, symbol, price, volume)
        await self.publish(event)
        
    async def publish_agent_signal(self, source: str, agent_name: str,
                                 signal_type: str, confidence: float,
                                 additional_data: Dict[str, Any] = None):
        """Convenience method to publish agent signal"""
        event = AgentSignalEvent(source, agent_name, signal_type, confidence)
        if additional_data:
            event.data.update(additional_data)
        await self.publish(event)


# Example usage and testing
if __name__ == "__main__":
    async def test_event_bus():
        bus = EventBus(persist_events=True)
        
        # Create event handlers
        async def market_handler(event: Event):
            print(f"Market tick: {event.data}")
            
        async def signal_handler(event: Event):
            print(f"Agent signal: {event.data}")
            
        # Subscribe to events
        bus.subscribe(EventType.MARKET_TICK, market_handler)
        bus.subscribe(EventType.AGENT_SIGNAL, signal_handler)
        
        # Start bus
        await bus.start()
        
        # Publish some events
        await bus.publish_market_tick("ibkr", "AAPL", 150.0, 1000)
        await bus.publish_agent_signal("technical", "technical", "buy_signal", 0.8)
        
        # Wait a bit for processing
        await asyncio.sleep(1)
        
        # Check history
        history = bus.get_event_history()
        print(f"Event history has {len(history)} events")
        
        await bus.stop()
        
    # Run test
    asyncio.run(test_event_bus())
