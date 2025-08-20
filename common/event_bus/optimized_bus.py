"""
Optimized Event Bus for Trading System

High-performance event bus with Kafka/Redpanda integration,
proper serialization, backpressure handling, and error recovery.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
import uuid
from dataclasses import asdict
import time

from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import redis.asyncio as redis

from schemas.contracts import (
    Signal, Opportunity, Intent, DecisionLog,
    validate_signal_contract, validate_opportunity_contract,
    validate_intent_contract, validate_decision_log_contract
)


logger = logging.getLogger(__name__)


class OptimizedEventBus:
    """
    High-performance event bus with Kafka/Redpanda integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Kafka configuration
        self.kafka_config = {
            'bootstrap.servers': config.get('kafka_bootstrap_servers', 'localhost:9092'),
            'client.id': config.get('client_id', 'trading-system'),
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'batch.size': 16384,  # 16KB batch size
            'linger.ms': 5,  # Wait up to 5ms for batching
            'compression.type': 'lz4',  # Fast compression
            'max.in.flight.requests.per.connection': 5,
        }
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap.servers': config.get('kafka_bootstrap_servers', 'localhost:9092'),
            'group.id': config.get('consumer_group_id', 'trading-system-group'),
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 1000,
            'max.poll.records': 500,  # Process up to 500 records per poll
            'max.poll.interval.ms': 300000,  # 5 minutes
            'session.timeout.ms': 30000,  # 30 seconds
            'heartbeat.interval.ms': 3000,  # 3 seconds
        }
        
        # Redis configuration for caching
        self.redis_config = {
            'host': config.get('redis_host', 'localhost'),
            'port': config.get('redis_port', 6379),
            'db': config.get('redis_db', 0),
            'decode_responses': True,
        }
        
        # Topic configuration
        self.topics = {
            'signals.raw': {
                'partitions': 6,
                'replication_factor': 1,
                'retention_ms': 7 * 24 * 60 * 60 * 1000,  # 7 days
            },
            'opportunities.raw': {
                'partitions': 6,
                'replication_factor': 1,
                'retention_ms': 7 * 24 * 60 * 60 * 1000,  # 7 days
            },
            'intents.trade': {
                'partitions': 6,
                'replication_factor': 1,
                'retention_ms': 30 * 24 * 60 * 60 * 1000,  # 30 days
            },
            'decisions.log': {
                'partitions': 6,
                'replication_factor': 1,
                'retention_ms': 90 * 24 * 60 * 60 * 1000,  # 90 days
            },
            'telemetry.metrics': {
                'partitions': 3,
                'replication_factor': 1,
                'retention_ms': 24 * 60 * 60 * 1000,  # 1 day
            }
        }
        
        # Initialize components
        self.producer: Optional[Producer] = None
        self.consumers: Dict[str, Consumer] = {}
        self.redis_client: Optional[redis.Redis] = None
        
        # Event handlers
        self.handlers: Dict[str, List[Callable]] = {
            'signals.raw': [],
            'opportunities.raw': [],
            'intents.trade': [],
            'decisions.log': [],
            'telemetry.metrics': []
        }
        
        # Performance metrics
        self.metrics = {
            'messages_published': 0,
            'messages_consumed': 0,
            'errors': 0,
            'latency_avg_ms': 0.0,
            'throughput_msgs_per_sec': 0.0,
        }
        
        # Backpressure control
        self.max_queue_size = config.get('max_queue_size', 10000)
        self.processing_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Health status
        self.is_healthy = False
        self.last_heartbeat = datetime.utcnow()
        
    async def start(self):
        """Start the event bus"""
        logger.info("🚀 Starting Optimized Event Bus...")
        
        try:
            # Initialize Redis
            await self._init_redis()
            
            # Initialize Kafka producer
            await self._init_producer()
            
            # Create topics
            await self._create_topics()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._metrics_collector())
            
            self.is_healthy = True
            logger.info("✅ Optimized Event Bus started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start event bus: {e}")
            raise
    
    async def stop(self):
        """Stop the event bus"""
        logger.info("🛑 Stopping Optimized Event Bus...")
        
        self.is_healthy = False
        
        # Stop consumers
        for topic, consumer in self.consumers.items():
            try:
                consumer.close()
                logger.info(f"Closed consumer for {topic}")
            except Exception as e:
                logger.error(f"Error closing consumer for {topic}: {e}")
        
        # Stop producer
        if self.producer:
            try:
                self.producer.flush(timeout=10)
                logger.info("Flushed producer")
            except Exception as e:
                logger.error(f"Error flushing producer: {e}")
        
        # Close Redis
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")
        
        logger.info("✅ Optimized Event Bus stopped")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def _init_producer(self):
        """Initialize Kafka producer"""
        try:
            self.producer = Producer(self.kafka_config)
            logger.info("✅ Kafka producer initialized")
        except Exception as e:
            logger.error(f"❌ Kafka producer initialization failed: {e}")
            raise
    
    async def _create_topics(self):
        """Create Kafka topics if they don't exist"""
        try:
            admin_client = AdminClient({'bootstrap.servers': self.kafka_config['bootstrap.servers']})
            
            new_topics = []
            for topic_name, topic_config in self.topics.items():
                new_topics.append(NewTopic(
                    topic_name,
                    num_partitions=topic_config['partitions'],
                    replication_factor=topic_config['replication_factor'],
                    config={
                        'retention.ms': str(topic_config['retention_ms']),
                        'cleanup.policy': 'delete',
                        'compression.type': 'lz4',
                    }
                ))
            
            # Create topics
            fs = admin_client.create_topics(new_topics)
            for topic, f in fs.items():
                try:
                    f.result()  # Wait for topic creation
                    logger.info(f"✅ Created topic: {topic}")
                except KafkaException as e:
                    if "already exists" in str(e):
                        logger.info(f"Topic already exists: {topic}")
                    else:
                        logger.error(f"❌ Failed to create topic {topic}: {e}")
                        raise
            
            admin_client.close()
            
        except Exception as e:
            logger.error(f"❌ Topic creation failed: {e}")
            raise
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish a signal to the event bus"""
        return await self._publish_message('signals.raw', signal)
    
    async def publish_opportunity(self, opportunity: Opportunity) -> bool:
        """Publish an opportunity to the event bus"""
        return await self._publish_message('opportunities.raw', opportunity)
    
    async def publish_intent(self, intent: Intent) -> bool:
        """Publish a trading intent to the event bus"""
        return await self._publish_message('intents.trade', intent)
    
    async def publish_decision_log(self, decision_log: DecisionLog) -> bool:
        """Publish a decision log to the event bus"""
        return await self._publish_message('decisions.log', decision_log)
    
    async def publish_telemetry(self, telemetry_data: Dict[str, Any]) -> bool:
        """Publish telemetry data to the event bus"""
        return await self._publish_message('telemetry.metrics', telemetry_data)
    
    async def _publish_message(self, topic: str, message: Any) -> bool:
        """Publish a message to a topic"""
        if not self.is_healthy:
            logger.error("Event bus not healthy, cannot publish")
            return False
        
        try:
            start_time = time.time()
            
            # Serialize message
            if hasattr(message, 'to_dict'):
                message_dict = message.to_dict()
            elif hasattr(message, 'dict'):
                message_dict = message.dict()
            else:
                message_dict = asdict(message) if hasattr(message, '__dict__') else message
            
            # Add metadata
            message_dict['_metadata'] = {
                'timestamp': datetime.utcnow().isoformat(),
                'producer_id': self.config.get('client_id', 'trading-system'),
                'message_id': str(uuid.uuid4()),
            }
            
            # Serialize to JSON
            message_json = json.dumps(message_dict, default=str)
            
            # Publish to Kafka
            def delivery_report(err, msg):
                if err is not None:
                    logger.error(f"Message delivery failed: {err}")
                    self.metrics['errors'] += 1
                else:
                    self.metrics['messages_published'] += 1
                    latency = (time.time() - start_time) * 1000
                    self.metrics['latency_avg_ms'] = (
                        (self.metrics['latency_avg_ms'] * (self.metrics['messages_published'] - 1) + latency) /
                        self.metrics['messages_published']
                    )
            
            self.producer.produce(
                topic=topic,
                value=message_json.encode('utf-8'),
                callback=delivery_report
            )
            
            # Trigger delivery reports
            self.producer.poll(0)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message to {topic}: {e}")
            self.metrics['errors'] += 1
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]):
        """Subscribe to a topic with a handler"""
        if topic not in self.handlers:
            raise ValueError(f"Unknown topic: {topic}")
        
        self.handlers[topic].append(handler)
        logger.info(f"✅ Subscribed to {topic}")
    
    async def start_consuming(self, topics: List[str]):
        """Start consuming from specified topics"""
        for topic in topics:
            if topic not in self.handlers:
                logger.warning(f"⚠️ No handlers registered for topic: {topic}")
                continue
            
            await self._start_consumer(topic)
    
    async def _start_consumer(self, topic: str):
        """Start a consumer for a specific topic"""
        try:
            consumer_config = self.consumer_config.copy()
            consumer_config['group.id'] = f"{self.consumer_config['group.id']}-{topic}"
            
            consumer = Consumer(consumer_config)
            consumer.subscribe([topic])
            
            self.consumers[topic] = consumer
            
            # Start consumer task
            asyncio.create_task(self._consume_messages(topic, consumer))
            
            logger.info(f"✅ Started consumer for {topic}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start consumer for {topic}: {e}")
            raise
    
    async def _consume_messages(self, topic: str, consumer: Consumer):
        """Consume messages from a topic"""
        logger.info(f"🔄 Starting message consumption for {topic}")
        
        while self.is_healthy:
            try:
                # Poll for messages
                msg = consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        self.metrics['errors'] += 1
                        continue
                
                # Parse message
                try:
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    # Validate message based on topic
                    if topic == 'signals.raw':
                        if not validate_signal_contract(message_data):
                            logger.warning(f"Invalid signal contract: {message_data}")
                            continue
                    elif topic == 'opportunities.raw':
                        if not validate_opportunity_contract(message_data):
                            logger.warning(f"Invalid opportunity contract: {message_data}")
                            continue
                    elif topic == 'intents.trade':
                        if not validate_intent_contract(message_data):
                            logger.warning(f"Invalid intent contract: {message_data}")
                            continue
                    elif topic == 'decisions.log':
                        if not validate_decision_log_contract(message_data):
                            logger.warning(f"Invalid decision log contract: {message_data}")
                            continue
                    
                    # Process message with handlers
                    for handler in self.handlers[topic]:
                        try:
                            await handler(message_data)
                        except Exception as e:
                            logger.error(f"Handler error for {topic}: {e}")
                            self.metrics['errors'] += 1
                    
                    self.metrics['messages_consumed'] += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    self.metrics['errors'] += 1
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    self.metrics['errors'] += 1
                
            except Exception as e:
                logger.error(f"Consumer error for {topic}: {e}")
                self.metrics['errors'] += 1
                await asyncio.sleep(1)  # Back off on error
        
        logger.info(f"🛑 Stopped consuming from {topic}")
    
    async def _heartbeat_monitor(self):
        """Monitor system health"""
        while self.is_healthy:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                # Check Redis connection
                if self.redis_client:
                    await self.redis_client.ping()
                
                # Check Kafka producer
                if self.producer:
                    # Trigger delivery reports
                    self.producer.poll(0)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                self.is_healthy = False
                break
    
    async def _metrics_collector(self):
        """Collect and publish performance metrics"""
        while self.is_healthy:
            try:
                # Calculate throughput
                if self.metrics['messages_published'] > 0:
                    self.metrics['throughput_msgs_per_sec'] = (
                        self.metrics['messages_published'] / 
                        (time.time() - self.last_heartbeat.timestamp())
                    )
                
                # Publish metrics
                await self.publish_telemetry({
                    'component': 'event_bus',
                    'metrics': self.metrics,
                    'timestamp': datetime.utcnow().isoformat(),
                    'health': self.is_healthy,
                })
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the event bus"""
        return {
            'healthy': self.is_healthy,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'metrics': self.metrics,
            'consumers': list(self.consumers.keys()),
            'handlers': {topic: len(handlers) for topic, handlers in self.handlers.items()},
            'queue_size': self.processing_queue.qsize(),
        }
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set a value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await self.redis_client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
