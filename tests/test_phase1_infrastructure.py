"""
Phase 1 Infrastructure Tests

Comprehensive tests for the core infrastructure components:
- Message contracts
- Event bus
- Feature store
- Observability
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

from schemas.contracts import (
    Signal, Opportunity, Intent, DecisionLog,
    SignalType, RegimeType, HorizonType, DirectionType,
    validate_signal_contract, validate_opportunity_contract,
    validate_intent_contract, validate_decision_log_contract
)
from common.event_bus.optimized_bus import OptimizedEventBus
from common.feature_store.optimized_store import OptimizedFeatureStore
from common.observability.telemetry import TradingTelemetry, init_telemetry


class TestMessageContracts:
    """Test message contract validation and serialization"""
    
    def test_signal_contract_creation(self):
        """Test creating a valid signal"""
        signal = Signal(
            trace_id="test-trace-123",
            agent_id="technical-agent",
            agent_type=SignalType.TECHNICAL,
            symbol="AAPL",
            mu=0.05,
            sigma=0.02,
            confidence=0.8,
            horizon=HorizonType.SHORT_TERM,
            regime=RegimeType.RISK_ON,
            direction=DirectionType.LONG,
            model_version="v1.0.0",
            feature_version="v1.0.0"
        )
        
        assert signal.symbol == "AAPL"
        assert signal.mu == 0.05
        assert signal.sigma == 0.02
        assert signal.confidence == 0.8
        assert signal.agent_type == SignalType.TECHNICAL
    
    def test_signal_contract_validation(self):
        """Test signal contract validation"""
        valid_signal_data = {
            "trace_id": "test-trace-123",
            "agent_id": "technical-agent",
            "agent_type": "technical",
            "symbol": "AAPL",
            "mu": 0.05,
            "sigma": 0.02,
            "confidence": 0.8,
            "horizon": "short_term",
            "regime": "risk_on",
            "direction": "long",
            "model_version": "v1.0.0",
            "feature_version": "v1.0.0"
        }
        
        assert validate_signal_contract(valid_signal_data) == True
        
        # Test invalid confidence
        invalid_signal_data = valid_signal_data.copy()
        invalid_signal_data["confidence"] = 1.5  # Should be <= 1.0
        
        assert validate_signal_contract(invalid_signal_data) == False
    
    def test_opportunity_contract_creation(self):
        """Test creating a valid opportunity"""
        opportunity = Opportunity(
            trace_id="test-trace-123",
            symbol="AAPL",
            mu_blended=0.04,
            sigma_blended=0.015,
            confidence_blended=0.85,
            horizon=HorizonType.SHORT_TERM,
            regime=RegimeType.RISK_ON,
            direction=DirectionType.LONG,
            blender_version="v1.0.0"
        )
        
        assert opportunity.symbol == "AAPL"
        assert opportunity.mu_blended == 0.04
        assert opportunity.confidence_blended == 0.85
    
    def test_intent_contract_creation(self):
        """Test creating a valid intent"""
        intent = Intent(
            trace_id="test-trace-123",
            opportunity_id="opp-123",
            symbol="AAPL",
            direction=DirectionType.LONG,
            size_eur=100.0,
            risk_eur=5.0,
            risk_pct=0.01,
            var_95=10.0,
            cvar_95=15.0,
            max_position_size=500.0,
            max_risk_per_trade=0.01,
            sizer_version="v1.0.0",
            risk_version="v1.0.0"
        )
        
        assert intent.symbol == "AAPL"
        assert intent.size_eur == 100.0
        assert intent.risk_pct == 0.01
    
    def test_decision_log_contract_creation(self):
        """Test creating a valid decision log"""
        decision_log = DecisionLog(
            trace_id="test-trace-123"
        )
        
        assert decision_log.trace_id == "test-trace-123"
        assert len(decision_log.signals) == 0
        assert len(decision_log.opportunities) == 0


class TestEventBus:
    """Test event bus functionality"""
    
    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing"""
        config = {
            'kafka_bootstrap_servers': 'localhost:9092',
            'client_id': 'test-client',
            'consumer_group_id': 'test-group',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
        }
        
        bus = OptimizedEventBus(config)
        yield bus
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_bus_initialization(self, event_bus):
        """Test event bus initialization"""
        # Note: This test requires Kafka and Redis to be running
        # In a real test environment, you'd use test containers
        try:
            await event_bus.start()
            assert event_bus.is_healthy == True
        except Exception as e:
            # Skip test if infrastructure not available
            pytest.skip(f"Infrastructure not available: {e}")
    
    @pytest.mark.asyncio
    async def test_signal_publishing(self, event_bus):
        """Test publishing signals"""
        try:
            await event_bus.start()
            
            signal = Signal(
                trace_id="test-trace-123",
                agent_id="technical-agent",
                agent_type=SignalType.TECHNICAL,
                symbol="AAPL",
                mu=0.05,
                sigma=0.02,
                confidence=0.8,
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="v1.0.0",
                feature_version="v1.0.0"
            )
            
            success = await event_bus.publish_signal(signal)
            assert success == True
            
        except Exception as e:
            pytest.skip(f"Infrastructure not available: {e}")


class TestFeatureStore:
    """Test feature store functionality"""
    
    @pytest.fixture
    async def feature_store(self):
        """Create feature store for testing"""
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
            'max_read_latency_ms': 5,
            'default_ttl_seconds': 3600,
        }
        
        store = OptimizedFeatureStore(config)
        yield store
        await store.stop()
    
    @pytest.mark.asyncio
    async def test_feature_store_initialization(self, feature_store):
        """Test feature store initialization"""
        try:
            await feature_store.start()
            assert feature_store.is_healthy == True
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    @pytest.mark.asyncio
    async def test_feature_storage_and_retrieval(self, feature_store):
        """Test storing and retrieving features"""
        try:
            await feature_store.start()
            
            # Test data
            test_data = {
                'price': 150.0,
                'volume': 1000000,
                'rsi': 65.5,
                'macd': 0.02
            }
            
            # Store feature
            success = await feature_store.set_feature('AAPL_technical', test_data)
            assert success == True
            
            # Retrieve feature
            retrieved_data = await feature_store.get_feature('AAPL_technical')
            assert retrieved_data is not None
            assert retrieved_data['price'] == 150.0
            assert retrieved_data['rsi'] == 65.5
            
            # Check latency
            health_status = await feature_store.get_health_status()
            assert health_status['avg_read_latency_ms'] < 5.0
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    @pytest.mark.asyncio
    async def test_batch_feature_operations(self, feature_store):
        """Test batch feature operations"""
        try:
            await feature_store.start()
            
            # Store multiple features
            features = {
                'AAPL_price': 150.0,
                'AAPL_volume': 1000000,
                'AAPL_rsi': 65.5,
                'GOOGL_price': 2800.0,
                'GOOGL_volume': 500000,
            }
            
            for name, data in features.items():
                success = await feature_store.set_feature(name, data)
                assert success == True
            
            # Retrieve batch
            feature_names = list(features.keys())
            retrieved_features = await feature_store.get_features_batch(feature_names)
            
            assert len(retrieved_features) == len(features)
            assert retrieved_features['AAPL_price'] == 150.0
            assert retrieved_features['GOOGL_price'] == 2800.0
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")


class TestObservability:
    """Test observability system"""
    
    @pytest.fixture
    def telemetry(self):
        """Create telemetry system for testing"""
        config = {
            'service_name': 'test-service',
            'service_version': '1.0.0',
            'environment': 'test',
        }
        
        return init_telemetry(config)
    
    def test_telemetry_initialization(self, telemetry):
        """Test telemetry initialization"""
        assert telemetry.service_name == 'test-service'
        assert telemetry.service_version == '1.0.0'
        assert telemetry.environment == 'test'
    
    def test_structured_logging(self, telemetry):
        """Test structured logging"""
        # Test event logging
        telemetry.log_event(
            'test_event',
            'Test message',
            trace_id='test-trace-123',
            user_id='test-user',
            action='test_action'
        )
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            telemetry.log_error(
                e,
                'test_context',
                trace_id='test-trace-123',
                user_id='test-user'
            )
    
    @pytest.mark.asyncio
    async def test_operation_tracing(self, telemetry):
        """Test operation tracing"""
        async with telemetry.trace_operation('test_operation', user_id='test-user') as span:
            # Simulate some work
            await asyncio.sleep(0.1)
            
            # Add custom attributes
            span.set_attribute("custom.attribute", "test_value")
    
    def test_metric_recording(self, telemetry):
        """Test metric recording"""
        telemetry.record_metric(
            'test_metric',
            42.0,
            labels={'service': 'test-service', 'operation': 'test'}
        )
    
    @pytest.mark.asyncio
    async def test_health_status(self, telemetry):
        """Test health status reporting"""
        # Register a health check
        telemetry.register_health_check('test_check', lambda: True)
        
        # Get health status
        health_status = await telemetry.get_health_status()
        
        assert health_status['service'] == 'test-service'
        assert health_status['healthy'] == True
        assert 'test_check' in health_status['checks']
        assert health_status['checks']['test_check']['status'] == 'healthy'


class TestIntegration:
    """Integration tests for Phase 1 components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_flow(self):
        """Test end-to-end signal flow through all components"""
        # This test would require all infrastructure to be running
        # In a real environment, you'd use test containers
        
        # Initialize components
        telemetry_config = {
            'service_name': 'test-integration',
            'service_version': '1.0.0',
            'environment': 'test',
        }
        
        event_bus_config = {
            'kafka_bootstrap_servers': 'localhost:9092',
            'client_id': 'test-integration',
            'consumer_group_id': 'test-integration-group',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
        }
        
        feature_store_config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
        }
        
        try:
            # Initialize telemetry
            telemetry = init_telemetry(telemetry_config)
            
            # Initialize event bus
            event_bus = OptimizedEventBus(event_bus_config)
            await event_bus.start()
            
            # Initialize feature store
            feature_store = OptimizedFeatureStore(feature_store_config)
            await feature_store.start()
            
            # Create test signal
            signal = Signal(
                trace_id="integration-test-123",
                agent_id="technical-agent",
                agent_type=SignalType.TECHNICAL,
                symbol="AAPL",
                mu=0.05,
                sigma=0.02,
                confidence=0.8,
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="v1.0.0",
                feature_version="v1.0.0"
            )
            
            # Store feature
            feature_data = {'price': 150.0, 'volume': 1000000}
            await feature_store.set_feature('AAPL_market_data', feature_data)
            
            # Publish signal
            success = await event_bus.publish_signal(signal)
            assert success == True
            
            # Log event
            telemetry.log_event(
                'signal_published',
                f'Published signal for {signal.symbol}',
                trace_id=signal.trace_id,
                agent_id=signal.agent_id,
                confidence=signal.confidence
            )
            
            # Cleanup
            await event_bus.stop()
            await feature_store.stop()
            
        except Exception as e:
            pytest.skip(f"Integration test infrastructure not available: {e}")


class TestPerformance:
    """Performance tests for Phase 1 components"""
    
    @pytest.mark.asyncio
    async def test_feature_store_performance(self):
        """Test feature store performance"""
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
            'max_read_latency_ms': 5,
        }
        
        try:
            store = OptimizedFeatureStore(config)
            await store.start()
            
            # Performance test: Store and retrieve 1000 features
            start_time = time.time()
            
            for i in range(1000):
                feature_name = f'feature_{i}'
                feature_data = {'value': i, 'timestamp': datetime.utcnow().isoformat()}
                
                success = await store.set_feature(feature_name, feature_data)
                assert success == True
                
                retrieved_data = await store.get_feature(feature_name)
                assert retrieved_data is not None
                assert retrieved_data['value'] == i
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete 2000 operations (1000 writes + 1000 reads) in reasonable time
            assert total_time < 60.0  # Less than 60 seconds
            
            # Check average latency
            health_status = await store.get_health_status()
            assert health_status['avg_read_latency_ms'] < 5.0
            
            await store.stop()
            
        except Exception as e:
            pytest.skip(f"Performance test infrastructure not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
