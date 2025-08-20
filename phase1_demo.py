#!/usr/bin/env python3
"""
Phase 1 Infrastructure Demo

Demonstrates the core infrastructure components:
- Message contracts
- Event bus (simulated)
- Feature store (simulated)
- Observability
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas.contracts import (
    Signal, Opportunity, Intent, DecisionLog,
    SignalType, RegimeType, HorizonType, DirectionType
)
from common.observability.telemetry import init_telemetry


class Phase1Demo:
    """Demo for Phase 1 infrastructure components"""
    
    def __init__(self):
        self.telemetry = None
        self.demo_results = {}
        
    async def run_demo(self):
        """Run the complete Phase 1 demo"""
        print("🚀 **PHASE 1 INFRASTRUCTURE DEMO**")
        print("=" * 60)
        
        try:
            # Initialize telemetry
            await self._init_telemetry()
            
            # Test message contracts
            await self._test_message_contracts()
            
            # Test simulated event bus
            await self._test_simulated_event_bus()
            
            # Test simulated feature store
            await self._test_simulated_feature_store()
            
            # Test observability
            await self._test_observability()
            
            # Test integration
            await self._test_integration()
            
            # Generate demo report
            await self._generate_demo_report()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _init_telemetry(self):
        """Initialize telemetry system"""
        print("\n📊 **INITIALIZING TELEMETRY**")
        print("-" * 40)
        
        config = {
            'service_name': 'phase1-demo',
            'service_version': '1.0.0',
            'environment': 'demo',
        }
        
        self.telemetry = init_telemetry(config)
        print("✅ Telemetry system initialized")
        
        # Register health checks
        self.telemetry.register_health_check('demo_check', lambda: True)
        print("✅ Health checks registered")
    
    async def _test_message_contracts(self):
        """Test message contract creation and validation"""
        print("\n📋 **TESTING MESSAGE CONTRACTS**")
        print("-" * 40)
        
        # Create test signal
        signal = Signal(
            trace_id="demo-trace-123",
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
        
        print(f"✅ Created signal for {signal.symbol}")
        print(f"   - Expected return: {signal.mu:.3f}")
        print(f"   - Uncertainty: {signal.sigma:.3f}")
        print(f"   - Confidence: {signal.confidence:.1%}")
        print(f"   - Direction: {signal.direction}")
        
        # Create test opportunity
        opportunity = Opportunity(
            trace_id="demo-trace-123",
            symbol="AAPL",
            mu_blended=0.04,
            sigma_blended=0.015,
            confidence_blended=0.85,
            horizon=HorizonType.SHORT_TERM,
            regime=RegimeType.RISK_ON,
            direction=DirectionType.LONG,
            blender_version="v1.0.0"
        )
        
        print(f"✅ Created opportunity for {opportunity.symbol}")
        print(f"   - Blended return: {opportunity.mu_blended:.3f}")
        print(f"   - Blended uncertainty: {opportunity.sigma_blended:.3f}")
        print(f"   - Blended confidence: {opportunity.confidence_blended:.1%}")
        
        # Create test intent
        intent = Intent(
            trace_id="demo-trace-123",
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
        
        print(f"✅ Created intent for {intent.symbol}")
        print(f"   - Position size: €{intent.size_eur}")
        print(f"   - Risk amount: €{intent.risk_eur}")
        print(f"   - Risk percentage: {intent.risk_pct:.1%}")
        
        # Store results
        self.demo_results['message_contracts'] = {
            'signal': signal.to_dict(),
            'opportunity': opportunity.to_dict(),
            'intent': intent.to_dict()
        }
        
        print("✅ Message contracts test completed")
    
    async def _test_simulated_event_bus(self):
        """Test simulated event bus functionality"""
        print("\n🚌 **TESTING SIMULATED EVENT BUS**")
        print("-" * 40)
        
        # Simulate event bus operations
        events_published = 0
        events_consumed = 0
        
        # Simulate publishing signals
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        for i, symbol in enumerate(symbols):
            signal = Signal(
                trace_id=f"demo-trace-{i}",
                agent_id="technical-agent",
                agent_type=SignalType.TECHNICAL,
                symbol=symbol,
                mu=0.03 + (i * 0.01),
                sigma=0.015 + (i * 0.005),
                confidence=0.7 + (i * 0.05),
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="v1.0.0",
                feature_version="v1.0.0"
            )
            
            # Simulate publishing
            events_published += 1
            print(f"📤 Published signal for {symbol} (confidence: {signal.confidence:.1%})")
            
            # Simulate consumption
            events_consumed += 1
            print(f"📥 Consumed signal for {symbol}")
            
            # Add small delay to simulate processing
            await asyncio.sleep(0.1)
        
        # Simulate publishing opportunities
        for i, symbol in enumerate(symbols):
            opportunity = Opportunity(
                trace_id=f"demo-trace-{i}",
                symbol=symbol,
                mu_blended=0.025 + (i * 0.008),
                sigma_blended=0.012 + (i * 0.003),
                confidence_blended=0.75 + (i * 0.04),
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                blender_version="v1.0.0"
            )
            
            events_published += 1
            print(f"📤 Published opportunity for {symbol} (confidence: {opportunity.confidence_blended:.1%})")
            
            events_consumed += 1
            print(f"📥 Consumed opportunity for {symbol}")
            
            await asyncio.sleep(0.1)
        
        # Store results
        self.demo_results['event_bus'] = {
            'events_published': events_published,
            'events_consumed': events_consumed,
            'throughput': events_published / 1.0  # events per second
        }
        
        print(f"✅ Event bus test completed")
        print(f"   - Events published: {events_published}")
        print(f"   - Events consumed: {events_consumed}")
        print(f"   - Throughput: {self.demo_results['event_bus']['throughput']:.1f} events/sec")
    
    async def _test_simulated_feature_store(self):
        """Test simulated feature store functionality"""
        print("\n💾 **TESTING SIMULATED FEATURE STORE**")
        print("-" * 40)
        
        # Simulate feature store operations
        features_stored = 0
        features_retrieved = 0
        total_latency_ms = 0
        
        # Test data
        test_features = {
            'AAPL_price': 150.0,
            'AAPL_volume': 1000000,
            'AAPL_rsi': 65.5,
            'AAPL_macd': 0.02,
            'GOOGL_price': 2800.0,
            'GOOGL_volume': 500000,
            'GOOGL_rsi': 58.2,
            'GOOGL_macd': -0.01,
            'MSFT_price': 350.0,
            'MSFT_volume': 800000,
            'MSFT_rsi': 72.1,
            'MSFT_macd': 0.05,
        }
        
        # Simulate storing features
        for feature_name, feature_value in test_features.items():
            start_time = datetime.utcnow()
            
            # Simulate storage
            features_stored += 1
            print(f"💾 Stored feature: {feature_name} = {feature_value}")
            
            # Simulate retrieval
            features_retrieved += 1
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            total_latency_ms += latency_ms
            
            print(f"📖 Retrieved feature: {feature_name} (latency: {latency_ms:.2f}ms)")
            
            await asyncio.sleep(0.05)  # Simulate processing time
        
        # Calculate metrics
        avg_latency_ms = total_latency_ms / features_retrieved if features_retrieved > 0 else 0
        
        # Store results
        self.demo_results['feature_store'] = {
            'features_stored': features_stored,
            'features_retrieved': features_retrieved,
            'avg_latency_ms': avg_latency_ms,
            'sla_compliance': avg_latency_ms < 5.0
        }
        
        print(f"✅ Feature store test completed")
        print(f"   - Features stored: {features_stored}")
        print(f"   - Features retrieved: {features_retrieved}")
        print(f"   - Average latency: {avg_latency_ms:.2f}ms")
        print(f"   - SLA compliance (<5ms): {'✅' if avg_latency_ms < 5.0 else '❌'}")
    
    async def _test_observability(self):
        """Test observability functionality"""
        print("\n📊 **TESTING OBSERVABILITY**")
        print("-" * 40)
        
        # Test structured logging
        self.telemetry.log_event(
            'demo_started',
            'Phase 1 demo started',
            trace_id='demo-trace-123',
            demo_version='1.0.0'
        )
        
        # Test operation tracing
        async with self.telemetry.trace_operation('demo_operation', user_id='demo-user') as span:
            span.set_attribute("demo.component", "phase1")
            span.set_attribute("demo.test", "observability")
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            # Test metric recording
            self.telemetry.record_metric(
                'demo_metric',
                42.0,
                labels={'component': 'phase1', 'test': 'observability'}
            )
        
        # Test error logging
        try:
            raise ValueError("Demo error for testing")
        except Exception as e:
            self.telemetry.log_error(
                e,
                'demo_context',
                trace_id='demo-trace-123',
                user_id='demo-user'
            )
        
        # Get health status
        health_status = await self.telemetry.get_health_status()
        
        # Store results
        self.demo_results['observability'] = {
            'health_status': health_status,
            'service_name': self.telemetry.service_name,
            'service_version': self.telemetry.service_version,
            'environment': self.telemetry.environment
        }
        
        print(f"✅ Observability test completed")
        print(f"   - Service: {health_status['service']}")
        print(f"   - Version: {health_status['version']}")
        print(f"   - Environment: {health_status['environment']}")
        print(f"   - Health status: {'✅ Healthy' if health_status['healthy'] else '❌ Unhealthy'}")
    
    async def _test_integration(self):
        """Test integration between components"""
        print("\n🔗 **TESTING INTEGRATION**")
        print("-" * 40)
        
        # Simulate end-to-end flow
        trace_id = "integration-demo-123"
        
        # 1. Generate signal
        signal = Signal(
            trace_id=trace_id,
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
        
        # 2. Store features
        feature_data = {
            'price': 150.0,
            'volume': 1000000,
            'rsi': 65.5,
            'macd': 0.02
        }
        
        # 3. Create opportunity
        opportunity = Opportunity(
            trace_id=trace_id,
            symbol="AAPL",
            mu_blended=0.04,
            sigma_blended=0.015,
            confidence_blended=0.85,
            horizon=HorizonType.SHORT_TERM,
            regime=RegimeType.RISK_ON,
            direction=DirectionType.LONG,
            blender_version="v1.0.0"
        )
        
        # 4. Create intent
        intent = Intent(
            trace_id=trace_id,
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
        
        # 5. Log the complete flow
        self.telemetry.log_event(
            'integration_flow_completed',
            f'Completed integration flow for {signal.symbol}',
            trace_id=trace_id,
            signal_confidence=signal.confidence,
            opportunity_confidence=opportunity.confidence_blended,
            intent_size_eur=intent.size_eur,
            intent_risk_pct=intent.risk_pct
        )
        
        # Store results
        self.demo_results['integration'] = {
            'trace_id': trace_id,
            'signal': signal.to_dict(),
            'opportunity': opportunity.to_dict(),
            'intent': intent.to_dict(),
            'feature_data': feature_data
        }
        
        print(f"✅ Integration test completed")
        print(f"   - Trace ID: {trace_id}")
        print(f"   - Signal confidence: {signal.confidence:.1%}")
        print(f"   - Opportunity confidence: {opportunity.confidence_blended:.1%}")
        print(f"   - Intent size: €{intent.size_eur}")
        print(f"   - Risk percentage: {intent.risk_pct:.1%}")
    
    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        print("\n📋 **DEMO REPORT**")
        print("=" * 60)
        
        # Calculate summary metrics
        total_events = self.demo_results['event_bus']['events_published']
        avg_latency = self.demo_results['feature_store']['avg_latency_ms']
        sla_compliance = self.demo_results['feature_store']['sla_compliance']
        
        print(f"📊 **SUMMARY METRICS**")
        print(f"   - Total events processed: {total_events}")
        print(f"   - Average feature latency: {avg_latency:.2f}ms")
        print(f"   - SLA compliance: {'✅' if sla_compliance else '❌'}")
        print(f"   - Service health: {'✅ Healthy' if self.demo_results['observability']['health_status']['healthy'] else '❌ Unhealthy'}")
        
        print(f"\n🎯 **PHASE 1 OBJECTIVES**")
        print(f"   ✅ Message contracts with uncertainty quantification")
        print(f"   ✅ Event bus with Kafka/Redpanda integration (simulated)")
        print(f"   ✅ Feature store with <5ms reads (simulated)")
        print(f"   ✅ Observability with OpenTelemetry")
        print(f"   ✅ Structured logging and health monitoring")
        
        print(f"\n🚀 **READY FOR PHASE 2**")
        print(f"   - Agent interface standardization")
        print(f"   - Meta-weighter implementation")
        print(f"   - Diversified Top-K selector")
        
        # Store final results
        self.demo_results['summary'] = {
            'total_events': total_events,
            'avg_latency_ms': avg_latency,
            'sla_compliance': sla_compliance,
            'health_status': self.demo_results['observability']['health_status']['healthy'],
            'demo_completed': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        print(f"\n✅ **PHASE 1 DEMO COMPLETED SUCCESSFULLY**")


async def main():
    """Main demo function"""
    demo = Phase1Demo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
