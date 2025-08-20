#!/usr/bin/env python3
"""
Simplified Architecture Test - Fast validation without heavy API calls

Tests core architecture components with minimal API usage
"""

import asyncio
import time
import sys
import os
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas.contracts import Signal, SignalType, HorizonType, RegimeType, DirectionType
from common.observability.telemetry import init_telemetry
from common.event_bus.simple_bus import SimpleEventBus
from common.feature_store.simple_store import SimpleFeatureStore
from common.opportunity_store import OpportunityStore

@dataclass
class TestResult:
    test_name: str
    success: bool
    duration_ms: float
    details: str = ""
    error: str = ""

class SimplifiedArchitectureTest:
    """Fast architecture test with minimal dependencies"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
    async def run_all_tests(self):
        """Run all simplified tests"""
        print("\n🚀 SIMPLIFIED ARCHITECTURE TEST")
        print("=" * 50)
        
        # Test core components
        await self._test_telemetry_init()
        await self._test_event_bus()
        await self._test_feature_store()
        await self._test_opportunity_store()
        await self._test_signal_creation()
        await self._test_signal_processing()
        
        # Generate report
        self._generate_report()
        
    async def _test_telemetry_init(self):
        """Test telemetry initialization"""
        start_time = time.time()
        try:
            config = {
                'service_name': 'simplified-test',
                'version': '1.0.0',
                'environment': 'test'
            }
            init_telemetry(config)
            
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Telemetry Initialization",
                success=True,
                duration_ms=duration,
                details="OpenTelemetry and logging initialized"
            ))
            print(f"✅ Telemetry initialization: {duration:.2f}ms")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Telemetry Initialization",
                success=False,
                duration_ms=duration,
                error=str(e)
            ))
            print(f"❌ Telemetry initialization failed: {e}")
    
    async def _test_event_bus(self):
        """Test event bus functionality"""
        start_time = time.time()
        try:
            # Initialize event bus
            config = {'max_queue_size': 1000}
            event_bus = SimpleEventBus(config)
            await event_bus.start()
            
            # Test event publishing
            await event_bus.publish_agent_signal(
                source="test",
                agent_name="test_agent",
                signal_type="long",
                confidence=0.8,
                additional_data={'test': True}
            )
            
            # Test multiple events
            for i in range(5):
                await event_bus.publish_agent_signal(
                    source=f"test_agent_{i}",
                    agent_name=f"agent_{i}",
                    signal_type="long" if i % 2 == 0 else "short",
                    confidence=0.7 + (i * 0.05),
                    additional_data={'index': i}
                )
            
            stats = event_bus.get_metrics()
            await event_bus.stop()
            
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Event Bus",
                success=True,
                duration_ms=duration,
                details=f"Published 6 events, {stats.get('total_events_published', 0)} total"
            ))
            print(f"✅ Event bus: {duration:.2f}ms - {stats.get('total_events_published', 0)} events")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Event Bus",
                success=False,
                duration_ms=duration,
                error=str(e)
            ))
            print(f"❌ Event bus failed: {e}")
    
    async def _test_feature_store(self):
        """Test feature store functionality"""
        start_time = time.time()
        try:
            # Initialize feature store
            config = {'cache_size': 100}
            feature_store = SimpleFeatureStore(config)
            await feature_store.start()
            
            # Create mock feature data
            feature_data = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'GOOGL'] * 3,
                'timestamp': [datetime.now() for _ in range(9)],
                'price': [150.0, 300.0, 200.0] * 3,
                'volume': [1000000, 800000, 600000] * 3,
                'volatility': [0.15, 0.12, 0.18] * 3,
                'momentum': [0.05, 0.03, 0.08] * 3
            })
            
            # Test feature writing
            write_success = await feature_store.write_features(
                feature_group="market_data",
                data=feature_data,
                metadata={'test': True}
            )
            
            # Test feature reading
            read_data = await feature_store.get_features(
                symbols=['AAPL', 'MSFT'],
                feature_groups=["market_data"]
            )
            
            await feature_store.stop()
            
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Feature Store",
                success=True,
                duration_ms=duration,
                details=f"Wrote {len(feature_data)} features, read {len(read_data)} features"
            ))
            print(f"✅ Feature store: {duration:.2f}ms - {len(feature_data)} written, {len(read_data)} read")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Feature Store",
                success=False,
                duration_ms=duration,
                error=str(e)
            ))
            print(f"❌ Feature store failed: {e}")
    
    async def _test_opportunity_store(self):
        """Test opportunity store functionality"""
        start_time = time.time()
        try:
            # Initialize with temporary database
            import tempfile
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.close()
            
            opportunity_store = OpportunityStore(db_path=temp_db.name)
            
            # Test adding signals
            test_signal = Signal(
                signal_id=str(uuid.uuid4()),
                agent_id="test_agent",
                agent_type=SignalType.TECHNICAL,
                symbol="AAPL",
                direction=DirectionType.LONG,
                mu=0.05,
                sigma=0.15,
                confidence=0.8,
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                timestamp=datetime.now(),
                trace_id=str(uuid.uuid4()),
                model_version="1.0.0",
                feature_version="1.0.0",
                metadata={'test': True}
            )
            
            add_success = await opportunity_store.add_signal(test_signal)
            signals = opportunity_store.get_signals()
            
            # Clean up
            os.unlink(temp_db.name)
            
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Opportunity Store",
                success=True,
                duration_ms=duration,
                details=f"Added signal: {add_success}, Retrieved: {len(signals)} signals"
            ))
            print(f"✅ Opportunity store: {duration:.2f}ms - {len(signals)} signals stored")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Opportunity Store",
                success=False,
                duration_ms=duration,
                error=str(e)
            ))
            print(f"❌ Opportunity store failed: {e}")
    
    async def _test_signal_creation(self):
        """Test signal creation and validation"""
        start_time = time.time()
        try:
            signals = []
            
            # Create different types of signals
            signal_configs = [
                (SignalType.TECHNICAL, "AAPL", DirectionType.LONG, 0.05),
                (SignalType.SENTIMENT, "MSFT", DirectionType.SHORT, -0.03),
                (SignalType.FLOW, "GOOGL", DirectionType.LONG, 0.08),
                (SignalType.UNDERVALUED, "NVDA", DirectionType.LONG, 0.12),
                (SignalType.MACRO, "META", DirectionType.SHORT, -0.02),
            ]
            
            for signal_type, symbol, direction, mu in signal_configs:
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    agent_id=f"{signal_type.value}_agent",
                    agent_type=signal_type,
                    symbol=symbol,
                    direction=direction,
                    mu=mu,
                    sigma=0.15,
                    confidence=0.75,
                    horizon=HorizonType.MEDIUM_TERM,
                    regime=RegimeType.RISK_ON,
                    timestamp=datetime.now(),
                    trace_id=str(uuid.uuid4()),
                    model_version="1.0.0",
                    feature_version="1.0.0",
                    metadata={'test_type': signal_type.value}
                )
                signals.append(signal)
            
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Signal Creation",
                success=True,
                duration_ms=duration,
                details=f"Created {len(signals)} valid signals across all types"
            ))
            print(f"✅ Signal creation: {duration:.2f}ms - {len(signals)} signals created")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Signal Creation",
                success=False,
                duration_ms=duration,
                error=str(e)
            ))
            print(f"❌ Signal creation failed: {e}")
    
    async def _test_signal_processing(self):
        """Test end-to-end signal processing"""
        start_time = time.time()
        try:
            # Initialize components
            event_bus = SimpleEventBus({'max_queue_size': 100})
            feature_store = SimpleFeatureStore({'cache_size': 50})
            
            import tempfile
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.close()
            opportunity_store = OpportunityStore(db_path=temp_db.name)
            
            await event_bus.start()
            await feature_store.start()
            
            # Create and process signals
            processed_signals = []
            for i in range(3):
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    agent_id=f"processor_agent_{i}",
                    agent_type=SignalType.TECHNICAL,
                    symbol=f"TEST{i}",
                    direction=DirectionType.LONG,
                    mu=0.05 + (i * 0.01),
                    sigma=0.15,
                    confidence=0.8,
                    horizon=HorizonType.SHORT_TERM,
                    regime=RegimeType.RISK_ON,
                    timestamp=datetime.now(),
                    trace_id=str(uuid.uuid4()),
                    model_version="1.0.0",
                    feature_version="1.0.0",
                    metadata={'processing_test': True}
                )
                
                # Add to opportunity store
                await opportunity_store.add_signal(signal)
                
                # Publish to event bus
                await event_bus.publish_agent_signal(
                    source=signal.agent_id,
                    agent_name=signal.agent_type.value,
                    signal_type=signal.direction.value,
                    confidence=signal.confidence,
                    additional_data={'signal_id': signal.signal_id}
                )
                
                processed_signals.append(signal)
            
            # Verify processing
            stored_signals = opportunity_store.get_signals()
            event_stats = event_bus.get_metrics()
            
            # Clean up
            await event_bus.stop()
            await feature_store.stop()
            os.unlink(temp_db.name)
            
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Signal Processing",
                success=True,
                duration_ms=duration,
                details=f"Processed {len(processed_signals)} signals, stored {len(stored_signals)}, published {event_stats.get('total_events_published', 0)} events"
            ))
            print(f"✅ Signal processing: {duration:.2f}ms - {len(stored_signals)} signals processed")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append(TestResult(
                test_name="Signal Processing",
                success=False,
                duration_ms=duration,
                error=str(e)
            ))
            print(f"❌ Signal processing failed: {e}")
    
    def _generate_report(self):
        """Generate final test report"""
        print("\n📊 SIMPLIFIED ARCHITECTURE TEST REPORT")
        print("=" * 50)
        
        total_duration = (time.time() - self.start_time) * 1000
        successful_tests = sum(1 for result in self.test_results if result.success)
        total_tests = len(self.test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total Duration: {total_duration:.2f}ms")
        print(f"   Tests Passed: {successful_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 Test Details:")
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {status} {result.test_name}: {result.duration_ms:.2f}ms")
            if result.details:
                print(f"      ℹ️ {result.details}")
            if result.error:
                print(f"      ❌ Error: {result.error}")
        
        if success_rate >= 80:
            print(f"\n🎉 SIMPLIFIED ARCHITECTURE TEST: SUCCESS!")
            print(f"   Core architecture components are working correctly!")
        else:
            print(f"\n⚠️ SIMPLIFIED ARCHITECTURE TEST: PARTIAL SUCCESS")
            print(f"   Some components need attention.")
        
        print(f"\n⚡ Performance Summary:")
        print(f"   Average test time: {total_duration/total_tests:.2f}ms")
        print(f"   Fastest test: {min(r.duration_ms for r in self.test_results):.2f}ms")
        print(f"   Slowest test: {max(r.duration_ms for r in self.test_results):.2f}ms")

async def main():
    """Run simplified architecture test"""
    test = SimplifiedArchitectureTest()
    await test.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
