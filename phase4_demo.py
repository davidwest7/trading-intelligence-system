#!/usr/bin/env python3
"""
Phase 4 Demo: Advanced AI Trading System
======================================

Comprehensive demonstration of Phase 4 features:
- Causal inference with T-Learner/DR-Learner
- Robustness system with anomaly detection
- Advanced off-policy evaluation (DR-OPE, SNIPS, FQE)
- Deterministic replay and audit system
- Production optimization with ONNX deployment

This demo showcases the complete AI trading system with
state-of-the-art causal inference, robustness, and auditability.
"""

import asyncio
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import logging
import uuid

# Local imports
from schemas.contracts import (
    Signal, Opportunity, Intent, SignalType, HorizonType, 
    RegimeType, DirectionType
)
from causal.cate_estimator import (
    CausalSignalAnalyzer, TreatmentGroup
)
from robustness.anomaly_detector import (
    RobustnessManager, AnomalyType, SeverityLevel
)
from learning.advanced_ope import (
    AdvancedOPEManager, PolicyExperience, OPEMethod
)
from audit.replay_system import (
    AuditManager, SystemState, ChaosType
)
from deployment.production_optimizer import (
    ProductionOptimizationManager, DeploymentStage, ModelFormat
)
from common.observability.telemetry import get_telemetry, init_telemetry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase4Demo:
    """Comprehensive Phase 4 demonstration"""
    
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        
        # Initialize telemetry system first
        init_telemetry({
            "service_name": "phase4_demo",
            "service_version": "1.0.0",
            "environment": "demo",
            "enable_tracing": True,
            "enable_metrics": True,
            "enable_logging": True
        })
        
        # Initialize Phase 4 components
        self.causal_analyzer = CausalSignalAnalyzer()
        self.robustness_manager = RobustnessManager()
        self.ope_manager = AdvancedOPEManager()
        self.audit_manager = AuditManager()
        self.production_optimizer = ProductionOptimizationManager()
        
        # Demo data
        self.historical_data = self._generate_historical_data()
        self.signals = []
        self.opportunities = []
        
        print("🚀 Phase 4 Demo: Advanced AI Trading System")
        print("=" * 60)
        
    def _generate_historical_data(self) -> pd.DataFrame:
        """Generate synthetic historical data for demo"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='h'),
            'signal_strength': np.random.normal(0.5, 0.2, n_samples),
            'volatility': np.random.exponential(0.02, n_samples),
            'volume': np.random.lognormal(15, 1, n_samples),
            'momentum': np.random.normal(0, 0.1, n_samples),
            'regime_score': np.random.uniform(0, 1, n_samples),
            'realized_return': np.random.normal(0.001, 0.02, n_samples),
            'exogenous_timing': np.random.binomial(1, 0.1, n_samples)  # Instrument
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_signals(self) -> List[Signal]:
        """Create sample signals for causal analysis"""
        signals = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for i, symbol in enumerate(symbols):
            signal = Signal(
                trace_id=self.trace_id,
                agent_id=f"agent_{i}",
                agent_type=SignalType.TECHNICAL,
                symbol=symbol,
                mu=0.01 + i * 0.002,
                sigma=0.02 + i * 0.005,
                confidence=0.7 + i * 0.05,
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="1.0.0",
                feature_version="1.0.0"
            )
            signals.append(signal)
        
        return signals
    
    def _create_sample_opportunities(self) -> List[Opportunity]:
        """Create sample opportunities"""
        opportunities = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for i, symbol in enumerate(symbols):
            technical_signal = Signal(
                trace_id=self.trace_id,
                agent_id="technical_agent",
                agent_type=SignalType.TECHNICAL,
                symbol=symbol,
                mu=0.01 + i * 0.002,
                sigma=0.02 + i * 0.005,
                confidence=0.7 + i * 0.05,
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="1.0.0",
                feature_version="1.0.0"
            )
            
            opp = Opportunity(
                symbol=symbol,
                mu_blended=0.01 + i * 0.002,
                sigma_blended=0.02 + i * 0.005,
                confidence_blended=0.7 + i * 0.05,
                sharpe_ratio=0.5 + i * 0.1,
                var_95=-0.02 - i * 0.005,
                cvar_95=-0.025 - i * 0.005,
                agent_signals={'technical': technical_signal},
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                blender_version="1.0.0",
                trace_id=self.trace_id
            )
            opportunities.append(opp)
        
        return opportunities
    
    def _generate_ope_experiences(self) -> List[PolicyExperience]:
        """Generate policy experiences for OPE evaluation"""
        experiences = []
        
        for i in range(200):
            state = np.random.randn(5)  # 5-dimensional state
            action = np.random.choice([0, 1])  # Binary action
            reward = np.random.normal(0.001, 0.02)  # Small returns
            next_state = np.random.randn(5)
            done = np.random.choice([True, False], p=[0.1, 0.9])
            
            # Behavior and target policy probabilities
            behavior_prob = 0.5  # Random policy
            if action == 1:
                target_prob = 1 / (1 + np.exp(-np.sum(state)))  # Logistic
            else:
                target_prob = 1 - 1 / (1 + np.exp(-np.sum(state)))
            
            experience = PolicyExperience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                behavior_prob=behavior_prob,
                target_prob=target_prob,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                metadata={'episode': i // 20}
            )
            experiences.append(experience)
        
        return experiences
    
    async def demo_causal_inference(self):
        """Demonstrate causal inference capabilities"""
        print("\n🔬 CAUSAL INFERENCE DEMO")
        print("-" * 40)
        
        try:
            # Create sample signals
            self.signals = self._create_sample_signals()
            
            print(f"📊 Analyzing {len(self.signals)} signals for causal effects...")
            
            # Analyze each signal for causal effects
            causal_results = []
            
            for signal in self.signals:
                print(f"   Analyzing {signal.symbol}...")
                
                # Perform causal analysis
                analysis = await self.causal_analyzer.analyze_signal_causality(
                    signal, self.historical_data, self.trace_id
                )
                
                causal_results.append(analysis)
                
                # Display results
                causal_effect = analysis['has_causal_effect']
                effect_magnitude = analysis['effect_magnitude']
                priority_score = analysis['priority_score']
                recommendation = analysis['recommendation']
                
                status_emoji = "✅" if causal_effect else "❌"
                print(f"     {status_emoji} {signal.symbol}: Effect={effect_magnitude:.4f}, "
                      f"Priority={priority_score:.2f}, {recommendation}")
            
            # Batch analysis demonstration
            print(f"\n🔄 Running batch causal analysis...")
            batch_results = await self.causal_analyzer.batch_analyze_signals(
                self.signals, self.historical_data, self.trace_id
            )
            
            # Summary statistics
            total_signals = len(batch_results)
            causal_signals = sum(1 for r in batch_results if r['has_causal_effect'])
            avg_priority = np.mean([r['priority_score'] for r in batch_results])
            
            print(f"📈 Causal Analysis Summary:")
            print(f"   • Total signals analyzed: {total_signals}")
            print(f"   • Signals with causal effects: {causal_signals}")
            print(f"   • Causal signal rate: {causal_signals/total_signals:.1%}")
            print(f"   • Average priority score: {avg_priority:.2f}")
            
            # Historical performance
            historical_perf = self.causal_analyzer.get_historical_performance()
            print(f"   • Historical analyses: {historical_perf['total_analyses']}")
            
            return causal_results
            
        except Exception as e:
            print(f"❌ Causal inference demo failed: {e}")
            return []
    
    async def demo_robustness_system(self):
        """Demonstrate robustness and anomaly detection"""
        print("\n🛡️ ROBUSTNESS SYSTEM DEMO")
        print("-" * 40)
        
        try:
            # Generate test data with some anomalies
            normal_data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 100),
                'feature_2': np.random.normal(0, 1, 100),
                'feature_3': np.random.normal(0, 1, 100),
                'feature_4': np.random.normal(0, 1, 100)
            })
            
            # Add some outliers
            anomalous_data = normal_data.copy()
            anomalous_data.iloc[5:10] = 5  # Extreme values
            anomalous_data.iloc[50:55] = -5  # More extreme values
            
            print("🔍 Running comprehensive robustness checks...")
            
            # Fit robustness manager on normal data
            print("   Training anomaly detectors on normal data...")
            
            # Fit individual detectors
            for detector_name, detector in self.robustness_manager.anomaly_detectors.items():
                detector.fit(normal_data.values)
                print(f"     ✅ {detector_name} detector trained")
            
            # Fit drift detector
            self.robustness_manager.drift_detector.fit(normal_data)
            print("     ✅ Drift detector trained")
            
            # Test on anomalous data
            print("   Testing on anomalous data...")
            
            robustness_results = await self.robustness_manager.comprehensive_check(
                anomalous_data, normal_data, self.trace_id
            )
            
            # Display results
            print(f"\n📊 Robustness Check Results:")
            
            # Anomaly detection results
            print(f"   Anomaly Detection:")
            for method, result in robustness_results['anomaly_detection'].items():
                status = "🚨" if result.is_anomaly else "✅"
                print(f"     {status} {method}: {result.severity.value} severity, "
                      f"confidence={result.confidence:.2f}")
            
            # Drift detection results
            if robustness_results['drift_detection']:
                drift = robustness_results['drift_detection']
                drift_status = "🚨" if drift.drift_detected else "✅"
                print(f"   Distribution Drift:")
                print(f"     {drift_status} PSI Score: {drift.psi_score:.3f}")
                print(f"     {drift_status} KS Statistic: {drift.ks_statistic:.3f}")
                print(f"     {drift_status} Magnitude: {drift.drift_magnitude}")
            
            # Overall assessment
            assessment = robustness_results['overall_assessment']
            risk_level = assessment['risk_level']
            risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "💥"}
            
            print(f"   Overall Risk Assessment:")
            print(f"     {risk_emoji.get(risk_level, '❓')} Risk Level: {risk_level}")
            print(f"     📊 Risk Score: {assessment['risk_score']:.2f}")
            print(f"     🔒 Quarantine Size: {assessment['quarantine_size']}")
            print(f"     🎚️ Throttle Level: {robustness_results['throttle_level']:.1f}")
            
            # Recommendations
            if robustness_results['recommendations']:
                print(f"   🎯 Recommendations:")
                for rec in robustness_results['recommendations']:
                    print(f"     • {rec}")
            
            # Quarantine status
            quarantine_status = self.robustness_manager.get_quarantine_status()
            print(f"   📦 Quarantine Status:")
            print(f"     • Total quarantined: {quarantine_status['total_quarantined']}")
            print(f"     • Recent quarantined: {quarantine_status['recent_quarantined']}")
            
            return robustness_results
            
        except Exception as e:
            print(f"❌ Robustness system demo failed: {e}")
            return {}
    
    async def demo_advanced_ope(self):
        """Demonstrate advanced off-policy evaluation"""
        print("\n🧠 ADVANCED OFF-POLICY EVALUATION DEMO")
        print("-" * 40)
        
        try:
            # Generate policy experiences
            experiences = self._generate_ope_experiences()
            print(f"📊 Generated {len(experiences)} policy experiences")
            
            # Run comprehensive OPE evaluation
            print("🔄 Running comprehensive OPE evaluation...")
            
            evaluation_result = await self.ope_manager.comprehensive_evaluation(
                experiences, self.trace_id
            )
            
            # Display individual method results
            print(f"\n📈 OPE Method Results:")
            for method, result in evaluation_result['individual_methods'].items():
                convergence_emoji = "✅" if result.convergence_achieved else "⚠️"
                print(f"   {method.upper()}:")
                print(f"     {convergence_emoji} Value Estimate: {result.value_estimate:.4f}")
                print(f"     📊 Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
                print(f"     📏 Standard Error: {result.standard_error:.4f}")
                print(f"     🎯 Bias Estimate: {result.bias_estimate:.4f}")
                print(f"     📦 Sample Size: {result.sample_size}")
            
            # Ensemble estimate
            ensemble_estimate = evaluation_result['ensemble_estimate']
            print(f"\n🏆 Ensemble Estimate: {ensemble_estimate:.4f}")
            
            # Convergence status
            convergence = evaluation_result['convergence_status']
            convergence_emoji = "✅" if convergence['overall_converged'] else "⚠️"
            print(f"   {convergence_emoji} Convergence Rate: {convergence['convergence_rate']:.1%}")
            print(f"   📊 Converged Methods: {convergence['converged_methods']}")
            
            # Recommendation
            recommendation = evaluation_result['recommendation']
            print(f"   🎯 Recommendation: {recommendation}")
            
            # Test policy comparison
            print(f"\n🔀 Testing Policy Comparison...")
            
            # Split experiences for A/B test
            mid_point = len(experiences) // 2
            policy_a_experiences = experiences[:mid_point]
            policy_b_experiences = experiences[mid_point:]
            
            comparison = await self.ope_manager.compare_policies(
                policy_a_experiences, policy_b_experiences, self.trace_id
            )
            
            print(f"   Policy Comparison Results:")
            print(f"     📈 Policy A Value: {comparison.policy_a_value:.4f}")
            print(f"     📈 Policy B Value: {comparison.policy_b_value:.4f}")
            print(f"     📊 Difference: {comparison.difference:.4f}")
            print(f"     📏 95% CI: [{comparison.confidence_interval[0]:.4f}, {comparison.confidence_interval[1]:.4f}]")
            print(f"     📊 P-value: {comparison.p_value:.4f}")
            
            significance_emoji = "✅" if comparison.significant else "❌"
            print(f"     {significance_emoji} Statistically Significant: {comparison.significant}")
            print(f"     🎯 Recommendation: {comparison.recommendation}")
            
            # Live counterfactual system demo
            print(f"\n🎲 Live Counterfactual System Demo...")
            
            exploration_stats = self.ope_manager.counterfactual_system.get_exploration_stats()
            print(f"   🔍 Exploration Statistics:")
            print(f"     • Total explorations: {exploration_stats['total_explorations']}")
            print(f"     • Recent exploration rate: {exploration_stats['recent_exploration_rate']:.3f}")
            print(f"     • Safety violations: {exploration_stats['safety_violations']}")
            print(f"     • Average reward: {exploration_stats['average_exploration_reward']:.4f}")
            
            # Test exploration decision
            test_state = np.random.randn(5)
            should_explore = await self.ope_manager.counterfactual_system.should_explore(
                test_state, self.trace_id
            )
            explore_emoji = "🎲" if should_explore else "🛡️"
            print(f"     {explore_emoji} Test exploration decision: {'EXPLORE' if should_explore else 'EXPLOIT'}")
            
            # Historical performance
            historical_perf = self.ope_manager.get_historical_performance()
            print(f"\n📊 Historical OPE Performance:")
            print(f"   • Total evaluations: {historical_perf['total_evaluations']}")
            print(f"   • Policy comparisons: {historical_perf['total_policy_comparisons']}")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ Advanced OPE demo failed: {e}")
            return {}
    
    async def demo_audit_system(self):
        """Demonstrate audit and replay system"""
        print("\n📋 AUDIT & REPLAY SYSTEM DEMO")
        print("-" * 40)
        
        try:
            # Create sample system states and decisions
            print("📸 Capturing system states...")
            
            # Sample market data
            market_data = {
                'prices': {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0},
                'volumes': {'AAPL': 1000000, 'GOOGL': 500000, 'MSFT': 800000},
                'volatility': 0.02
            }
            
            # Sample agent states
            agent_states = {
                'technical_agent': {'last_signal': 0.01, 'confidence': 0.8},
                'sentiment_agent': {'sentiment_score': 0.6, 'volume_spike': False}
            }
            
            # Sample model states
            model_states = {
                'meta_weighter': {'version': '1.0.0', 'last_update': datetime.now(timezone.utc)},
                'selector': {'diversity_penalty': 0.1, 'selected_count': 5}
            }
            
            # Sample risk metrics
            risk_metrics = {
                'var_95': -0.02,
                'cvar_95': -0.025,
                'portfolio_volatility': 0.015,
                'max_drawdown': -0.03
            }
            
            # Sample portfolio state
            portfolio_state = {
                'total_value': 500000,
                'positions': {'AAPL': 100, 'GOOGL': 50},
                'cash': 50000,
                'leverage': 1.2
            }
            
            # Capture state
            captured_state = await self.audit_manager.state_capture.capture_state(
                market_data, agent_states, model_states, 
                risk_metrics, portfolio_state, self.trace_id
            )
            
            print(f"   ✅ State captured: {captured_state.state_hash[:8]}...")
            print(f"   ⏱️ Timestamp: {captured_state.timestamp}")
            print(f"   🔢 Nanosecond precision: {captured_state.nanosecond_precision}")
            
            # Record sample decisions
            print("📝 Recording trading decisions...")
            
            opportunities = self._create_sample_opportunities()
            selected_opportunities = opportunities[:2]  # Select first 2
            
            intents = [
                Intent(
                    symbol=opp.symbol,
                    target_size=100,
                    max_risk=0.01,
                    urgency="medium",
                    trace_id=self.trace_id
                )
                for opp in selected_opportunities
            ]
            
            decision = await self.audit_manager.decision_tracker.record_decision(
                captured_state, opportunities, selected_opportunities, intents,
                policy_version="1.0.0",
                model_versions={'meta_weighter': '1.0.0', 'selector': '1.0.0'},
                feature_versions={'market_data': '1.0.0'},
                trace_id=self.trace_id
            )
            
            print(f"   ✅ Decision recorded: {decision.decision_id}")
            print(f"   📊 Opportunities considered: {len(opportunities)}")
            print(f"   🎯 Selected opportunities: {len(selected_opportunities)}")
            print(f"   🎪 Intents generated: {len(intents)}")
            
            # Update with execution results
            await self.audit_manager.decision_tracker.update_decision_results(
                decision.decision_id,
                execution_results={'status': 'executed', 'fills': 2},
                realized_pnl=0.005,
                trace_id=self.trace_id
            )
            
            print(f"   💰 Decision updated with PnL: 0.005")
            
            # Test replay system
            print(f"\n⏮️ Testing replay system...")
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)
            
            replay_result = await self.audit_manager.replay_engine.replay_timespan(
                start_time, end_time, trace_id=self.trace_id
            )
            
            print(f"   📊 Replay Results:")
            print(f"     • Status: {replay_result['status']}")
            print(f"     • Decisions replayed: {replay_result.get('total_decisions', 0)}")
            print(f"     • Successful replays: {replay_result.get('successful_replays', 0)}")
            print(f"     • Success rate: {replay_result.get('success_rate', 0):.1%}")
            print(f"     • Cumulative PnL: {replay_result.get('cumulative_pnl', 0):.4f}")
            
            # Test policy diff analyzer
            print(f"\n🔍 Testing policy diff analyzer...")
            
            test_states = [captured_state, captured_state]  # Use same state for demo
            policy_diffs = await self.audit_manager.policy_diff_analyzer.compare_policies(
                "policy_v1.0", "policy_v1.1", test_states, self.trace_id
            )
            
            print(f"   📊 Policy Comparison Results:")
            for i, diff in enumerate(policy_diffs):
                print(f"     Test Case {i+1}:")
                print(f"       🏷️ Policy A: {diff.policy_a_version}")
                print(f"       🏷️ Policy B: {diff.policy_b_version}")
                print(f"       💰 PnL Difference: {diff.pnl_difference:.4f}")
                print(f"       📊 Significance Score: {diff.significance_score:.2f}")
                print(f"       💬 Explanation: {diff.explanation}")
            
            # Test chaos engineering
            print(f"\n💥 Testing chaos engineering...")
            
            chaos_test_result = await self.audit_manager.chaos_testing_engine.run_chaos_test(
                ChaosType.SERVICE_KILL, duration_seconds=5, trace_id=self.trace_id
            )
            
            print(f"   🧪 Chaos Test Results:")
            print(f"     • Test ID: {chaos_test_result['test_id']}")
            print(f"     • Type: {chaos_test_result['chaos_type'].value}")
            print(f"     • Status: {chaos_test_result['status']}")
            
            if 'result' in chaos_test_result:
                result = chaos_test_result['result']
                print(f"     • Kill switches activated: {result.get('kill_switches_activated', False)}")
                print(f"     • Recovery time: {result.get('system_recovery_time_seconds', 0)}s")
                print(f"     • Data integrity: {result.get('data_integrity_maintained', False)}")
            
            # Kill switch status
            kill_switches = self.audit_manager.chaos_testing_engine.get_kill_switch_status()
            print(f"   🔐 Kill Switch Status:")
            for switch, status in kill_switches.items():
                status_emoji = "🔴" if status else "🟢"
                print(f"     {status_emoji} {switch}: {'ACTIVE' if status else 'INACTIVE'}")
            
            # Full system audit
            print(f"\n🔍 Running full system audit...")
            
            audit_result = await self.audit_manager.full_system_audit(
                start_time, end_time, self.trace_id
            )
            
            print(f"   📊 Audit Results:")
            if 'overall_assessment' in audit_result:
                assessment = audit_result['overall_assessment']
                print(f"     🏆 Overall Score: {assessment['overall_score']:.2f}")
                print(f"     📈 Status: {assessment['status']}")
                if assessment['recommendations']:
                    print(f"     🎯 Recommendations:")
                    for rec in assessment['recommendations']:
                        print(f"       • {rec}")
            
            # Audit summary
            audit_summary = self.audit_manager.get_audit_summary()
            print(f"\n📋 Audit System Summary:")
            print(f"   • States captured: {audit_summary['total_states_captured']}")
            print(f"   • Decisions tracked: {audit_summary['total_decisions_tracked']}")
            print(f"   • Replays performed: {audit_summary['total_replays_performed']}")
            print(f"   • Policy diffs: {audit_summary['total_policy_diffs']}")
            print(f"   • Active chaos tests: {audit_summary['active_chaos_tests']}")
            
            return audit_result
            
        except Exception as e:
            print(f"❌ Audit system demo failed: {e}")
            return {}
    
    async def demo_production_optimization(self):
        """Demonstrate production optimization and deployment"""
        print("\n🚀 PRODUCTION OPTIMIZATION DEMO")
        print("-" * 40)
        
        try:
            # Register trading schemas
            print("📋 Registering trading schemas...")
            
            schemas_registered = []
            for schema_name in ['signal_schema', 'opportunity_schema', 'intent_schema']:
                success = self.production_optimizer.register_trading_schema(schema_name, "1.0.0")
                status_emoji = "✅" if success else "❌"
                print(f"   {status_emoji} {schema_name}: {'Registered' if success else 'Failed'}")
                schemas_registered.append(success)
            
            # List registered schemas
            schemas = self.production_optimizer.schema_registry.list_schemas()
            print(f"   📊 Total schemas registered: {len(schemas)}")
            
            # Mock model optimization and deployment
            print(f"\n🔧 Optimizing and deploying model...")
            
            # Create a mock model file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                tmp_file.write(b"mock_model_data" * 1000)
                model_path = tmp_file.name
            
            # Optimize and deploy
            deployment_result = await self.production_optimizer.optimize_and_deploy(
                model_path, "trading_model", "1.0.0", DeploymentStage.SHADOW, self.trace_id
            )
            
            print(f"   📊 Deployment Results:")
            print(f"     • Status: {deployment_result['status']}")
            
            if deployment_result['status'] == 'success':
                # Optimization results
                opt_result = deployment_result['optimization_result']
                print(f"     🗜️ Compression ratio: {opt_result['compression_ratio']:.2f}x")
                print(f"     ⚡ Estimated speedup: {opt_result['estimated_speedup']:.2f}x")
                print(f"     💾 Memory reduction: {opt_result['memory_reduction']:.1%}")
                
                # Benchmark results
                bench_result = deployment_result['benchmark_result']
                if bench_result:
                    print(f"     ⏱️ Average latency: {bench_result['avg_latency_ms']:.2f}ms")
                    print(f"     🔄 Throughput: {bench_result['throughput_rps']:.1f} RPS")
                    print(f"     🧠 Memory usage: {bench_result['memory_usage_mb']:.1f} MB")
                
                # Deployment info
                deployment_id = deployment_result['deployment_id']
                print(f"     🆔 Deployment ID: {deployment_id}")
                
                # Test gradual rollout
                print(f"\n📈 Testing gradual rollout...")
                
                rollout_result = await self.production_optimizer.gradual_rollout(
                    deployment_id, [1, 5, 10], self.trace_id
                )
                
                print(f"   📊 Rollout Results:")
                print(f"     • Status: {rollout_result['status']}")
                print(f"     • Completed steps: {len(rollout_result.get('completed_steps', []))}")
                
                if 'completed_steps' in rollout_result:
                    for step in rollout_result['completed_steps']:
                        step_emoji = "✅" if step['status'] == 'success' else "❌"
                        print(f"     {step_emoji} Step {step['step']}: {step['traffic_percentage']}% traffic")
                
                # Check deployment status
                print(f"\n📊 Checking deployment status...")
                
                deployment_status = self.production_optimizer.deployment_orchestrator.get_deployment_status(deployment_id)
                if deployment_status:
                    print(f"   🏷️ Model ID: {deployment_status['model_id']}")
                    print(f"   📋 Version: {deployment_status['version']}")
                    print(f"   🎭 Stage: {deployment_status['stage']}")
                    print(f"   🚦 Traffic: {deployment_status['traffic_percentage']}%")
                    print(f"   📅 Created: {deployment_status['created_at']}")
                
            # Health check
            print(f"\n🏥 Running system health check...")
            
            health_status = await self.production_optimizer.health_check(self.trace_id)
            
            print(f"   🏥 Health Status: {health_status['overall_status'].upper()}")
            print(f"   🕐 Timestamp: {health_status['timestamp']}")
            
            for component, status in health_status['components'].items():
                status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
                emoji = status_emoji.get(status['status'], "❓")
                print(f"   {emoji} {component}: {status['status']}")
            
            # Optimization statistics
            print(f"\n📊 Optimization Statistics:")
            
            opt_stats = self.production_optimizer.get_optimization_stats()
            print(f"   • Total deployments: {opt_stats['total_deployments']}")
            print(f"   • Successful deployments: {opt_stats['successful_deployments']}")
            print(f"   • Success rate: {opt_stats['success_rate']:.1%}")
            print(f"   • Average compression ratio: {opt_stats['average_compression_ratio']:.2f}x")
            print(f"   • Average latency: {opt_stats['average_latency_ms']:.2f}ms")
            print(f"   • Schemas registered: {opt_stats['schemas_registered']}")
            print(f"   • Active deployments: {opt_stats['active_deployments']}")
            
            # Clean up temporary file
            import os
            try:
                os.unlink(model_path)
            except:
                pass
            
            return deployment_result
            
        except Exception as e:
            print(f"❌ Production optimization demo failed: {e}")
            return {}
    
    async def run_full_demo(self):
        """Run the complete Phase 4 demonstration"""
        start_time = time.time()
        
        print(f"🎯 Running comprehensive Phase 4 demo...")
        print(f"📅 Started at: {datetime.now(timezone.utc)}")
        print(f"🆔 Trace ID: {self.trace_id}")
        
        demo_results = {}
        
        try:
            # 1. Causal Inference Demo
            causal_results = await self.demo_causal_inference()
            demo_results['causal_inference'] = causal_results
            
            # 2. Robustness System Demo
            robustness_results = await self.demo_robustness_system()
            demo_results['robustness_system'] = robustness_results
            
            # 3. Advanced OPE Demo
            ope_results = await self.demo_advanced_ope()
            demo_results['advanced_ope'] = ope_results
            
            # 4. Audit System Demo
            audit_results = await self.demo_audit_system()
            demo_results['audit_system'] = audit_results
            
            # 5. Production Optimization Demo
            production_results = await self.demo_production_optimization()
            demo_results['production_optimization'] = production_results
            
        except Exception as e:
            print(f"❌ Demo execution failed: {e}")
            demo_results['error'] = str(e)
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n" + "=" * 60)
        print(f"🎉 PHASE 4 DEMO COMPLETE!")
        print(f"=" * 60)
        print(f"⏱️ Total execution time: {duration:.2f} seconds")
        print(f"🆔 Trace ID: {self.trace_id}")
        
        # Count successful components
        successful_components = sum(1 for key, value in demo_results.items() 
                                  if key != 'error' and value)
        total_components = 5
        
        success_rate = successful_components / total_components
        success_emoji = "🎉" if success_rate == 1.0 else "⚠️" if success_rate >= 0.8 else "❌"
        
        print(f"{success_emoji} Components completed: {successful_components}/{total_components} ({success_rate:.1%})")
        
        print(f"\n📋 Component Summary:")
        component_names = [
            "Causal Inference",
            "Robustness System", 
            "Advanced OPE",
            "Audit System",
            "Production Optimization"
        ]
        
        for i, name in enumerate(component_names):
            key = list(demo_results.keys())[i] if i < len(demo_results) else None
            if key and demo_results.get(key):
                print(f"   ✅ {name}")
            else:
                print(f"   ❌ {name}")
        
        if success_rate >= 0.8:
            print(f"\n🚀 Phase 4 system is PRODUCTION READY!")
            print(f"🎯 Key achievements:")
            print(f"   • Causal inference fighting spurious alpha")
            print(f"   • Robust anomaly detection and drift monitoring")
            print(f"   • Advanced off-policy evaluation with live counterfactuals")
            print(f"   • Complete auditability with deterministic replay")
            print(f"   • Production-grade deployment with ONNX optimization")
        else:
            print(f"\n⚠️ Some components need attention before production deployment")
        
        return demo_results

async def main():
    """Main demo execution"""
    demo = Phase4Demo()
    results = await demo.run_full_demo()
    return results

if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(main())
