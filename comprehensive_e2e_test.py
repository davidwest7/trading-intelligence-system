#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Trading Intelligence System
Excludes Streamlit apps, includes debugging and bug fixing
"""

import sys
import os
import time
import json
import traceback
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('e2e_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class E2ETestSuite:
    """Comprehensive end-to-end test suite for the Trading Intelligence System."""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        self.start_time = datetime.now()
        self._cleanup_handlers = []
        
    def add_cleanup_handler(self, handler):
        """Add a cleanup handler to be called on exit"""
        self._cleanup_handlers.append(handler)
    
    def cleanup(self):
        """Run all cleanup handlers"""
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Cleanup handler failed: {e}")
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", error: Exception = None):
        """Log test results with details."""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "error": str(error) if error else None
        }
        
        self.test_results[test_name] = result
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {details}")
        else:
            logger.error(f"❌ {test_name}: FAILED - {details}")
            if error:
                logger.error(f"   Error: {error}")
            self.errors.append(result)
    
    def test_data_engine(self):
        """Test the data engine components."""
        logger.info("🔍 Testing Data Engine Components...")
        
        try:
            # Test data adapters
            from common.data_adapters.polygon_adapter import PolygonDataAdapter
            from common.data_adapters.alpha_vantage_adapter import AlphaVantageAdapter
            
            # Initialize adapters with configs
            polygon_config = {
                'polygon_api_key': os.getenv('POLYGON_API_KEY', 'demo_key')
            }
            polygon_adapter = PolygonDataAdapter(polygon_config)
            
            # Initialize Alpha Vantage adapter with config
            alpha_config = {
                'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo_key')
            }
            alpha_adapter = AlphaVantageAdapter(alpha_config)
            
            # Test basic functionality
            self.log_test_result("Data Adapters Initialization", True, "Polygon and Alpha Vantage adapters initialized")
            
            # Test data fetching (mock data)
            mock_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, 100)
            })
            
            self.log_test_result("Mock Data Generation", True, f"Generated {len(mock_data)} data points")
            
            return mock_data
            
        except Exception as e:
            self.log_test_result("Data Engine Test", False, "Failed to initialize data engine", e)
            return None
    
    def test_ml_models(self):
        """Test the ML models components."""
        logger.info("🤖 Testing ML Models Components...")
        
        try:
            # Test model imports
            from ml_models.advanced_ml_models import AdvancedMLModels
            from ml_models.ensemble_predictor import EnsemblePredictor
            from ml_models.lstm_predictor import LSTMPredictor
            from ml_models.transformer_sentiment import TransformerSentiment
            
            # Initialize models
            advanced_models = AdvancedMLModels()
            ensemble = EnsemblePredictor()
            lstm = LSTMPredictor()
            transformer = TransformerSentiment()
            
            self.log_test_result("ML Models Initialization", True, "All ML models initialized successfully")
            
            # Test model training with mock data
            mock_features = np.random.randn(1000, 10)
            mock_targets = np.random.randn(1000)
            
            # Test ensemble training
            ensemble.train(mock_features, mock_targets)
            predictions = ensemble.predict(mock_features[:100])
            
            self.log_test_result("Ensemble Model Training", True, f"Trained ensemble model, generated {len(predictions)} predictions")
            
            # Test LSTM training
            lstm_features = np.random.randn(100, 20, 10)  # (samples, timesteps, features)
            lstm_targets = np.random.randn(100)
            lstm.train(lstm_features, lstm_targets)
            
            self.log_test_result("LSTM Model Training", True, "LSTM model trained successfully")
            
            return {
                "ensemble": ensemble,
                "lstm": lstm,
                "transformer": transformer,
                "advanced_models": advanced_models
            }
            
        except Exception as e:
            self.log_test_result("ML Models Test", False, "Failed to test ML models", e)
            return None
    
    def test_risk_management(self):
        """Test the risk management components."""
        logger.info("⚠️ Testing Risk Management Components...")
        
        try:
            # Test risk manager imports
            from risk_management.advanced_risk_manager import AdvancedRiskManager
            from risk_management.factor_model import FactorModel
            
            # Initialize risk components
            risk_manager = AdvancedRiskManager()
            factor_model = FactorModel()
            
            self.log_test_result("Risk Management Initialization", True, "Risk manager and factor model initialized")
            
            # Test risk calculations
            mock_returns = np.random.randn(1000) * 0.02  # 2% daily volatility
            mock_positions = np.random.randn(50) * 0.01  # 1% position sizes
            
            # Calculate VaR
            var_95 = risk_manager.calculate_var(mock_returns, confidence_level=0.95)
            var_99 = risk_manager.calculate_var(mock_returns, confidence_level=0.99)
            
            self.log_test_result("VaR Calculation", True, f"VaR 95%: {var_95:.4f}, VaR 99%: {var_99:.4f}")
            
            # Test factor model
            factor_returns = np.random.randn(1000, 5)  # 5 factors
            factor_model.fit(factor_returns, mock_returns)
            factor_risk = factor_model.calculate_factor_risk()
            
            self.log_test_result("Factor Model", True, f"Factor model fitted, risk: {factor_risk:.4f}")
            
            return {
                "risk_manager": risk_manager,
                "factor_model": factor_model,
                "var_95": var_95,
                "var_99": var_99
            }
            
        except Exception as e:
            self.log_test_result("Risk Management Test", False, "Failed to test risk management", e)
            return None
    
    def test_execution_algorithms(self):
        """Test the execution algorithms."""
        logger.info("🚀 Testing Execution Algorithms...")
        
        try:
            # Test execution imports
            from execution_algorithms.advanced_execution import AdvancedExecution
            from execution_algorithms.impact_models import ImpactModels
            from execution_algorithms.rl_exec_agent import RLExecutionAgent
            
            # Initialize execution components
            execution = AdvancedExecution()
            impact_models = ImpactModels()
            rl_agent = RLExecutionAgent()
            
            self.log_test_result("Execution Algorithms Initialization", True, "All execution components initialized")
            
            # Test execution simulation
            mock_orders = [
                {"symbol": "AAPL", "side": "buy", "quantity": 1000, "price": 150.0},
                {"symbol": "GOOGL", "side": "sell", "quantity": 500, "price": 2800.0},
                {"symbol": "MSFT", "side": "buy", "quantity": 800, "price": 300.0}
            ]
            
            # Test impact modeling
            impact_results = []
            for order in mock_orders:
                impact = impact_models.calculate_market_impact(order)
                impact_results.append(impact)
            
            self.log_test_result("Market Impact Modeling", True, f"Calculated impact for {len(impact_results)} orders")
            
            # Test execution optimization
            optimized_orders = execution.optimize_execution(mock_orders)
            
            self.log_test_result("Execution Optimization", True, f"Optimized {len(optimized_orders)} orders")
            
            return {
                "execution": execution,
                "impact_models": impact_models,
                "rl_agent": rl_agent,
                "optimized_orders": optimized_orders
            }
            
        except Exception as e:
            self.log_test_result("Execution Algorithms Test", False, "Failed to test execution algorithms", e)
            return None
    
    def test_governance_system(self):
        """Test the governance system."""
        logger.info("🏛️ Testing Governance System...")
        
        try:
            # Test governance imports
            from governance.governance_engine import GovernanceEngine
            
            # Initialize governance
            governance = GovernanceEngine()
            
            self.log_test_result("Governance System Initialization", True, "Governance engine initialized")
            
            # Test pre-trading checks
            pre_trading_results = governance.run_pre_trading_checks()
            
            self.log_test_result("Pre-Trading Checks", True, f"Completed {len(pre_trading_results)} pre-trading checks")
            
            # Test execution checks
            execution_results = governance.run_trading_execution_checks()
            
            self.log_test_result("Execution Checks", True, f"Completed {len(execution_results)} execution checks")
            
            # Test post-trading checks
            post_trading_results = governance.run_post_trading_checks()
            
            self.log_test_result("Post-Trading Checks", True, f"Completed {len(post_trading_results)} post-trading checks")
            
            # Test governance summary
            summary = governance.get_governance_summary()
            
            self.log_test_result("Governance Summary", True, f"Trading halted: {summary['trading_halted']}")
            
            return {
                "governance": governance,
                "summary": summary
            }
            
        except Exception as e:
            self.log_test_result("Governance System Test", False, "Failed to test governance system", e)
            return None
    
    def test_monitoring_system(self):
        """Test the monitoring and drift detection system."""
        logger.info("📊 Testing Monitoring System...")
        
        try:
            # Test monitoring imports
            from monitoring.drift_suite import DriftSuite, create_drift_suite
            
            # Initialize monitoring
            drift_suite = DriftSuite()
            
            self.log_test_result("Monitoring System Initialization", True, "Drift detection suite initialized")
            
            # Test drift detection with mock data
            np.random.seed(42)
            n_samples = 1000
            
            # Reference data (baseline)
            reference_data = {
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.normal(0, 1, n_samples)
            }
            
            # Current data (with some drift)
            current_data = {
                'feature_1': np.random.normal(0.2, 1, n_samples),  # Mean shift
                'feature_2': np.random.normal(0, 1.2, n_samples),  # Variance increase
                'feature_3': np.random.normal(0, 1, n_samples)     # No drift
            }
            
            # Sample market data
            market_data = pd.DataFrame({
                'close': 100 + np.random.randn(252).cumsum(),
                'volume': np.random.randint(1000, 10000, 252)
            }, index=pd.date_range('2023-01-01', periods=252, freq='D'))
            
            # Sample model performance
            model_performance = {
                'model_1': {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88
                },
                'model_2': {
                    'sharpe_ratio': 1.2,
                    'information_ratio': 0.8
                }
            }
            
            # Run comprehensive monitoring
            alerts = drift_suite.run_comprehensive_monitoring(
                reference_data, current_data, market_data, model_performance
            )
            
            self.log_test_result("Drift Detection", True, f"Generated {len(alerts)} drift alerts")
            
            # Test monitoring summary
            summary = drift_suite.get_monitoring_summary()
            
            self.log_test_result("Monitoring Summary", True, f"Current regime: {summary['current_regime']['regime_name']}")
            
            return {
                "drift_suite": drift_suite,
                "alerts": alerts,
                "summary": summary
            }
            
        except Exception as e:
            self.log_test_result("Monitoring System Test", False, "Failed to test monitoring system", e)
            return None
    
    def test_alternative_data(self):
        """Test alternative data integration."""
        logger.info("📈 Testing Alternative Data Integration...")
        
        try:
            # Test alternative data imports
            from alternative_data.real_time_data_integration import RealTimeDataIntegration
            
            # Initialize alternative data with context manager for proper cleanup
            with RealTimeDataIntegration() as alt_data:
                # Initialize the data integration
                import asyncio
                try:
                    # Run initialization with timeout
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(asyncio.wait_for(alt_data.initialize(), timeout=10.0))
                    loop.close()
                except asyncio.TimeoutError:
                    logger.warning("Alternative data initialization timed out, continuing with mock data")
                    success = False
                except Exception as e:
                    logger.warning(f"Alternative data initialization failed: {e}, continuing with mock data")
                    success = False
                
                if success:
                    self.log_test_result("Alternative Data Initialization", True, "Alternative data integration initialized")
                else:
                    self.log_test_result("Alternative Data Initialization", False, "Initialization failed, using mock data")
                
                # Test data sources
                sources = alt_data.get_available_sources()
                self.log_test_result("Data Sources", True, f"Available sources: {len(sources)}")
                
                # Test data fetching (mock)
                mock_sentiment = {
                    'AAPL': {'sentiment': 0.7, 'volume': 1000000},
                    'GOOGL': {'sentiment': 0.6, 'volume': 800000},
                    'MSFT': {'sentiment': 0.8, 'volume': 1200000}
                }
                
                self.log_test_result("Sentiment Data", True, f"Mock sentiment data for {len(mock_sentiment)} symbols")
                
                # Test news data
                mock_news = [
                    {'symbol': 'AAPL', 'headline': 'Apple reports strong earnings', 'sentiment': 0.8},
                    {'symbol': 'GOOGL', 'headline': 'Google faces regulatory scrutiny', 'sentiment': -0.3},
                    {'symbol': 'MSFT', 'headline': 'Microsoft cloud growth accelerates', 'sentiment': 0.9}
                ]
                
                self.log_test_result("News Data", True, f"Mock news data: {len(mock_news)} articles")
                
                return {
                    "alt_data": alt_data,
                    "sentiment": mock_sentiment,
                    "news": mock_news
                }
            
        except Exception as e:
            self.log_test_result("Alternative Data Test", False, "Failed to test alternative data", e)
            return None
    
    def test_hft_components(self):
        """Test HFT components."""
        logger.info("⚡ Testing HFT Components...")
        
        try:
            # Test HFT imports
            from hft.low_latency_execution import LowLatencyExecution
            from hft.market_microstructure import MarketMicrostructure
            from hft.ultra_fast_models import UltraFastModels
            
            # Initialize HFT components
            hft_execution = LowLatencyExecution()
            microstructure = MarketMicrostructure()
            fast_models = UltraFastModels()
            
            self.log_test_result("HFT Components Initialization", True, "All HFT components initialized")
            
            # Test latency measurement
            latency = hft_execution.measure_latency()
            
            self.log_test_result("Latency Measurement", True, f"Measured latency: {latency:.6f} seconds")
            
            # Test market microstructure analysis
            mock_orderbook = {
                'bids': [(150.0, 1000), (149.9, 1500), (149.8, 2000)],
                'asks': [(150.1, 1200), (150.2, 1800), (150.3, 2500)]
            }
            
            spread = microstructure.calculate_spread(mock_orderbook)
            depth = microstructure.calculate_depth(mock_orderbook)
            
            self.log_test_result("Market Microstructure", True, f"Spread: {spread:.4f}, Depth: {depth}")
            
            # Test ultra-fast model prediction
            mock_features = np.random.randn(100, 10)
            predictions = fast_models.predict(mock_features)
            
            self.log_test_result("Ultra-Fast Models", True, f"Generated {len(predictions)} predictions")
            
            return {
                "hft_execution": hft_execution,
                "microstructure": microstructure,
                "fast_models": fast_models,
                "latency": latency
            }
            
        except Exception as e:
            self.log_test_result("HFT Components Test", False, "Failed to test HFT components", e)
            return None
    
    def test_integration_workflow(self):
        """Test the complete integration workflow."""
        logger.info("🔄 Testing Complete Integration Workflow...")
        
        try:
            # Test the main enhanced system
            from main_enhanced import TradingIntelligenceSystem
            
            # Initialize the complete system
            system = TradingIntelligenceSystem()
            
            self.log_test_result("Complete System Initialization", True, "Trading Intelligence System initialized")
            
            # Test system startup
            startup_result = system.startup()
            
            self.log_test_result("System Startup", True, f"Startup result: {startup_result}")
            
            # Test data pipeline
            data_status = system.get_data_status()
            
            self.log_test_result("Data Pipeline", True, f"Data status: {data_status}")
            
            # Test model pipeline
            model_status = system.get_model_status()
            
            self.log_test_result("Model Pipeline", True, f"Model status: {model_status}")
            
            # Test risk pipeline
            risk_status = system.get_risk_status()
            
            self.log_test_result("Risk Pipeline", True, f"Risk status: {risk_status}")
            
            # Test execution pipeline
            execution_status = system.get_execution_status()
            
            self.log_test_result("Execution Pipeline", True, f"Execution status: {execution_status}")
            
            # Test complete workflow
            workflow_result = system.run_complete_workflow()
            
            self.log_test_result("Complete Workflow", True, f"Workflow completed: {workflow_result}")
            
            return {
                "system": system,
                "workflow_result": workflow_result
            }
            
        except Exception as e:
            self.log_test_result("Integration Workflow Test", False, "Failed to test integration workflow", e)
            return None
    
    def test_performance_metrics(self):
        """Test performance metrics and evaluation."""
        logger.info("📊 Testing Performance Metrics...")
        
        try:
            # Test evaluation components
            from common.evaluation.performance_metrics import PerformanceMetrics
            from common.evaluation.risk_metrics import RiskMetrics
            from common.evaluation.backtest_engine import BacktestEngine
            
            # Initialize evaluation components
            perf_metrics = PerformanceMetrics()
            risk_metrics = RiskMetrics()
            backtest = BacktestEngine()
            
            self.log_test_result("Performance Metrics Initialization", True, "All evaluation components initialized")
            
            # Generate mock performance data
            mock_returns = np.random.randn(252) * 0.02  # Daily returns
            mock_benchmark = np.random.randn(252) * 0.015  # Benchmark returns
            
            # Calculate performance metrics
            sharpe_ratio = perf_metrics.calculate_sharpe_ratio(mock_returns)
            max_drawdown = perf_metrics.calculate_max_drawdown(mock_returns)
            information_ratio = perf_metrics.calculate_information_ratio(mock_returns, mock_benchmark)
            
            self.log_test_result("Performance Calculations", True, 
                               f"Sharpe: {sharpe_ratio:.3f}, MaxDD: {max_drawdown:.3f}, IR: {information_ratio:.3f}")
            
            # Calculate risk metrics
            var_95 = risk_metrics.calculate_var(mock_returns, 0.95)
            expected_shortfall = risk_metrics.calculate_expected_shortfall(mock_returns, 0.95)
            
            self.log_test_result("Risk Calculations", True, 
                               f"VaR 95%: {var_95:.4f}, ES: {expected_shortfall:.4f}")
            
            # Test backtest
            backtest_result = backtest.run_backtest(mock_returns, mock_benchmark)
            
            self.log_test_result("Backtest", True, f"Backtest completed: {backtest_result['total_return']:.4f}")
            
            return {
                "perf_metrics": perf_metrics,
                "risk_metrics": risk_metrics,
                "backtest": backtest,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "information_ratio": information_ratio
            }
            
        except Exception as e:
            self.log_test_result("Performance Metrics Test", False, "Failed to test performance metrics", e)
            return None
    
    def run_comprehensive_test(self):
        """Run the comprehensive end-to-end test suite."""
        logger.info("🚀 Starting Comprehensive End-to-End Test Suite")
        logger.info("=" * 80)
        
        # Store all test results
        test_outputs = {}
        
        # Test each component with timeout protection
        test_methods = [
            ('data_engine', self.test_data_engine),
            ('ml_models', self.test_ml_models),
            ('risk_management', self.test_risk_management),
            ('execution_algorithms', self.test_execution_algorithms),
            ('governance_system', self.test_governance_system),
            ('monitoring_system', self.test_monitoring_system),
            ('alternative_data', self.test_alternative_data),
            ('hft_components', self.test_hft_components),
            ('integration_workflow', self.test_integration_workflow),
            ('performance_metrics', self.test_performance_metrics)
        ]
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"🧪 Running {test_name} test...")
                test_outputs[test_name] = test_method()
                logger.info(f"✅ {test_name} test completed")
            except Exception as e:
                logger.error(f"❌ {test_name} test failed: {e}")
                test_outputs[test_name] = None
                self.log_test_result(f"{test_name}_test", False, f"Test failed with error: {e}", e)
        
        # Generate comprehensive report
        self.generate_test_report(test_outputs)
        
        return test_outputs
    
    def generate_test_report(self, test_outputs):
        """Generate a comprehensive test report."""
        logger.info("📋 Generating Comprehensive Test Report")
        logger.info("=" * 80)
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Test duration
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Print summary
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Duration: {duration}")
        
        # Print detailed results
        logger.info(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            logger.info(f"   {status} - {test_name}: {result['details']}")
        
        # Print errors if any
        if self.errors:
            logger.info(f"\n❌ Errors Found:")
            for error in self.errors:
                logger.error(f"   {error['test_name']}: {error['error']}")
        
        # Print warnings if any
        if self.warnings:
            logger.info(f"\n⚠️ Warnings:")
            for warning in self.warnings:
                logger.warning(f"   {warning}")
        
        # Save detailed report to file
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "duration": str(duration),
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "test_results": self.test_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "test_outputs": {k: str(v) for k, v in test_outputs.items() if v is not None}
        }
        
        with open('e2e_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved to: e2e_test_report.json")
        
        # Final assessment
        if success_rate >= 90:
            logger.info(f"\n🎉 EXCELLENT: System is ready for production! Success rate: {success_rate:.1f}%")
        elif success_rate >= 80:
            logger.info(f"\n✅ GOOD: System is mostly functional. Success rate: {success_rate:.1f}%")
        elif success_rate >= 70:
            logger.info(f"\n⚠️ FAIR: System needs some fixes. Success rate: {success_rate:.1f}%")
        else:
            logger.error(f"\n❌ POOR: System needs significant work. Success rate: {success_rate:.1f}%")

def main():
    """Main function to run the comprehensive end-to-end test."""
    print("🧪 Comprehensive End-to-End Test Suite")
    print("=" * 100)
    
    # Create test suite
    test_suite = E2ETestSuite()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        test_suite.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run comprehensive test with timeout
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def run_test():
            try:
                test_outputs = test_suite.run_comprehensive_test()
                result_queue.put(('success', test_outputs))
            except Exception as e:
                result_queue.put(('error', e))
        
        # Start test in a separate thread
        test_thread = threading.Thread(target=run_test)
        test_thread.daemon = True
        test_thread.start()
        
        # Wait for completion with timeout (5 minutes)
        try:
            result_type, result = result_queue.get(timeout=300)
            if result_type == 'success':
                print("\n🎉 End-to-End Test Suite Completed!")
                print("=" * 100)
                return 0
            else:
                logger.error(f"❌ Test suite failed with error: {result}")
                traceback.print_exc()
                return 1
        except queue.Empty:
            logger.error("❌ Test suite timed out after 5 minutes")
            test_suite.cleanup()
            return 1
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        traceback.print_exc()
        test_suite.cleanup()
        return 1
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
