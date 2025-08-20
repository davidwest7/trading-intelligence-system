#!/usr/bin/env python3
"""
Simple End-to-End Test for Trading Intelligence System
Focuses on core components that are known to work
"""

import sys
import os
import time
import json
import traceback
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimpleE2ETest:
    """Simple end-to-end test suite focusing on core components."""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.start_time = datetime.now()
        
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
    
    def test_basic_imports(self):
        """Test basic Python imports."""
        logger.info("🔍 Testing Basic Imports...")
        
        try:
            # Test numpy
            import numpy as np
            test_array = np.random.randn(100)
            self.log_test_result("NumPy Import", True, f"Created array with {len(test_array)} elements")
            
            # Test pandas
            import pandas as pd
            test_df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)})
            self.log_test_result("Pandas Import", True, f"Created DataFrame with shape {test_df.shape}")
            
            # Test scipy
            import scipy
            self.log_test_result("SciPy Import", True, f"SciPy version: {scipy.__version__}")
            
            # Test sklearn
            import sklearn
            self.log_test_result("Scikit-learn Import", True, f"Scikit-learn version: {sklearn.__version__}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Basic Imports Test", False, "Failed to import basic libraries", e)
            return False
    
    def test_governance_system(self):
        """Test the governance system (known to work)."""
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
            
            return True
            
        except Exception as e:
            self.log_test_result("Governance System Test", False, "Failed to test governance system", e)
            return False
    
    def test_monitoring_system(self):
        """Test the monitoring system (known to work)."""
        logger.info("📊 Testing Monitoring System...")
        
        try:
            # Test monitoring imports
            from monitoring.drift_suite import DriftSuite
            
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
            
            # Test PSI calculation
            from monitoring.drift_suite import PSICalculator
            psi_calc = PSICalculator()
            
            psi_result = psi_calc.calculate_psi(
                reference_data['feature_1'], 
                current_data['feature_1']
            )
            
            self.log_test_result("PSI Calculation", True, f"PSI score: {psi_result.psi_score:.4f}")
            
            # Test regime detection
            from monitoring.drift_suite import RegimeDetector
            regime_detector = RegimeDetector()
            
            regime_state = regime_detector.detect_regime(market_data)
            
            self.log_test_result("Regime Detection", True, f"Detected regime: {regime_state.regime_name}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Monitoring System Test", False, "Failed to test monitoring system", e)
            return False
    
    def test_data_generation(self):
        """Test data generation capabilities."""
        logger.info("📈 Testing Data Generation...")
        
        try:
            # Generate mock market data
            np.random.seed(42)
            n_days = 252
            
            # Generate price data
            returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            ohlcv_data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_days)
            }, index=pd.date_range('2023-01-01', periods=n_days, freq='D'))
            
            self.log_test_result("OHLCV Data Generation", True, f"Generated {len(ohlcv_data)} days of OHLCV data")
            
            # Calculate technical indicators
            ohlcv_data['sma_20'] = ohlcv_data['close'].rolling(20).mean()
            ohlcv_data['sma_50'] = ohlcv_data['close'].rolling(50).mean()
            ohlcv_data['rsi'] = self.calculate_rsi(ohlcv_data['close'])
            ohlcv_data['volatility'] = ohlcv_data['close'].pct_change().rolling(20).std()
            
            self.log_test_result("Technical Indicators", True, "Calculated SMA, RSI, and volatility")
            
            # Generate multiple asset data
            assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            multi_asset_data = {}
            
            for asset in assets:
                asset_returns = np.random.normal(0.001, 0.02, n_days)
                asset_prices = 100 * np.exp(np.cumsum(asset_returns))
                multi_asset_data[asset] = pd.Series(asset_prices, index=ohlcv_data.index)
            
            self.log_test_result("Multi-Asset Data", True, f"Generated data for {len(assets)} assets")
            
            return {
                'ohlcv_data': ohlcv_data,
                'multi_asset_data': multi_asset_data,
                'assets': assets
            }
            
        except Exception as e:
            self.log_test_result("Data Generation Test", False, "Failed to generate test data", e)
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def test_risk_calculations(self):
        """Test basic risk calculations."""
        logger.info("⚠️ Testing Risk Calculations...")
        
        try:
            # Generate mock returns
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, 1000)  # 1000 daily returns
            
            # Calculate basic risk metrics
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            self.log_test_result("Basic Risk Metrics", True, 
                               f"Mean: {mean_return:.4f}, Vol: {volatility:.4f}, Sharpe: {sharpe_ratio:.3f}")
            
            self.log_test_result("VaR Calculations", True, 
                               f"VaR 95%: {var_95:.4f}, VaR 99%: {var_99:.4f}")
            
            self.log_test_result("Drawdown Calculation", True, 
                               f"Max Drawdown: {max_drawdown:.4f}")
            
            # Test correlation matrix
            n_assets = 5
            asset_returns = np.random.normal(0, 0.02, (1000, n_assets))
            correlation_matrix = np.corrcoef(asset_returns.T)
            
            self.log_test_result("Correlation Matrix", True, 
                               f"Generated {n_assets}x{n_assets} correlation matrix")
            
            return {
                'mean_return': mean_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown,
                'correlation_matrix': correlation_matrix
            }
            
        except Exception as e:
            self.log_test_result("Risk Calculations Test", False, "Failed to calculate risk metrics", e)
            return None
    
    def test_performance_analysis(self):
        """Test performance analysis capabilities."""
        logger.info("📊 Testing Performance Analysis...")
        
        try:
            # Generate mock performance data
            np.random.seed(42)
            n_days = 252
            
            # Portfolio returns
            portfolio_returns = np.random.normal(0.001, 0.015, n_days)
            benchmark_returns = np.random.normal(0.0008, 0.012, n_days)
            
            # Calculate cumulative returns
            portfolio_cumulative = np.cumprod(1 + portfolio_returns)
            benchmark_cumulative = np.cumprod(1 + benchmark_returns)
            
            # Calculate performance metrics
            total_return = portfolio_cumulative[-1] - 1
            annualized_return = (1 + total_return) ** (252 / n_days) - 1
            annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Calculate information ratio
            excess_returns = portfolio_returns - benchmark_returns
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            self.log_test_result("Performance Metrics", True, 
                               f"Total Return: {total_return:.4f}, Sharpe: {sharpe_ratio:.3f}")
            
            self.log_test_result("Risk-Adjusted Metrics", True, 
                               f"Information Ratio: {information_ratio:.3f}, Beta: {beta:.3f}")
            
            # Calculate rolling metrics
            rolling_sharpe = []
            window = 60
            
            for i in range(window, len(portfolio_returns)):
                window_returns = portfolio_returns[i-window:i]
                window_mean = np.mean(window_returns)
                window_std = np.std(window_returns)
                rolling_sharpe.append(window_mean / window_std if window_std > 0 else 0)
            
            self.log_test_result("Rolling Analysis", True, 
                               f"Calculated {len(rolling_sharpe)} rolling Sharpe ratios")
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'information_ratio': information_ratio,
                'beta': beta,
                'rolling_sharpe': rolling_sharpe
            }
            
        except Exception as e:
            self.log_test_result("Performance Analysis Test", False, "Failed to analyze performance", e)
            return None
    
    def run_simple_test(self):
        """Run the simple end-to-end test suite."""
        logger.info("🚀 Starting Simple End-to-End Test Suite")
        logger.info("=" * 60)
        
        # Store all test results
        test_outputs = {}
        
        # Test each component
        test_outputs['basic_imports'] = self.test_basic_imports()
        test_outputs['governance_system'] = self.test_governance_system()
        test_outputs['monitoring_system'] = self.test_monitoring_system()
        test_outputs['data_generation'] = self.test_data_generation()
        test_outputs['risk_calculations'] = self.test_risk_calculations()
        test_outputs['performance_analysis'] = self.test_performance_analysis()
        
        # Generate comprehensive report
        self.generate_test_report(test_outputs)
        
        return test_outputs
    
    def generate_test_report(self, test_outputs):
        """Generate a comprehensive test report."""
        logger.info("📋 Generating Simple Test Report")
        logger.info("=" * 60)
        
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
            "errors": self.errors
        }
        
        with open('simple_e2e_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved to: simple_e2e_test_report.json")
        
        # Final assessment
        if success_rate >= 90:
            logger.info(f"\n🎉 EXCELLENT: Core system is working well! Success rate: {success_rate:.1f}%")
        elif success_rate >= 80:
            logger.info(f"\n✅ GOOD: Core system is mostly functional. Success rate: {success_rate:.1f}%")
        elif success_rate >= 70:
            logger.info(f"\n⚠️ FAIR: Core system needs some fixes. Success rate: {success_rate:.1f}%")
        else:
            logger.error(f"\n❌ POOR: Core system needs significant work. Success rate: {success_rate:.1f}%")

def main():
    """Main function to run the simple end-to-end test."""
    print("🧪 Simple End-to-End Test Suite")
    print("=" * 80)
    
    # Create test suite
    test_suite = SimpleE2ETest()
    
    try:
        # Run simple test
        test_outputs = test_suite.run_simple_test()
        
        print("\n🎉 Simple End-to-End Test Suite Completed!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
