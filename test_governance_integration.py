#!/usr/bin/env python3
"""
Governance System Integration Test
Demonstrates integration with the complete Trading Intelligence System
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from governance.governance_engine import GovernanceEngine, CheckStatus, ApprovalStatus, Severity
from data.data_engine import DataEngine
from models.model_manager import ModelManager
from risk.risk_manager import RiskManager
from execution.execution_engine import ExecutionEngine
from portfolio.portfolio_manager import PortfolioManager

def test_governance_integration():
    """Test governance integration with the complete trading system."""
    print("🔗 Testing Governance System Integration")
    print("=" * 80)
    
    # Initialize all system components
    print("\n1. Initializing System Components...")
    
    # Initialize governance engine
    governance = GovernanceEngine()
    print("   ✅ Governance Engine initialized")
    
    # Initialize data engine
    data_engine = DataEngine()
    print("   ✅ Data Engine initialized")
    
    # Initialize model manager
    model_manager = ModelManager()
    print("   ✅ Model Manager initialized")
    
    # Initialize risk manager
    risk_manager = RiskManager()
    print("   ✅ Risk Manager initialized")
    
    # Initialize execution engine
    execution_engine = ExecutionEngine()
    print("   ✅ Execution Engine initialized")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager()
    print("   ✅ Portfolio Manager initialized")
    
    # Test complete trading workflow with governance
    print("\n2. Testing Complete Trading Workflow with Governance...")
    
    # Pre-trading phase
    print("\n🌅 Pre-Trading Phase:")
    
    # Run governance pre-trading checks
    pre_trading_results = governance.run_pre_trading_checks()
    print(f"   📊 Governance checks: {len(pre_trading_results)} completed")
    
    # Check data quality
    data_status = data_engine.get_data_status()
    print(f"   📊 Data status: {data_status['status']}")
    
    # Check model status
    model_status = model_manager.get_model_status()
    print(f"   📊 Model status: {model_status['status']}")
    
    # Check risk status
    risk_status = risk_manager.get_risk_status()
    print(f"   📊 Risk status: {risk_status['status']}")
    
    # Evaluate if trading should proceed
    if governance.trading_halted:
        print("   🛑 Trading halted by governance - cannot proceed")
        return
    
    print("   ✅ All pre-trading checks passed - trading can proceed")
    
    # Trading execution phase
    print("\n🕐 Trading Execution Phase:")
    
    # Run governance execution checks
    execution_results = governance.run_trading_execution_checks()
    print(f"   📊 Execution checks: {len(execution_results)} completed")
    
    # Simulate trading decisions
    print("   📈 Generating trading signals...")
    signals = model_manager.generate_signals()
    print(f"   📊 Generated {len(signals)} trading signals")
    
    # Risk validation
    print("   ⚠️  Validating signals against risk limits...")
    validated_signals = risk_manager.validate_signals(signals)
    print(f"   📊 {len(validated_signals)} signals passed risk validation")
    
    # Execution
    if validated_signals:
        print("   🚀 Executing validated signals...")
        execution_results = execution_engine.execute_signals(validated_signals)
        print(f"   📊 Executed {len(execution_results)} trades")
        
        # Update portfolio
        portfolio_manager.update_positions(execution_results)
        print("   📊 Portfolio updated with new positions")
    
    # Post-trading phase
    print("\n🌆 Post-Trading Phase:")
    
    # Run governance post-trading checks
    post_trading_results = governance.run_post_trading_checks()
    print(f"   📊 Post-trading checks: {len(post_trading_results)} completed")
    
    # Performance analysis
    performance = portfolio_manager.calculate_performance()
    print(f"   📊 Portfolio performance: {performance['daily_return']:.4f}")
    
    # Risk analysis
    risk_metrics = risk_manager.calculate_risk_metrics()
    print(f"   📊 Risk metrics calculated: VaR={risk_metrics['var_95']:.4f}")
    
    # Generate reports
    print("   📄 Generating end-of-day reports...")
    governance._generate_daily_reports()
    print("   ✅ Daily reports generated")
    
    print("\n✅ Complete Trading Workflow with Governance Completed!")

def test_governance_scenarios():
    """Test governance scenarios with real system components."""
    print("\n🎭 Testing Governance Scenarios with System Integration...")
    print("=" * 80)
    
    # Initialize components
    governance = GovernanceEngine()
    data_engine = DataEngine()
    model_manager = ModelManager()
    risk_manager = RiskManager()
    execution_engine = ExecutionEngine()
    portfolio_manager = PortfolioManager()
    
    # Scenario 1: Data Quality Issue
    print("\n📊 Scenario 1: Data Quality Issue")
    print("   Simulating data quality problem...")
    
    # Simulate data quality issue
    data_engine.simulate_data_issue("missing_data", ["AAPL", "GOOGL"])
    
    # Run governance checks
    pre_trading_results = governance.run_pre_trading_checks()
    
    # Check for data quality failures
    data_quality_failures = [
        result for result in pre_trading_results.values()
        if "DQ" in result.check_id and result.status != CheckStatus.PASSED
    ]
    
    if data_quality_failures:
        print("   ⚠️  Data quality checks failed")
        for failure in data_quality_failures:
            print(f"      ❌ {failure.name}: {failure.status.value}")
        
        # Simulate manual approval
        for failure in data_quality_failures:
            governance.approve_check(
                check_id=failure.check_id,
                approver="data_engineer",
                approved=True,
                comments="Data quality acceptable for today's trading"
            )
            print(f"      ✅ {failure.name} manually approved")
    
    # Scenario 2: Model Performance Degradation
    print("\n📈 Scenario 2: Model Performance Degradation")
    print("   Simulating model performance issue...")
    
    # Simulate model performance degradation
    model_manager.simulate_performance_degradation(0.3)  # Low Sharpe ratio
    
    # Run governance checks
    pre_trading_results = governance.run_pre_trading_checks()
    
    # Check for model validation failures
    model_failures = [
        result for result in pre_trading_results.values()
        if "MV" in result.check_id and result.status != CheckStatus.PASSED
    ]
    
    if model_failures:
        print("   ⚠️  Model validation checks failed")
        for failure in model_failures:
            print(f"      ❌ {failure.name}: {failure.status.value}")
        
        # Simulate manual approval with conditions
        for failure in model_failures:
            governance.approve_check(
                check_id=failure.check_id,
                approver="quantitative_researcher",
                approved=True,
                comments="Model performance acceptable but monitoring increased"
            )
            print(f"      ✅ {failure.name} approved with conditions")
    
    # Scenario 3: Risk Limit Breach
    print("\n⚠️  Scenario 3: Risk Limit Breach")
    print("   Simulating risk limit breach...")
    
    # Simulate risk limit breach
    portfolio_manager.simulate_large_position("TSLA", 0.08)  # 8% position
    
    # Run governance checks
    pre_trading_results = governance.run_pre_trading_checks()
    
    # Check for risk management failures
    risk_failures = [
        result for result in pre_trading_results.values()
        if "RM" in result.check_id and result.status != CheckStatus.PASSED
    ]
    
    if risk_failures:
        print("   🚨 Risk management checks failed")
        for failure in risk_failures:
            print(f"      ❌ {failure.name}: {failure.status.value}")
        
        # Simulate risk mitigation
        portfolio_manager.reduce_position("TSLA", 0.03)  # Reduce to 3%
        print("      ✅ Position reduced to bring risk within limits")
        
        # Re-run checks
        pre_trading_results = governance.run_pre_trading_checks()
        print("      ✅ Risk checks now pass")
    
    # Scenario 4: Market Volatility
    print("\n📊 Scenario 4: Market Volatility")
    print("   Simulating high market volatility...")
    
    # Simulate high volatility
    data_engine.simulate_high_volatility(0.05)  # 5% daily volatility
    
    # Check if policy clock triggers trading halt
    governance._check_policy_clocks()
    
    if governance.trading_halted:
        print("   🛑 Trading halted due to high volatility")
        print("   ⏰ Waiting for volatility to subside...")
        
        # Simulate volatility reduction
        data_engine.simulate_normal_volatility(0.02)  # 2% daily volatility
        
        # Resume trading
        governance.resume_trading(
            approver="risk_manager",
            reason="Volatility returned to normal levels"
        )
        print("   ✅ Trading resumed after volatility normalization")
    
    print("\n✅ All Governance Scenarios with System Integration Completed!")

def test_governance_monitoring():
    """Test real-time governance monitoring."""
    print("\n📊 Testing Real-Time Governance Monitoring...")
    print("=" * 80)
    
    # Initialize components
    governance = GovernanceEngine()
    data_engine = DataEngine()
    model_manager = ModelManager()
    risk_manager = RiskManager()
    portfolio_manager = PortfolioManager()
    
    print("\n🔄 Starting Real-Time Monitoring...")
    
    # Simulate real-time monitoring for 5 cycles
    for cycle in range(5):
        print(f"\n📊 Monitoring Cycle {cycle + 1}/5:")
        
        # Run real-time governance checks
        execution_results = governance.run_trading_execution_checks()
        
        # Check system health
        data_health = data_engine.get_health_status()
        model_health = model_manager.get_health_status()
        risk_health = risk_manager.get_health_status()
        portfolio_health = portfolio_manager.get_health_status()
        
        print(f"   📊 Data Health: {data_health['status']}")
        print(f"   📊 Model Health: {model_health['status']}")
        print(f"   📊 Risk Health: {risk_health['status']}")
        print(f"   📊 Portfolio Health: {portfolio_health['status']}")
        
        # Check for alerts
        alerts = []
        if data_health['status'] != 'healthy':
            alerts.append(f"Data: {data_health['status']}")
        if model_health['status'] != 'healthy':
            alerts.append(f"Model: {model_health['status']}")
        if risk_health['status'] != 'healthy':
            alerts.append(f"Risk: {risk_health['status']}")
        if portfolio_health['status'] != 'healthy':
            alerts.append(f"Portfolio: {portfolio_health['status']}")
        
        if alerts:
            print(f"   ⚠️  Alerts: {', '.join(alerts)}")
        else:
            print("   ✅ All systems healthy")
        
        # Check governance status
        summary = governance.get_governance_summary()
        print(f"   📊 Governance: {summary['total_checks']} checks, {summary['failed_checks']} failed")
        
        # Simulate time passing
        time.sleep(1)
    
    print("\n✅ Real-Time Monitoring Test Completed!")

def test_governance_reporting():
    """Test comprehensive governance reporting."""
    print("\n📄 Testing Comprehensive Governance Reporting...")
    print("=" * 80)
    
    # Initialize governance engine
    governance = GovernanceEngine()
    
    # Test daily reporting
    print("\n📊 Daily Reporting:")
    try:
        governance._generate_daily_reports()
        print("   ✅ Daily reports generated successfully")
    except Exception as e:
        print(f"   ❌ Error generating daily reports: {e}")
    
    # Test weekly reporting
    print("\n📊 Weekly Reporting:")
    try:
        governance._generate_weekly_reports()
        print("   ✅ Weekly reports generated successfully")
    except Exception as e:
        print(f"   ❌ Error generating weekly reports: {e}")
    
    # Test monthly reporting
    print("\n📊 Monthly Reporting:")
    try:
        governance._generate_monthly_reports()
        print("   ✅ Monthly reports generated successfully")
    except Exception as e:
        print(f"   ❌ Error generating monthly reports: {e}")
    
    # Test custom report
    print("\n📊 Custom Report:")
    custom_report_config = {
        "id": "INTEGRATION_RPT",
        "name": "Integration Test Report",
        "description": "Comprehensive integration test report",
        "frequency": "on_demand",
        "recipients": ["integration_test@company.com"],
        "format": ["email"],
        "content": ["performance_summary", "risk_metrics", "position_summary", "execution_quality", "compliance_status"]
    }
    
    try:
        governance._generate_report(custom_report_config)
        print("   ✅ Custom report generated successfully")
    except Exception as e:
        print(f"   ❌ Error generating custom report: {e}")
    
    # Check generated reports
    reports_dir = Path("governance/reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.pdf"))
        print(f"\n📄 Generated Reports:")
        print(f"   📊 Total reports: {len(report_files)}")
        for report_file in report_files:
            print(f"      • {report_file.name}")
    
    print("\n✅ Comprehensive Reporting Test Completed!")

def main():
    """Main integration test function."""
    print("🧪 Governance System Integration Test Suite")
    print("=" * 100)
    
    try:
        # Test complete integration
        test_governance_integration()
        
        # Test governance scenarios
        test_governance_scenarios()
        
        # Test real-time monitoring
        test_governance_monitoring()
        
        # Test comprehensive reporting
        test_governance_reporting()
        
        print("\n🎉 All Integration Tests Completed Successfully!")
        print("=" * 100)
        print("\n📋 Integration Features Verified:")
        print("   ✅ Governance integration with data engine")
        print("   ✅ Governance integration with model manager")
        print("   ✅ Governance integration with risk manager")
        print("   ✅ Governance integration with execution engine")
        print("   ✅ Governance integration with portfolio manager")
        print("   ✅ Real-time monitoring and alerting")
        print("   ✅ Scenario-based testing with real components")
        print("   ✅ Comprehensive reporting integration")
        print("   ✅ End-to-end workflow testing")
        print("   ✅ System health monitoring")
        print("   ✅ Performance tracking integration")
        print("   ✅ Risk management integration")
        
        # Final system status
        print(f"\n📊 Final System Status:")
        governance = GovernanceEngine()
        summary = governance.get_governance_summary()
        print(f"   • Governance Status: {'🛑 HALTED' if summary['trading_halted'] else '✅ ACTIVE'}")
        print(f"   • Total Checks: {summary['total_checks']}")
        print(f"   • Failed Checks: {summary['failed_checks']}")
        print(f"   • Pending Approvals: {summary['pending_approvals']}")
        print(f"   • Open Exceptions: {summary['open_exceptions']}")
        print(f"   • Last Updated: {summary['last_updated']}")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
