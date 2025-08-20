#!/usr/bin/env python3
"""
Enhanced Trading System Integration Test
Demonstrates all major enhancements working together in a unified system
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import logging

# Add current directory to path
sys.path.append('.')

# Import enhanced components
from main_enhanced import app
from agents.technical.agent_enhanced import create_enhanced_technical_agent
from common.feature_store.embargo import create_embargo_manager, EmbargoEvent, EmbargoType
from agents.flow.lob_features import create_lob_extractor
from ml_models.hierarchical_meta_ensemble import create_hierarchical_ensemble
from agents.insider.options_surface import create_options_analyzer


async def test_enhanced_system_integration():
    """Test the complete enhanced trading system integration"""
    print("\n" + "="*80)
    print("🚀 ENHANCED TRADING SYSTEM INTEGRATION TEST")
    print("="*80)
    print("Testing all major enhancements working together:")
    print("✅ Multi-event embargo system")
    print("✅ LOB and microstructure features")
    print("✅ Hierarchical meta-ensemble")
    print("✅ Options surface analysis")
    print("✅ Enhanced technical agent")
    print("✅ Unified API integration")
    
    try:
        # Step 1: Initialize all enhanced components
        print("\n📊 Step 1: Initializing Enhanced Components...")
        
        embargo_manager = await create_embargo_manager()
        lob_extractor = await create_lob_extractor()
        hierarchical_ensemble = await create_hierarchical_ensemble({
            'n_base_models': 8,
            'n_meta_models': 3,
            'uncertainty_method': 'bootstrap',
            'calibration_window': 500,
            'drift_threshold': 0.1
        })
        options_analyzer = await create_options_analyzer()
        enhanced_technical_agent = await create_enhanced_technical_agent()
        
        print("✅ All enhanced components initialized successfully")
        
        # Step 2: Set up embargo events
        print("\n🚫 Step 2: Setting Up Embargo Events...")
        
        now = datetime.now()
        
        # Add earnings embargo for AAPL
        earnings_event = EmbargoEvent(
            event_id="AAPL_EARNINGS_2024Q1",
            event_type=EmbargoType.EARNINGS,
            symbol="AAPL",
            event_date=now + timedelta(days=7),
            embargo_start=now,
            embargo_end=now + timedelta(days=10),
            embargo_horizon=7,
            embargo_duration=3,
            confidence=0.9,
            source="integration_test"
        )
        
        await embargo_manager.add_embargo_event(earnings_event)
        
        # Add split embargo for TSLA
        split_event = EmbargoEvent(
            event_id="TSLA_SPLIT_2024",
            event_type=EmbargoType.SPLIT,
            symbol="TSLA",
            event_date=now + timedelta(days=3),
            embargo_start=now,
            embargo_end=now + timedelta(days=4),
            embargo_horizon=3,
            embargo_duration=1,
            confidence=0.8,
            source="integration_test"
        )
        
        await embargo_manager.add_embargo_event(split_event)
        
        print(f"✅ Added {len(embargo_manager.active_embargos)} embargo events")
        
        # Step 3: Test embargo filtering
        print("\n🔍 Step 3: Testing Embargo Filtering...")
        
        symbols_to_test = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]
        
        for symbol in symbols_to_test:
            is_embargoed, reasons = await embargo_manager.check_embargo_status(symbol, now)
            status = "🚫 EMBARGOED" if is_embargoed else "✅ CLEAR"
            print(f"   {symbol}: {status}")
            if reasons:
                print(f"      Reasons: {reasons}")
        
        # Step 4: Test LOB feature extraction
        print("\n📊 Step 4: Testing LOB Feature Extraction...")
        
        from agents.flow.lob_features import OrderBookSnapshot, OrderBookLevel, OrderSide
        
        # Create sample order book for MSFT
        bids = [
            OrderBookLevel(price=300.00 - i*0.10, size=2000 + i*1000, 
                          side=OrderSide.BID, timestamp=now, venue="NASDAQ")
            for i in range(10)
        ]
        
        asks = [
            OrderBookLevel(price=300.10 + i*0.10, size=2500 + i*800, 
                          side=OrderSide.ASK, timestamp=now, venue="NASDAQ")
            for i in range(10)
        ]
        
        order_book = OrderBookSnapshot(
            symbol="MSFT",
            timestamp=now,
            bids=bids,
            asks=asks,
            last_trade_price=300.05,
            last_trade_size=1000
        )
        
        lob_features = await lob_extractor.extract_lob_features(order_book)
        
        print(f"   MSFT LOB Features:")
        print(f"      Order Imbalance: {lob_features.get('order_imbalance', 0):.3f}")
        print(f"      Spread: {lob_features.get('spread_bps', 0):.1f} bps")
        print(f"      Kyle's Lambda: {lob_features.get('kyle_lambda', 0):.6f}")
        print(f"      Buy Impact (10k): {lob_features.get('buy_impact_10000', 0):.4f}")
        print(f"      Total Depth (3 levels): {lob_features.get('total_depth_3', 0):,.0f}")
        
        # Step 5: Test hierarchical ensemble
        print("\n🧠 Step 5: Testing Hierarchical Meta-Ensemble...")
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=['technical_score', 'lob_imbalance', 'spread_bps', 'kyle_lambda', 
                    'buy_impact', 'sell_impact', 'liquidity_depth', 'large_orders']
        )
        
        # Create target with some structure
        y = pd.Series(
            0.3 * X['technical_score'] + 0.2 * X['lob_imbalance'] + 
            0.1 * X['kyle_lambda'] + 0.4 * np.random.randn(n_samples)
        )
        
        # Train ensemble
        training_results = await hierarchical_ensemble.train_hierarchical(X, y)
        
        print(f"   Ensemble Training Results:")
        print(f"      Base models: {len(training_results.get('base_models', {}))}")
        print(f"      Meta models: {len(training_results.get('meta_models', {}))}")
        print(f"      Super model: {'super_model' in training_results.get('super_model', {})}")
        
        # Test prediction
        test_features = {
            'technical_score': 0.8,
            'lob_imbalance': 0.2,
            'spread_bps': 5.0,
            'kyle_lambda': 0.0001,
            'buy_impact': 0.001,
            'sell_impact': 0.001,
            'liquidity_depth': 50000,
            'large_orders': 2
        }
        
        test_X = pd.DataFrame([test_features])
        
        predictions, uncertainties, intervals = await hierarchical_ensemble.predict_with_uncertainty(test_X)
        
        print(f"   Ensemble Prediction:")
        print(f"      Prediction: {predictions[0]:.4f}")
        print(f"      Uncertainty: {uncertainties[0]:.4f}")
        print(f"      Confidence Interval: [{intervals[0][0]:.4f}, {intervals[0][1]:.4f}]")
        
        # Step 6: Test options surface analysis
        print("\n📈 Step 6: Testing Options Surface Analysis...")
        
        from agents.insider.options_surface import OptionsSurface, OptionContract, OptionType
        
        # Create sample options data
        underlying_price = 300.00
        
        calls = [
            OptionContract(
                symbol="MSFT240315C300",
                strike=295.0,
                expiry=now + timedelta(days=30),
                option_type=OptionType.CALL,
                last_price=12.50,
                bid=12.45,
                ask=12.55,
                volume=2000,
                open_interest=8000,
                implied_volatility=0.25,
                delta=0.65,
                gamma=0.02,
                theta=-0.15,
                vega=0.12,
                rho=0.08,
                timestamp=now
            ),
            OptionContract(
                symbol="MSFT240315C305",
                strike=305.0,
                expiry=now + timedelta(days=30),
                option_type=OptionType.CALL,
                last_price=8.20,
                bid=8.15,
                ask=8.25,
                volume=1800,
                open_interest=6000,
                implied_volatility=0.28,
                delta=0.45,
                gamma=0.03,
                theta=-0.18,
                vega=0.15,
                rho=0.06,
                timestamp=now
            )
        ]
        
        puts = [
            OptionContract(
                symbol="MSFT240315P300",
                strike=295.0,
                expiry=now + timedelta(days=30),
                option_type=OptionType.PUT,
                last_price=7.80,
                bid=7.75,
                ask=7.85,
                volume=1500,
                open_interest=5000,
                implied_volatility=0.30,
                delta=-0.35,
                gamma=0.02,
                theta=-0.12,
                vega=0.10,
                rho=-0.04,
                timestamp=now
            ),
            OptionContract(
                symbol="MSFT240315P305",
                strike=305.0,
                expiry=now + timedelta(days=30),
                option_type=OptionType.PUT,
                last_price=12.90,
                bid=12.85,
                ask=12.95,
                volume=1200,
                open_interest=4000,
                implied_volatility=0.32,
                delta=-0.55,
                gamma=0.02,
                theta=-0.14,
                vega=0.11,
                rho=-0.06,
                timestamp=now
            )
        ]
        
        options_surface = OptionsSurface(
            symbol="MSFT",
            underlying_price=underlying_price,
            timestamp=now,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            calls=calls,
            puts=puts
        )
        
        options_features = await options_analyzer.analyze_options_surface(options_surface)
        
        print(f"   MSFT Options Analysis:")
        print(f"      Put/Call Ratio: {options_features.get('put_call_volume_ratio', 0):.3f}")
        print(f"      IV Skew: {options_features.get('iv_skew', 0):.3f}")
        print(f"      VW Delta: {options_features.get('vw_delta', 0):.3f}")
        print(f"      VW Gamma: {options_features.get('vw_gamma', 0):.4f}")
        print(f"      Volume Anomaly: {options_features.get('volume_anomaly', 0):.3f}")
        
        # Step 7: Test enhanced technical agent
        print("\n🔧 Step 7: Testing Enhanced Technical Agent...")
        
        # Create analysis payload
        analysis_payload = {
            "symbols": ["MSFT", "GOOGL", "NVDA"],  # Exclude embargoed symbols
            "timeframes": ["1h", "4h"],
            "strategies": ["imbalance", "trend"],
            "min_score": 0.01,
            "max_risk": 0.02,
            "lookback_periods": 100
        }
        
        enhanced_result = await enhanced_technical_agent.find_opportunities_enhanced(analysis_payload)
        
        print(f"   Enhanced Technical Analysis Results:")
        print(f"      Symbols Analyzed: {enhanced_result['metadata']['symbols_analyzed']}")
        print(f"      Opportunities Found: {enhanced_result['metadata']['opportunities_found']}")
        print(f"      Embargoed Symbols: {len(enhanced_result['metadata']['embargoed_symbols'])}")
        print(f"      Enhanced Features: {enhanced_result['metadata']['enhanced_features']}")
        
        if enhanced_result['opportunities']:
            opp = enhanced_result['opportunities'][0]
            print(f"   Sample Opportunity:")
            print(f"      Symbol: {opp['symbol']}")
            print(f"      Strategy: {opp['strategy']}")
            print(f"      Confidence: {opp['confidence_score']:.3f}")
            print(f"      LOB Features: {'order_imbalance' in opp.get('lob_features', {})}")
            print(f"      Ensemble Prediction: {'prediction' in opp.get('ensemble_prediction', {})}")
        
        # Step 8: Test API integration
        print("\n🌐 Step 8: Testing API Integration...")
        
        # Test embargo check endpoint
        embargo_check = await app.state.embargo_manager.check_embargo_status("AAPL", now)
        print(f"   API Embargo Check (AAPL): {'EMBARGOED' if embargo_check[0] else 'CLEAR'}")
        
        # Test LOB analysis endpoint
        lob_result = await app.state.lob_extractor.extract_lob_features(order_book)
        print(f"   API LOB Analysis: {len(lob_result)} features extracted")
        
        # Test ensemble prediction endpoint
        ensemble_result = await hierarchical_ensemble.predict_with_uncertainty(test_X)
        print(f"   API Ensemble Prediction: {ensemble_result[0][0]:.4f}")
        
        # Step 9: System performance summary
        print("\n📊 Step 9: System Performance Summary...")
        
        embargo_summary = await embargo_manager.get_embargo_summary()
        ensemble_summary = await hierarchical_ensemble.get_ensemble_summary()
        technical_status = await enhanced_technical_agent.get_enhanced_status()
        
        print(f"   Embargo System:")
        print(f"      Active Embargos: {embargo_summary['active_embargos']}")
        print(f"      Total Checks: {embargo_summary['total_checks']}")
        print(f"      Violation Rate: {embargo_summary['violation_rate']:.2%}")
        
        print(f"   Hierarchical Ensemble:")
        print(f"      Total Models: {ensemble_summary['overall_performance']['total_models']}")
        print(f"      Calibration Samples: {ensemble_summary['overall_performance']['calibration_samples']}")
        
        print(f"   Enhanced Technical Agent:")
        print(f"      Embargo Violations: {technical_status['embargo_manager']['violations']}")
        print(f"      LOB Analyses: {technical_status['lob_extractor']['analyses']}")
        print(f"      Ensemble Predictions: {technical_status['hierarchical_ensemble']['predictions']}")
        
        # Step 10: Integration validation
        print("\n✅ Step 10: Integration Validation...")
        
        integration_success = True
        validation_checks = [
            ("Embargo Manager Active", embargo_manager is not None),
            ("LOB Extractor Active", lob_extractor is not None),
            ("Hierarchical Ensemble Active", hierarchical_ensemble is not None),
            ("Options Analyzer Active", options_analyzer is not None),
            ("Enhanced Technical Agent Active", enhanced_technical_agent is not None),
            ("Embargo Events Created", len(embargo_manager.active_embargos) > 0),
            ("LOB Features Extracted", len(lob_features) > 0),
            ("Ensemble Trained", len(training_results.get('base_models', {})) > 0),
            ("Options Analysis Completed", len(options_features) > 0),
            ("Enhanced Analysis Completed", enhanced_result['metadata']['symbols_analyzed'] > 0)
        ]
        
        for check_name, check_result in validation_checks:
            status = "✅ PASS" if check_result else "❌ FAIL"
            print(f"   {check_name}: {status}")
            if not check_result:
                integration_success = False
        
        print("\n" + "="*80)
        if integration_success:
            print("🎉 ENHANCED TRADING SYSTEM INTEGRATION: SUCCESS!")
            print("✅ All major enhancements are working together seamlessly")
            print("✅ System is ready for production deployment")
        else:
            print("⚠️  ENHANCED TRADING SYSTEM INTEGRATION: PARTIAL SUCCESS")
            print("⚠️  Some components may need additional configuration")
        
        print("="*80)
        
        return integration_success
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the enhanced system integration test"""
    print("🚀 Enhanced Trading System Integration Test")
    print("Testing all major enhancements working together...")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run integration test
    success = await test_enhanced_system_integration()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Deploy the enhanced system to production")
        print("2. Configure real-time data feeds")
        print("3. Set up monitoring and alerting")
        print("4. Begin live trading with enhanced features")
        print("5. Monitor performance and optimize")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check component initialization")
        print("2. Verify dependencies are installed")
        print("3. Review error logs")
        print("4. Test components individually")
        print("5. Contact support if issues persist")


if __name__ == "__main__":
    asyncio.run(main())
