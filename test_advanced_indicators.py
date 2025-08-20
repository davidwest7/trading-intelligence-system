#!/usr/bin/env python3
"""
Test Advanced Technical Indicators
"""

import pandas as pd
import numpy as np
from agents.technical.advanced_indicators import AdvancedTechnicalIndicators

def create_test_data():
    """Create sample market data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create realistic price data with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.normal(0, 2, 100)
    prices = base_price + trend + noise
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 100),
        'high': prices + np.abs(np.random.normal(1, 0.5, 100)),
        'low': prices - np.abs(np.random.normal(1, 0.5, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    return data

def test_advanced_indicators():
    """Test all advanced technical indicators"""
    print("🧪 Testing Advanced Technical Indicators...")
    
    # Create test data
    data = create_test_data()
    indicators = AdvancedTechnicalIndicators()
    
    print(f"📊 Test data shape: {data.shape}")
    print(f"📈 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    # Test Ichimoku Cloud
    print("\n☁️ Testing Ichimoku Cloud...")
    ichimoku = indicators.calculate_ichimoku_cloud(data)
    print(f"   Tenkan-sen: {ichimoku.tenkan_sen.iloc[-1]:.2f}")
    print(f"   Kijun-sen: {ichimoku.kijun_sen.iloc[-1]:.2f}")
    print(f"   Senkou Span A: {ichimoku.senkou_span_a.iloc[-1]:.2f}")
    print(f"   Senkou Span B: {ichimoku.senkou_span_b.iloc[-1]:.2f}")
    
    # Test Fibonacci Levels
    print("\n📐 Testing Fibonacci Retracements...")
    fibonacci = indicators.calculate_fibonacci_levels(data)
    print(f"   Level 0%: ${fibonacci.level_0:.2f}")
    print(f"   Level 23.6%: ${fibonacci.level_236:.2f}")
    print(f"   Level 38.2%: ${fibonacci.level_382:.2f}")
    print(f"   Level 50%: ${fibonacci.level_500:.2f}")
    print(f"   Level 61.8%: ${fibonacci.level_618:.2f}")
    print(f"   Level 78.6%: ${fibonacci.level_786:.2f}")
    print(f"   Level 100%: ${fibonacci.level_100:.2f}")
    
    # Test Elliott Wave
    print("\n🌊 Testing Elliott Wave Analysis...")
    elliott = indicators.detect_elliott_waves(data)
    print(f"   Wave Count: {elliott.wave_count}")
    print(f"   Current Wave: {elliott.current_wave}")
    print(f"   Pattern: {elliott.wave_pattern}")
    print(f"   Confidence: {elliott.confidence:.2f}")
    print(f"   Target Levels: {[f'${level:.2f}' for level in elliott.target_levels]}")
    
    # Test Harmonic Patterns
    print("\n🎵 Testing Harmonic Patterns...")
    harmonics = indicators.detect_harmonic_patterns(data)
    print(f"   Patterns Found: {len(harmonics)}")
    for i, pattern in enumerate(harmonics):
        print(f"   Pattern {i+1}: {pattern.pattern_type} (Confidence: {pattern.confidence:.2f})")
    
    # Test Volume Profile
    print("\n📊 Testing Volume Profile...")
    volume_profile = indicators.calculate_volume_profile(data)
    print(f"   Point of Control: ${volume_profile['poc']:.2f}")
    print(f"   Value Area High: ${volume_profile['value_area_high']:.2f}")
    print(f"   Value Area Low: ${volume_profile['value_area_low']:.2f}")
    print(f"   VWAP: ${volume_profile['volume_weighted_price']:.2f}")
    
    # Test Market Microstructure
    print("\n🔬 Testing Market Microstructure...")
    microstructure = indicators.calculate_market_microstructure(data)
    print(f"   Bid-Ask Spread: {microstructure['bid_ask_spread']:.4f}")
    print(f"   Order Flow Imbalance: {microstructure['order_flow_imbalance']:.4f}")
    print(f"   Market Depth: {microstructure['market_depth']:.2f}")
    print(f"   Price Impact: {microstructure['price_impact']:.4f}")
    
    # Test Advanced Oscillators
    print("\n📈 Testing Advanced Oscillators...")
    oscillators = indicators.calculate_advanced_oscillators(data)
    print(f"   Williams %R: {oscillators['williams_r'].iloc[-1]:.2f}")
    print(f"   Stochastic K: {oscillators['stoch_k'].iloc[-1]:.2f}")
    print(f"   Stochastic D: {oscillators['stoch_d'].iloc[-1]:.2f}")
    print(f"   CCI: {oscillators['cci'].iloc[-1]:.2f}")
    print(f"   MFI: {oscillators['mfi'].iloc[-1]:.2f}")
    print(f"   ROC: {oscillators['roc'].iloc[-1]:.4f}")
    print(f"   Ultimate Oscillator: {oscillators['ultimate_osc'].iloc[-1]:.2f}")
    print(f"   ADX: {oscillators['adx'].iloc[-1]:.2f}")
    
    # Test Statistical Arbitrage
    print("\n📊 Testing Statistical Arbitrage...")
    stat_arb = indicators.calculate_statistical_arbitrage_signals(data)
    print(f"   Z-Score: {stat_arb['z_score']:.2f}")
    print(f"   Mean Reversion Probability: {stat_arb['mean_reversion_probability']:.2f}")
    print(f"   Momentum Probability: {stat_arb['momentum_probability']:.2f}")
    print(f"   Volatility Regime: {stat_arb['volatility_regime']}")
    
    # Test Composite Signal
    print("\n🎯 Testing Composite Signal...")
    composite = indicators.calculate_composite_signal(data)
    print(f"   Composite Score: {composite['composite_score']:.3f}")
    print(f"   Signal Strength: {composite['signal_strength']:.3f}")
    print(f"   Confidence: {composite['confidence']:.3f}")
    print(f"   Risk Level: {composite['risk_level']}")
    
    print("\n✅ All advanced technical indicators tested successfully!")
    return True

if __name__ == "__main__":
    test_advanced_indicators()
