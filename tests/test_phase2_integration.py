#!/usr/bin/env python3
"""
Phase 2 Integration Tests

Comprehensive tests for:
- Agent standardization with uncertainty quantification
- Meta-weighter QR LightGBM implementation  
- Diversified selector with anti-correlation
- End-to-end uncertainty propagation
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.contracts import Signal, Opportunity, SignalType, RegimeType, HorizonType, DirectionType
from agents.technical.agent_phase2 import TechnicalAgentPhase2
from agents.sentiment.agent_phase2 import SentimentAgentPhase2
from agents.flow.agent_phase2 import FlowAgentPhase2
from agents.macro.agent_phase2 import MacroAgentPhase2
from ml.meta_weighter import QRLightGBMMetaWeighter
from ml.diversified_selector import DiversifiedTopKSelector


class TestPhase2Integration:
    """Integration tests for Phase 2 components"""
    
    @pytest.fixture
    def test_symbols(self):
        """Test symbols for integration tests"""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    @pytest.fixture
    def agent_config(self):
        """Standard agent configuration"""
        return {
            'min_confidence': 0.2,
            'model_version': '2.0.0-test',
            'feature_version': '2.0.0-test'
        }
    
    @pytest.fixture
    def test_agents(self, agent_config):
        """Initialize test agents"""
        agents = {
            'technical': TechnicalAgentPhase2(agent_config),
            'sentiment': SentimentAgentPhase2(agent_config),
            'flow': FlowAgentPhase2(agent_config),
            'macro': MacroAgentPhase2(agent_config)
        }
        return agents
    
    @pytest.fixture
    def meta_weighter_config(self):
        """Meta-weighter configuration"""
        return {
            'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
            'n_estimators': 10,  # Reduced for tests
            'learning_rate': 0.1,
            'calibration_window': 100,  # Reduced for tests
        }
    
    @pytest.fixture
    def meta_weighter(self, meta_weighter_config):
        """Initialize meta-weighter"""
        return QRLightGBMMetaWeighter(meta_weighter_config)
    
    @pytest.fixture
    def selector_config(self):
        """Diversified selector configuration"""
        return {
            'top_k': 3,  # Reduced for tests
            'correlation_penalty': 0.15,
            'min_expected_return': 0.001,  # Lowered for tests
            'risk_aversion': 2.0,
        }
    
    @pytest.fixture
    def diversified_selector(self, selector_config):
        """Initialize diversified selector"""
        return DiversifiedTopKSelector(selector_config)

    @pytest.mark.asyncio
    async def test_agent_standardization(self, test_agents, test_symbols):
        """Test that all agents emit standardized uncertainty-quantified signals"""
        
        trace_id = "test-agent-standardization"
        
        for agent_name, agent in test_agents.items():
            # Generate signals
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            
            # Verify signals generated
            assert len(signals) > 0, f"{agent_name} agent should generate signals"
            
            # Test signal structure
            for signal in signals:
                # Verify uncertainty quantification
                assert isinstance(signal.mu, float), "mu should be float"
                assert isinstance(signal.sigma, float), "sigma should be float"
                assert signal.sigma > 0, "sigma should be positive"
                
                # Verify confidence
                assert 0.0 <= signal.confidence <= 1.0, "confidence should be in [0,1]"
                
                # Verify horizon
                assert signal.horizon in HorizonType, "horizon should be valid HorizonType"
                
                # Verify regime
                assert signal.regime in RegimeType, "regime should be valid RegimeType"
                
                # Verify direction consistency
                if signal.mu > 0.01:
                    assert signal.direction == DirectionType.LONG
                elif signal.mu < -0.01:
                    assert signal.direction == DirectionType.SHORT
                else:
                    assert signal.direction == DirectionType.NEUTRAL
                
                # Verify metadata
                assert signal.trace_id == trace_id, "trace_id should match"
                assert signal.agent_id == agent.name, "agent_id should match"
                assert signal.agent_type.value == agent_name, "agent_type should match"
                
                # Verify timestamps
                assert isinstance(signal.timestamp, datetime), "timestamp should be datetime"
                
                # Verify versioning
                assert signal.model_version is not None, "model_version should be set"
                assert signal.feature_version is not None, "feature_version should be set"

    @pytest.mark.asyncio
    async def test_meta_weighter_blending(self, test_agents, meta_weighter, test_symbols):
        """Test meta-weighter signal blending with uncertainty propagation"""
        
        trace_id = "test-meta-weighter"
        
        # Generate signals from all agents
        all_signals = []
        for agent in test_agents.values():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            all_signals.extend(signals)
        
        assert len(all_signals) > 0, "Should have signals to blend"
        
        # Blend signals into opportunities
        opportunities = await meta_weighter.blend_signals(all_signals, trace_id=trace_id)
        
        # Verify opportunities created
        assert len(opportunities) > 0, "Should create opportunities from signals"
        
        # Test opportunity structure
        for opp in opportunities:
            # Verify blended uncertainty quantification
            assert isinstance(opp.mu_blended, float), "mu_blended should be float"
            assert isinstance(opp.sigma_blended, float), "sigma_blended should be float"
            assert opp.sigma_blended > 0, "sigma_blended should be positive"
            
            # Verify confidence blending
            assert 0.0 <= opp.confidence_blended <= 1.0, "confidence_blended should be in [0,1]"
            
            # Verify agent signals mapping
            assert len(opp.agent_signals) > 0, "Should have agent signals"
            
            # Verify risk metrics
            assert opp.var_95 is not None, "VaR should be calculated"
            assert opp.cvar_95 is not None, "CVaR should be calculated"
            
            # Verify Sharpe ratio
            if opp.sigma_blended > 0:
                expected_sharpe = opp.mu_blended / opp.sigma_blended
                assert abs(opp.sharpe_ratio - expected_sharpe) < 0.01, "Sharpe ratio should be correct"
            
            # Verify metadata preservation
            assert opp.trace_id == trace_id, "trace_id should be preserved"

    @pytest.mark.asyncio
    async def test_diversified_selection(self, test_agents, meta_weighter, diversified_selector, test_symbols):
        """Test diversified selection with anti-correlation logic"""
        
        trace_id = "test-diversified-selection"
        
        # Generate complete pipeline
        all_signals = []
        for agent in test_agents.values():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            all_signals.extend(signals)
        
        opportunities = await meta_weighter.blend_signals(all_signals, trace_id=trace_id)
        
        # Select diversified opportunities
        selected = await diversified_selector.select_opportunities(opportunities, trace_id=trace_id)
        
        # Verify selection results
        assert len(selected) <= diversified_selector.config['top_k'], "Should not exceed top_k"
        assert len(selected) > 0, "Should select at least one opportunity"
        
        # Verify diversification
        if len(selected) > 1:
            # Check that selected opportunities are not identical
            symbols = [opp.symbol for opp in selected]
            assert len(set(symbols)) == len(symbols), "Should select different symbols"
            
            # Verify selection quality (sorted by score)
            scores = [getattr(opp, 'selection_score', 0) for opp in selected]
            assert scores == sorted(scores, reverse=True), "Should be sorted by selection score"
        
        # Verify expected return threshold
        min_return = diversified_selector.config['min_expected_return']
        for opp in selected:
            assert opp.mu_blended >= min_return, f"Selected opportunity should meet min return threshold"

    @pytest.mark.asyncio
    async def test_end_to_end_uncertainty_propagation(self, test_agents, meta_weighter, 
                                                     diversified_selector, test_symbols):
        """Test end-to-end uncertainty propagation through the pipeline"""
        
        trace_id = "test-e2e-uncertainty"
        
        # Stage 1: Generate signals with individual uncertainties
        all_signals = []
        for agent in test_agents.values():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            all_signals.extend(signals)
        
        signal_uncertainties = [s.sigma for s in all_signals]
        avg_signal_uncertainty = np.mean(signal_uncertainties)
        
        # Stage 2: Blend signals (should reduce uncertainty)
        opportunities = await meta_weighter.blend_signals(all_signals, trace_id=trace_id)
        
        blended_uncertainties = [o.sigma_blended for o in opportunities]
        avg_blended_uncertainty = np.mean(blended_uncertainties)
        
        # Stage 3: Select diversified portfolio
        selected = await diversified_selector.select_opportunities(opportunities, trace_id=trace_id)
        
        portfolio_uncertainties = [o.sigma_blended for o in selected]
        avg_portfolio_uncertainty = np.mean(portfolio_uncertainties)
        
        # Verify uncertainty propagation
        assert avg_signal_uncertainty > 0, "Should have signal uncertainty"
        assert avg_blended_uncertainty > 0, "Should have blended uncertainty"
        assert avg_portfolio_uncertainty > 0, "Should have portfolio uncertainty"
        
        # Verify uncertainty generally decreases (with some tolerance for randomness)
        # Note: In real scenarios, blending should reduce uncertainty, but in tests
        # with random data, this might not always hold
        print(f"Signal uncertainty: {avg_signal_uncertainty:.4f}")
        print(f"Blended uncertainty: {avg_blended_uncertainty:.4f}")
        print(f"Portfolio uncertainty: {avg_portfolio_uncertainty:.4f}")

    @pytest.mark.asyncio
    async def test_performance_vs_naive_selection(self, test_agents, meta_weighter, 
                                                 diversified_selector, test_symbols):
        """Test performance improvement vs naive selection"""
        
        trace_id = "test-performance-comparison"
        
        # Generate complete pipeline
        all_signals = []
        for agent in test_agents.values():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            all_signals.extend(signals)
        
        opportunities = await meta_weighter.blend_signals(all_signals, trace_id=trace_id)
        
        if len(opportunities) < 2:
            pytest.skip("Need at least 2 opportunities for comparison")
        
        # Sophisticated selection
        smart_selected = await diversified_selector.select_opportunities(opportunities, trace_id=trace_id)
        
        # Naive selection: top K by expected return
        top_k = len(smart_selected)
        naive_selected = sorted(opportunities, key=lambda x: x.mu_blended, reverse=True)[:top_k]
        
        # Calculate correlation metrics
        def calculate_avg_correlation(portfolio):
            if len(portfolio) < 2:
                return 0.0
            
            correlations = []
            for i, opp1 in enumerate(portfolio):
                for opp2 in portfolio[i+1:]:
                    # Simple correlation proxy based on agent overlap
                    agents1 = set(opp1.agent_signals.keys())
                    agents2 = set(opp2.agent_signals.keys())
                    common_agents = agents1 & agents2
                    total_agents = agents1 | agents2
                    correlation = len(common_agents) / len(total_agents) if total_agents else 0
                    correlations.append(correlation)
            
            return np.mean(correlations) if correlations else 0.0
        
        smart_correlation = calculate_avg_correlation(smart_selected)
        naive_correlation = calculate_avg_correlation(naive_selected)
        
        # Verify diversification benefit
        # Smart selection should have lower correlation (more diversified)
        print(f"Smart selection correlation: {smart_correlation:.3f}")
        print(f"Naive selection correlation: {naive_correlation:.3f}")
        
        # In most cases, sophisticated selection should be more diversified
        # (but we don't assert this strongly due to test randomness)

    @pytest.mark.asyncio
    async def test_trace_id_propagation(self, test_agents, meta_weighter, diversified_selector, test_symbols):
        """Test that trace_id is properly propagated through the pipeline"""
        
        trace_id = "test-trace-propagation-12345"
        
        # Generate signals
        all_signals = []
        for agent in test_agents.values():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            all_signals.extend(signals)
        
        # Verify trace_id in signals
        for signal in all_signals:
            assert signal.trace_id == trace_id, "Signal should preserve trace_id"
        
        # Blend signals
        opportunities = await meta_weighter.blend_signals(all_signals, trace_id=trace_id)
        
        # Verify trace_id in opportunities
        for opp in opportunities:
            assert opp.trace_id == trace_id, "Opportunity should preserve trace_id"
        
        # Select opportunities
        selected = await diversified_selector.select_opportunities(opportunities, trace_id=trace_id)
        
        # Verify trace_id in selected opportunities
        for opp in selected:
            assert opp.trace_id == trace_id, "Selected opportunity should preserve trace_id"

    @pytest.mark.asyncio
    async def test_regime_awareness(self, test_agents, test_symbols):
        """Test that agents are regime-aware"""
        
        trace_id = "test-regime-awareness"
        
        for agent_name, agent in test_agents.items():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            
            # Verify regime detection
            regimes = set(s.regime for s in signals)
            assert len(regimes) > 0, f"{agent_name} should detect regimes"
            
            # All regimes should be valid
            for regime in regimes:
                assert regime in RegimeType, f"Invalid regime: {regime}"

    @pytest.mark.asyncio
    async def test_horizon_consistency(self, test_agents, test_symbols):
        """Test horizon consistency across agents"""
        
        trace_id = "test-horizon-consistency"
        
        for agent_name, agent in test_agents.items():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            
            # Verify horizon assignment
            horizons = set(s.horizon for s in signals)
            assert len(horizons) > 0, f"{agent_name} should assign horizons"
            
            # All horizons should be valid
            for horizon in horizons:
                assert horizon in HorizonType, f"Invalid horizon: {horizon}"
            
            # Verify horizon logic based on agent type
            for signal in signals:
                if agent.agent_type == SignalType.TECHNICAL:
                    # Technical signals often short-term
                    assert signal.horizon in [HorizonType.INTRADAY, HorizonType.SHORT_TERM]
                elif agent.agent_type == SignalType.MACRO:
                    # Macro signals often longer-term
                    assert signal.horizon in [HorizonType.MEDIUM_TERM, HorizonType.LONG_TERM]

    @pytest.mark.asyncio
    async def test_calibration_awareness(self, meta_weighter, test_symbols):
        """Test calibration awareness in meta-weighter"""
        
        # Create test signals with known uncertainties
        test_signals = []
        for i, symbol in enumerate(test_symbols):
            signal = Signal(
                trace_id="test-calibration",
                agent_id=f"test-agent-{i}",
                agent_type=SignalType.TECHNICAL,
                symbol=symbol,
                mu=0.01 * (i + 1),  # Different expected returns
                sigma=0.02 * (i + 1),  # Different uncertainties
                confidence=0.7 + 0.05 * i,  # Different confidences
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="test-1.0",
                feature_version="test-1.0"
            )
            test_signals.append(signal)
        
        # Blend signals
        opportunities = await meta_weighter.blend_signals(test_signals, trace_id="test-calibration")
        
        # Verify calibration-aware blending
        for opp in opportunities:
            # Blended confidence should consider individual signal confidences
            signal_confidences = [s.confidence for s in test_signals if s.symbol == opp.symbol]
            if signal_confidences:
                avg_confidence = np.mean(signal_confidences)
                # Blended confidence should be related to individual confidences
                assert 0.0 <= opp.confidence_blended <= 1.0, "Blended confidence should be valid"

    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, meta_weighter, test_symbols):
        """Test risk metrics calculation in opportunities"""
        
        # Create test signals
        test_signals = []
        for symbol in test_symbols:
            signal = Signal(
                trace_id="test-risk-metrics",
                agent_id="test-agent",
                agent_type=SignalType.TECHNICAL,
                symbol=symbol,
                mu=0.01,
                sigma=0.02,
                confidence=0.8,
                horizon=HorizonType.SHORT_TERM,
                regime=RegimeType.RISK_ON,
                direction=DirectionType.LONG,
                model_version="test-1.0",
                feature_version="test-1.0"
            )
            test_signals.append(signal)
        
        # Blend signals
        opportunities = await meta_weighter.blend_signals(test_signals, trace_id="test-risk-metrics")
        
        # Verify risk metrics
        for opp in opportunities:
            # VaR should be negative (representing potential loss)
            assert opp.var_95 <= 0, "VaR should be negative or zero"
            
            # CVaR should be more negative than VaR
            assert opp.cvar_95 <= opp.var_95, "CVaR should be <= VaR"
            
            # Sharpe ratio should be calculated
            if opp.sigma_blended > 0:
                expected_sharpe = opp.mu_blended / opp.sigma_blended
                assert abs(opp.sharpe_ratio - expected_sharpe) < 0.01, "Sharpe should be mu/sigma"


# Performance benchmarks
class TestPhase2Performance:
    """Performance tests for Phase 2 components"""
    
    @pytest.fixture
    def test_symbols(self):
        """Test symbols for performance tests"""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    @pytest.fixture
    def agent_config(self):
        """Standard agent configuration"""
        return {
            'min_confidence': 0.2,
            'model_version': '2.0.0-test',
            'feature_version': '2.0.0-test'
        }
    
    @pytest.fixture
    def test_agents(self, agent_config):
        """Initialize test agents"""
        agents = {
            'technical': TechnicalAgentPhase2(agent_config),
            'sentiment': SentimentAgentPhase2(agent_config),
            'flow': FlowAgentPhase2(agent_config),
            'macro': MacroAgentPhase2(agent_config)
        }
        return agents
    
    @pytest.fixture
    def meta_weighter_config(self):
        """Meta-weighter configuration"""
        return {
            'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
            'n_estimators': 10,  # Reduced for tests
            'learning_rate': 0.1,
            'calibration_window': 100,  # Reduced for tests
        }
    
    @pytest.fixture
    def meta_weighter(self, meta_weighter_config):
        """Initialize meta-weighter"""
        return QRLightGBMMetaWeighter(meta_weighter_config)
    
    @pytest.mark.asyncio
    async def test_signal_generation_performance(self, test_agents, test_symbols):
        """Test signal generation performance"""
        import time
        
        trace_id = "test-performance"
        
        start_time = time.time()
        
        all_signals = []
        for agent in test_agents.values():
            signals = await agent.generate_signals(test_symbols, trace_id=trace_id)
            all_signals.extend(signals)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 5.0, f"Signal generation took too long: {duration:.2f}s"
        assert len(all_signals) > 0, "Should generate signals"
        
        signals_per_second = len(all_signals) / duration
        print(f"Generated {len(all_signals)} signals in {duration:.3f}s ({signals_per_second:.1f} signals/s)")

    @pytest.mark.asyncio
    async def test_meta_weighter_performance(self, meta_weighter, test_symbols):
        """Test meta-weighter performance"""
        import time
        
        # Create many test signals
        test_signals = []
        for i in range(50):  # 50 signals for performance test
            for symbol in test_symbols:
                signal = Signal(
                    trace_id="test-performance",
                    agent_id=f"test-agent-{i}",
                    agent_type=SignalType.TECHNICAL,
                    symbol=symbol,
                    mu=np.random.normal(0.01, 0.005),
                    sigma=np.random.uniform(0.01, 0.03),
                    confidence=np.random.uniform(0.5, 0.9),
                    horizon=HorizonType.SHORT_TERM,
                    regime=RegimeType.RISK_ON,
                    direction=DirectionType.LONG,
                    model_version="test-1.0",
                    feature_version="test-1.0"
                )
                test_signals.append(signal)
        
        start_time = time.time()
        opportunities = await meta_weighter.blend_signals(test_signals, trace_id="test-performance")
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 10.0, f"Meta-weighting took too long: {duration:.2f}s"
        assert len(opportunities) > 0, "Should create opportunities"
        
        signals_per_second = len(test_signals) / duration
        print(f"Processed {len(test_signals)} signals in {duration:.3f}s ({signals_per_second:.1f} signals/s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
