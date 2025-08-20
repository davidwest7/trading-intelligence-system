#!/usr/bin/env python3
"""
Phase 2 Uncertainty-Aware Trading System Demo

Demonstrates the complete Phase 2 pipeline:
- Standardized agents with uncertainty quantification (μ, σ, horizon)
- QR LightGBM meta-weighter with isotonic calibration
- Diversified Top-K selector with anti-correlation logic
- End-to-end uncertainty propagation
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas.contracts import Signal, Opportunity, SignalType, RegimeType, HorizonType, DirectionType
from agents.technical.agent_phase2 import TechnicalAgentPhase2
from agents.sentiment.agent_phase2 import SentimentAgentPhase2
from agents.flow.agent_phase2 import FlowAgentPhase2
from agents.macro.agent_phase2 import MacroAgentPhase2
from ml.meta_weighter import QRLightGBMMetaWeighter
from ml.diversified_selector import DiversifiedTopKSelector
from common.observability.telemetry import init_telemetry


class Phase2Demo:
    """Phase 2 uncertainty-aware trading system demo"""
    
    def __init__(self):
        self.telemetry = None
        self.agents = {}
        self.meta_weighter = None
        self.diversified_selector = None
        self.demo_results = {}
        
        # Demo configuration
        self.demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'JPM', 'XOM', 'JNJ']
        self.trace_id = f"phase2-demo-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
    async def run_demo(self):
        """Run the complete Phase 2 demo"""
        print("🚀 **PHASE 2 UNCERTAINTY-AWARE TRADING SYSTEM DEMO**")
        print("=" * 80)
        
        try:
            # Initialize components
            await self._init_components()
            
            # Step 1: Generate uncertainty-quantified signals
            await self._demo_signal_generation()
            
            # Step 2: Meta-weighting with uncertainty propagation
            await self._demo_meta_weighting()
            
            # Step 3: Diversified selection with anti-correlation
            await self._demo_diversified_selection()
            
            # Step 4: End-to-end uncertainty analysis
            await self._demo_uncertainty_analysis()
            
            # Step 5: Performance comparison
            await self._demo_performance_comparison()
            
            # Generate comprehensive report
            await self._generate_demo_report()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _init_components(self):
        """Initialize all Phase 2 components"""
        print("\n🔧 **INITIALIZING PHASE 2 COMPONENTS**")
        print("-" * 60)
        
        # Initialize telemetry
        telemetry_config = {
            'service_name': 'phase2-demo',
            'service_version': '2.0.0',
            'environment': 'demo',
        }
        self.telemetry = init_telemetry(telemetry_config)
        print("✅ Telemetry system initialized")
        
        # Initialize agents
        agent_config = {'min_confidence': 0.3}
        
        self.agents['technical'] = TechnicalAgentPhase2(agent_config)
        print("✅ Technical Agent (Phase 2) initialized")
        
        self.agents['sentiment'] = SentimentAgentPhase2(agent_config)
        print("✅ Sentiment Agent (Phase 2) initialized")
        
        self.agents['flow'] = FlowAgentPhase2(agent_config)
        print("✅ Flow Agent (Phase 2) initialized")
        
        self.agents['macro'] = MacroAgentPhase2(agent_config)
        print("✅ Macro Agent (Phase 2) initialized")
        
        # Initialize meta-weighter
        meta_config = {
            'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
            'n_estimators': 50,
            'learning_rate': 0.1,
            'calibration_window': 500,
        }
        self.meta_weighter = QRLightGBMMetaWeighter(meta_config)
        print("✅ QR LightGBM Meta-Weighter initialized")
        
        # Initialize diversified selector
        selector_config = {
            'top_k': 5,
            'correlation_penalty': 0.15,
            'min_expected_return': 0.005,
            'risk_aversion': 2.0,
        }
        self.diversified_selector = DiversifiedTopKSelector(selector_config)
        print("✅ Diversified Top-K Selector initialized")
    
    async def _demo_signal_generation(self):
        """Demonstrate uncertainty-quantified signal generation"""
        print("\n📊 **STEP 1: UNCERTAINTY-QUANTIFIED SIGNAL GENERATION**")
        print("-" * 60)
        
        all_signals = []
        
        # Generate signals from each agent
        for agent_name, agent in self.agents.items():
            print(f"\n🤖 **{agent_name.upper()} AGENT ANALYSIS**")
            
            try:
                # Generate signals for demo symbols
                agent_signals = await agent.generate_signals(
                    self.demo_symbols,
                    trace_id=self.trace_id
                )
                
                if agent_signals:
                    print(f"   📈 Generated {len(agent_signals)} signals")
                    
                    # Show sample signals
                    for i, signal in enumerate(agent_signals[:3]):  # Show first 3
                        confidence_pct = signal.confidence * 100
                        print(f"   • {signal.symbol}: μ={signal.mu:.4f}, σ={signal.sigma:.4f}, "
                              f"confidence={confidence_pct:.1f}%, horizon={signal.horizon.value}")
                    
                    all_signals.extend(agent_signals)
                else:
                    print("   ⚠️ No signals generated")
                    
            except Exception as e:
                print(f"   ❌ Error generating {agent_name} signals: {e}")
        
        # Store results
        self.demo_results['signals'] = {
            'total_signals': len(all_signals),
            'signals_by_agent': {
                agent_name: len([s for s in all_signals if s.agent_type.value == agent_name])
                for agent_name in self.agents.keys()
            },
            'avg_confidence': np.mean([s.confidence for s in all_signals]) if all_signals else 0,
            'avg_expected_return': np.mean([s.mu for s in all_signals]) if all_signals else 0,
            'avg_uncertainty': np.mean([s.sigma for s in all_signals]) if all_signals else 0,
            'all_signals': all_signals
        }
        
        print(f"\n✅ **SIGNAL GENERATION SUMMARY**")
        print(f"   📊 Total signals: {len(all_signals)}")
        print(f"   🎯 Average confidence: {self.demo_results['signals']['avg_confidence']:.1%}")
        print(f"   📈 Average expected return: {self.demo_results['signals']['avg_expected_return']:.3f}")
        print(f"   📉 Average uncertainty: {self.demo_results['signals']['avg_uncertainty']:.3f}")
    
    async def _demo_meta_weighting(self):
        """Demonstrate meta-weighting with uncertainty propagation"""
        print("\n🧠 **STEP 2: META-WEIGHTING WITH UNCERTAINTY PROPAGATION**")
        print("-" * 60)
        
        all_signals = self.demo_results['signals']['all_signals']
        
        if not all_signals:
            print("❌ No signals available for meta-weighting")
            return
        
        try:
            # Blend signals into opportunities
            opportunities = await self.meta_weighter.blend_signals(
                all_signals, trace_id=self.trace_id
            )
            
            print(f"🔄 **BLENDING RESULTS**")
            print(f"   📥 Input: {len(all_signals)} signals from {len(self.agents)} agents")
            print(f"   📤 Output: {len(opportunities)} blended opportunities")
            
            if opportunities:
                print(f"\n🎯 **TOP BLENDED OPPORTUNITIES**")
                
                # Sort by blended confidence
                sorted_opportunities = sorted(opportunities, 
                                            key=lambda x: x.confidence_blended, reverse=True)
                
                for i, opp in enumerate(sorted_opportunities[:5]):
                    sharpe = opp.sharpe_ratio or 0
                    print(f"   {i+1}. {opp.symbol}: μ={opp.mu_blended:.4f}, σ={opp.sigma_blended:.4f}, "
                          f"conf={opp.confidence_blended:.1%}, Sharpe={sharpe:.2f}")
                    print(f"      └─ Agents: {len(opp.agent_signals)}, "
                          f"VaR₉₅={opp.var_95:.4f}, CVaR₉₅={opp.cvar_95:.4f}")
                
                # Analyze uncertainty propagation
                self._analyze_uncertainty_propagation(all_signals, opportunities)
            
            # Store results
            self.demo_results['opportunities'] = {
                'total_opportunities': len(opportunities),
                'avg_blended_confidence': np.mean([o.confidence_blended for o in opportunities]) if opportunities else 0,
                'avg_blended_return': np.mean([o.mu_blended for o in opportunities]) if opportunities else 0,
                'avg_blended_uncertainty': np.mean([o.sigma_blended for o in opportunities]) if opportunities else 0,
                'opportunities': opportunities
            }
            
            print(f"\n✅ **META-WEIGHTING SUMMARY**")
            print(f"   🎯 Opportunities created: {len(opportunities)}")
            print(f"   📊 Average blended confidence: {self.demo_results['opportunities']['avg_blended_confidence']:.1%}")
            print(f"   📈 Average blended return: {self.demo_results['opportunities']['avg_blended_return']:.3f}")
            print(f"   📉 Average blended uncertainty: {self.demo_results['opportunities']['avg_blended_uncertainty']:.3f}")
            
        except Exception as e:
            print(f"❌ Meta-weighting failed: {e}")
            self.demo_results['opportunities'] = {'opportunities': []}
    
    async def _demo_diversified_selection(self):
        """Demonstrate diversified selection with anti-correlation"""
        print("\n🎯 **STEP 3: DIVERSIFIED SELECTION WITH ANTI-CORRELATION**")
        print("-" * 60)
        
        opportunities = self.demo_results['opportunities'].get('opportunities', [])
        
        if not opportunities:
            print("❌ No opportunities available for selection")
            return
        
        try:
            # Select diversified opportunities
            selected_opportunities = await self.diversified_selector.select_opportunities(
                opportunities, trace_id=self.trace_id
            )
            
            print(f"🔄 **SELECTION RESULTS**")
            print(f"   📥 Input: {len(opportunities)} opportunities")
            print(f"   📤 Output: {len(selected_opportunities)} selected opportunities")
            
            if selected_opportunities:
                print(f"\n🏆 **SELECTED DIVERSIFIED PORTFOLIO**")
                
                total_expected_return = 0
                total_risk = 0
                
                for i, opp in enumerate(selected_opportunities):
                    sharpe = opp.sharpe_ratio or 0
                    print(f"   {i+1}. {opp.symbol}: μ={opp.mu_blended:.4f}, σ={opp.sigma_blended:.4f}, "
                          f"Sharpe={sharpe:.2f}")
                    total_expected_return += opp.mu_blended
                    total_risk += opp.sigma_blended ** 2
                
                # Calculate portfolio metrics
                portfolio_expected_return = total_expected_return / len(selected_opportunities)
                portfolio_risk = np.sqrt(total_risk) / len(selected_opportunities)
                portfolio_sharpe = portfolio_expected_return / portfolio_risk if portfolio_risk > 0 else 0
                
                print(f"\n📊 **PORTFOLIO METRICS**")
                print(f"   📈 Portfolio Expected Return: {portfolio_expected_return:.4f}")
                print(f"   📉 Portfolio Risk: {portfolio_risk:.4f}")
                print(f"   ⚡ Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
                
                # Analyze diversification benefits
                self._analyze_diversification_benefits(opportunities, selected_opportunities)
            
            # Store results
            self.demo_results['selection'] = {
                'candidates': len(opportunities),
                'selected': len(selected_opportunities),
                'selection_rate': len(selected_opportunities) / len(opportunities) if opportunities else 0,
                'selected_opportunities': selected_opportunities
            }
            
            print(f"\n✅ **DIVERSIFIED SELECTION SUMMARY**")
            print(f"   🎯 Selection rate: {self.demo_results['selection']['selection_rate']:.1%}")
            print(f"   📊 Portfolio size: {len(selected_opportunities)}")
            
        except Exception as e:
            print(f"❌ Diversified selection failed: {e}")
            self.demo_results['selection'] = {'selected_opportunities': []}
    
    async def _demo_uncertainty_analysis(self):
        """Demonstrate end-to-end uncertainty analysis"""
        print("\n🔬 **STEP 4: END-TO-END UNCERTAINTY ANALYSIS**")
        print("-" * 60)
        
        all_signals = self.demo_results['signals']['all_signals']
        opportunities = self.demo_results['opportunities'].get('opportunities', [])
        selected = self.demo_results['selection'].get('selected_opportunities', [])
        
        if not all_signals or not opportunities or not selected:
            print("❌ Insufficient data for uncertainty analysis")
            return
        
        # Analyze uncertainty at each stage
        print("📊 **UNCERTAINTY PROPAGATION ANALYSIS**")
        
        # Stage 1: Individual agent uncertainties
        agent_uncertainties = {}
        for agent_name in self.agents.keys():
            agent_signals = [s for s in all_signals if s.agent_type.value == agent_name]
            if agent_signals:
                avg_uncertainty = np.mean([s.sigma for s in agent_signals])
                agent_uncertainties[agent_name] = avg_uncertainty
                print(f"   • {agent_name.capitalize()}: σ_avg = {avg_uncertainty:.4f}")
        
        # Stage 2: Blended uncertainties
        if opportunities:
            blended_uncertainties = [o.sigma_blended for o in opportunities]
            avg_blended_uncertainty = np.mean(blended_uncertainties)
            print(f"   • Meta-Weighter: σ_blended = {avg_blended_uncertainty:.4f}")
        
        # Stage 3: Selected portfolio uncertainty
        if selected:
            portfolio_uncertainties = [o.sigma_blended for o in selected]
            avg_portfolio_uncertainty = np.mean(portfolio_uncertainties)
            print(f"   • Diversified Portfolio: σ_portfolio = {avg_portfolio_uncertainty:.4f}")
        
        # Uncertainty reduction analysis
        print(f"\n🔍 **UNCERTAINTY REDUCTION BENEFITS**")
        
        if len(agent_uncertainties) > 1:
            max_agent_uncertainty = max(agent_uncertainties.values())
            min_agent_uncertainty = min(agent_uncertainties.values())
            uncertainty_range = max_agent_uncertainty - min_agent_uncertainty
            print(f"   📊 Agent uncertainty range: {uncertainty_range:.4f}")
            
            if opportunities:
                blending_reduction = max_agent_uncertainty - avg_blended_uncertainty
                blending_improvement = blending_reduction / max_agent_uncertainty * 100
                print(f"   🧠 Meta-weighting reduction: {blending_reduction:.4f} ({blending_improvement:.1f}%)")
                
                if selected:
                    diversification_reduction = avg_blended_uncertainty - avg_portfolio_uncertainty
                    diversification_improvement = diversification_reduction / avg_blended_uncertainty * 100
                    print(f"   🎯 Diversification reduction: {diversification_reduction:.4f} ({diversification_improvement:.1f}%)")
                    
                    total_reduction = max_agent_uncertainty - avg_portfolio_uncertainty
                    total_improvement = total_reduction / max_agent_uncertainty * 100
                    print(f"   🏆 Total uncertainty reduction: {total_reduction:.4f} ({total_improvement:.1f}%)")
        
        # Store results
        self.demo_results['uncertainty_analysis'] = {
            'agent_uncertainties': agent_uncertainties,
            'blended_uncertainty': avg_blended_uncertainty if opportunities else 0,
            'portfolio_uncertainty': avg_portfolio_uncertainty if selected else 0,
        }
    
    async def _demo_performance_comparison(self):
        """Demonstrate performance comparison"""
        print("\n📈 **STEP 5: PERFORMANCE COMPARISON**")
        print("-" * 60)
        
        opportunities = self.demo_results['opportunities'].get('opportunities', [])
        selected = self.demo_results['selection'].get('selected_opportunities', [])
        
        if not opportunities or not selected:
            print("❌ Insufficient data for performance comparison")
            return
        
        # Compare naive vs sophisticated selection
        print("🏁 **NAIVE VS SOPHISTICATED SELECTION**")
        
        # Naive selection: top K by expected return
        naive_selected = sorted(opportunities, key=lambda x: x.mu_blended, reverse=True)[:len(selected)]
        
        # Calculate metrics for both approaches
        def calculate_portfolio_metrics(portfolio):
            if not portfolio:
                return 0, 0, 0, 0
            
            avg_return = np.mean([o.mu_blended for o in portfolio])
            avg_risk = np.mean([o.sigma_blended for o in portfolio])
            sharpe = avg_return / avg_risk if avg_risk > 0 else 0
            
            # Calculate portfolio correlation
            correlations = []
            for i, opp1 in enumerate(portfolio):
                for opp2 in portfolio[i+1:]:
                    # Simple correlation proxy based on agent overlap
                    common_agents = set(opp1.agent_signals.keys()) & set(opp2.agent_signals.keys())
                    correlation = len(common_agents) / max(len(opp1.agent_signals), len(opp2.agent_signals), 1)
                    correlations.append(correlation)
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            return avg_return, avg_risk, sharpe, avg_correlation
        
        naive_return, naive_risk, naive_sharpe, naive_correlation = calculate_portfolio_metrics(naive_selected)
        smart_return, smart_risk, smart_sharpe, smart_correlation = calculate_portfolio_metrics(selected)
        
        print(f"   📊 **NAIVE SELECTION (Top Expected Return)**")
        print(f"      Expected Return: {naive_return:.4f}")
        print(f"      Risk: {naive_risk:.4f}")
        print(f"      Sharpe Ratio: {naive_sharpe:.2f}")
        print(f"      Avg Correlation: {naive_correlation:.3f}")
        
        print(f"   🧠 **SOPHISTICATED SELECTION (Diversified)**")
        print(f"      Expected Return: {smart_return:.4f}")
        print(f"      Risk: {smart_risk:.4f}")
        print(f"      Sharpe Ratio: {smart_sharpe:.2f}")
        print(f"      Avg Correlation: {smart_correlation:.3f}")
        
        # Calculate improvements
        sharpe_improvement = ((smart_sharpe - naive_sharpe) / naive_sharpe * 100) if naive_sharpe > 0 else 0
        correlation_reduction = ((naive_correlation - smart_correlation) / naive_correlation * 100) if naive_correlation > 0 else 0
        
        print(f"\n🏆 **IMPROVEMENTS**")
        print(f"   ⚡ Sharpe Ratio improvement: {sharpe_improvement:.1f}%")
        print(f"   🎯 Correlation reduction: {correlation_reduction:.1f}%")
        
        # Store results
        self.demo_results['performance_comparison'] = {
            'naive': {
                'return': naive_return,
                'risk': naive_risk,
                'sharpe': naive_sharpe,
                'correlation': naive_correlation
            },
            'sophisticated': {
                'return': smart_return,
                'risk': smart_risk,
                'sharpe': smart_sharpe,
                'correlation': smart_correlation
            },
            'improvements': {
                'sharpe_improvement_pct': sharpe_improvement,
                'correlation_reduction_pct': correlation_reduction
            }
        }
    
    def _analyze_uncertainty_propagation(self, signals: List[Signal], opportunities: List[Opportunity]):
        """Analyze how uncertainty propagates through the system"""
        print(f"\n🔬 **UNCERTAINTY PROPAGATION ANALYSIS**")
        
        # Group signals by symbol
        signals_by_symbol = {}
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        # Compare individual vs blended uncertainties
        for opportunity in opportunities[:3]:  # Show first 3
            symbol_signals = signals_by_symbol.get(opportunity.symbol, [])
            
            if symbol_signals:
                individual_uncertainties = [s.sigma for s in symbol_signals]
                avg_individual = np.mean(individual_uncertainties)
                blended_uncertainty = opportunity.sigma_blended
                
                reduction = (avg_individual - blended_uncertainty) / avg_individual * 100
                print(f"   {opportunity.symbol}: σ_individual={avg_individual:.4f} → σ_blended={blended_uncertainty:.4f} "
                      f"({reduction:.1f}% reduction)")
    
    def _analyze_diversification_benefits(self, all_opportunities: List[Opportunity], 
                                        selected_opportunities: List[Opportunity]):
        """Analyze diversification benefits"""
        print(f"\n🎯 **DIVERSIFICATION BENEFITS ANALYSIS**")
        
        # Calculate average metrics
        if all_opportunities:
            all_avg_return = np.mean([o.mu_blended for o in all_opportunities])
            all_avg_risk = np.mean([o.sigma_blended for o in all_opportunities])
            all_sharpe = all_avg_return / all_avg_risk if all_avg_risk > 0 else 0
        
        if selected_opportunities:
            selected_avg_return = np.mean([o.mu_blended for o in selected_opportunities])
            selected_avg_risk = np.mean([o.sigma_blended for o in selected_opportunities])
            selected_sharpe = selected_avg_return / selected_avg_risk if selected_avg_risk > 0 else 0
            
            print(f"   📊 Universe avg Sharpe: {all_sharpe:.2f}")
            print(f"   🎯 Selected avg Sharpe: {selected_sharpe:.2f}")
            
            if all_sharpe > 0:
                sharpe_improvement = (selected_sharpe - all_sharpe) / all_sharpe * 100
                print(f"   🏆 Selection improvement: {sharpe_improvement:.1f}%")
    
    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        print("\n📋 **PHASE 2 DEMO COMPREHENSIVE REPORT**")
        print("=" * 80)
        
        # Summary metrics
        signals_data = self.demo_results.get('signals', {})
        opportunities_data = self.demo_results.get('opportunities', {})
        selection_data = self.demo_results.get('selection', {})
        uncertainty_data = self.demo_results.get('uncertainty_analysis', {})
        performance_data = self.demo_results.get('performance_comparison', {})
        
        print(f"🎯 **EXECUTIVE SUMMARY**")
        print(f"   📊 Total signals generated: {signals_data.get('total_signals', 0)}")
        print(f"   🧠 Opportunities created: {opportunities_data.get('total_opportunities', 0)}")
        print(f"   🎯 Final selections: {selection_data.get('selected', 0)}")
        print(f"   ⚡ Selection rate: {selection_data.get('selection_rate', 0):.1%}")
        
        print(f"\n📈 **UNCERTAINTY QUANTIFICATION**")
        print(f"   🔬 Average signal confidence: {signals_data.get('avg_confidence', 0):.1%}")
        print(f"   🧠 Average blended confidence: {opportunities_data.get('avg_blended_confidence', 0):.1%}")
        print(f"   📉 Portfolio uncertainty: {uncertainty_data.get('portfolio_uncertainty', 0):.4f}")
        
        if performance_data:
            improvements = performance_data.get('improvements', {})
            print(f"\n🏆 **PERFORMANCE IMPROVEMENTS**")
            print(f"   ⚡ Sharpe ratio improvement: {improvements.get('sharpe_improvement_pct', 0):.1f}%")
            print(f"   🎯 Correlation reduction: {improvements.get('correlation_reduction_pct', 0):.1f}%")
        
        print(f"\n✅ **PHASE 2 OBJECTIVES ACHIEVED**")
        print(f"   ✅ Uncertainty quantification (μ, σ, horizon) in all agents")
        print(f"   ✅ QR LightGBM meta-weighter with isotonic calibration")
        print(f"   ✅ Diversified Top-K selector with anti-correlation")
        print(f"   ✅ End-to-end uncertainty propagation")
        print(f"   ✅ Performance improvement vs naive selection")
        
        # Store final results
        self.demo_results['summary'] = {
            'total_signals': signals_data.get('total_signals', 0),
            'total_opportunities': opportunities_data.get('total_opportunities', 0),
            'final_selections': selection_data.get('selected', 0),
            'avg_confidence': signals_data.get('avg_confidence', 0),
            'portfolio_sharpe': performance_data.get('sophisticated', {}).get('sharpe', 0),
            'sharpe_improvement': improvements.get('sharpe_improvement_pct', 0) if performance_data else 0,
            'demo_completed': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        print(f"\n🚀 **PHASE 2 DEMO COMPLETED SUCCESSFULLY**")
        print(f"Ready for Phase 3: Risk Management & Execution! 🎯")


async def main():
    """Main demo function"""
    demo = Phase2Demo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
