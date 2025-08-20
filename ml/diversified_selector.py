"""
Diversified Top-K Selector with Submodular Greedy Selection

Implements diversified slate bandits with:
- Submodular greedy selection for anti-correlation
- Rolling correlation penalty on agent-blended alphas
- Mean-variance utility optimization
- CVaR-aware selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import deque

from schemas.contracts import Opportunity, Intent, DirectionType


logger = logging.getLogger(__name__)


@dataclass
class SelectionMetrics:
    """Metrics for selection performance"""
    opportunities_considered: int
    opportunities_selected: int
    avg_correlation: float
    avg_expected_return: float
    avg_uncertainty: float
    portfolio_sharpe: float
    diversification_ratio: float
    timestamp: datetime


@dataclass
class CorrelationData:
    """Correlation data for diversification"""
    symbol_pair: Tuple[str, str]
    correlation: float
    window_size: int
    last_updated: datetime


class DiversifiedTopKSelector:
    """
    Diversified Top-K Selector using Submodular Greedy Selection
    
    Features:
    - Anti-correlation by design through submodular optimization
    - Rolling correlation penalty on agent-blended alphas
    - Mean-variance utility with CVaR constraints
    - Thompson Sampling on blended posteriors
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Selection parameters
        self.k = config.get('top_k', 10)
        self.correlation_penalty = config.get('correlation_penalty', 0.1)
        self.min_expected_return = config.get('min_expected_return', 0.005)
        self.max_correlation = config.get('max_correlation', 0.7)
        self.risk_aversion = config.get('risk_aversion', 2.0)
        
        # Correlation tracking
        self.correlation_window = config.get('correlation_window', 252)  # ~1 year
        self.min_correlation_samples = config.get('min_correlation_samples', 30)
        self.correlation_data: Dict[Tuple[str, str], CorrelationData] = {}
        self.returns_history: Dict[str, deque] = {}
        
        # CVaR parameters
        self.cvar_alpha = config.get('cvar_alpha', 0.95)
        self.max_portfolio_cvar = config.get('max_portfolio_cvar', 0.05)
        
        # Thompson Sampling parameters
        self.use_thompson_sampling = config.get('use_thompson_sampling', True)
        self.exploration_factor = config.get('exploration_factor', 1.0)
        
        # Performance tracking
        self.selection_history: List[SelectionMetrics] = []
        self.selected_count = 0
        self.avg_portfolio_return = 0.0
        self.avg_portfolio_risk = 0.0
        
        # Diversification constraints
        self.max_sector_concentration = config.get('max_sector_concentration', 0.3)
        self.max_single_position = config.get('max_single_position', 0.15)
        
    async def select_opportunities(self, opportunities: List[Opportunity],
                                 portfolio_state: Optional[Dict[str, Any]] = None,
                                 trace_id: Optional[str] = None) -> List[Opportunity]:
        """
        Select diversified opportunities using submodular greedy selection
        
        Args:
            opportunities: List of opportunities to select from
            portfolio_state: Current portfolio state for context
            trace_id: Optional trace ID for tracking
            
        Returns:
            List of selected opportunities (up to k)
        """
        if not opportunities:
            return []
        
        try:
            # Filter opportunities by basic criteria
            filtered_opportunities = self._filter_opportunities(opportunities)
            
            if not filtered_opportunities:
                logger.warning("No opportunities passed initial filtering")
                return []
            
            # Update correlation data
            await self._update_correlations(filtered_opportunities)
            
            # Perform submodular greedy selection
            selected_opportunities = await self._submodular_greedy_selection(
                filtered_opportunities, portfolio_state
            )
            
            # Apply final constraints
            final_opportunities = self._apply_final_constraints(
                selected_opportunities, portfolio_state
            )
            
            # Record metrics
            self._record_selection_metrics(
                filtered_opportunities, final_opportunities
            )
            
            self.selected_count += len(final_opportunities)
            
            logger.info(f"Selected {len(final_opportunities)} opportunities from "
                       f"{len(opportunities)} candidates")
            
            return final_opportunities
            
        except Exception as e:
            logger.error(f"Error selecting opportunities: {e}")
            return []
    
    def _filter_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """Apply basic filtering criteria"""
        filtered = []
        
        for opp in opportunities:
            # Minimum expected return
            if opp.mu_blended < self.min_expected_return:
                continue
            
            # Maximum uncertainty (risk filter)
            if opp.sigma_blended > 0.15:  # Max 15% uncertainty
                continue
            
            # Minimum confidence
            if opp.confidence_blended < 0.3:
                continue
            
            # Valid direction
            if opp.direction == DirectionType.NEUTRAL:
                continue
            
            filtered.append(opp)
        
        return filtered
    
    async def _update_correlations(self, opportunities: List[Opportunity]):
        """Update rolling correlations between opportunities"""
        try:
            # Update returns history
            for opp in opportunities:
                if opp.symbol not in self.returns_history:
                    self.returns_history[opp.symbol] = deque(maxlen=self.correlation_window)
                
                # Use blended mu as proxy for expected return
                self.returns_history[opp.symbol].append(opp.mu_blended)
            
            # Calculate pairwise correlations
            symbols = [opp.symbol for opp in opportunities]
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    await self._calculate_pairwise_correlation(symbol1, symbol2)
                    
        except Exception as e:
            logger.error(f"Error updating correlations: {e}")
    
    async def _calculate_pairwise_correlation(self, symbol1: str, symbol2: str):
        """Calculate correlation between two symbols"""
        try:
            if (symbol1 not in self.returns_history or 
                symbol2 not in self.returns_history):
                return
            
            returns1 = list(self.returns_history[symbol1])
            returns2 = list(self.returns_history[symbol2])
            
            if (len(returns1) < self.min_correlation_samples or 
                len(returns2) < self.min_correlation_samples):
                return
            
            # Align series length
            min_length = min(len(returns1), len(returns2))
            returns1 = returns1[-min_length:]
            returns2 = returns2[-min_length:]
            
            # Calculate correlation
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Handle NaN correlations
            if np.isnan(correlation):
                correlation = 0.0
            
            # Store correlation data
            symbol_pair = tuple(sorted([symbol1, symbol2]))
            self.correlation_data[symbol_pair] = CorrelationData(
                symbol_pair=symbol_pair,
                correlation=correlation,
                window_size=min_length,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation for {symbol1}-{symbol2}: {e}")
    
    async def _submodular_greedy_selection(self, opportunities: List[Opportunity],
                                         portfolio_state: Optional[Dict[str, Any]] = None) -> List[Opportunity]:
        """
        Perform submodular greedy selection for maximum diversification
        """
        if len(opportunities) <= self.k:
            return opportunities
        
        try:
            selected = []
            remaining = opportunities.copy()
            
            # Sort by initial utility (expected return / risk)
            remaining.sort(key=lambda x: x.mu_blended / x.sigma_blended, reverse=True)
            
            for _ in range(min(self.k, len(remaining))):
                if not remaining:
                    break
                
                best_opp = None
                best_utility = float('-inf')
                
                for opp in remaining:
                    # Calculate marginal utility of adding this opportunity
                    marginal_utility = self._calculate_marginal_utility(
                        opp, selected, portfolio_state
                    )
                    
                    if marginal_utility > best_utility:
                        best_utility = marginal_utility
                        best_opp = opp
                
                if best_opp:
                    selected.append(best_opp)
                    remaining.remove(best_opp)
                else:
                    break
            
            return selected
            
        except Exception as e:
            logger.error(f"Error in submodular greedy selection: {e}")
            return opportunities[:self.k]
    
    def _calculate_marginal_utility(self, candidate: Opportunity, 
                                  selected: List[Opportunity],
                                  portfolio_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate marginal utility of adding candidate to selected opportunities
        
        Utility = Expected Return - Risk Penalty - Correlation Penalty
        """
        try:
            # Base utility: risk-adjusted return
            base_utility = candidate.mu_blended - (self.risk_aversion * candidate.sigma_blended)
            
            # Correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(candidate, selected)
            
            # CVaR penalty
            cvar_penalty = self._calculate_cvar_penalty(candidate, selected)
            
            # Concentration penalty
            concentration_penalty = self._calculate_concentration_penalty(
                candidate, selected, portfolio_state
            )
            
            # Thompson Sampling exploration bonus
            exploration_bonus = 0.0
            if self.use_thompson_sampling:
                exploration_bonus = self._calculate_exploration_bonus(candidate)
            
            # Total marginal utility
            total_utility = (base_utility - 
                           correlation_penalty - 
                           cvar_penalty - 
                           concentration_penalty + 
                           exploration_bonus)
            
            return total_utility
            
        except Exception as e:
            logger.error(f"Error calculating marginal utility: {e}")
            return 0.0
    
    def _calculate_correlation_penalty(self, candidate: Opportunity, 
                                     selected: List[Opportunity]) -> float:
        """Calculate correlation penalty for candidate"""
        if not selected:
            return 0.0
        
        total_penalty = 0.0
        
        for selected_opp in selected:
            correlation = self._get_correlation(candidate.symbol, selected_opp.symbol)
            
            # Penalty increases with correlation
            if abs(correlation) > 0.3:  # Only penalize significant correlations
                penalty = self.correlation_penalty * (abs(correlation) ** 2)
                total_penalty += penalty
        
        return total_penalty
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if symbol1 == symbol2:
            return 1.0
        
        symbol_pair = tuple(sorted([symbol1, symbol2]))
        
        if symbol_pair in self.correlation_data:
            corr_data = self.correlation_data[symbol_pair]
            
            # Check if correlation is recent enough
            age = (datetime.utcnow() - corr_data.last_updated).total_seconds()
            if age < 3600:  # 1 hour
                return corr_data.correlation
        
        # Default correlation for unknown pairs
        return 0.0
    
    def _calculate_cvar_penalty(self, candidate: Opportunity, 
                              selected: List[Opportunity]) -> float:
        """Calculate CVaR penalty"""
        if not selected:
            return 0.0
        
        # Estimate portfolio CVaR with candidate added
        all_opportunities = selected + [candidate]
        
        # Simple portfolio CVaR approximation
        portfolio_cvar = self._estimate_portfolio_cvar(all_opportunities)
        
        if portfolio_cvar > self.max_portfolio_cvar:
            return (portfolio_cvar - self.max_portfolio_cvar) * 10.0  # Strong penalty
        
        return 0.0
    
    def _estimate_portfolio_cvar(self, opportunities: List[Opportunity]) -> float:
        """Estimate portfolio CVaR"""
        if not opportunities:
            return 0.0
        
        # Equal weight approximation
        weight = 1.0 / len(opportunities)
        
        # Portfolio expected return and risk
        portfolio_mu = sum(opp.mu_blended * weight for opp in opportunities)
        portfolio_sigma = np.sqrt(sum((opp.sigma_blended * weight) ** 2 for opp in opportunities))
        
        # CVaR approximation (assuming normal distribution)
        from scipy.stats import norm
        var_95 = portfolio_mu - norm.ppf(self.cvar_alpha) * portfolio_sigma
        cvar_95 = portfolio_mu - (norm.pdf(norm.ppf(self.cvar_alpha)) / (1 - self.cvar_alpha)) * portfolio_sigma
        
        return abs(cvar_95)
    
    def _calculate_concentration_penalty(self, candidate: Opportunity,
                                       selected: List[Opportunity],
                                       portfolio_state: Optional[Dict[str, Any]] = None) -> float:
        """Calculate concentration penalty"""
        if not portfolio_state:
            return 0.0
        
        # Check sector concentration
        current_positions = portfolio_state.get('positions', {})
        candidate_sector = self._get_sector(candidate.symbol)
        
        if candidate_sector:
            sector_exposure = 0.0
            total_portfolio_value = sum(pos.get('value', 0) for pos in current_positions.values())
            
            if total_portfolio_value > 0:
                for symbol, position in current_positions.items():
                    if self._get_sector(symbol) == candidate_sector:
                        sector_exposure += position.get('value', 0) / total_portfolio_value
                
                if sector_exposure > self.max_sector_concentration:
                    return (sector_exposure - self.max_sector_concentration) * 5.0
        
        return 0.0
    
    def _get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for symbol (simplified)"""
        # This would typically use a sector mapping service
        # For now, use simple heuristics
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META']:
            return 'Technology'
        elif symbol in ['JPM', 'BAC', 'GS']:
            return 'Financials'
        elif symbol in ['JNJ', 'PFE', 'UNH']:
            return 'Healthcare'
        else:
            return 'Other'
    
    def _calculate_exploration_bonus(self, candidate: Opportunity) -> float:
        """Calculate Thompson Sampling exploration bonus"""
        # Bonus based on uncertainty (higher uncertainty = more exploration value)
        uncertainty_bonus = candidate.sigma_blended * self.exploration_factor
        
        # Bonus for less frequently selected symbols
        symbol_frequency = self._get_symbol_frequency(candidate.symbol)
        frequency_bonus = max(0, (1.0 - symbol_frequency)) * 0.01
        
        return uncertainty_bonus + frequency_bonus
    
    def _get_symbol_frequency(self, symbol: str) -> float:
        """Get relative frequency of symbol selection"""
        if not self.selection_history:
            return 0.0
        
        # This would track selection frequency over time
        # For now, return a default
        return 0.5
    
    def _apply_final_constraints(self, opportunities: List[Opportunity],
                               portfolio_state: Optional[Dict[str, Any]] = None) -> List[Opportunity]:
        """Apply final constraints before selection"""
        if not opportunities:
            return []
        
        # Remove highly correlated pairs
        final_opportunities = []
        
        for opp in opportunities:
            should_add = True
            
            for existing_opp in final_opportunities:
                correlation = self._get_correlation(opp.symbol, existing_opp.symbol)
                
                if abs(correlation) > self.max_correlation:
                    # Keep the one with better risk-adjusted return
                    if (opp.mu_blended / opp.sigma_blended <= 
                        existing_opp.mu_blended / existing_opp.sigma_blended):
                        should_add = False
                        break
                    else:
                        # Remove existing and add new
                        final_opportunities.remove(existing_opp)
                        break
            
            if should_add:
                final_opportunities.append(opp)
        
        return final_opportunities
    
    def _record_selection_metrics(self, candidates: List[Opportunity], 
                                selected: List[Opportunity]):
        """Record selection performance metrics"""
        if not candidates or not selected:
            return
        
        try:
            # Calculate correlations among selected
            correlations = []
            for i, opp1 in enumerate(selected):
                for opp2 in selected[i+1:]:
                    corr = self._get_correlation(opp1.symbol, opp2.symbol)
                    correlations.append(abs(corr))
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            # Calculate portfolio metrics
            avg_return = np.mean([opp.mu_blended for opp in selected])
            avg_uncertainty = np.mean([opp.sigma_blended for opp in selected])
            portfolio_sharpe = avg_return / avg_uncertainty if avg_uncertainty > 0 else 0.0
            
            # Diversification ratio
            portfolio_risk = self._estimate_portfolio_risk(selected)
            avg_risk = avg_uncertainty
            diversification_ratio = avg_risk / portfolio_risk if portfolio_risk > 0 else 1.0
            
            metrics = SelectionMetrics(
                opportunities_considered=len(candidates),
                opportunities_selected=len(selected),
                avg_correlation=avg_correlation,
                avg_expected_return=avg_return,
                avg_uncertainty=avg_uncertainty,
                portfolio_sharpe=portfolio_sharpe,
                diversification_ratio=diversification_ratio,
                timestamp=datetime.utcnow()
            )
            
            self.selection_history.append(metrics)
            
            # Keep only recent history
            if len(self.selection_history) > 1000:
                self.selection_history = self.selection_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording selection metrics: {e}")
    
    def _estimate_portfolio_risk(self, opportunities: List[Opportunity]) -> float:
        """Estimate portfolio risk considering correlations"""
        if not opportunities:
            return 0.0
        
        n = len(opportunities)
        weight = 1.0 / n
        
        # Variance calculation with correlations
        portfolio_variance = 0.0
        
        for i, opp1 in enumerate(opportunities):
            for j, opp2 in enumerate(opportunities):
                if i == j:
                    correlation = 1.0
                else:
                    correlation = self._get_correlation(opp1.symbol, opp2.symbol)
                
                portfolio_variance += (weight * weight * 
                                     opp1.sigma_blended * opp2.sigma_blended * 
                                     correlation)
        
        return np.sqrt(portfolio_variance)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.selection_history:
            return {}
        
        recent_metrics = self.selection_history[-10:]  # Last 10 selections
        
        return {
            'selections_made': self.selected_count,
            'avg_opportunities_considered': np.mean([m.opportunities_considered for m in recent_metrics]),
            'avg_opportunities_selected': np.mean([m.opportunities_selected for m in recent_metrics]),
            'avg_correlation': np.mean([m.avg_correlation for m in recent_metrics]),
            'avg_expected_return': np.mean([m.avg_expected_return for m in recent_metrics]),
            'avg_portfolio_sharpe': np.mean([m.portfolio_sharpe for m in recent_metrics]),
            'avg_diversification_ratio': np.mean([m.diversification_ratio for m in recent_metrics]),
            'correlation_pairs_tracked': len(self.correlation_data),
            'symbols_tracked': len(self.returns_history),
        }
