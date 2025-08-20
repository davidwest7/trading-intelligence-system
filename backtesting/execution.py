#!/usr/bin/env python3
"""
Execution Engine
===============

Realistic execution simulation for backtesting:
- Transaction costs and slippage
- Market impact modeling
- Partial fills and volume constraints
- Risk controls and position limits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    """Configuration for execution simulation"""
    # Transaction costs
    entry_bps: float = 1.5
    exit_bps: float = 1.5
    slippage_bps: float = 1.0
    
    # Market impact
    impact_model: str = "sqrt"  # sqrt, linear, square
    k_by_bucket: Dict[str, float] = None
    alpha: float = 0.5
    
    # Volume constraints
    max_participation: float = 0.1  # 10% of bar volume
    
    # Risk controls
    max_gross: float = 1.0
    max_per_name: float = 0.1
    dd_kill_switch: float = 0.15
    
    def __post_init__(self):
        if self.k_by_bucket is None:
            self.k_by_bucket = {
                'mega': 0.05,
                'large': 0.08,
                'mid': 0.12,
                'small': 0.20
            }

class ExecutionEngine:
    """
    Realistic execution simulation engine
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.position_history = []
        self.trade_history = []
        
        logger.info("🚀 Execution Engine initialized")
    
    def apply_costs(self, weights: pd.DataFrame, prices: pd.DataFrame, 
                   cost_bps: Optional[float] = None) -> pd.DataFrame:
        """
        Apply transaction costs to weights
        
        Args:
            weights: DataFrame with target weights (symbols as columns, timestamps as index)
            prices: DataFrame with prices (symbols as columns, timestamps as index)
            cost_bps: Cost in basis points (overrides config)
        
        Returns:
            DataFrame with costs applied
        """
        if weights.empty or prices.empty:
            return weights
        
        # Use config cost if not specified
        if cost_bps is None:
            cost_bps = self.config.entry_bps
        
        # Calculate costs
        cost_multiplier = 1 - (cost_bps / 10000)
        
        # Apply costs to weights
        adjusted_weights = weights * cost_multiplier
        
        logger.info(f"Applied {cost_bps} bps transaction costs")
        return adjusted_weights
    
    def apply_slippage(self, weights: pd.DataFrame, prices: pd.DataFrame,
                      slippage_bps: Optional[float] = None) -> pd.DataFrame:
        """
        Apply slippage to execution prices
        
        Args:
            weights: Target weights
            prices: Market prices
            slippage_bps: Slippage in basis points
        
        Returns:
            DataFrame with slippage-adjusted prices
        """
        if weights.empty or prices.empty:
            return prices
        
        if slippage_bps is None:
            slippage_bps = self.config.slippage_bps
        
        # Calculate slippage multiplier
        slippage_multiplier = 1 + (slippage_bps / 10000)
        
        # Apply slippage to prices (buy orders pay more, sell orders receive less)
        adjusted_prices = prices.copy()
        
        for symbol in weights.columns:
            if symbol in prices.columns:
                # Buy orders (positive weights) pay more
                buy_mask = weights[symbol] > 0
                adjusted_prices.loc[buy_mask, symbol] = prices.loc[buy_mask, symbol] * slippage_multiplier
                
                # Sell orders (negative weights) receive less
                sell_mask = weights[symbol] < 0
                adjusted_prices.loc[sell_mask, symbol] = prices.loc[sell_mask, symbol] / slippage_multiplier
        
        logger.info(f"Applied {slippage_bps} bps slippage")
        return adjusted_prices
    
    def calculate_market_impact(self, participation: pd.DataFrame, 
                              liquidity_bucket: str = 'large') -> pd.DataFrame:
        """
        Calculate market impact based on participation rate
        
        Args:
            participation: Participation rate as fraction of volume
            liquidity_bucket: Liquidity bucket (mega, large, mid, small)
        
        Returns:
            Impact multiplier
        """
        if participation.empty:
            return participation
        
        k = self.config.k_by_bucket.get(liquidity_bucket, 0.1)
        alpha = self.config.alpha
        
        # Calculate impact based on model
        if self.config.impact_model == "sqrt":
            impact = 1 + k * np.sign(participation) * np.sqrt(np.abs(participation))
        elif self.config.impact_model == "linear":
            impact = 1 + k * participation
        elif self.config.impact_model == "square":
            impact = 1 + k * np.sign(participation) * (participation ** 2)
        else:
            # Default to sqrt
            impact = 1 + k * np.sign(participation) * np.sqrt(np.abs(participation))
        
        logger.info(f"Calculated market impact using {self.config.impact_model} model")
        return impact
    
    def cap_by_volume(self, target_shares: pd.DataFrame, bar_volume: pd.DataFrame,
                     max_participation: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cap position sizes by volume constraints
        
        Args:
            target_shares: Target shares to trade
            bar_volume: Bar volume for each symbol
            max_participation: Maximum participation rate
        
        Returns:
            Tuple of (filled_shares, residual_shares)
        """
        if target_shares.empty or bar_volume.empty:
            return target_shares, pd.DataFrame()
        
        if max_participation is None:
            max_participation = self.config.max_participation
        
        # Calculate volume cap
        volume_cap = bar_volume * max_participation
        
        # Cap shares by volume
        filled_shares = target_shares.copy()
        residual_shares = target_shares.copy()
        
        for symbol in target_shares.columns:
            if symbol in bar_volume.columns:
                # Cap positive positions
                positive_mask = target_shares[symbol] > 0
                filled_shares.loc[positive_mask, symbol] = np.minimum(
                    target_shares.loc[positive_mask, symbol],
                    volume_cap.loc[positive_mask, symbol]
                )
                
                # Cap negative positions
                negative_mask = target_shares[symbol] < 0
                filled_shares.loc[negative_mask, symbol] = np.maximum(
                    target_shares.loc[negative_mask, symbol],
                    -volume_cap.loc[negative_mask, symbol]
                )
                
                # Calculate residuals
                residual_shares[symbol] = target_shares[symbol] - filled_shares[symbol]
        
        logger.info(f"Capped positions by {max_participation*100:.1f}% volume participation")
        return filled_shares, residual_shares
    
    def apply_risk_controls(self, weights: pd.DataFrame, 
                          current_positions: Dict[str, float],
                          portfolio_value: float) -> pd.DataFrame:
        """
        Apply risk controls to weights
        
        Args:
            weights: Target weights
            current_positions: Current positions
            portfolio_value: Current portfolio value
        
        Returns:
            Risk-adjusted weights
        """
        if weights.empty:
            return weights
        
        adjusted_weights = weights.copy()
        
        # Check gross exposure
        gross_exposure = weights.abs().sum(axis=1)
        if (gross_exposure > self.config.max_gross).any():
            logger.warning(f"Gross exposure exceeds {self.config.max_gross*100:.1f}% limit")
            # Scale down weights
            scale_factor = self.config.max_gross / gross_exposure
            adjusted_weights = adjusted_weights.multiply(scale_factor, axis=0)
        
        # Check per-name limits
        for symbol in weights.columns:
            if symbol in weights.columns:
                max_weight = self.config.max_per_name
                if (weights[symbol].abs() > max_weight).any():
                    logger.warning(f"Position in {symbol} exceeds {max_weight*100:.1f}% limit")
                    # Cap individual positions
                    adjusted_weights[symbol] = weights[symbol].clip(-max_weight, max_weight)
        
        logger.info("Applied risk controls")
        return adjusted_weights
    
    def execute_trades(self, target_weights: pd.DataFrame, 
                      current_weights: pd.DataFrame,
                      prices: pd.DataFrame,
                      volumes: pd.DataFrame,
                      portfolio_value: float) -> Dict[str, Any]:
        """
        Execute trades with realistic constraints
        
        Args:
            target_weights: Target portfolio weights
            current_weights: Current portfolio weights
            prices: Market prices
            volumes: Bar volumes
            portfolio_value: Current portfolio value
        
        Returns:
            Execution results
        """
        if target_weights.empty or current_weights.empty:
            return {
                'executed_weights': pd.DataFrame(),
                'trades': pd.DataFrame(),
                'costs': 0.0,
                'slippage': 0.0,
                'impact': 0.0
            }
        
        # Calculate target shares
        target_shares = (target_weights * portfolio_value) / prices
        
        # Apply risk controls
        target_shares = self.apply_risk_controls(target_shares, {}, portfolio_value)
        
        # Calculate trade sizes
        current_shares = (current_weights * portfolio_value) / prices
        trade_shares = target_shares - current_shares
        
        # Cap by volume
        filled_shares, residual_shares = self.cap_by_volume(
            trade_shares, volumes, self.config.max_participation
        )
        
        # Calculate participation rate
        participation = filled_shares / volumes
        participation = participation.fillna(0)
        
        # Calculate market impact
        impact_multiplier = self.calculate_market_impact(participation)
        
        # Apply impact to prices
        execution_prices = prices * impact_multiplier
        
        # Apply slippage
        execution_prices = self.apply_slippage(filled_shares, execution_prices)
        
        # Calculate trade values
        trade_values = filled_shares * execution_prices
        
        # Calculate costs
        total_costs = self._calculate_total_costs(trade_values, prices)
        
        # Create trade records
        trades = self._create_trade_records(
            filled_shares, execution_prices, trade_values, 
            target_weights.index, target_weights.columns
        )
        
        # Calculate new weights
        new_portfolio_value = portfolio_value - total_costs
        executed_weights = (filled_shares * execution_prices) / new_portfolio_value
        
        results = {
            'executed_weights': executed_weights,
            'trades': trades,
            'costs': total_costs,
            'slippage': self._calculate_slippage_cost(trade_values, prices),
            'impact': self._calculate_impact_cost(trade_values, prices, impact_multiplier),
            'residual_shares': residual_shares
        }
        
        logger.info(f"Executed trades with ${total_costs:.2f} total costs")
        return results
    
    def _calculate_total_costs(self, trade_values: pd.DataFrame, 
                             market_prices: pd.DataFrame) -> float:
        """Calculate total transaction costs"""
        if trade_values.empty:
            return 0.0
        
        # Calculate costs based on trade direction
        buy_costs = trade_values[trade_values > 0].sum().sum() * (self.config.entry_bps / 10000)
        sell_costs = trade_values[trade_values < 0].abs().sum().sum() * (self.config.exit_bps / 10000)
        
        return buy_costs + sell_costs
    
    def _calculate_slippage_cost(self, trade_values: pd.DataFrame, 
                               market_prices: pd.DataFrame) -> float:
        """Calculate slippage costs"""
        if trade_values.empty:
            return 0.0
        
        # Simplified slippage calculation
        return trade_values.abs().sum().sum() * (self.config.slippage_bps / 10000)
    
    def _calculate_impact_cost(self, trade_values: pd.DataFrame, 
                             market_prices: pd.DataFrame,
                             impact_multiplier: pd.DataFrame) -> float:
        """Calculate market impact costs"""
        if trade_values.empty:
            return 0.0
        
        # Calculate impact cost as difference from market prices
        impact_cost = 0.0
        for symbol in trade_values.columns:
            if symbol in market_prices.columns:
                symbol_trades = trade_values[symbol]
                symbol_impact = impact_multiplier[symbol]
                symbol_prices = market_prices[symbol]
                
                # Impact cost = trade_value * (impact_multiplier - 1)
                impact_cost += (symbol_trades * symbol_prices * (symbol_impact - 1)).sum()
        
        return impact_cost
    
    def _create_trade_records(self, shares: pd.DataFrame, prices: pd.DataFrame,
                            values: pd.DataFrame, timestamps: pd.DatetimeIndex,
                            symbols: List[str]) -> pd.DataFrame:
        """Create trade records for analysis"""
        trades = []
        
        for timestamp in timestamps:
            for symbol in symbols:
                if symbol in shares.columns:
                    share_amount = shares.loc[timestamp, symbol]
                    if share_amount != 0:
                        trade = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'shares': share_amount,
                            'price': prices.loc[timestamp, symbol],
                            'value': values.loc[timestamp, symbol],
                            'side': 'buy' if share_amount > 0 else 'sell'
                        }
                        trades.append(trade)
        
        return pd.DataFrame(trades)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics"""
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        summary = {
            'total_trades': len(trades_df),
            'total_volume': trades_df['value'].abs().sum(),
            'avg_trade_size': trades_df['value'].abs().mean(),
            'buy_volume': trades_df[trades_df['side'] == 'buy']['value'].sum(),
            'sell_volume': trades_df[trades_df['side'] == 'sell']['value'].abs().sum(),
            'avg_slippage': trades_df.get('slippage', 0).mean(),
            'avg_impact': trades_df.get('impact', 0).mean()
        }
        
        return summary
