#!/usr/bin/env python3
"""
Backtesting System Demo
=======================

This script demonstrates the complete backtesting system with:
- Realistic market data simulation
- Multiple strategy implementations
- Comprehensive performance analysis
- Error handling and retry mechanisms
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')

# Import backtesting components
from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.execution import ExecutionConfig
from backtesting.metrics import BacktestMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_realistic_market_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate realistic market data for demonstration
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with realistic price data
    """
    logger.info("📊 Generating realistic market data...")
    
    # Create date range
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start_dt, end_dt, freq='D')
    
    # Generate realistic price data for each symbol
    price_data = pd.DataFrame(index=dates)
    
    for symbol in symbols:
        # Start with realistic base prices
        if symbol == 'SPY':
            base_price = 400
        elif symbol == 'QQQ':
            base_price = 350
        elif symbol in ['AAPL', 'MSFT']:
            base_price = 150
        elif symbol in ['GOOGL', 'AMZN']:
            base_price = 2500
        else:
            base_price = 100
        
        # Generate price series with realistic characteristics
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        
        # Daily returns with realistic volatility
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
        
        # Add some trend and mean reversion
        trend = np.linspace(0, 0.1, len(dates))  # 10% trend over period
        daily_returns += trend / len(dates)
        
        # Add some correlation with market (SPY)
        if symbol != 'SPY':
            market_returns = price_data.get('SPY', daily_returns)
            correlation = 0.6  # 60% correlation with market
            daily_returns = correlation * market_returns + np.sqrt(1 - correlation**2) * daily_returns
        
        # Convert to prices
        prices = base_price * np.exp(np.cumsum(daily_returns))
        price_data[symbol] = prices
    
    logger.info(f"✅ Generated {len(price_data)} days of data for {len(symbols)} symbols")
    return price_data

def momentum_strategy(data: pd.DataFrame, current_date: datetime, 
                     current_prices: pd.Series, lookback_days: int = 20) -> pd.Series:
    """
    Momentum strategy - buy stocks with positive momentum
    
    Args:
        data: Historical price data
        current_date: Current date
        current_prices: Current prices
        lookback_days: Lookback period for momentum calculation
    
    Returns:
        Target weights
    """
    try:
        # Get historical data up to current date
        historical_data = data[data.index <= current_date]
        
        if len(historical_data) < lookback_days:
            # Not enough data, return equal weights
            return pd.Series(1.0 / len(current_prices), index=current_prices.index)
        
        # Calculate momentum (price change over lookback period)
        momentum = {}
        for symbol in current_prices.index:
            if symbol in historical_data.columns:
                symbol_data = historical_data[symbol].dropna()
                if len(symbol_data) >= lookback_days:
                    # Calculate momentum as percentage change
                    start_price = symbol_data.iloc[-lookback_days]
                    end_price = symbol_data.iloc[-1]
                    momentum[symbol] = (end_price - start_price) / start_price
                else:
                    momentum[symbol] = 0.0
            else:
                momentum[symbol] = 0.0
        
        momentum_series = pd.Series(momentum)
        
        # Select top 3 momentum stocks
        top_momentum = momentum_series.nlargest(3)
        
        # Create weights (equal weight among top momentum stocks)
        weights = pd.Series(0.0, index=current_prices.index)
        if len(top_momentum) > 0:
            weight_per_stock = 1.0 / len(top_momentum)
            for symbol in top_momentum.index:
                weights[symbol] = weight_per_stock
        
        return weights
        
    except Exception as e:
        logger.error(f"Error in momentum strategy: {e}")
        # Return equal weights as fallback
        return pd.Series(1.0 / len(current_prices), index=current_prices.index)

def mean_reversion_strategy(data: pd.DataFrame, current_date: datetime,
                           current_prices: pd.Series, lookback_days: int = 60) -> pd.Series:
    """
    Mean reversion strategy - buy stocks below their moving average
    
    Args:
        data: Historical price data
        current_date: Current date
        current_prices: Current prices
        lookback_days: Lookback period for mean calculation
    
    Returns:
        Target weights
    """
    try:
        # Get historical data
        historical_data = data[data.index <= current_date]
        
        if len(historical_data) < lookback_days:
            return pd.Series(1.0 / len(current_prices), index=current_prices.index)
        
        # Calculate mean reversion signals
        signals = {}
        for symbol in current_prices.index:
            if symbol in historical_data.columns:
                symbol_data = historical_data[symbol].dropna()
                if len(symbol_data) >= lookback_days:
                    # Calculate z-score relative to moving average
                    moving_avg = symbol_data.iloc[-lookback_days:].mean()
                    moving_std = symbol_data.iloc[-lookback_days:].std()
                    current_price = symbol_data.iloc[-1]
                    
                    if moving_std > 0:
                        z_score = (current_price - moving_avg) / moving_std
                        # Negative z-score means price is below mean (buy signal)
                        signals[symbol] = -z_score
                    else:
                        signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.0
            else:
                signals[symbol] = 0.0
        
        signals_series = pd.Series(signals)
        
        # Select stocks with negative z-score (below mean)
        buy_signals = signals_series[signals_series < -0.5]  # Z-score < -0.5
        
        # Create weights
        weights = pd.Series(0.0, index=current_prices.index)
        if len(buy_signals) > 0:
            weight_per_stock = 1.0 / len(buy_signals)
            for symbol in buy_signals.index:
                weights[symbol] = weight_per_stock
        
        return weights
        
    except Exception as e:
        logger.error(f"Error in mean reversion strategy: {e}")
        return pd.Series(1.0 / len(current_prices), index=current_prices.index)

def equal_weight_strategy(data: pd.DataFrame, current_date: datetime,
                         current_prices: pd.Series, **kwargs) -> pd.Series:
    """
    Equal weight strategy (baseline)
    """
    return pd.Series(1.0 / len(current_prices), index=current_prices.index)

def run_strategy_backtest(strategy_name: str, strategy_func, data: pd.DataFrame) -> dict:
    """
    Run backtest for a specific strategy
    
    Args:
        strategy_name: Name of the strategy
        strategy_func: Strategy function
        data: Market data
    
    Returns:
        Backtest results
    """
    logger.info(f"🚀 Running {strategy_name} strategy backtest...")
    
    # Create execution configuration
    execution_config = ExecutionConfig(
        entry_bps=1.5,
        exit_bps=1.5,
        slippage_bps=1.0,
        impact_model="sqrt",
        max_participation=0.1,
        max_gross=1.0,
        max_per_name=0.2
    )
    
    # Create backtest configuration
    config = BacktestConfig(
        symbols=list(data.columns),
        start_date=data.index[0].strftime('%Y-%m-%d'),
        end_date=data.index[-1].strftime('%Y-%m-%d'),
        timeframe="1d",
        initial_capital=1000000.0,
        rebalance_frequency="1w",  # Weekly rebalancing
        execution_config=execution_config,
        polygon_api_key=None,  # Use generated data
        strategy_function=strategy_func,
        strategy_params={"lookback_days": 20},
        max_drawdown=0.15,
        benchmark_symbol="SPY"
    )
    
    # Initialize and run backtest
    engine = BacktestEngine(config)
    
    # Mock the data loading to use our generated data
    engine._load_data = lambda: data
    
    # Run backtest
    results = engine.run_backtest(max_retries=1)
    
    logger.info(f"✅ {strategy_name} backtest completed")
    return results

def analyze_and_report_results(results: dict, strategy_name: str):
    """
    Analyze results and generate comprehensive report
    
    Args:
        results: Backtest results
        strategy_name: Strategy name
    """
    logger.info(f"📊 Analyzing {strategy_name} results...")
    
    # Extract key metrics
    performance_metrics = results.get('performance_metrics', {})
    execution_metrics = results.get('execution_metrics', {})
    
    # Create summary
    summary = {
        'strategy': strategy_name,
        'total_return': performance_metrics.get('total_return', 0),
        'annualized_return': performance_metrics.get('annualized_return', 0),
        'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
        'max_drawdown': performance_metrics.get('max_drawdown', 0),
        'volatility': performance_metrics.get('volatility', 0),
        'win_rate': performance_metrics.get('win_rate', 0),
        'profit_factor': performance_metrics.get('profit_factor', 0),
        'total_trades': execution_metrics.get('total_trades', 0),
        'total_costs': execution_metrics.get('total_volume', 0) * 0.0015
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name.upper()}")
    print(f"{'='*60}")
    print(f"Total Return: {summary['total_return']:.2%}")
    print(f"Annualized Return: {summary['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {summary['max_drawdown']:.2%}")
    print(f"Volatility: {summary['volatility']:.2%}")
    print(f"Win Rate: {summary['win_rate']:.2%}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Total Costs: ${summary['total_costs']:,.2f}")
    
    return summary

def main():
    """
    Main demo function
    """
    logger.info("🚀 Starting Backtesting System Demo")
    
    # Configuration
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    
    # Generate realistic market data
    market_data = generate_realistic_market_data(symbols, start_date, end_date)
    
    # Define strategies
    strategies = {
        "Momentum": momentum_strategy,
        "Mean Reversion": mean_reversion_strategy,
        "Equal Weight": equal_weight_strategy
    }
    
    # Run backtests
    all_results = {}
    all_summaries = []
    
    for strategy_name, strategy_func in strategies.items():
        try:
            # Run backtest
            results = run_strategy_backtest(strategy_name, strategy_func, market_data)
            
            # Analyze results
            summary = analyze_and_report_results(results, strategy_name)
            
            # Store results
            all_results[strategy_name] = results
            all_summaries.append(summary)
            
        except Exception as e:
            logger.error(f"❌ Failed to run {strategy_name} strategy: {e}")
    
    # Generate comparison report
    if all_summaries:
        print(f"\n{'='*80}")
        print("STRATEGY COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        comparison_df = pd.DataFrame(all_summaries)
        comparison_df = comparison_df.set_index('strategy')
        
        # Display comparison
        print(comparison_df.round(4))
        
        # Find best performing strategy
        best_strategy = comparison_df['sharpe_ratio'].idxmax()
        print(f"\n🏆 Best Strategy (by Sharpe Ratio): {best_strategy}")
        print(f"   Sharpe Ratio: {comparison_df.loc[best_strategy, 'sharpe_ratio']:.2f}")
        print(f"   Total Return: {comparison_df.loc[best_strategy, 'total_return']:.2%}")
        
        # Save results
        results_dir = Path("demo_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison
        comparison_df.to_csv(results_dir / "strategy_comparison.csv")
        
        # Save detailed results
        for strategy_name, results in all_results.items():
            results_file = results_dir / f"{strategy_name.lower().replace(' ', '_')}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to {results_dir}")
    
    logger.info("🎉 Backtesting System Demo completed!")

if __name__ == "__main__":
    main()
