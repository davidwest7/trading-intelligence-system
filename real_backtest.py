#!/usr/bin/env python3
"""
Real Backtest with Polygon Data
===============================

This script runs a comprehensive backtest using real market data from Polygon.io
with realistic execution simulation and professional performance analysis.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the backtesting module to path
sys.path.append(str(Path(__file__).parent))

from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.execution import ExecutionConfig
from backtesting.metrics import BacktestMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_polygon_api_key():
    """Check if Polygon API key is available"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("❌ POLYGON_API_KEY environment variable not set!")
        logger.info("Please set your Polygon API key:")
        logger.info("export POLYGON_API_KEY='your_api_key_here'")
        return False
    
    logger.info(f"✅ Polygon API key found: {api_key[:8]}...")
    return True

def momentum_strategy(data, current_date, current_prices, lookback_days=20):
    """
    Momentum strategy - buy stocks with positive momentum
    
    Args:
        data: Historical price data for all symbols
        current_date: Current trading date
        current_prices: Current prices for all symbols
        lookback_days: Number of days to calculate momentum
    
    Returns:
        dict: Target weights for each symbol
    """
    try:
        # Calculate momentum for each symbol
        weights = {}
        valid_symbols = []
        
        for symbol in data.keys():
            if symbol in current_prices and symbol in data:
                symbol_data = data[symbol]
                
                # Get historical prices for momentum calculation
                if len(symbol_data) >= lookback_days:
                    # Calculate momentum as price change over lookback period
                    start_price = symbol_data.iloc[-lookback_days]['close']
                    end_price = current_prices[symbol]
                    momentum = (end_price - start_price) / start_price
                    
                    if momentum > 0:  # Only positive momentum
                        valid_symbols.append((symbol, momentum))
        
        # Sort by momentum and select top performers
        valid_symbols.sort(key=lambda x: x[1], reverse=True)
        
        # Equal weight among top momentum stocks (max 5 stocks)
        top_stocks = valid_symbols[:5]
        if top_stocks:
            weight_per_stock = 1.0 / len(top_stocks)
            for symbol, _ in top_stocks:
                weights[symbol] = weight_per_stock
        
        logger.info(f"📈 Momentum strategy: {len(weights)} stocks selected")
        return weights
        
    except Exception as e:
        logger.error(f"❌ Error in momentum strategy: {e}")
        return {}

def mean_reversion_strategy(data, current_date, current_prices, lookback_days=60):
    """
    Mean reversion strategy - buy stocks below moving average
    
    Args:
        data: Historical price data for all symbols
        current_date: Current trading date
        current_prices: Current prices for all symbols
        lookback_days: Number of days for moving average
    
    Returns:
        dict: Target weights for each symbol
    """
    try:
        weights = {}
        valid_symbols = []
        
        for symbol in data.keys():
            if symbol in current_prices and symbol in data:
                symbol_data = data[symbol]
                
                if len(symbol_data) >= lookback_days:
                    # Calculate moving average
                    ma = symbol_data['close'].rolling(window=lookback_days).mean().iloc[-1]
                    current_price = current_prices[symbol]
                    
                    # Calculate z-score (how far below mean)
                    if ma > 0:
                        z_score = (current_price - ma) / ma
                        
                        # Buy if significantly below mean (negative z-score)
                        if z_score < -0.1:  # 10% below moving average
                            valid_symbols.append((symbol, abs(z_score)))
        
        # Sort by z-score and select top mean reversion candidates
        valid_symbols.sort(key=lambda x: x[1], reverse=True)
        
        # Equal weight among selected stocks (max 5 stocks)
        top_stocks = valid_symbols[:5]
        if top_stocks:
            weight_per_stock = 1.0 / len(top_stocks)
            for symbol, _ in top_stocks:
                weights[symbol] = weight_per_stock
        
        logger.info(f"📉 Mean reversion strategy: {len(weights)} stocks selected")
        return weights
        
    except Exception as e:
        logger.error(f"❌ Error in mean reversion strategy: {e}")
        return {}

def value_strategy(data, current_date, current_prices, lookback_days=252):
    """
    Value strategy - buy stocks with low volatility and consistent performance
    
    Args:
        data: Historical price data for all symbols
        current_date: Current trading date
        current_prices: Current prices for all symbols
        lookback_days: Number of days for analysis
    
    Returns:
        dict: Target weights for each symbol
    """
    try:
        weights = {}
        valid_symbols = []
        
        for symbol in data.keys():
            if symbol in current_prices and symbol in data:
                symbol_data = data[symbol]
                
                if len(symbol_data) >= lookback_days:
                    # Calculate volatility
                    returns = symbol_data['close'].pct_change().dropna()
                    volatility = returns.std()
                    
                    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
                    avg_return = returns.mean()
                    sharpe = (avg_return - 0.02/252) / volatility if volatility > 0 else 0
                    
                    # Select stocks with low volatility and positive Sharpe
                    if volatility < 0.02 and sharpe > 0.5:  # 2% daily vol, Sharpe > 0.5
                        valid_symbols.append((symbol, sharpe))
        
        # Sort by Sharpe ratio
        valid_symbols.sort(key=lambda x: x[1], reverse=True)
        
        # Equal weight among selected stocks (max 5 stocks)
        top_stocks = valid_symbols[:5]
        if top_stocks:
            weight_per_stock = 1.0 / len(top_stocks)
            for symbol, _ in top_stocks:
                weights[symbol] = weight_per_stock
        
        logger.info(f"💰 Value strategy: {len(weights)} stocks selected")
        return weights
        
    except Exception as e:
        logger.error(f"❌ Error in value strategy: {e}")
        return {}

def run_real_backtest():
    """Run comprehensive backtest with real Polygon data"""
    
    logger.info("🚀 Starting Real Backtest with Polygon Data")
    logger.info("=" * 60)
    
    # Check API key
    if not check_polygon_api_key():
        return
    
    # Configuration for real backtest
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "VTI", "VOO", "NVDA"]
    start_date = "2021-01-01"  # 3+ years of data
    end_date = "2024-12-31"
    initial_capital = 1000000.0  # $1M starting capital
    
    # Execution configuration for realistic trading
    execution_config = ExecutionConfig(
        entry_bps=2.0,           # 2 bps entry costs
        exit_bps=2.0,            # 2 bps exit costs
        slippage_bps=1.5,        # 1.5 bps slippage
        impact_model="sqrt",     # Square root market impact
        max_participation=0.05,  # 5% volume participation
        max_gross=1.0,           # 100% gross exposure
        max_per_name=0.15        # 15% per name limit
    )
    
    # Strategy configurations
    strategies = {
        "Momentum": momentum_strategy,
        "Mean_Reversion": mean_reversion_strategy,
        "Value": value_strategy
    }
    
    results = {}
    
    # Run backtest for each strategy
    for strategy_name, strategy_func in strategies.items():
        logger.info(f"🎯 Running {strategy_name} strategy...")
        
        try:
            # Create backtest configuration
            config = BacktestConfig(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d",
                initial_capital=initial_capital,
                rebalance_frequency="1w",  # Weekly rebalancing
                execution_config=execution_config,
                polygon_api_key=os.getenv("POLYGON_API_KEY"),
                strategy_function=strategy_func,
                max_drawdown=0.20,  # 20% max drawdown limit
                local_path="real_data"  # Store real data locally
            )
            
            # Initialize and run backtest
            engine = BacktestEngine(config)
            result = engine.run_backtest(max_retries=3)
            
            if result:
                results[strategy_name] = result
                logger.info(f"✅ {strategy_name} backtest completed successfully")
                
                # Display key metrics
                metrics = result.get('metrics', {})
                logger.info(f"📊 {strategy_name} Results:")
                logger.info(f"   Total Return: {metrics.get('total_return', 0):.2%}")
                logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                logger.info(f"   Volatility: {metrics.get('volatility', 0):.2%}")
                logger.info(f"   Total Trades: {metrics.get('total_trades', 0)}")
                logger.info(f"   Total Costs: ${metrics.get('total_costs', 0):,.2f}")
                
            else:
                logger.error(f"❌ {strategy_name} backtest failed")
                
        except Exception as e:
            logger.error(f"❌ Error running {strategy_name} backtest: {e}")
            continue
    
    # Generate strategy comparison
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("🏆 STRATEGY COMPARISON")
        logger.info("=" * 60)
        
        comparison_data = []
        for strategy_name, result in results.items():
            metrics = result.get('metrics', {})
            comparison_data.append({
                'strategy': strategy_name,
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'volatility': metrics.get('volatility', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_trades': metrics.get('total_trades', 0),
                'total_costs': metrics.get('total_costs', 0)
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best strategy by Sharpe ratio
        if not comparison_df.empty:
            best_strategy = comparison_df.loc[comparison_df['sharpe_ratio'].idxmax()]
            logger.info(f"🏆 Best Strategy: {best_strategy['strategy']}")
            logger.info(f"   Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
            logger.info(f"   Total Return: {best_strategy['total_return']:.2%}")
            logger.info(f"   Max Drawdown: {best_strategy['max_drawdown']:.2%}")
        
        # Save results
        output_dir = Path("real_backtest_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save comparison
        comparison_df.to_csv(output_dir / "strategy_comparison.csv", index=False)
        
        # Save detailed results
        for strategy_name, result in results.items():
            result_file = output_dir / f"{strategy_name.lower()}_results.json"
            import json
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        logger.info(f"\n📁 Results saved to: {output_dir}")
        logger.info("✅ Real backtest completed successfully!")
        
    else:
        logger.error("❌ No successful backtest results to compare")

if __name__ == "__main__":
    run_real_backtest()
