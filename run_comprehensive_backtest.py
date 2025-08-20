#!/usr/bin/env python3
"""
Comprehensive Backtesting System Demo
====================================

This script demonstrates the complete backtesting system with:
- Polygon Pro data integration
- Realistic execution simulation
- Error handling and retries
- Performance analysis
- Results reporting
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def momentum_strategy(data: pd.DataFrame, current_date: datetime, 
                     current_prices: pd.Series, lookback_days: int = 20) -> pd.Series:
    """
    Simple momentum strategy for demonstration
    
    Args:
        data: Historical price data
        current_date: Current date
        current_prices: Current prices
        lookback_days: Lookback period for momentum calculation
    
    Returns:
        Target weights
    """
    try:
        # Calculate momentum (price change over lookback period)
        lookback_date = current_date - timedelta(days=lookback_days)
        
        # Get historical data up to current date
        historical_data = data[data.index <= current_date]
        
        if len(historical_data) < lookback_days:
            # Not enough data, return equal weights
            return pd.Series(1.0 / len(current_prices), index=current_prices.index)
        
        # Calculate momentum
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
        
        # Select top 5 momentum stocks
        top_momentum = momentum_series.nlargest(5)
        
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
    Simple mean reversion strategy
    
    Args:
        data: Historical price data
        current_date: Current date
        current_prices: Current prices
        lookback_days: Lookback period for mean calculation
    
    Returns:
        Target weights
    """
    try:
        # Calculate mean reversion signal
        lookback_date = current_date - timedelta(days=lookback_days)
        
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
                    # Calculate z-score
                    mean_price = symbol_data.iloc[-lookback_days:].mean()
                    std_price = symbol_data.iloc[-lookback_days:].std()
                    current_price = symbol_data.iloc[-1]
                    
                    if std_price > 0:
                        z_score = (current_price - mean_price) / std_price
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

def create_backtest_config(strategy_name: str = "momentum") -> BacktestConfig:
    """
    Create backtest configuration
    
    Args:
        strategy_name: Strategy to use (momentum, mean_reversion, equal_weight)
    
    Returns:
        BacktestConfig object
    """
    # Strategy mapping
    strategies = {
        "momentum": momentum_strategy,
        "mean_reversion": mean_reversion_strategy,
        "equal_weight": equal_weight_strategy
    }
    
    # Execution configuration
    execution_config = ExecutionConfig(
        entry_bps=1.5,
        exit_bps=1.5,
        slippage_bps=1.0,
        impact_model="sqrt",
        max_participation=0.1,
        max_gross=1.0,
        max_per_name=0.1
    )
    
    # Create backtest config
    config = BacktestConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "SPY", "QQQ"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        timeframe="1d",
        initial_capital=1000000.0,
        rebalance_frequency="1w",  # Weekly rebalancing
        execution_config=execution_config,
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
        s3_bucket=os.getenv("S3_BUCKET"),  # Optional
        strategy_function=strategies.get(strategy_name, equal_weight_strategy),
        strategy_params={"lookback_days": 20},
        max_drawdown=0.15,
        benchmark_symbol="SPY"
    )
    
    return config

def run_backtest_with_retry(config: BacktestConfig, max_retries: int = 3) -> dict:
    """
    Run backtest with comprehensive error handling and retries
    
    Args:
        config: Backtest configuration
        max_retries: Maximum number of retry attempts
    
    Returns:
        Backtest results
    """
    logger.info(f"🚀 Starting backtest with {max_retries} retry attempts")
    
    for attempt in range(max_retries):
        try:
            # Initialize backtest engine
            engine = BacktestEngine(config)
            
            # Run backtest
            results = engine.run_backtest(max_retries=2)  # Internal retries
            
            logger.info(f"✅ Backtest completed successfully on attempt {attempt + 1}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed on attempt {attempt + 1}/{max_retries}: {e}")
            
            if attempt == max_retries - 1:
                logger.error("❌ All retry attempts failed")
                raise
            
            # Wait before retry
            wait_time = 2 ** attempt
            logger.info(f"⏳ Waiting {wait_time}s before retry...")
            import time
            time.sleep(wait_time)
    
    raise RuntimeError("Backtest failed after all retry attempts")

def analyze_results(results: dict) -> dict:
    """
    Analyze backtest results and generate insights
    
    Args:
        results: Backtest results
    
    Returns:
        Analysis summary
    """
    try:
        # Extract key metrics
        performance_metrics = results.get('performance_metrics', {})
        execution_metrics = results.get('execution_metrics', {})
        summary = results.get('summary', {})
        
        # Create analysis
        analysis = {
            'performance_summary': {
                'total_return': performance_metrics.get('total_return', 0),
                'annualized_return': performance_metrics.get('annualized_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'volatility': performance_metrics.get('volatility', 0),
                'win_rate': performance_metrics.get('win_rate', 0),
                'profit_factor': performance_metrics.get('profit_factor', 0)
            },
            'execution_summary': {
                'total_trades': execution_metrics.get('total_trades', 0),
                'total_volume': execution_metrics.get('total_volume', 0),
                'avg_trade_size': execution_metrics.get('avg_trade_size', 0),
                'buy_volume': execution_metrics.get('buy_volume', 0),
                'sell_volume': execution_metrics.get('sell_volume', 0)
            },
            'risk_metrics': {
                'var_95': performance_metrics.get('var_95', 0),
                'cvar_95': performance_metrics.get('cvar_95', 0),
                'sortino_ratio': performance_metrics.get('sortino_ratio', 0),
                'calmar_ratio': performance_metrics.get('calmar_ratio', 0)
            },
            'trading_metrics': {
                'avg_win': performance_metrics.get('avg_win', 0),
                'avg_loss': performance_metrics.get('avg_loss', 0),
                'max_consecutive_wins': performance_metrics.get('max_consecutive_wins', 0),
                'max_consecutive_losses': performance_metrics.get('max_consecutive_losses', 0)
            }
        }
        
        # Add performance assessment
        analysis['assessment'] = assess_performance(analysis['performance_summary'])
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Error analyzing results: {e}")
        return {}

def assess_performance(performance: dict) -> str:
    """
    Assess performance and provide recommendations
    
    Args:
        performance: Performance metrics
    
    Returns:
        Assessment string
    """
    total_return = performance.get('total_return', 0)
    sharpe_ratio = performance.get('sharpe_ratio', 0)
    max_drawdown = performance.get('max_drawdown', 0)
    win_rate = performance.get('win_rate', 0)
    
    assessment = []
    
    # Return assessment
    if total_return > 0.2:
        assessment.append("Excellent total return (>20%)")
    elif total_return > 0.1:
        assessment.append("Good total return (10-20%)")
    elif total_return > 0:
        assessment.append("Positive but modest return")
    else:
        assessment.append("Negative return - strategy needs improvement")
    
    # Risk-adjusted return assessment
    if sharpe_ratio > 1.5:
        assessment.append("Excellent risk-adjusted returns (Sharpe > 1.5)")
    elif sharpe_ratio > 1.0:
        assessment.append("Good risk-adjusted returns (Sharpe > 1.0)")
    elif sharpe_ratio > 0.5:
        assessment.append("Acceptable risk-adjusted returns")
    else:
        assessment.append("Poor risk-adjusted returns")
    
    # Drawdown assessment
    if abs(max_drawdown) < 0.05:
        assessment.append("Very low drawdown (<5%)")
    elif abs(max_drawdown) < 0.10:
        assessment.append("Low drawdown (5-10%)")
    elif abs(max_drawdown) < 0.15:
        assessment.append("Moderate drawdown (10-15%)")
    else:
        assessment.append("High drawdown (>15%) - consider risk management")
    
    # Win rate assessment
    if win_rate > 0.6:
        assessment.append("High win rate (>60%)")
    elif win_rate > 0.5:
        assessment.append("Positive win rate (>50%)")
    else:
        assessment.append("Low win rate - consider strategy refinement")
    
    return "; ".join(assessment)

def save_results(results: dict, analysis: dict, strategy_name: str):
    """
    Save backtest results and analysis
    
    Args:
        results: Backtest results
        analysis: Analysis summary
        strategy_name: Strategy name
    """
    try:
        # Create results directory
        results_dir = Path("backtest_results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = results_dir / f"{strategy_name}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save analysis
        analysis_file = results_dir / f"{strategy_name}_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = results_dir / f"{strategy_name}_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"BACKTEST SUMMARY - {strategy_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in analysis['performance_summary'].items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\nEXECUTION METRICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in analysis['execution_summary'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            f.write(analysis['assessment'])
        
        logger.info(f"✅ Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")

def main():
    """
    Main function to run comprehensive backtesting
    """
    logger.info("🚀 Starting Comprehensive Backtesting System")
    
    # Strategies to test
    strategies = ["momentum", "mean_reversion", "equal_weight"]
    
    all_results = {}
    
    for strategy_name in strategies:
        try:
            logger.info(f"📊 Testing {strategy_name} strategy...")
            
            # Create configuration
            config = create_backtest_config(strategy_name)
            
            # Run backtest with retries
            results = run_backtest_with_retry(config, max_retries=3)
            
            # Analyze results
            analysis = analyze_results(results)
            
            # Save results
            save_results(results, analysis, strategy_name)
            
            # Store for comparison
            all_results[strategy_name] = {
                'results': results,
                'analysis': analysis
            }
            
            logger.info(f"✅ {strategy_name} strategy completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to test {strategy_name} strategy: {e}")
            all_results[strategy_name] = {'error': str(e)}
    
    # Generate comparison report
    try:
        comparison_file = Path("backtest_results") / f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(comparison_file, 'w') as f:
            f.write("STRATEGY COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for strategy_name, data in all_results.items():
                f.write(f"{strategy_name.upper()} STRATEGY:\n")
                f.write("-" * 30 + "\n")
                
                if 'error' in data:
                    f.write(f"ERROR: {data['error']}\n\n")
                else:
                    performance = data['analysis']['performance_summary']
                    f.write(f"Total Return: {performance['total_return']:.4f}\n")
                    f.write(f"Sharpe Ratio: {performance['sharpe_ratio']:.4f}\n")
                    f.write(f"Max Drawdown: {performance['max_drawdown']:.4f}\n")
                    f.write(f"Win Rate: {performance['win_rate']:.4f}\n")
                    f.write(f"Assessment: {data['analysis']['assessment']}\n\n")
        
        logger.info(f"✅ Strategy comparison saved to {comparison_file}")
        
    except Exception as e:
        logger.error(f"❌ Error generating comparison report: {e}")
    
    logger.info("🎉 Comprehensive backtesting completed!")

if __name__ == "__main__":
    main()
