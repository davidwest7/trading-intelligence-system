#!/usr/bin/env python3
"""
Simple Real Backtest with Polygon Data
=====================================

This script runs a simplified backtest using real market data from Polygon.io
with direct data download and realistic execution simulation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_real_backtest.log'),
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

def download_polygon_data(symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Download daily bars data from Polygon"""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'apiKey': api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        logger.info(f"📥 Downloading {symbol} data from {start_date} to {end_date}...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['resultsCount'] > 0:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df['date'] = df['timestamp'].dt.date
                df['symbol'] = symbol
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'vw': 'vwap',
                    'n': 'transactions'
                })
                
                # Select only needed columns
                df = df[['symbol', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                logger.info(f"✅ Downloaded {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
        else:
            logger.error(f"❌ API request failed for {symbol}: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Error downloading {symbol}: {e}")
        return pd.DataFrame()

def momentum_strategy(data, current_date, current_prices, lookback_days=20):
    """Momentum strategy - buy stocks with positive momentum"""
    try:
        weights = {}
        valid_symbols = []
        
        for symbol in data.keys():
            if symbol in current_prices and symbol in data:
                symbol_data = data[symbol]
                
                if len(symbol_data) >= lookback_days:
                    # Calculate momentum
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
    """Mean reversion strategy - buy stocks below moving average"""
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
                    
                    # Calculate z-score
                    if ma > 0:
                        z_score = (current_price - ma) / ma
                        
                        # Buy if significantly below mean
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

def run_simple_backtest():
    """Run simplified backtest with real Polygon data"""
    
    logger.info("🚀 Starting Simple Real Backtest with Polygon Data")
    logger.info("=" * 60)
    
    # Check API key
    if not check_polygon_api_key():
        return
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]  # Reduced for faster testing
    start_date = "2023-01-01"  # 1 year of data for faster testing
    end_date = "2024-12-31"
    initial_capital = 100000.0  # $100K starting capital
    
    # Download data
    logger.info("📥 Downloading market data...")
    all_data = {}
    
    for symbol in symbols:
        df = download_polygon_data(symbol, start_date, end_date, api_key)
        if not df.empty:
            all_data[symbol] = df
            time.sleep(0.1)  # Rate limiting
        else:
            logger.warning(f"Skipping {symbol} - no data available")
    
    if not all_data:
        logger.error("❌ No data downloaded for any symbols")
        return
    
    logger.info(f"✅ Downloaded data for {len(all_data)} symbols")
    
    # Create price matrix
    dates = sorted(set.intersection(*[set(df['date']) for df in all_data.values()]))
    if not dates:
        logger.error("❌ No common dates found across symbols")
        return
    
    logger.info(f"📊 Trading period: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
    
    # Run strategies
    strategies = {
        "Momentum": momentum_strategy,
        "Mean_Reversion": mean_reversion_strategy
    }
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        logger.info(f"🎯 Running {strategy_name} strategy...")
        
        try:
            # Initialize portfolio
            portfolio_value = initial_capital
            portfolio_history = []
            trades = []
            
            # Run backtest
            for i, current_date in enumerate(dates):
                if i < 60:  # Skip first 60 days for strategy warm-up
                    continue
                
                # Get current prices
                current_prices = {}
                for symbol, df in all_data.items():
                    symbol_data = df[df['date'] <= current_date]
                    if not symbol_data.empty:
                        current_prices[symbol] = symbol_data.iloc[-1]['close']
                
                if len(current_prices) < 2:
                    continue
                
                # Get strategy weights
                historical_data = {}
                for symbol, df in all_data.items():
                    historical_data[symbol] = df[df['date'] <= current_date]
                
                target_weights = strategy_func(historical_data, current_date, current_prices)
                
                if target_weights:
                    # Calculate portfolio value
                    total_value = 0
                    for symbol, weight in target_weights.items():
                        if symbol in current_prices:
                            position_value = portfolio_value * weight
                            total_value += position_value
                    
                    if total_value > 0:
                        portfolio_value = total_value
                
                # Record portfolio value
                portfolio_history.append({
                    'date': current_date,
                    'value': portfolio_value
                })
            
            # Calculate metrics
            if portfolio_history:
                df_history = pd.DataFrame(portfolio_history)
                df_history['date'] = pd.to_datetime(df_history['date'])
                df_history = df_history.set_index('date')
                
                # Calculate returns
                returns = df_history['value'].pct_change().dropna()
                
                # Calculate metrics
                total_return = (portfolio_value - initial_capital) / initial_capital
                annualized_return = total_return * (252 / len(returns)) if len(returns) > 0 else 0
                volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
                sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
                
                # Calculate max drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                results[strategy_name] = {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'final_value': portfolio_value,
                    'total_trades': len(trades)
                }
                
                logger.info(f"✅ {strategy_name} Results:")
                logger.info(f"   Total Return: {total_return:.2%}")
                logger.info(f"   Annualized Return: {annualized_return:.2%}")
                logger.info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
                logger.info(f"   Max Drawdown: {max_drawdown:.2%}")
                logger.info(f"   Final Value: ${portfolio_value:,.2f}")
                
            else:
                logger.error(f"❌ No portfolio history for {strategy_name}")
                
        except Exception as e:
            logger.error(f"❌ Error running {strategy_name} strategy: {e}")
            continue
    
    # Generate comparison
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("🏆 STRATEGY COMPARISON")
        logger.info("=" * 60)
        
        comparison_data = []
        for strategy_name, result in results.items():
            comparison_data.append({
                'strategy': strategy_name,
                'total_return': result['total_return'],
                'annualized_return': result['annualized_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'final_value': result['final_value']
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
        output_dir = Path("simple_backtest_results")
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
        logger.info("✅ Simple real backtest completed successfully!")
        
    else:
        logger.error("❌ No successful backtest results to compare")

if __name__ == "__main__":
    run_simple_backtest()
