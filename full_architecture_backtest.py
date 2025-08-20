#!/usr/bin/env python3
"""
Full Architecture Backtest
==========================

This script runs a comprehensive backtest using the entire trading intelligence architecture:
- All agent types (technical, sentiment, flow, causal, etc.)
- Coordination and meta-weighting
- Real Polygon data integration
- Portfolio optimization
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
import json
from typing import Dict, List, Any, Optional

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

# Import architecture components
try:
    from agents.technical.agent_complete import TechnicalAgent
    from agents.sentiment.agent_complete import SentimentAgent
    from agents.flow.agent_complete import FlowAgent
    from agents.causal.agent_optimized import CausalAgent
    from agents.macro.agent_complete import MacroAgent
    from agents.top_performers.agent_complete import TopPerformersAgent
    from agents.undervalued.agent_complete import UndervaluedAgent
    from agents.learning.advanced_learning_methods_fixed import LearningAgent
    from coordination.meta_weighter import MetaWeighter
    from coordination.opportunity_builder import OpportunityBuilder
    from coordination.top_k_selector import TopKSelector
    from common.models import Signal, Opportunity, Portfolio
    from common.scoring.unified_score import UnifiedScore
    from common.opportunity_store import OpportunityStore
    ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some architecture components not available: {e}")
    ARCHITECTURE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_architecture_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_polygon_api_key():
    """Check if Polygon API key is available"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("❌ POLYGON_API_KEY environment variable not set!")
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

class FullArchitectureBacktest:
    """Full architecture backtest integrating all components"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.initial_capital = 1000000.0  # $1M starting capital
        self.portfolio_value = self.initial_capital
        self.portfolio_history = []
        self.trades = []
        
        # Initialize architecture components if available
        if ARCHITECTURE_AVAILABLE:
            self._initialize_architecture()
        else:
            logger.warning("⚠️  Architecture components not available, using simplified backtest")
    
    def _initialize_architecture(self):
        """Initialize all architecture components"""
        try:
            logger.info("🏗️  Initializing trading intelligence architecture...")
            
            # Initialize agents
            self.technical_agent = TechnicalAgent()
            self.sentiment_agent = SentimentAgent()
            self.flow_agent = FlowAgent()
            self.causal_agent = CausalAgent()
            self.macro_agent = MacroAgent()
            self.top_performers_agent = TopPerformersAgent()
            self.undervalued_agent = UndervaluedAgent()
            self.learning_agent = LearningAgent()
            
            # Initialize coordination components
            self.meta_weighter = MetaWeighter()
            self.opportunity_builder = OpportunityBuilder()
            self.top_k_selector = TopKSelector()
            
            # Initialize common components
            self.unified_score = UnifiedScore()
            self.opportunity_store = OpportunityStore()
            
            logger.info("✅ Architecture components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing architecture: {e}")
            raise
    
    def get_agent_signals(self, data: Dict[str, pd.DataFrame], current_date: str) -> List:
        """Get signals from all agents"""
        signals = []
        
        if not ARCHITECTURE_AVAILABLE:
            return signals
        
        try:
            # Technical analysis signals
            for symbol, df in data.items():
                symbol_data = df[df['date'] <= current_date]
                if len(symbol_data) >= 60:  # Need enough data
                    try:
                        tech_signals = self.technical_agent.analyze(symbol_data)
                        if tech_signals:
                            signals.extend(tech_signals)
                    except Exception as e:
                        logger.debug(f"Technical analysis failed for {symbol}: {e}")
            
            # Sentiment analysis signals
            try:
                sentiment_signals = self.sentiment_agent.analyze_market_sentiment(current_date)
                if sentiment_signals:
                    signals.extend(sentiment_signals)
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
            
            # Flow analysis signals
            try:
                flow_signals = self.flow_agent.analyze_money_flows(data, current_date)
                if flow_signals:
                    signals.extend(flow_signals)
            except Exception as e:
                logger.debug(f"Flow analysis failed: {e}")
            
            # Causal analysis signals
            try:
                causal_signals = self.causal_agent.analyze_causal_relationships(data, current_date)
                if causal_signals:
                    signals.extend(causal_signals)
            except Exception as e:
                logger.debug(f"Causal analysis failed: {e}")
            
            # Macro analysis signals
            try:
                macro_signals = self.macro_agent.analyze_macro_environment(current_date)
                if macro_signals:
                    signals.extend(macro_signals)
            except Exception as e:
                logger.debug(f"Macro analysis failed: {e}")
            
            # Top performers analysis
            try:
                top_performer_signals = self.top_performers_agent.identify_top_performers(data, current_date)
                if top_performer_signals:
                    signals.extend(top_performer_signals)
            except Exception as e:
                logger.debug(f"Top performers analysis failed: {e}")
            
            # Undervalued analysis
            try:
                undervalued_signals = self.undervalued_agent.identify_undervalued_stocks(data, current_date)
                if undervalued_signals:
                    signals.extend(undervalued_signals)
            except Exception as e:
                logger.debug(f"Undervalued analysis failed: {e}")
            
            # Learning agent signals
            try:
                learning_signals = self.learning_agent.generate_signals(data, current_date)
                if learning_signals:
                    signals.extend(learning_signals)
            except Exception as e:
                logger.debug(f"Learning agent failed: {e}")
            
            logger.info(f"📊 Generated {len(signals)} signals from all agents")
            
        except Exception as e:
            logger.error(f"❌ Error getting agent signals: {e}")
        
        return signals
    
    def build_opportunities(self, signals: List, current_date: str) -> List:
        """Build opportunities from signals"""
        opportunities = []
        
        if not ARCHITECTURE_AVAILABLE or not signals:
            return opportunities
        
        try:
            # Use opportunity builder to create opportunities from signals
            opportunities = self.opportunity_builder.build_opportunities(signals, current_date)
            
            # Score opportunities using unified scoring
            for opportunity in opportunities:
                opportunity.score = self.unified_score.calculate_score(opportunity)
            
            # Sort by score
            opportunities.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"🎯 Built {len(opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"❌ Error building opportunities: {e}")
        
        return opportunities
    
    def select_portfolio(self, opportunities: List, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Select portfolio using meta-weighting and top-k selection"""
        weights = {}
        
        if not ARCHITECTURE_AVAILABLE or not opportunities:
            return weights
        
        try:
            # Use top-k selector to get best opportunities
            top_opportunities = self.top_k_selector.select_top_k(opportunities, k=10)
            
            if top_opportunities:
                # Use meta-weighter to determine weights
                weights = self.meta_weighter.calculate_weights(top_opportunities, current_prices)
                
                # Normalize weights to sum to 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
                
                logger.info(f"📈 Selected portfolio with {len(weights)} positions")
            
        except Exception as e:
            logger.error(f"❌ Error selecting portfolio: {e}")
        
        return weights
    
    def fallback_strategy(self, data: Dict[str, pd.DataFrame], current_date: str, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Fallback strategy when architecture is not available"""
        weights = {}
        
        try:
            # Simple momentum strategy as fallback
            valid_symbols = []
            
            for symbol, df in data.items():
                if symbol in current_prices:
                    symbol_data = df[df['date'] <= current_date]
                    
                    if len(symbol_data) >= 20:
                        # Calculate momentum
                        start_price = symbol_data.iloc[-20]['close']
                        end_price = current_prices[symbol]
                        momentum = (end_price - start_price) / start_price
                        
                        if momentum > 0.05:  # 5% positive momentum
                            valid_symbols.append((symbol, momentum))
            
            # Select top 5 momentum stocks
            valid_symbols.sort(key=lambda x: x[1], reverse=True)
            top_stocks = valid_symbols[:5]
            
            if top_stocks:
                weight_per_stock = 1.0 / len(top_stocks)
                for symbol, _ in top_stocks:
                    weights[symbol] = weight_per_stock
                
                logger.info(f"📈 Fallback strategy: {len(weights)} stocks selected")
        
        except Exception as e:
            logger.error(f"❌ Error in fallback strategy: {e}")
        
        return weights
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str):
        """Run the full architecture backtest"""
        
        logger.info("🚀 Starting Full Architecture Backtest")
        logger.info("=" * 60)
        
        # Download data
        logger.info("📥 Downloading market data...")
        all_data = {}
        
        for symbol in symbols:
            df = download_polygon_data(symbol, start_date, end_date, self.api_key)
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
            
            # Get agent signals
            signals = self.get_agent_signals(all_data, current_date)
            
            # Build opportunities
            opportunities = self.build_opportunities(signals, current_date)
            
            # Select portfolio
            if ARCHITECTURE_AVAILABLE and opportunities:
                weights = self.select_portfolio(opportunities, current_prices)
            else:
                weights = self.fallback_strategy(all_data, current_date, current_prices)
            
            # Update portfolio value
            if weights:
                total_value = 0
                for symbol, weight in weights.items():
                    if symbol in current_prices:
                        position_value = self.portfolio_value * weight
                        total_value += position_value
                
                if total_value > 0:
                    self.portfolio_value = total_value
            
            # Record portfolio value
            self.portfolio_history.append({
                'date': current_date,
                'value': self.portfolio_value,
                'num_positions': len(weights),
                'num_signals': len(signals),
                'num_opportunities': len(opportunities)
            })
            
            # Log progress every 50 days
            if i % 50 == 0:
                logger.info(f"📅 {current_date}: Portfolio Value: ${self.portfolio_value:,.2f}, Positions: {len(weights)}")
        
        # Calculate final results
        self._calculate_results()
    
    def _calculate_results(self):
        """Calculate and display final results"""
        if not self.portfolio_history:
            logger.error("❌ No portfolio history to analyze")
            return
        
        try:
            df_history = pd.DataFrame(self.portfolio_history)
            df_history['date'] = pd.to_datetime(df_history['date'])
            df_history = df_history.set_index('date')
            
            # Calculate returns
            returns = df_history['value'].pct_change().dropna()
            
            # Calculate metrics
            total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            annualized_return = total_return * (252 / len(returns)) if len(returns) > 0 else 0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate additional metrics
            avg_positions = df_history['num_positions'].mean()
            avg_signals = df_history['num_signals'].mean()
            avg_opportunities = df_history['num_opportunities'].mean()
            
            results = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': self.portfolio_value,
                'avg_positions': avg_positions,
                'avg_signals': avg_signals,
                'avg_opportunities': avg_opportunities,
                'total_trades': len(self.trades)
            }
            
            # Display results
            logger.info("\n" + "=" * 60)
            logger.info("🏆 FULL ARCHITECTURE BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"📈 Total Return: {total_return:.2%}")
            logger.info(f"📊 Annualized Return: {annualized_return:.2%}")
            logger.info(f"📉 Volatility: {volatility:.2%}")
            logger.info(f"🎯 Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"📉 Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"💰 Final Value: ${self.portfolio_value:,.2f}")
            logger.info(f"📊 Average Positions: {avg_positions:.1f}")
            logger.info(f"📡 Average Signals: {avg_signals:.1f}")
            logger.info(f"🎯 Average Opportunities: {avg_opportunities:.1f}")
            logger.info(f"🏗️  Architecture Available: {ARCHITECTURE_AVAILABLE}")
            
            # Save results
            output_dir = Path("full_architecture_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            with open(output_dir / "backtest_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save portfolio history
            df_history.to_csv(output_dir / "portfolio_history.csv")
            
            logger.info(f"\n📁 Results saved to: {output_dir}")
            logger.info("✅ Full architecture backtest completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error calculating results: {e}")

def run_full_architecture_backtest():
    """Run the full architecture backtest"""
    
    # Check API key
    if not check_polygon_api_key():
        return
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    # Configuration
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "SPY", "QQQ", "VTI", "VOO", "NVDA",
        "META", "NFLX", "CRM", "ADBE", "PYPL"
    ]  # 15 major stocks/ETFs
    start_date = "2023-01-01"  # 1 year for faster testing
    end_date = "2024-12-31"
    
    # Create and run backtest
    backtest = FullArchitectureBacktest(api_key)
    backtest.run_backtest(symbols, start_date, end_date)

if __name__ == "__main__":
    run_full_architecture_backtest()
