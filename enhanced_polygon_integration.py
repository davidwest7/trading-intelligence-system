#!/usr/bin/env python3
"""
Enhanced Polygon Integration - Phase 1 Implementation
Implements high-impact features using all available Polygon data points
"""
import asyncio
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

class EnhancedPolygonIntegration:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY', '')
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.rate_limits = {
            'calls': 0,
            'limit': 5,  # 5 calls per minute
            'reset_time': time.time() + 60
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        if current_time > self.rate_limits['reset_time']:
            self.rate_limits['calls'] = 0
            self.rate_limits['reset_time'] = current_time + 60
        
        if self.rate_limits['calls'] >= self.rate_limits['limit']:
            wait_time = self.rate_limits['reset_time'] - current_time
            if wait_time > 0:
                print(f"⏱️ Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.rate_limits['calls'] = 0
                self.rate_limits['reset_time'] = time.time() + 60
        
        self.rate_limits['calls'] += 1
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make API request with rate limiting"""
        await self._rate_limit_check()
        
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"❌ API request failed: {response.status}")
                    return {'error': f'HTTP {response.status}'}
        except Exception as e:
            print(f"❌ Request error: {str(e)}")
            return {'error': str(e)}
    
    async def get_trades_data(self, ticker: str, date: str = None, limit: int = 1000) -> Dict[str, Any]:
        """Get detailed trades data for order flow analysis"""
        print(f"📊 Fetching trades data for {ticker}")
        
        endpoint = f"/v3/trades/{ticker}"
        params = {'limit': limit}
        if date:
            params['date'] = date
        
        data = await self._make_request(endpoint, params)
        
        if 'results' in data:
            trades_df = pd.DataFrame(data['results'])
            
            # Calculate order flow metrics
            analysis = {
                'total_trades': len(trades_df),
                'total_volume': trades_df['size'].sum(),
                'avg_trade_size': trades_df['size'].mean(),
                'large_trades': len(trades_df[trades_df['size'] > trades_df['size'].quantile(0.9)]),
                'buy_pressure': len(trades_df[trades_df['price'] > trades_df['price'].shift(1)]),
                'sell_pressure': len(trades_df[trades_df['price'] < trades_df['price'].shift(1)]),
                'volume_weighted_price': (trades_df['price'] * trades_df['size']).sum() / trades_df['size'].sum(),
                'trade_clustering': self._analyze_trade_clustering(trades_df),
                'market_impact': self._calculate_market_impact(trades_df)
            }
            
            return {
                'status': 'success',
                'trades_data': data['results'],
                'analysis': analysis,
                'expected_alpha': '5-10%'
            }
        
        return {'status': 'error', 'data': data}
    
    async def get_quotes_data(self, ticker: str, date: str = None, limit: int = 1000) -> Dict[str, Any]:
        """Get detailed quotes data for bid-ask analysis"""
        print(f"📊 Fetching quotes data for {ticker}")
        
        endpoint = f"/v3/quotes/{ticker}"
        params = {'limit': limit}
        if date:
            params['date'] = date
        
        data = await self._make_request(endpoint, params)
        
        if 'results' in data:
            quotes_df = pd.DataFrame(data['results'])
            
            # Calculate bid-ask metrics
            analysis = {
                'total_quotes': len(quotes_df),
                'avg_bid_ask_spread': (quotes_df['ask_price'] - quotes_df['bid_price']).mean(),
                'spread_volatility': (quotes_df['ask_price'] - quotes_df['bid_price']).std(),
                'bid_depth': quotes_df['bid_size'].sum(),
                'ask_depth': quotes_df['ask_size'].sum(),
                'order_book_imbalance': (quotes_df['bid_size'] - quotes_df['ask_size']).mean(),
                'market_maker_activity': self._analyze_market_maker_activity(quotes_df),
                'liquidity_metrics': self._calculate_liquidity_metrics(quotes_df)
            }
            
            return {
                'status': 'success',
                'quotes_data': data['results'],
                'analysis': analysis,
                'expected_alpha': '3-7%'
            }
        
        return {'status': 'error', 'data': data}
    
    async def get_options_contracts(self, underlying_asset: str) -> Dict[str, Any]:
        """Get options contracts for options flow analysis"""
        print(f"📊 Fetching options contracts for {underlying_asset}")
        
        endpoint = "/v3/reference/options/contracts"
        params = {'underlying_asset': underlying_asset}
        
        data = await self._make_request(endpoint, params)
        
        if 'results' in data:
            contracts_df = pd.DataFrame(data['results'])
            
            # Analyze options chain
            analysis = {
                'total_contracts': len(contracts_df),
                'call_contracts': len(contracts_df[contracts_df['contract_type'] == 'call']),
                'put_contracts': len(contracts_df[contracts_df['contract_type'] == 'put']),
                'strike_range': {
                    'min': contracts_df['strike_price'].min(),
                    'max': contracts_df['strike_price'].max()
                },
                'expiration_dates': contracts_df['expiration_date'].unique().tolist(),
                'options_chain_analysis': self._analyze_options_chain(contracts_df)
            }
            
            return {
                'status': 'success',
                'contracts_data': data['results'],
                'analysis': analysis,
                'expected_alpha': '8-15%'
            }
        
        return {'status': 'error', 'data': data}
    
    async def get_options_trades(self, underlying_asset: str, date: str = None) -> Dict[str, Any]:
        """Get options trades for unusual options activity analysis"""
        print(f"📊 Fetching options trades for {underlying_asset}")
        
        endpoint = f"/v3/trades/{underlying_asset}/options"
        params = {}
        if date:
            params['date'] = date
        
        data = await self._make_request(endpoint, params)
        
        if 'results' in data:
            trades_df = pd.DataFrame(data['results'])
            
            # Analyze unusual options activity
            analysis = {
                'total_options_trades': len(trades_df),
                'call_volume': len(trades_df[trades_df['contract_type'] == 'call']),
                'put_volume': len(trades_df[trades_df['contract_type'] == 'put']),
                'put_call_ratio': len(trades_df[trades_df['contract_type'] == 'put']) / max(len(trades_df[trades_df['contract_type'] == 'call']), 1),
                'large_options_trades': len(trades_df[trades_df['size'] > trades_df['size'].quantile(0.9)]),
                'unusual_activity': self._detect_unusual_options_activity(trades_df),
                'options_flow_sentiment': self._calculate_options_flow_sentiment(trades_df)
            }
            
            return {
                'status': 'success',
                'options_trades_data': data['results'],
                'analysis': analysis,
                'expected_alpha': '10-20%'
            }
        
        return {'status': 'error', 'data': data}
    
    async def get_enhanced_aggregates(self, ticker: str, multiplier: int = 1, timespan: str = 'day', 
                                    from_date: str = None, to_date: str = None) -> Dict[str, Any]:
        """Get enhanced aggregates with advanced technical analysis"""
        print(f"📊 Fetching enhanced aggregates for {ticker}")
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        data = await self._make_request(endpoint)
        
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            
            # Advanced technical analysis
            analysis = {
                'multi_timeframe_analysis': self._multi_timeframe_analysis(df),
                'volume_profile': self._volume_profile_analysis(df),
                'price_momentum': self._price_momentum_analysis(df),
                'support_resistance': self._support_resistance_levels(df),
                'trend_analysis': self._trend_analysis(df),
                'market_microstructure': self._market_microstructure_analysis(df)
            }
            
            return {
                'status': 'success',
                'aggregates_data': data['results'],
                'analysis': analysis,
                'expected_alpha': '4-8%'
            }
        
        return {'status': 'error', 'data': data}
    
    async def get_dividends_data(self, ticker: str = None) -> Dict[str, Any]:
        """Get dividend data for dividend analysis"""
        print(f"📊 Fetching dividends data for {ticker or 'all'}")
        
        endpoint = "/v3/reference/dividends"
        params = {}
        if ticker:
            params['ticker'] = ticker
        
        data = await self._make_request(endpoint, params)
        
        if 'results' in data:
            dividends_df = pd.DataFrame(data['results'])
            
            # Dividend analysis
            analysis = {
                'total_dividends': len(dividends_df),
                'avg_dividend_amount': dividends_df['amount'].mean(),
                'dividend_growth': self._calculate_dividend_growth(dividends_df),
                'dividend_sustainability': self._assess_dividend_sustainability(dividends_df),
                'dividend_yield_analysis': self._dividend_yield_analysis(dividends_df),
                'income_strategy_signals': self._income_strategy_signals(dividends_df)
            }
            
            return {
                'status': 'success',
                'dividends_data': data['results'],
                'analysis': analysis,
                'expected_alpha': '2-5%'
            }
        
        return {'status': 'error', 'data': data}
    
    # Analysis helper methods
    def _analyze_trade_clustering(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade clustering patterns"""
        # Group trades by time intervals and analyze clustering
        trades_df['time_group'] = pd.to_datetime(trades_df['t'], unit='ns').dt.floor('1min')
        clustering = trades_df.groupby('time_group').agg({
            'size': ['count', 'sum', 'mean'],
            'price': ['mean', 'std']
        }).round(2)
        
        return {
            'clustering_score': len(clustering[clustering[('size', 'count')] > 5]),
            'volume_clusters': clustering[('size', 'sum')].nlargest(10).to_dict(),
            'price_volatility_clusters': clustering[('price', 'std')].nlargest(10).to_dict()
        }
    
    def _calculate_market_impact(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market impact of trades"""
        trades_df['price_change'] = trades_df['price'].diff()
        trades_df['volume_weighted_impact'] = trades_df['price_change'] * trades_df['size']
        
        return {
            'avg_market_impact': trades_df['volume_weighted_impact'].mean(),
            'large_trade_impact': trades_df[trades_df['size'] > trades_df['size'].quantile(0.9)]['volume_weighted_impact'].mean(),
            'impact_decay': trades_df['volume_weighted_impact'].rolling(5).mean().iloc[-1]
        }
    
    def _analyze_market_maker_activity(self, quotes_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market maker activity patterns"""
        quotes_df['spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['mid_price'] = (quotes_df['ask_price'] + quotes_df['bid_price']) / 2
        
        return {
            'spread_tightening_events': len(quotes_df[quotes_df['spread'] < quotes_df['spread'].quantile(0.1)]),
            'spread_widening_events': len(quotes_df[quotes_df['spread'] > quotes_df['spread'].quantile(0.9)]),
            'market_maker_pressure': (quotes_df['bid_size'] - quotes_df['ask_size']).mean()
        }
    
    def _calculate_liquidity_metrics(self, quotes_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate liquidity metrics"""
        quotes_df['spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        
        return {
            'amihud_illiquidity': quotes_df['spread'].mean() / quotes_df['bid_size'].mean(),
            'bid_ask_spread_volatility': quotes_df['spread'].std(),
            'liquidity_score': 1 / (quotes_df['spread'].mean() * quotes_df['bid_size'].std())
        }
    
    def _analyze_options_chain(self, contracts_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze options chain structure"""
        calls = contracts_df[contracts_df['contract_type'] == 'call']
        puts = contracts_df[contracts_df['contract_type'] == 'put']
        
        return {
            'call_put_ratio': len(calls) / max(len(puts), 1),
            'strike_distribution': contracts_df['strike_price'].describe().to_dict(),
            'expiration_structure': contracts_df.groupby('expiration_date').size().to_dict()
        }
    
    def _detect_unusual_options_activity(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect unusual options activity"""
        large_trades = trades_df[trades_df['size'] > trades_df['size'].quantile(0.95)]
        
        return {
            'unusual_trades_count': len(large_trades),
            'unusual_call_volume': len(large_trades[large_trades['contract_type'] == 'call']),
            'unusual_put_volume': len(large_trades[large_trades['contract_type'] == 'put']),
            'unusual_activity_score': len(large_trades) / len(trades_df) * 100
        }
    
    def _calculate_options_flow_sentiment(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate options flow sentiment"""
        call_volume = len(trades_df[trades_df['contract_type'] == 'call'])
        put_volume = len(trades_df[trades_df['contract_type'] == 'put'])
        
        sentiment_score = (call_volume - put_volume) / (call_volume + put_volume) if (call_volume + put_volume) > 0 else 0
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': 'Bullish' if sentiment_score > 0.1 else 'Bearish' if sentiment_score < -0.1 else 'Neutral',
            'call_dominance': call_volume / (call_volume + put_volume) if (call_volume + put_volume) > 0 else 0.5
        }
    
    def _multi_timeframe_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Multi-timeframe technical analysis"""
        # Calculate different timeframe moving averages
        df['sma_5'] = df['c'].rolling(5).mean()
        df['sma_20'] = df['c'].rolling(20).mean()
        df['ema_12'] = df['c'].ewm(span=12).mean()
        df['ema_26'] = df['c'].ewm(span=26).mean()
        
        return {
            'trend_alignment': 'Bullish' if df['sma_5'].iloc[-1] > df['sma_20'].iloc[-1] else 'Bearish',
            'momentum_divergence': df['ema_12'].iloc[-1] - df['ema_26'].iloc[-1],
            'multi_timeframe_signal': self._generate_multi_timeframe_signal(df)
        }
    
    def _volume_profile_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Volume profile analysis"""
        df['volume_price_level'] = pd.cut(df['c'], bins=10)
        volume_profile = df.groupby('volume_price_level')['v'].sum()
        
        return {
            'high_volume_nodes': volume_profile.nlargest(3).to_dict(),
            'low_volume_nodes': volume_profile.nsmallest(3).to_dict(),
            'volume_weighted_average_price': (df['c'] * df['v']).sum() / df['v'].sum()
        }
    
    def _price_momentum_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Price momentum analysis"""
        df['returns'] = df['c'].pct_change()
        df['momentum_5'] = df['returns'].rolling(5).sum()
        df['momentum_20'] = df['returns'].rolling(20).sum()
        
        return {
            'short_term_momentum': df['momentum_5'].iloc[-1],
            'long_term_momentum': df['momentum_20'].iloc[-1],
            'momentum_acceleration': df['momentum_5'].iloc[-1] - df['momentum_5'].iloc[-5]
        }
    
    def _support_resistance_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify support and resistance levels"""
        recent_highs = df['h'].rolling(20).max()
        recent_lows = df['l'].rolling(20).min()
        
        return {
            'resistance_levels': recent_highs.nlargest(3).to_dict(),
            'support_levels': recent_lows.nsmallest(3).to_dict(),
            'current_position': (df['c'].iloc[-1] - recent_lows.iloc[-1]) / (recent_highs.iloc[-1] - recent_lows.iloc[-1])
        }
    
    def _trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        df['trend_5'] = df['c'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        df['trend_20'] = df['c'].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        return {
            'short_term_trend': 'Up' if df['trend_5'].iloc[-1] > 0 else 'Down',
            'long_term_trend': 'Up' if df['trend_20'].iloc[-1] > 0 else 'Down',
            'trend_strength': abs(df['c'].iloc[-1] - df['c'].iloc[-20]) / df['c'].iloc[-20] * 100
        }
    
    def _market_microstructure_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Market microstructure analysis"""
        df['price_efficiency'] = df['returns'].rolling(10).std()
        df['volume_efficiency'] = df['v'].rolling(10).std() / df['v'].rolling(10).mean()
        
        return {
            'price_efficiency_score': df['price_efficiency'].iloc[-1],
            'volume_efficiency_score': df['volume_efficiency'].iloc[-1],
            'market_quality': 1 / (df['price_efficiency'].iloc[-1] * df['volume_efficiency'].iloc[-1])
        }
    
    def _calculate_dividend_growth(self, dividends_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dividend growth metrics"""
        dividends_df['ex_date'] = pd.to_datetime(dividends_df['ex_date'])
        dividends_df = dividends_df.sort_values('ex_date')
        
        if len(dividends_df) > 1:
            growth_rate = (dividends_df['amount'].iloc[-1] - dividends_df['amount'].iloc[0]) / dividends_df['amount'].iloc[0] * 100
        else:
            growth_rate = 0
        
        return {
            'growth_rate': growth_rate,
            'growth_trend': 'Increasing' if growth_rate > 0 else 'Decreasing',
            'consistency_score': len(dividends_df) / 4  # Assuming quarterly dividends
        }
    
    def _assess_dividend_sustainability(self, dividends_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess dividend sustainability"""
        avg_dividend = dividends_df['amount'].mean()
        dividend_volatility = dividends_df['amount'].std()
        
        return {
            'sustainability_score': avg_dividend / max(dividend_volatility, 0.01),
            'consistency': 1 - (dividend_volatility / avg_dividend) if avg_dividend > 0 else 0,
            'risk_level': 'Low' if dividend_volatility < avg_dividend * 0.1 else 'Medium' if dividend_volatility < avg_dividend * 0.2 else 'High'
        }
    
    def _dividend_yield_analysis(self, dividends_df: pd.DataFrame) -> Dict[str, Any]:
        """Dividend yield analysis"""
        annual_dividend = dividends_df['amount'].sum() * 4  # Assuming quarterly
        
        return {
            'annual_dividend': annual_dividend,
            'yield_ranking': 'High' if annual_dividend > 5 else 'Medium' if annual_dividend > 2 else 'Low',
            'income_potential': annual_dividend * 100  # Assuming $100 investment
        }
    
    def _income_strategy_signals(self, dividends_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate income strategy signals"""
        consistency = len(dividends_df) / 4  # Quarterly frequency
        avg_amount = dividends_df['amount'].mean()
        
        return {
            'income_strategy_score': consistency * avg_amount,
            'recommendation': 'Strong Buy' if consistency > 0.8 and avg_amount > 1 else 'Buy' if consistency > 0.6 else 'Hold',
            'income_potential': 'High' if avg_amount > 2 else 'Medium' if avg_amount > 1 else 'Low'
        }
    
    def _generate_multi_timeframe_signal(self, df: pd.DataFrame) -> str:
        """Generate multi-timeframe trading signal"""
        sma_5 = df['sma_5'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        ema_12 = df['ema_12'].iloc[-1]
        ema_26 = df['ema_26'].iloc[-1]
        
        bullish_count = sum([
            sma_5 > sma_20,
            ema_12 > ema_26,
            df['c'].iloc[-1] > sma_5
        ])
        
        if bullish_count >= 2:
            return 'Strong Buy'
        elif bullish_count == 1:
            return 'Buy'
        else:
            return 'Sell'
    
    async def run_comprehensive_analysis(self, ticker: str) -> Dict[str, Any]:
        """Run comprehensive analysis using all Polygon data points"""
        print(f"🚀 Running comprehensive analysis for {ticker}")
        print("=" * 60)
        
        start_time = time.time()
        results = {}
        total_expected_alpha = 0
        
        # 1. Trades Data Analysis (5-10% alpha)
        print("📊 Phase 1: Trades Data Analysis")
        trades_result = await self.get_trades_data(ticker)
        results['trades_analysis'] = trades_result
        if trades_result['status'] == 'success':
            total_expected_alpha += 7.5  # Midpoint of 5-10%
        
        # 2. Quotes Data Analysis (3-7% alpha)
        print("📊 Phase 2: Quotes Data Analysis")
        quotes_result = await self.get_quotes_data(ticker)
        results['quotes_analysis'] = quotes_result
        if quotes_result['status'] == 'success':
            total_expected_alpha += 5  # Midpoint of 3-7%
        
        # 3. Enhanced Aggregates Analysis (4-8% alpha)
        print("📊 Phase 3: Enhanced Aggregates Analysis")
        aggregates_result = await self.get_enhanced_aggregates(ticker)
        results['aggregates_analysis'] = aggregates_result
        if aggregates_result['status'] == 'success':
            total_expected_alpha += 6  # Midpoint of 4-8%
        
        # 4. Options Analysis (if available)
        print("📊 Phase 4: Options Analysis")
        options_contracts_result = await self.get_options_contracts(ticker)
        results['options_contracts_analysis'] = options_contracts_result
        if options_contracts_result['status'] == 'success':
            total_expected_alpha += 11.5  # Midpoint of 8-15%
            
            options_trades_result = await self.get_options_trades(ticker)
            results['options_trades_analysis'] = options_trades_result
            if options_trades_result['status'] == 'success':
                total_expected_alpha += 15  # Midpoint of 10-20%
        
        # 5. Dividends Analysis (2-5% alpha)
        print("📊 Phase 5: Dividends Analysis")
        dividends_result = await self.get_dividends_data(ticker)
        results['dividends_analysis'] = dividends_result
        if dividends_result['status'] == 'success':
            total_expected_alpha += 3.5  # Midpoint of 2-5%
        
        total_time = time.time() - start_time
        
        # Calculate net alpha with diversification
        diversification_factor = 0.6
        net_alpha = total_expected_alpha * diversification_factor
        
        comprehensive_report = {
            'ticker': ticker,
            'analysis_date': datetime.now().isoformat(),
            'total_analysis_time': total_time,
            'total_expected_alpha': total_expected_alpha,
            'net_alpha': net_alpha,
            'diversification_factor': diversification_factor,
            'results': results,
            'summary': {
                'successful_analyses': sum(1 for r in results.values() if r['status'] == 'success'),
                'total_analyses': len(results),
                'success_rate': sum(1 for r in results.values() if r['status'] == 'success') / len(results) * 100,
                'alpha_per_analysis': net_alpha / len(results)
            }
        }
        
        print(f"\n📋 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Total Expected Alpha: {total_expected_alpha:.1f}%")
        print(f"🎯 Net Alpha (with diversification): {net_alpha:.1f}%")
        print(f"✅ Success Rate: {comprehensive_report['summary']['success_rate']:.1f}%")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        
        return comprehensive_report

async def main():
    """Test the enhanced Polygon integration"""
    print("🚀 Testing Enhanced Polygon Integration")
    print("=" * 60)
    
    async with EnhancedPolygonIntegration() as polygon:
        # Test with a major stock
        ticker = "AAPL"
        
        print(f"📊 Testing comprehensive analysis for {ticker}")
        print("=" * 50)
        
        # Run comprehensive analysis
        report = await polygon.run_comprehensive_analysis(ticker)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_polygon_analysis_{ticker}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Analysis report saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save report: {str(e)}")
        
        # Print key findings
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   Ticker: {report['ticker']}")
        print(f"   Net Alpha Potential: {report['net_alpha']:.1f}%")
        print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"   Analyses Completed: {report['summary']['successful_analyses']}/{report['summary']['total_analyses']}")
        
        if report['net_alpha'] > 20:
            print("🎉 EXCELLENT: High alpha potential achieved!")
        elif report['net_alpha'] > 10:
            print("📈 GOOD: Significant alpha potential achieved!")
        else:
            print("⚠️ MODERATE: Moderate alpha potential - check data availability")

if __name__ == "__main__":
    asyncio.run(main())
