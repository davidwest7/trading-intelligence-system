"""
Real Data Undervalued Agent
Uses Polygon.io adapter for financial data and valuation metrics
"""
import asyncio
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

# Add current directory to path
sys.path.append('.')
from common.models import BaseAgent
from common.data_adapters.polygon_adapter import PolygonAdapter

class RealDataUndervaluedAgent(BaseAgent):
    """Undervalued Agent with real market data from Polygon.io"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataUndervaluedAgent", config)
        self.polygon_adapter = PolygonAdapter(config)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        tickers = kwargs.get('tickers', args[0] if args else ['AAPL', 'TSLA', 'MSFT', 'GOOGL'])
        return await self.analyze_undervalued_stocks(tickers, **kwargs)
    
    async def analyze_undervalued_stocks(self, tickers: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze undervalued stocks using real market data"""
        print(f"💎 Real Data Undervalued Agent: Analyzing {len(tickers)} tickers for value opportunities")
        
        results = {}
        
        for ticker in tickers:
            try:
                # Use cache if fresh
                if ticker in self.cache and time.time() - self.cache[ticker]['timestamp'] < self.cache_ttl:
                    results[ticker] = self.cache[ticker]['data']
                    continue

                # Fetch financials and quote concurrently
                financials, quote = await asyncio.gather(
                    self.polygon_adapter.get_financial_statements(ticker),
                    self.polygon_adapter.get_real_time_quote(ticker)
                )

                # Compute valuation locally to avoid duplicate adapter calls
                valuation = self._compute_valuation_metrics_from(financials, quote, ticker)
                
                # Analyze value metrics
                value_analysis = await self._analyze_value_metrics(
                    ticker, financials, valuation, quote
                )
                
                results[ticker] = value_analysis
                # Update cache
                self.cache[ticker] = {
                    'data': value_analysis,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {ticker}: {e}")
                results[ticker] = self._create_empty_value_analysis(ticker)
        
        # Generate overall value signals
        overall_value = await self._generate_overall_value_signals(results)
        
        return {
            'timestamp': datetime.now(),
            'tickers_analyzed': len(tickers),
            'value_analysis': results,
            'overall_value': overall_value,
            'data_source': 'Polygon.io (Real Market Data)'
        }
    
    async def _analyze_value_metrics(self, ticker: str, financials: Dict[str, Any],
                                   valuation: Dict[str, Any], quote: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze value metrics from real data"""
        
        analysis = {
            'ticker': ticker,
            'current_price': quote['price'],
            'timestamp': datetime.now()
        }
        
        # Financial metrics
        analysis['revenue'] = financials.get('revenue', 0)
        analysis['net_income'] = financials.get('net_income', 0)
        analysis['total_assets'] = financials.get('total_assets', 0)
        analysis['total_liabilities'] = financials.get('total_liabilities', 0)
        analysis['cash_flow'] = financials.get('cash_flow', 0)
        
        # Valuation metrics
        analysis['pe_ratio'] = valuation.get('pe_ratio', 0)
        analysis['pb_ratio'] = valuation.get('pb_ratio', 0)
        analysis['market_cap'] = valuation.get('market_cap', 0)
        analysis['enterprise_value'] = valuation.get('enterprise_value', 0)
        
        # Calculate additional ratios
        analysis['debt_to_equity'] = analysis['total_liabilities'] / (analysis['total_assets'] - analysis['total_liabilities']) if (analysis['total_assets'] - analysis['total_liabilities']) > 0 else 0
        analysis['return_on_equity'] = analysis['net_income'] / (analysis['total_assets'] - analysis['total_liabilities']) if (analysis['total_assets'] - analysis['total_liabilities']) > 0 else 0
        analysis['profit_margin'] = analysis['net_income'] / analysis['revenue'] if analysis['revenue'] > 0 else 0
        
        # Value signals
        analysis['value_signals'] = self._generate_value_signals(analysis)
        
        # Value score
        analysis['value_score'] = self._calculate_value_score(analysis)
        
        return analysis

    def _compute_valuation_metrics_from(self, financials: Dict[str, Any], quote: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Compute valuation metrics locally to avoid redundant API calls.
        Note: Uses simplified proxies consistent with adapter behavior.
        """
        try:
            price = float(quote.get('price', 0.0) or 0.0)
            net_income = float(financials.get('net_income', 0) or 0)
            total_assets = float(financials.get('total_assets', 0) or 0)
            total_liabilities = float(financials.get('total_liabilities', 0) or 0)

            pe_ratio = price / (net_income / 1_000_000) if price > 0 and net_income > 0 else 0.0
            pb_ratio = price / (total_assets / 1_000_000) if price > 0 and total_assets > 0 else 0.0
            market_cap = price * 1_000_000 if price > 0 else 0
            enterprise_value = market_cap + total_liabilities if market_cap > 0 else 0

            return {
                'symbol': ticker,
                'price': price,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'market_cap': market_cap,
                'enterprise_value': enterprise_value,
                'timestamp': datetime.now()
            }
        except Exception:
            return {
                'symbol': ticker,
                'price': 0.0,
                'pe_ratio': 0.0,
                'pb_ratio': 0.0,
                'market_cap': 0,
                'enterprise_value': 0,
                'timestamp': datetime.now()
            }
    
    def _generate_value_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate value-based trading signals"""
        signals = []
        
        # P/E ratio signals
        if analysis['pe_ratio'] > 0 and analysis['pe_ratio'] < 15:
            signals.append({
                'type': 'LOW_PE_RATIO',
                'strength': 'strong',
                'message': f"Low P/E ratio ({analysis['pe_ratio']:.1f}) - potentially undervalued"
            })
        elif analysis['pe_ratio'] > 25:
            signals.append({
                'type': 'HIGH_PE_RATIO',
                'strength': 'medium',
                'message': f"High P/E ratio ({analysis['pe_ratio']:.1f}) - potentially overvalued"
            })
        
        # P/B ratio signals
        if analysis['pb_ratio'] > 0 and analysis['pb_ratio'] < 1.5:
            signals.append({
                'type': 'LOW_PB_RATIO',
                'strength': 'strong',
                'message': f"Low P/B ratio ({analysis['pb_ratio']:.2f}) - trading below book value"
            })
        
        # Debt signals
        if analysis['debt_to_equity'] < 0.5:
            signals.append({
                'type': 'LOW_DEBT',
                'strength': 'medium',
                'message': f"Low debt-to-equity ratio ({analysis['debt_to_equity']:.2f}) - strong balance sheet"
            })
        elif analysis['debt_to_equity'] > 1.0:
            signals.append({
                'type': 'HIGH_DEBT',
                'strength': 'medium',
                'message': f"High debt-to-equity ratio ({analysis['debt_to_equity']:.2f}) - financial risk"
            })
        
        # Profitability signals
        if analysis['profit_margin'] > 0.15:
            signals.append({
                'type': 'HIGH_PROFIT_MARGIN',
                'strength': 'strong',
                'message': f"High profit margin ({analysis['profit_margin']:.1%}) - strong profitability"
            })
        
        if analysis['return_on_equity'] > 0.15:
            signals.append({
                'type': 'HIGH_ROE',
                'strength': 'strong',
                'message': f"High ROE ({analysis['return_on_equity']:.1%}) - efficient use of capital"
            })
        
        # Cash flow signals
        if analysis['cash_flow'] > analysis['net_income'] * 1.2:
            signals.append({
                'type': 'STRONG_CASH_FLOW',
                'strength': 'medium',
                'message': f"Strong cash flow - good operational efficiency"
            })
        
        return signals
    
    def _calculate_value_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate a composite value score (0-100)"""
        score = 50  # Base score
        
        # P/E ratio scoring
        if analysis['pe_ratio'] > 0:
            if analysis['pe_ratio'] < 10:
                score += 20
            elif analysis['pe_ratio'] < 15:
                score += 10
            elif analysis['pe_ratio'] > 25:
                score -= 10
        
        # P/B ratio scoring
        if analysis['pb_ratio'] > 0:
            if analysis['pb_ratio'] < 1.0:
                score += 15
            elif analysis['pb_ratio'] < 1.5:
                score += 5
            elif analysis['pb_ratio'] > 3.0:
                score -= 10
        
        # Debt scoring
        if analysis['debt_to_equity'] < 0.3:
            score += 10
        elif analysis['debt_to_equity'] > 1.0:
            score -= 10
        
        # Profitability scoring
        if analysis['profit_margin'] > 0.15:
            score += 10
        elif analysis['profit_margin'] < 0.05:
            score -= 10
        
        # ROE scoring
        if analysis['return_on_equity'] > 0.15:
            score += 10
        elif analysis['return_on_equity'] < 0.05:
            score -= 10
        
        return max(0, min(100, score))
    
    async def _generate_overall_value_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall value signals"""
        high_value_count = 0
        low_value_count = 0
        total_value_score = 0
        value_opportunities = []
        
        for ticker, analysis in results.items():
            value_score = analysis.get('value_score', 50)
            total_value_score += value_score
            
            if value_score > 70:
                high_value_count += 1
                value_opportunities.append({
                    'ticker': ticker,
                    'value_score': value_score,
                    'pe_ratio': analysis.get('pe_ratio', 0),
                    'pb_ratio': analysis.get('pb_ratio', 0)
                })
            elif value_score < 30:
                low_value_count += 1
        
        avg_value_score = total_value_score / len(results) if results else 50
        
        # Sort opportunities by value score
        value_opportunities.sort(key=lambda x: x['value_score'], reverse=True)
        
        return {
            'avg_value_score': avg_value_score,
            'high_value_count': high_value_count,
            'low_value_count': low_value_count,
            'value_opportunities': value_opportunities[:5],
            'market_value_regime': 'undervalued' if avg_value_score > 60 else 'overvalued' if avg_value_score < 40 else 'fair_value'
        }
    
    def _create_empty_value_analysis(self, ticker: str) -> Dict[str, Any]:
        """Create empty value analysis for failed tickers"""
        return {
            'ticker': ticker,
            'current_price': 0.0,
            'revenue': 0,
            'net_income': 0,
            'total_assets': 0,
            'total_liabilities': 0,
            'cash_flow': 0,
            'pe_ratio': 0.0,
            'pb_ratio': 0.0,
            'market_cap': 0,
            'enterprise_value': 0,
            'debt_to_equity': 0.0,
            'return_on_equity': 0.0,
            'profit_margin': 0.0,
            'value_signals': [],
            'value_score': 50.0,
            'timestamp': datetime.now()
        }
