"""
Enhanced Undervalued Agent with Realistic Fundamental Analysis
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf

from .models import UndervaluedOpportunity, ValuationMetrics, FinancialMetrics
from common.data_adapters.yfinance_adapter_fixed import FixedYFinanceAdapter
from common.opportunity_store import OpportunityStore
from common.unified_opportunity_scorer_enhanced import EnhancedUnifiedOpportunityScorer


class EnhancedUndervaluedAgent:
    """
    Enhanced Undervalued Agent with realistic fundamental analysis
    """
    
    def __init__(self):
        self.data_adapter = FixedYFinanceAdapter({})
        self.opportunity_store = OpportunityStore()
        self.scorer = EnhancedUnifiedOpportunityScorer()
        self.min_upside = 0.15  # 15% minimum upside potential
        self.max_pe_ratio = 25  # Maximum P/E ratio for value stocks
        
    async def process(self, universe: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process fundamental analysis to find undervalued opportunities
        """
        try:
            if universe is None:
                universe = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'AMD', 'INTC']
            
            print(f"🔍 Enhanced Value Analysis: {len(universe)} symbols")
            
            opportunities = []
            
            for symbol in universe:
                try:
                    opportunity = await self._analyze_symbol_fundamentals(symbol)
                    if opportunity and opportunity.upside_potential >= self.min_upside:
                        opportunities.append(opportunity)
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    continue
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            # Calculate priority scores
            for opportunity in opportunities:
                opportunity.priority_score = self.scorer.calculate_priority_score(opportunity)
                self.opportunity_store.add_opportunity(opportunity)
            
            opportunities.sort(key=lambda x: x.upside_potential, reverse=True)
            
            return {
                'undervalued_analysis': {
                    'identified_opportunities': [self._opportunity_to_dict(opp) for opp in opportunities],
                    'analysis_summary': {
                        'total_analyzed': len(universe),
                        'opportunities_found': len(opportunities),
                        'average_upside': np.mean([opp.upside_potential for opp in opportunities]) if opportunities else 0,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                },
                'success': True
            }
            
        except Exception as e:
            print(f"Error in enhanced value analysis: {e}")
            return {
                'undervalued_analysis': {
                    'identified_opportunities': [],
                    'analysis_summary': {'error': str(e)}
                },
                'success': False
            }
    
    async def _analyze_symbol_fundamentals(self, symbol: str) -> Optional[UndervaluedOpportunity]:
        """
        Analyze fundamental metrics for a symbol
        """
        try:
            # Get market data
            quote = await self.data_adapter.get_quote(symbol)
            current_price = quote['price']
            
            # Get fundamental data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Calculate valuation metrics
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            ps_ratio = info.get('priceToSalesTrailing12Months', 0)
            market_cap = info.get('marketCap', 0)
            
            # Get financial metrics
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            
            # Calculate fair value using multiple methods
            fair_value_dcf = await self._calculate_dcf_value(ticker, current_price)
            fair_value_pe = await self._calculate_pe_value(ticker, current_price)
            fair_value_pb = await self._calculate_pb_value(ticker, current_price)
            
            # Weighted average fair value
            fair_value = (fair_value_dcf * 0.4 + fair_value_pe * 0.4 + fair_value_pb * 0.2)
            
            # Calculate upside potential
            upside_potential = (fair_value - current_price) / current_price if current_price > 0 else 0
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(info, financials, balance_sheet)
            
            # Create valuation metrics
            valuation_metrics = ValuationMetrics(
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                ps_ratio=ps_ratio,
                fair_value=fair_value,
                upside_potential=upside_potential,
                valuation_methods={
                    'dcf': fair_value_dcf,
                    'pe': fair_value_pe,
                    'pb': fair_value_pb
                }
            )
            
            # Create financial metrics
            financial_metrics = FinancialMetrics(
                revenue_growth=self._calculate_revenue_growth(financials),
                profit_margin=self._calculate_profit_margin(financials),
                debt_to_equity=self._calculate_debt_to_equity(balance_sheet),
                return_on_equity=self._calculate_roe(financials, balance_sheet)
            )
            
            return UndervaluedOpportunity(
                ticker=symbol,
                current_price=current_price,
                fair_value=fair_value,
                upside_potential=upside_potential,
                confidence_score=confidence,
                valuation_metrics=valuation_metrics,
                financial_metrics=financial_metrics,
                analysis_timestamp=datetime.now(),
                entry_reason=f"Undervalued based on fundamental analysis. Fair value: ${fair_value:.2f}, Upside: {upside_potential:.1%}",
                opportunity_type="value_analysis",
                time_horizon="6-12 months",
                agent_type="value_analysis"
            )
            
        except Exception as e:
            print(f"Error analyzing fundamentals for {symbol}: {e}")
            return None
    
    async def _calculate_dcf_value(self, ticker: yf.Ticker, current_price: float) -> float:
        """Calculate DCF value"""
        try:
            # Get financial data
            financials = ticker.financials
            if financials.empty:
                return current_price * 1.1  # 10% premium if no data
            
            # Get free cash flow
            if 'Free Cash Flow' in financials.index:
                fcf = financials.loc['Free Cash Flow'].iloc[0]
            else:
                # Estimate FCF from net income
                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                fcf = net_income * 0.8  # Assume 80% conversion
            
            # Simple DCF with 10% discount rate and 3% growth
            discount_rate = 0.10
            growth_rate = 0.03
            
            # Terminal value
            terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
            
            # Present value
            pv_fcf = fcf / (1 + discount_rate)
            pv_terminal = terminal_value / (1 + discount_rate)
            
            dcf_value = pv_fcf + pv_terminal
            
            return max(dcf_value, current_price * 0.5)  # Minimum 50% of current price
            
        except Exception as e:
            print(f"Error calculating DCF for {ticker.ticker}: {e}")
            return current_price * 1.1
    
    async def _calculate_pe_value(self, ticker: yf.Ticker, current_price: float) -> float:
        """Calculate value based on P/E ratio"""
        try:
            info = ticker.info
            pe_ratio = info.get('trailingPE', 0)
            
            if pe_ratio <= 0:
                return current_price * 1.1
            
            # Get earnings
            financials = ticker.financials
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income'].iloc[0]
            else:
                return current_price * 1.1
            
            # Calculate fair P/E (industry average or 15)
            fair_pe = 15.0
            
            # Fair value based on P/E
            fair_value = (net_income / fair_pe) * 1000000  # Convert to dollars
            
            return max(fair_value, current_price * 0.5)
            
        except Exception as e:
            print(f"Error calculating P/E value for {ticker.ticker}: {e}")
            return current_price * 1.1
    
    async def _calculate_pb_value(self, ticker: yf.Ticker, current_price: float) -> float:
        """Calculate value based on P/B ratio"""
        try:
            info = ticker.info
            pb_ratio = info.get('priceToBook', 0)
            
            if pb_ratio <= 0:
                return current_price * 1.1
            
            # Get book value
            balance_sheet = ticker.balance_sheet
            if 'Total Stockholder Equity' in balance_sheet.index:
                book_value = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            else:
                return current_price * 1.1
            
            # Fair P/B ratio (industry average or 1.5)
            fair_pb = 1.5
            
            # Fair value based on P/B
            fair_value = (book_value / fair_pb) * 1000000  # Convert to dollars
            
            return max(fair_value, current_price * 0.5)
            
        except Exception as e:
            print(f"Error calculating P/B value for {ticker.ticker}: {e}")
            return current_price * 1.1
    
    def _calculate_confidence(self, info: Dict, financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """Calculate confidence based on data quality"""
        confidence = 0.5  # Base confidence
        
        # Data availability bonus
        if not financials.empty:
            confidence += 0.2
        if not balance_sheet.empty:
            confidence += 0.2
        
        # Key metrics availability
        if info.get('trailingPE'):
            confidence += 0.1
        if info.get('priceToBook'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_revenue_growth(self, financials: pd.DataFrame) -> float:
        """Calculate revenue growth rate"""
        try:
            if 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue']
                if len(revenue) >= 2:
                    growth = (revenue.iloc[0] - revenue.iloc[1]) / revenue.iloc[1]
                    return growth
        except:
            pass
        return 0.05  # Default 5% growth
    
    def _calculate_profit_margin(self, financials: pd.DataFrame) -> float:
        """Calculate profit margin"""
        try:
            if 'Net Income' in financials.index and 'Total Revenue' in financials.index:
                net_income = financials.loc['Net Income'].iloc[0]
                revenue = financials.loc['Total Revenue'].iloc[0]
                if revenue > 0:
                    return net_income / revenue
        except:
            pass
        return 0.15  # Default 15% margin
    
    def _calculate_debt_to_equity(self, balance_sheet: pd.DataFrame) -> float:
        """Calculate debt to equity ratio"""
        try:
            if 'Total Liab' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                debt = balance_sheet.loc['Total Liab'].iloc[0]
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                if equity > 0:
                    return debt / equity
        except:
            pass
        return 0.5  # Default 0.5 ratio
    
    def _calculate_roe(self, financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """Calculate return on equity"""
        try:
            if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index:
                net_income = financials.loc['Net Income'].iloc[0]
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                if equity > 0:
                    return net_income / equity
        except:
            pass
        return 0.12  # Default 12% ROE
    
    def _opportunity_to_dict(self, opportunity: UndervaluedOpportunity) -> Dict[str, Any]:
        """Convert opportunity to dictionary"""
        return {
            'ticker': opportunity.ticker,
            'current_price': opportunity.current_price,
            'fair_value': opportunity.fair_value,
            'upside_potential': opportunity.upside_potential,
            'confidence_score': opportunity.confidence_score,
            'pe_ratio': opportunity.valuation_metrics.pe_ratio,
            'pb_ratio': opportunity.valuation_metrics.pb_ratio,
            'analysis_timestamp': opportunity.analysis_timestamp.isoformat()
        }
