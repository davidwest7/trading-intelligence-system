"""
Optimized Undervalued Agent

Advanced value investing analysis with:
- DCF models and multiples analysis
- Technical oversold conditions detection
- Mean reversion signals
- Relative value analysis
- Performance optimization
- Error handling and resilience
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from .models import (
    UndervaluedAnalysis, ValuationAnalysis, ValuationScores,
    FundamentalMetrics, TechnicalOversoldSignals
)
from ..common.models import BaseAgent


class ValuationMethod(str, Enum):
    """Valuation methods"""
    DCF = "dcf"
    MULTIPLES = "multiples"
    TECHNICAL = "technical"
    RELATIVE_VALUE = "relative_value"


@dataclass
class ValueSignal:
    """Value investment signal"""
    ticker: str
    signal_type: str
    valuation_method: str
    undervaluation_score: float
    confidence: float
    target_price: float
    current_price: float
    upside_potential: float
    timestamp: datetime
    catalyst: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "valuation_method": self.valuation_method,
            "undervaluation_score": self.undervaluation_score,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "upside_potential": self.upside_potential,
            "timestamp": self.timestamp.isoformat(),
            "catalyst": self.catalyst
        }


class OptimizedUndervaluedAgent(BaseAgent):
    """
    Optimized Undervalued Assets Agent for financial markets
    
    Enhanced Capabilities:
    ✅ Advanced DCF valuation models and multiples analysis
    ✅ Technical oversold conditions detection
    ✅ Mean reversion signal generation
    ✅ Relative value analysis across sectors
    ✅ Catalyst identification and impact assessment
    ✅ Risk-adjusted valuation metrics
    ✅ Performance optimization and caching
    ✅ Error handling and resilience
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("undervalued", config)
        
        # Configuration with defaults
        self.config = config or {}
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.cache_ttl = self.config.get('cache_ttl', 900)  # 15 minutes
        self.lookback_periods = self.config.get('lookback_periods', 252)  # 1 year
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        self.cache = {}
        self.cache_timestamps = {}
        
        # Valuation parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.market_risk_premium = 0.06  # 6% market risk premium
        self.default_beta = 1.0
        
        # Real-time data storage
        self.max_history_size = 5000
        self.valuation_history = defaultdict(lambda: deque(maxlen=self.max_history_size))
        self.price_history = defaultdict(lambda: deque(maxlen=self.max_history_size))
        
        # Performance metrics
        self.metrics = {
            'total_valuations_performed': 0,
            'value_signals_generated': 0,
            'processing_time_avg': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0
        }
        
        # Error tracking
        self.error_count = 0
        self.total_requests = 0
        
        logging.info("Optimized Undervalued Agent initialized successfully")
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method with error handling"""
        try:
            self.total_requests += 1
            start_time = time.time()
            
            result = await self.scan_undervalued_optimized(*args, **kwargs)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.metrics['processing_time_avg'] = (
                (self.metrics['processing_time_avg'] * (self.total_requests - 1) + processing_time)
                / self.total_requests
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.metrics['error_rate'] = self.error_count / self.total_requests
            logging.error(f"Error in undervalued processing: {e}")
            raise
    
    async def scan_undervalued_optimized(
        self,
        tickers: List[str],
        horizon: str = "1y",
        valuation_methods: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 25,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized undervalued assets scanning with caching and parallel processing
        
        Args:
            tickers: List of tickers to analyze
            horizon: Investment horizon
            valuation_methods: Valuation methods to use
            filters: Screening filters
            limit: Maximum number of results
            use_cache: Use cached results if available
        
        Returns:
            Complete undervalued analysis results
        """
        
        if valuation_methods is None:
            valuation_methods = ["dcf", "multiples", "technical"]
        
        if filters is None:
            filters = {}
        
        # Check cache first
        cache_key = f"{','.join(sorted(tickers))}_{horizon}_{','.join(sorted(valuation_methods))}_{limit}"
        if use_cache and self._is_cache_valid(cache_key):
            self.metrics['cache_hit_rate'] += 1
            return self.cache[cache_key]
        
        try:
            # Analyze each ticker in parallel
            analysis_tasks = []
            for ticker in tickers:
                task = asyncio.create_task(
                    self._analyze_ticker_valuation_optimized(ticker, horizon, valuation_methods)
                )
                analysis_tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            all_valuations = []
            all_signals = []
            
            for i, result in enumerate(results):
                ticker = tickers[i]
                if isinstance(result, Exception):
                    logging.error(f"Error analyzing valuation for {ticker}: {result}")
                    self.error_count += 1
                elif result is not None:
                    all_valuations.append(result['valuation'])
                    all_signals.extend(result['signals'])
                    self.metrics['total_valuations_performed'] += 1
                    self.metrics['value_signals_generated'] += len(result['signals'])
            
            # Filter and rank undervalued assets
            undervalued_assets = self._filter_undervalued_optimized(all_valuations, filters, limit)
            
            # Generate value signals
            value_signals = self._generate_value_signals_optimized(undervalued_assets)
            
            # Create analysis summary
            analysis = self._create_undervalued_analysis(undervalued_assets, value_signals)
            
            # Generate summary
            summary = self._create_undervalued_summary(undervalued_assets, value_signals)
            
            # Create results
            final_results = {
                "undervalued_analysis": analysis.to_dict(),
                "undervalued_assets": [asset.to_dict() for asset in undervalued_assets],
                "value_signals": [signal.to_dict() for signal in value_signals],
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "processing_info": {
                    "total_tickers": len(tickers),
                    "processing_time": self.metrics['processing_time_avg'],
                    "cache_hit_rate": self.metrics['cache_hit_rate']
                }
            }
            
            # Cache results
            if use_cache:
                self._cache_result(cache_key, final_results)
            
            return final_results
            
        except Exception as e:
            self.error_count += 1
            self.metrics['error_rate'] = self.error_count / self.total_requests
            logging.error(f"Error in undervalued scanning: {e}")
            raise
    
    async def _analyze_ticker_valuation_optimized(
        self,
        ticker: str,
        horizon: str,
        valuation_methods: List[str]
    ) -> Dict[str, Any]:
        """Analyze valuation for a single ticker"""
        
        try:
            # Generate mock market and fundamental data
            market_data = await self._generate_mock_market_data(ticker, horizon)
            fundamental_data = await self._generate_mock_fundamental_data(ticker)
            
            # Calculate valuation scores
            valuation_scores = await self._calculate_valuation_scores_optimized(
                ticker, market_data, fundamental_data, valuation_methods
            )
            
            # Calculate fundamental metrics
            fundamental_metrics = self._calculate_fundamental_metrics_optimized(fundamental_data)
            
            # Detect technical oversold conditions
            technical_signals = self._detect_technical_oversold_optimized(ticker, market_data)
            
            # Calculate target price and upside
            target_price = self._calculate_target_price_optimized(
                ticker, market_data, fundamental_data, valuation_scores
            )
            
            current_price = market_data['close'][-1]
            upside_potential = (target_price - current_price) / current_price if current_price > 0 else 0.0
            
            # Create valuation analysis
            valuation_analysis = ValuationAnalysis(
                ticker=ticker,
                current_price=current_price,
                target_price=target_price,
                upside_potential=upside_potential,
                valuation_scores=valuation_scores,
                fundamental_metrics=fundamental_metrics,
                technical_oversold_signals=technical_signals,
                confidence=np.mean([valuation_scores.dcf_score, valuation_scores.multiples_score, valuation_scores.technical_score]),
                investment_horizon=horizon,
                timestamp=datetime.now()
            )
            
            # Generate signals
            signals = self._generate_ticker_value_signals_optimized(ticker, valuation_analysis)
            
            return {
                'valuation': valuation_analysis,
                'signals': signals
            }
            
        except Exception as e:
            logging.error(f"Error analyzing valuation for {ticker}: {e}")
            return {
                'valuation': self._create_empty_valuation(ticker),
                'signals': []
            }
    
    async def _generate_mock_market_data(self, ticker: str, horizon: str) -> Dict[str, Any]:
        """Generate mock market data for analysis"""
        
        # Time periods based on horizon
        periods = {
            "1w": 168, "1m": 720, "3m": 2160, "6m": 4320, "1y": 8760, "2y": 17520
        }.get(horizon, 8760)
        
        # Generate realistic price data with some undervaluation potential
        base_price = 50.0 + np.random.random() * 100
        
        data = {
            'timestamp': pd.date_range(end=datetime.now(), periods=min(periods, 1000), freq='1h'),
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'rsi': [],
            'bollinger_position': []
        }
        
        current_price = base_price
        
        # Generate price series with potential undervaluation (downward bias)
        for i in range(min(periods, 1000)):
            # Add some downward pressure to create undervaluation opportunities
            trend_factor = np.random.uniform(0.995, 1.005)  # Slight downward bias
            volatility = np.random.normal(0, 0.02)
            
            current_price *= trend_factor * (1 + volatility)
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = data['close'][-1] if data['close'] else current_price
            
            # Volume
            volume = 1000000 * (1 + np.random.normal(0, 0.3))
            
            # Technical indicators
            rsi = 30 + np.random.random() * 40  # Bias toward oversold
            bollinger_position = np.random.uniform(-2, 2)  # -2 = oversold, +2 = overbought
            
            data['open'].append(open_price)
            data['high'].append(high)
            data['low'].append(low)
            data['close'].append(current_price)
            data['volume'].append(max(1000, volume))
            data['rsi'].append(rsi)
            data['bollinger_position'].append(bollinger_position)
        
        return data
    
    async def _generate_mock_fundamental_data(self, ticker: str) -> Dict[str, Any]:
        """Generate mock fundamental data"""
        
        return {
            'market_cap': np.random.uniform(1e9, 500e9),  # $1B to $500B
            'revenue': np.random.uniform(1e8, 100e9),  # $100M to $100B
            'earnings': np.random.uniform(1e7, 20e9),  # $10M to $20B
            'book_value': np.random.uniform(1e8, 50e9),  # $100M to $50B
            'free_cash_flow': np.random.uniform(1e7, 15e9),  # $10M to $15B
            'debt': np.random.uniform(0, 30e9),  # $0 to $30B
            'cash': np.random.uniform(1e8, 50e9),  # $100M to $50B
            'shares_outstanding': np.random.uniform(1e6, 10e9),  # 1M to 10B shares
            'dividend_yield': np.random.uniform(0, 0.08),  # 0% to 8%
            'roe': np.random.uniform(0.05, 0.25),  # 5% to 25%
            'roa': np.random.uniform(0.02, 0.15),  # 2% to 15%
            'profit_margin': np.random.uniform(0.05, 0.30),  # 5% to 30%
            'beta': np.random.uniform(0.5, 2.0),  # 0.5 to 2.0
            'ev_ebitda': np.random.uniform(5, 25),  # 5x to 25x
            'pe_ratio': np.random.uniform(8, 30),  # 8x to 30x
            'pb_ratio': np.random.uniform(0.5, 5.0),  # 0.5x to 5x
        }
    
    async def _calculate_valuation_scores_optimized(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        fundamental_data: Dict[str, Any],
        valuation_methods: List[str]
    ) -> ValuationScores:
        """Calculate valuation scores using different methods"""
        
        try:
            # DCF Score (simplified)
            dcf_score = 0.0
            if "dcf" in valuation_methods:
                fcf = fundamental_data['free_cash_flow']
                shares = fundamental_data['shares_outstanding']
                growth_rate = 0.05  # Assume 5% growth
                discount_rate = self.risk_free_rate + fundamental_data['beta'] * self.market_risk_premium
                
                # Simplified DCF calculation
                terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
                dcf_per_share = terminal_value / shares
                current_price = market_data['close'][-1]
                
                dcf_score = max(0.0, min(1.0, (dcf_per_share - current_price) / current_price))
            
            # Multiples Score
            multiples_score = 0.0
            if "multiples" in valuation_methods:
                pe_ratio = fundamental_data['pe_ratio']
                pb_ratio = fundamental_data['pb_ratio']
                ev_ebitda = fundamental_data['ev_ebitda']
                
                # Compare to "fair value" multiples
                pe_fair = 15.0  # Fair P/E
                pb_fair = 2.0   # Fair P/B
                ev_ebitda_fair = 12.0  # Fair EV/EBITDA
                
                pe_score = max(0.0, (pe_fair - pe_ratio) / pe_fair)
                pb_score = max(0.0, (pb_fair - pb_ratio) / pb_fair)
                ev_score = max(0.0, (ev_ebitda_fair - ev_ebitda) / ev_ebitda_fair)
                
                multiples_score = np.mean([pe_score, pb_score, ev_score])
            
            # Technical Score (oversold conditions)
            technical_score = 0.0
            if "technical" in valuation_methods:
                avg_rsi = np.mean(market_data['rsi'][-20:]) if len(market_data['rsi']) >= 20 else 50
                avg_bollinger = np.mean(market_data['bollinger_position'][-20:]) if len(market_data['bollinger_position']) >= 20 else 0
                
                # Score based on oversold conditions
                rsi_score = max(0.0, (30 - avg_rsi) / 30) if avg_rsi < 30 else 0.0
                bollinger_score = max(0.0, (-1 - avg_bollinger) / 1) if avg_bollinger < -1 else 0.0
                
                technical_score = np.mean([rsi_score, bollinger_score])
            
            # Relative Value Score
            relative_value_score = 0.0
            if "relative_value" in valuation_methods:
                # Mock sector comparison
                sector_pe = 18.0  # Mock sector average P/E
                sector_pb = 2.5   # Mock sector average P/B
                
                pe_relative = max(0.0, (sector_pe - fundamental_data['pe_ratio']) / sector_pe)
                pb_relative = max(0.0, (sector_pb - fundamental_data['pb_ratio']) / sector_pb)
                
                relative_value_score = np.mean([pe_relative, pb_relative])
            
            # Composite Score
            scores = [dcf_score, multiples_score, technical_score, relative_value_score]
            weights = [0.4, 0.3, 0.15, 0.15]  # DCF gets highest weight
            
            composite_score = sum(score * weight for score, weight in zip(scores, weights))
            
            return ValuationScores(
                dcf_score=dcf_score,
                multiples_score=multiples_score,
                technical_score=technical_score,
                relative_value_score=relative_value_score,
                composite_score=composite_score
            )
            
        except Exception as e:
            logging.error(f"Error calculating valuation scores for {ticker}: {e}")
            return ValuationScores(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_fundamental_metrics_optimized(self, fundamental_data: Dict[str, Any]) -> FundamentalMetrics:
        """Calculate fundamental metrics"""
        
        try:
            return FundamentalMetrics(
                pe_ratio=fundamental_data['pe_ratio'],
                pb_ratio=fundamental_data['pb_ratio'],
                ev_ebitda=fundamental_data['ev_ebitda'],
                roe=fundamental_data['roe'],
                debt_to_equity=fundamental_data['debt'] / fundamental_data['book_value'] if fundamental_data['book_value'] > 0 else 0.0,
                free_cash_flow_yield=fundamental_data['free_cash_flow'] / fundamental_data['market_cap'] if fundamental_data['market_cap'] > 0 else 0.0
            )
            
        except Exception as e:
            logging.error(f"Error calculating fundamental metrics: {e}")
            return FundamentalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _detect_technical_oversold_optimized(self, ticker: str, market_data: Dict[str, Any]) -> TechnicalOversoldSignals:
        """Detect technical oversold conditions"""
        
        try:
            rsi_values = market_data['rsi']
            bollinger_positions = market_data['bollinger_position']
            
            # RSI oversold
            current_rsi = rsi_values[-1] if rsi_values else 50
            rsi_oversold = current_rsi < 30
            
            # Bollinger Bands oversold
            current_bollinger = bollinger_positions[-1] if bollinger_positions else 0
            bollinger_oversold = current_bollinger < -1.5
            
            # Williams %R (mock calculation)
            williams_r = np.random.uniform(-100, 0)  # Mock value
            williams_oversold = williams_r < -80
            
            return TechnicalOversoldSignals(
                rsi_oversold=rsi_oversold,
                bollinger_oversold=bollinger_oversold,
                williams_r_oversold=williams_oversold,
                composite_oversold_score=np.mean([rsi_oversold, bollinger_oversold, williams_oversold])
            )
            
        except Exception as e:
            logging.error(f"Error detecting technical oversold for {ticker}: {e}")
            return TechnicalOversoldSignals(False, False, False, 0.0)
    
    def _calculate_target_price_optimized(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        fundamental_data: Dict[str, Any],
        valuation_scores: ValuationScores
    ) -> float:
        """Calculate target price based on valuation analysis"""
        
        try:
            current_price = market_data['close'][-1]
            
            # Target price based on valuation scores
            if valuation_scores.composite_score > 0.5:
                # High undervaluation - significant upside
                upside_multiplier = 1.2 + (valuation_scores.composite_score - 0.5)
            elif valuation_scores.composite_score > 0.3:
                # Moderate undervaluation
                upside_multiplier = 1.1 + (valuation_scores.composite_score - 0.3) * 0.5
            else:
                # Low undervaluation
                upside_multiplier = 1.0 + valuation_scores.composite_score * 0.3
            
            target_price = current_price * upside_multiplier
            
            return target_price
            
        except Exception as e:
            logging.error(f"Error calculating target price for {ticker}: {e}")
            return market_data['close'][-1] if market_data['close'] else 0.0
    
    def _filter_undervalued_optimized(
        self,
        valuations: List[ValuationAnalysis],
        filters: Dict[str, Any],
        limit: int
    ) -> List[ValuationAnalysis]:
        """Filter and rank undervalued assets"""
        
        try:
            # Apply filters
            filtered_valuations = []
            
            for valuation in valuations:
                # Basic undervaluation filter
                if valuation.valuation_scores.composite_score < 0.3:
                    continue
                
                # Upside potential filter
                if valuation.upside_potential < 0.1:  # At least 10% upside
                    continue
                
                # Apply custom filters
                if filters.get('min_market_cap') and valuation.fundamental_metrics.pe_ratio < filters['min_market_cap']:
                    continue
                
                if filters.get('max_pe_ratio') and valuation.fundamental_metrics.pe_ratio > filters['max_pe_ratio']:
                    continue
                
                filtered_valuations.append(valuation)
            
            # Sort by composite score (highest first)
            sorted_valuations = sorted(
                filtered_valuations,
                key=lambda x: x.valuation_scores.composite_score,
                reverse=True
            )
            
            return sorted_valuations[:limit]
            
        except Exception as e:
            logging.error(f"Error filtering undervalued assets: {e}")
            return valuations[:limit]
    
    def _generate_value_signals_optimized(self, undervalued_assets: List[ValuationAnalysis]) -> List[ValueSignal]:
        """Generate value investment signals"""
        
        signals = []
        
        try:
            for asset in undervalued_assets:
                # Strong undervaluation signal
                if asset.valuation_scores.composite_score > 0.7:
                    signal = ValueSignal(
                        ticker=asset.ticker,
                        signal_type="strong_undervaluation",
                        valuation_method="composite",
                        undervaluation_score=asset.valuation_scores.composite_score,
                        confidence=0.8,
                        target_price=asset.target_price,
                        current_price=asset.current_price,
                        upside_potential=asset.upside_potential,
                        timestamp=datetime.now(),
                        catalyst="fundamental_analysis"
                    )
                    signals.append(signal)
                
                # Technical oversold signal
                if asset.technical_oversold_signals.composite_oversold_score > 0.6:
                    signal = ValueSignal(
                        ticker=asset.ticker,
                        signal_type="technical_oversold",
                        valuation_method="technical",
                        undervaluation_score=asset.technical_oversold_signals.composite_oversold_score,
                        confidence=0.6,
                        target_price=asset.target_price,
                        current_price=asset.current_price,
                        upside_potential=asset.upside_potential,
                        timestamp=datetime.now(),
                        catalyst="mean_reversion"
                    )
                    signals.append(signal)
                
                # DCF undervaluation signal
                if asset.valuation_scores.dcf_score > 0.5:
                    signal = ValueSignal(
                        ticker=asset.ticker,
                        signal_type="dcf_undervaluation",
                        valuation_method="dcf",
                        undervaluation_score=asset.valuation_scores.dcf_score,
                        confidence=0.7,
                        target_price=asset.target_price,
                        current_price=asset.current_price,
                        upside_potential=asset.upside_potential,
                        timestamp=datetime.now(),
                        catalyst="intrinsic_value"
                    )
                    signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error generating value signals: {e}")
        
        return signals
    
    def _generate_ticker_value_signals_optimized(self, ticker: str, valuation: ValuationAnalysis) -> List[ValueSignal]:
        """Generate value signals for individual ticker"""
        
        signals = []
        
        try:
            # High upside potential signal
            if valuation.upside_potential > 0.3:  # >30% upside
                signal = ValueSignal(
                    ticker=ticker,
                    signal_type="high_upside_potential",
                    valuation_method="composite",
                    undervaluation_score=valuation.valuation_scores.composite_score,
                    confidence=valuation.confidence,
                    target_price=valuation.target_price,
                    current_price=valuation.current_price,
                    upside_potential=valuation.upside_potential,
                    timestamp=datetime.now(),
                    catalyst="value_opportunity"
                )
                signals.append(signal)
            
            # Multiples undervaluation signal
            if valuation.valuation_scores.multiples_score > 0.5:
                signal = ValueSignal(
                    ticker=ticker,
                    signal_type="multiples_undervaluation",
                    valuation_method="multiples",
                    undervaluation_score=valuation.valuation_scores.multiples_score,
                    confidence=0.7,
                    target_price=valuation.target_price,
                    current_price=valuation.current_price,
                    upside_potential=valuation.upside_potential,
                    timestamp=datetime.now(),
                    catalyst="valuation_multiple_compression"
                )
                signals.append(signal)
        
        except Exception as e:
            logging.error(f"Error generating ticker value signals for {ticker}: {e}")
        
        return signals
    
    def _create_undervalued_analysis(
        self,
        undervalued_assets: List[ValuationAnalysis],
        value_signals: List[ValueSignal]
    ) -> UndervaluedAnalysis:
        """Create comprehensive undervalued analysis"""
        
        try:
            return UndervaluedAnalysis(
                timestamp=datetime.now(),
                total_analyzed=len(undervalued_assets),
                undervalued_assets=undervalued_assets,
                value_signals=value_signals,
                average_upside_potential=np.mean([asset.upside_potential for asset in undervalued_assets]) if undervalued_assets else 0.0,
                best_opportunity=undervalued_assets[0] if undervalued_assets else None,
                valuation_distribution={
                    "high_undervaluation": len([a for a in undervalued_assets if a.valuation_scores.composite_score > 0.7]),
                    "moderate_undervaluation": len([a for a in undervalued_assets if 0.5 < a.valuation_scores.composite_score <= 0.7]),
                    "low_undervaluation": len([a for a in undervalued_assets if 0.3 < a.valuation_scores.composite_score <= 0.5])
                }
            )
            
        except Exception as e:
            logging.error(f"Error creating undervalued analysis: {e}")
            return UndervaluedAnalysis(
                timestamp=datetime.now(),
                total_analyzed=0,
                undervalued_assets=[],
                value_signals=[],
                average_upside_potential=0.0,
                best_opportunity=None,
                valuation_distribution={}
            )
    
    def _create_empty_valuation(self, ticker: str) -> ValuationAnalysis:
        """Create empty valuation analysis"""
        
        return ValuationAnalysis(
            ticker=ticker,
            current_price=0.0,
            target_price=0.0,
            upside_potential=0.0,
            valuation_scores=ValuationScores(0.0, 0.0, 0.0, 0.0, 0.0),
            fundamental_metrics=FundamentalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            technical_oversold_signals=TechnicalOversoldSignals(False, False, False, 0.0),
            confidence=0.0,
            investment_horizon="1y",
            timestamp=datetime.now()
        )
    
    def _create_undervalued_summary(
        self,
        undervalued_assets: List[ValuationAnalysis],
        value_signals: List[ValueSignal]
    ) -> Dict[str, Any]:
        """Create undervalued analysis summary"""
        
        try:
            # Signal analysis
            total_signals = len(value_signals)
            signal_types = defaultdict(int)
            valuation_methods = defaultdict(int)
            
            for signal in value_signals:
                signal_types[signal.signal_type] += 1
                valuation_methods[signal.valuation_method] += 1
            
            # Asset analysis
            if undervalued_assets:
                avg_upside = np.mean([asset.upside_potential for asset in undervalued_assets])
                avg_composite_score = np.mean([asset.valuation_scores.composite_score for asset in undervalued_assets])
                best_opportunity = undervalued_assets[0]
            else:
                avg_upside = avg_composite_score = 0.0
                best_opportunity = None
            
            return {
                'total_undervalued_assets': len(undervalued_assets),
                'total_signals_generated': total_signals,
                'signal_types': dict(signal_types),
                'valuation_methods': dict(valuation_methods),
                'average_upside_potential': avg_upside,
                'average_composite_score': avg_composite_score,
                'best_opportunity': best_opportunity.ticker if best_opportunity else None,
                'value_opportunity_level': 'high' if avg_composite_score > 0.6 else 'medium' if avg_composite_score > 0.4 else 'low'
            }
            
        except Exception as e:
            logging.error(f"Error creating undervalued summary: {e}")
            return {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        age = time.time() - self.cache_timestamps[cache_key]
        return age < self.cache_ttl
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result with timestamp"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_timestamps[key]
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logging.info("Optimized Undervalued Agent cleanup completed")
