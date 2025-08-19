"""
Options Surface Analysis and Greeks Calculation
Advanced options analytics for insider trading detection and flow analysis
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import math
from scipy.stats import norm
from scipy.optimize import minimize


class OptionType(str, Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Individual option contract"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "option_type": self.option_type.value,
            "last_price": self.last_price,
            "bid": self.bid,
            "ask": self.ask,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptionsSurface:
    """Complete options surface for a symbol"""
    symbol: str
    underlying_price: float
    timestamp: datetime
    risk_free_rate: float
    dividend_yield: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "underlying_price": self.underlying_price,
            "timestamp": self.timestamp.isoformat(),
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "calls": [call.to_dict() for call in self.calls],
            "puts": [put.to_dict() for put in self.puts]
        }


class OptionsSurfaceAnalyzer:
    """
    Advanced options surface analysis for insider trading detection
    
    Features:
    - Implied volatility surface modeling
    - Greeks calculation and analysis
    - Options flow analysis
    - Volatility skew and term structure
    - Insider activity detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration
        self.min_volume = self.config.get('min_volume', 10)
        self.min_open_interest = self.config.get('min_open_interest', 100)
        self.vol_surface_points = self.config.get('vol_surface_points', 50)
        
        # Storage
        self.surface_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.flow_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def analyze_options_surface(self, surface: OptionsSurface) -> Dict[str, float]:
        """Analyze options surface for trading signals"""
        try:
            features = {}
            
            # Basic surface metrics
            features.update(await self._extract_basic_metrics(surface))
            
            # Volatility surface analysis
            features.update(await self._analyze_volatility_surface(surface))
            
            # Greeks analysis
            features.update(await self._analyze_greeks(surface))
            
            # Options flow analysis
            features.update(await self._analyze_options_flow(surface))
            
            # Insider activity detection
            features.update(await self._detect_insider_activity(surface))
            
            # Store for history
            self.surface_history[surface.symbol].append(surface)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing options surface: {e}")
            return {}
    
    async def _extract_basic_metrics(self, surface: OptionsSurface) -> Dict[str, float]:
        """Extract basic options surface metrics"""
        try:
            features = {}
            
            # Filter liquid options
            liquid_calls = [c for c in surface.calls 
                          if c.volume >= self.min_volume and c.open_interest >= self.min_open_interest]
            liquid_puts = [p for p in surface.puts 
                          if p.volume >= self.min_volume and p.open_interest >= self.min_open_interest]
            
            if not liquid_calls or not liquid_puts:
                return features
            
            # Basic statistics
            features["total_call_volume"] = sum(c.volume for c in liquid_calls)
            features["total_put_volume"] = sum(p.volume for p in liquid_puts)
            features["put_call_volume_ratio"] = features["total_put_volume"] / features["total_call_volume"] if features["total_call_volume"] > 0 else 0
            
            features["total_call_oi"] = sum(c.open_interest for c in liquid_calls)
            features["total_put_oi"] = sum(p.open_interest for p in liquid_puts)
            features["put_call_oi_ratio"] = features["total_put_oi"] / features["total_call_oi"] if features["total_call_oi"] > 0 else 0
            
            # Average implied volatility
            features["avg_call_iv"] = np.mean([c.implied_volatility for c in liquid_calls])
            features["avg_put_iv"] = np.mean([p.implied_volatility for p in liquid_puts])
            features["iv_skew"] = features["avg_put_iv"] - features["avg_call_iv"]
            
            # Moneyness analysis
            atm_calls = [c for c in liquid_calls if 0.95 <= c.strike / surface.underlying_price <= 1.05]
            atm_puts = [p for p in liquid_puts if 0.95 <= p.strike / surface.underlying_price <= 1.05]
            
            if atm_calls and atm_puts:
                features["atm_call_iv"] = np.mean([c.implied_volatility for c in atm_calls])
                features["atm_put_iv"] = np.mean([p.implied_volatility for p in atm_puts])
                features["atm_iv_skew"] = features["atm_put_iv"] - features["atm_call_iv"]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting basic metrics: {e}")
            return {}
    
    async def _analyze_volatility_surface(self, surface: OptionsSurface) -> Dict[str, float]:
        """Analyze implied volatility surface"""
        try:
            features = {}
            
            # Group options by expiry
            expiry_groups = defaultdict(lambda: {"calls": [], "puts": []})
            
            for call in surface.calls:
                if call.volume >= self.min_volume:
                    expiry_groups[call.expiry]["calls"].append(call)
            
            for put in surface.puts:
                if put.volume >= self.min_volume:
                    expiry_groups[put.expiry]["puts"].append(put)
            
            # Analyze term structure
            term_structure = []
            for expiry, options in expiry_groups.items():
                if options["calls"] and options["puts"]:
                    days_to_expiry = (expiry - surface.timestamp).days
                    avg_iv = (np.mean([c.implied_volatility for c in options["calls"]]) + 
                             np.mean([p.implied_volatility for p in options["puts"]])) / 2
                    term_structure.append((days_to_expiry, avg_iv))
            
            if len(term_structure) > 1:
                term_structure.sort(key=lambda x: x[0])
                days, ivs = zip(*term_structure)
                
                # Term structure slope
                if len(days) > 1:
                    slope = np.polyfit(days, ivs, 1)[0]
                    features["vol_term_slope"] = slope
                    features["vol_term_curvature"] = np.polyfit(days, ivs, 2)[0]
                
                # Volatility of volatility
                features["vol_of_vol"] = np.std(ivs)
            
            # Analyze skew across different expiries
            skew_by_expiry = []
            for expiry, options in expiry_groups.items():
                if options["calls"] and options["puts"]:
                    call_iv = np.mean([c.implied_volatility for c in options["calls"]])
                    put_iv = np.mean([p.implied_volatility for p in options["puts"]])
                    skew = put_iv - call_iv
                    skew_by_expiry.append(skew)
            
            if skew_by_expiry:
                features["skew_mean"] = np.mean(skew_by_expiry)
                features["skew_std"] = np.std(skew_by_expiry)
                features["skew_trend"] = np.polyfit(range(len(skew_by_expiry)), skew_by_expiry, 1)[0]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility surface: {e}")
            return {}
    
    async def _analyze_greeks(self, surface: OptionsSurface) -> Dict[str, float]:
        """Analyze options Greeks"""
        try:
            features = {}
            
            # Aggregate Greeks across all options
            all_options = surface.calls + surface.puts
            liquid_options = [opt for opt in all_options 
                            if opt.volume >= self.min_volume and opt.open_interest >= self.min_open_interest]
            
            if not liquid_options:
                return features
            
            # Volume-weighted Greeks
            total_volume = sum(opt.volume for opt in liquid_options)
            
            if total_volume > 0:
                features["vw_delta"] = sum(opt.delta * opt.volume for opt in liquid_options) / total_volume
                features["vw_gamma"] = sum(opt.gamma * opt.volume for opt in liquid_options) / total_volume
                features["vw_theta"] = sum(opt.theta * opt.volume for opt in liquid_options) / total_volume
                features["vw_vega"] = sum(opt.vega * opt.volume for opt in liquid_options) / total_volume
                features["vw_rho"] = sum(opt.rho * opt.volume for opt in liquid_options) / total_volume
            
            # OI-weighted Greeks
            total_oi = sum(opt.open_interest for opt in liquid_options)
            
            if total_oi > 0:
                features["oiw_delta"] = sum(opt.delta * opt.open_interest for opt in liquid_options) / total_oi
                features["oiw_gamma"] = sum(opt.gamma * opt.open_interest for opt in liquid_options) / total_oi
                features["oiw_theta"] = sum(opt.theta * opt.open_interest for opt in liquid_options) / total_oi
                features["oiw_vega"] = sum(opt.vega * opt.open_interest for opt in liquid_options) / total_oi
                features["oiw_rho"] = sum(opt.rho * opt.open_interest for opt in liquid_options) / total_oi
            
            # Greeks by moneyness
            atm_options = [opt for opt in liquid_options 
                          if 0.95 <= opt.strike / surface.underlying_price <= 1.05]
            itm_options = [opt for opt in liquid_options 
                          if opt.strike / surface.underlying_price < 0.95]
            otm_options = [opt for opt in liquid_options 
                          if opt.strike / surface.underlying_price > 1.05]
            
            if atm_options:
                features["atm_gamma"] = np.mean([opt.gamma for opt in atm_options])
                features["atm_vega"] = np.mean([opt.vega for opt in atm_options])
            
            if itm_options:
                features["itm_delta"] = np.mean([opt.delta for opt in itm_options])
                features["itm_theta"] = np.mean([opt.theta for opt in itm_options])
            
            if otm_options:
                features["otm_delta"] = np.mean([opt.delta for opt in otm_options])
                features["otm_vega"] = np.mean([opt.vega for opt in otm_options])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing Greeks: {e}")
            return {}
    
    async def _analyze_options_flow(self, surface: OptionsSurface) -> Dict[str, float]:
        """Analyze options flow patterns"""
        try:
            features = {}
            
            # Unusual activity detection
            liquid_calls = [c for c in surface.calls 
                          if c.volume >= self.min_volume and c.open_interest >= self.min_open_interest]
            liquid_puts = [p for p in surface.puts 
                          if p.volume >= self.min_volume and p.open_interest >= self.min_open_interest]
            
            if not liquid_calls or not liquid_puts:
                return features
            
            # Volume vs OI ratios
            call_volume_oi_ratios = [c.volume / c.open_interest for c in liquid_calls if c.open_interest > 0]
            put_volume_oi_ratios = [p.volume / p.open_interest for p in liquid_puts if p.open_interest > 0]
            
            if call_volume_oi_ratios:
                features["avg_call_volume_oi_ratio"] = np.mean(call_volume_oi_ratios)
                features["max_call_volume_oi_ratio"] = max(call_volume_oi_ratios)
            
            if put_volume_oi_ratios:
                features["avg_put_volume_oi_ratio"] = np.mean(put_volume_oi_ratios)
                features["max_put_volume_oi_ratio"] = max(put_volume_oi_ratios)
            
            # Large trades detection
            avg_call_volume = np.mean([c.volume for c in liquid_calls])
            avg_put_volume = np.mean([p.volume for p in liquid_puts])
            
            large_call_trades = sum(1 for c in liquid_calls if c.volume > 3 * avg_call_volume)
            large_put_trades = sum(1 for p in liquid_puts if p.volume > 3 * avg_put_volume)
            
            features["large_call_trades"] = large_call_trades
            features["large_put_trades"] = large_put_trades
            features["large_trades_total"] = large_call_trades + large_put_trades
            
            # Bid-ask spread analysis
            call_spreads = [(c.ask - c.bid) / c.bid for c in liquid_calls if c.bid > 0]
            put_spreads = [(p.ask - p.bid) / p.bid for p in liquid_puts if p.bid > 0]
            
            if call_spreads:
                features["avg_call_spread"] = np.mean(call_spreads)
                features["call_spread_std"] = np.std(call_spreads)
            
            if put_spreads:
                features["avg_put_spread"] = np.mean(put_spreads)
                features["put_spread_std"] = np.std(put_spreads)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing options flow: {e}")
            return {}
    
    async def _detect_insider_activity(self, surface: OptionsSurface) -> Dict[str, float]:
        """Detect potential insider trading activity"""
        try:
            features = {}
            
            # Compare with historical patterns
            if len(self.surface_history[surface.symbol]) > 5:
                recent_surfaces = list(self.surface_history[surface.symbol])[-5:]
                
                # Volume anomaly detection
                current_put_call_ratio = features.get("put_call_volume_ratio", 0)
                historical_ratios = []
                
                for hist_surface in recent_surfaces:
                    hist_calls = [c for c in hist_surface.calls if c.volume >= self.min_volume]
                    hist_puts = [p for p in hist_surface.puts if p.volume >= self.min_volume]
                    
                    if hist_calls:
                        hist_ratio = sum(p.volume for p in hist_puts) / sum(c.volume for c in hist_calls)
                        historical_ratios.append(hist_ratio)
                
                if historical_ratios:
                    avg_ratio = np.mean(historical_ratios)
                    ratio_std = np.std(historical_ratios)
                    
                    if ratio_std > 0:
                        z_score = (current_put_call_ratio - avg_ratio) / ratio_std
                        features["put_call_ratio_zscore"] = z_score
                        features["volume_anomaly"] = abs(z_score)
                
                # IV skew anomaly
                current_skew = features.get("iv_skew", 0)
                historical_skews = []
                
                for hist_surface in recent_surfaces:
                    hist_calls = [c for c in hist_surface.calls if c.volume >= self.min_volume]
                    hist_puts = [p for p in hist_surface.puts if p.volume >= self.min_volume]
                    
                    if hist_calls and hist_puts:
                        hist_call_iv = np.mean([c.implied_volatility for c in hist_calls])
                        hist_put_iv = np.mean([p.implied_volatility for p in hist_puts])
                        hist_skew = hist_put_iv - hist_call_iv
                        historical_skews.append(hist_skew)
                
                if historical_skews:
                    avg_skew = np.mean(historical_skews)
                    skew_std = np.std(historical_skews)
                    
                    if skew_std > 0:
                        skew_z_score = (current_skew - avg_skew) / skew_std
                        features["skew_anomaly"] = abs(skew_z_score)
            
            # Large OTM put activity (potential hedging)
            otm_puts = [p for p in surface.puts 
                       if p.strike / surface.underlying_price > 1.05 and p.volume >= self.min_volume]
            
            if otm_puts:
                total_otm_put_volume = sum(p.volume for p in otm_puts)
                total_put_volume = sum(p.volume for p in surface.puts if p.volume >= self.min_volume)
                
                if total_put_volume > 0:
                    features["otm_put_volume_ratio"] = total_otm_put_volume / total_put_volume
            
            # Expiry concentration (potential event-driven activity)
            expiry_volumes = defaultdict(int)
            for opt in surface.calls + surface.puts:
                if opt.volume >= self.min_volume:
                    expiry_volumes[opt.expiry] += opt.volume
            
            if expiry_volumes:
                total_volume = sum(expiry_volumes.values())
                max_expiry_volume = max(expiry_volumes.values())
                features["expiry_concentration"] = max_expiry_volume / total_volume if total_volume > 0 else 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error detecting insider activity: {e}")
            return {}
    
    async def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: OptionType, q: float = 0) -> Dict[str, float]:
        """Calculate Black-Scholes Greeks"""
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == OptionType.CALL:
                delta = np.exp(-q * T) * norm.cdf(d1)
                gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2) + 
                        q * S * np.exp(-q * T) * norm.cdf(d1))
                vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
                gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2) - 
                        q * S * np.exp(-q * T) * norm.cdf(-d1))
                vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            return {
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}


# Factory function for easy integration
async def create_options_analyzer(config: Optional[Dict[str, Any]] = None) -> OptionsSurfaceAnalyzer:
    """Create and initialize options surface analyzer"""
    return OptionsSurfaceAnalyzer(config)
