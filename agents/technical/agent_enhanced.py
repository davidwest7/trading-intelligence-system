"""
Enhanced Technical Strategy Agent
Integrates embargo system, LOB features, and hierarchical ensemble for best-in-class performance
"""

import time
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .models import (
    TechnicalOpportunity, AnalysisPayload, AnalysisMetadata, 
    MarketRegime, Direction
)
from .strategies import (
    ImbalanceStrategy, FairValueGapStrategy, LiquiditySweepStrategy,
    IDFPStrategy, TrendStrategy, BreakoutStrategy, MeanReversionStrategy
)
from .backtest import PurgedCrossValidationBacktester

# Enhanced imports
from common.feature_store.embargo import MultiEventEmbargoManager
from agents.flow.lob_features import LOBFeatureExtractor
from ml_models.hierarchical_meta_ensemble import HierarchicalMetaEnsemble


class EnhancedTechnicalAgent:
    """
    Enhanced Technical Strategy Agent with best-in-class features
    
    Features:
    - Multi-event embargo integration for data leakage prevention
    - LOB and microstructure feature integration
    - Hierarchical meta-ensemble for signal combination
    - Enhanced risk management with uncertainty quantification
    - Real-time adaptation and drift detection
    """
    
    def __init__(self, data_adapter=None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core strategies
        self.strategies = {
            "imbalance": ImbalanceStrategy(),
            "fvg": FairValueGapStrategy(),
            "liquidity_sweep": LiquiditySweepStrategy(),
            "idfp": IDFPStrategy(),
            "trend": TrendStrategy(),
            "breakout": BreakoutStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
        
        # Enhanced components
        self.embargo_manager: Optional[MultiEventEmbargoManager] = None
        self.lob_extractor: Optional[LOBFeatureExtractor] = None
        self.hierarchical_ensemble: Optional[HierarchicalMetaEnsemble] = None
        
        # Core components
        self.data_adapter = data_adapter
        self.backtester = PurgedCrossValidationBacktester()
        
        # Performance tracking
        self.analysis_history = []
        self.embargo_violations = 0
        self.lob_analyses = 0
        self.ensemble_predictions = 0
        
    async def initialize_enhanced_features(self):
        """Initialize enhanced features"""
        try:
            # Initialize embargo manager
            if not self.embargo_manager:
                from common.feature_store.embargo import create_embargo_manager
                self.embargo_manager = await create_embargo_manager()
            
            # Initialize LOB extractor
            if not self.lob_extractor:
                from agents.flow.lob_features import create_lob_extractor
                self.lob_extractor = await create_lob_extractor()
            
            # Initialize hierarchical ensemble
            if not self.hierarchical_ensemble:
                from ml_models.hierarchical_meta_ensemble import create_hierarchical_ensemble
                self.hierarchical_ensemble = await create_hierarchical_ensemble({
                    'n_base_models': 8,
                    'n_meta_models': 3,
                    'uncertainty_method': 'bootstrap',
                    'calibration_window': 500,
                    'drift_threshold': 0.1
                })
            
            print("✅ Enhanced technical agent initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing enhanced features: {e}")
    
    async def find_opportunities_enhanced(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced opportunity finding with embargo, LOB, and ensemble integration
        
        Args:
            payload: Analysis payload with enhanced options
            
        Returns:
            Enhanced analysis results with uncertainty quantification
        """
        start_time = time.time()
        
        # Initialize enhanced features if needed
        if not self.embargo_manager:
            await self.initialize_enhanced_features()
        
        # Parse payload
        analysis_payload = AnalysisPayload(**payload)
        
        # Extract enhanced options
        use_embargo = payload.get('use_embargo', True)
        use_lob_features = payload.get('use_lob_features', True)
        use_ensemble = payload.get('use_ensemble', True)
        
        # Step 1: Embargo filtering
        embargoed_symbols = []
        available_symbols = analysis_payload.symbols.copy()
        
        if use_embargo and self.embargo_manager:
            for symbol in analysis_payload.symbols:
                is_embargoed, reasons = await self.embargo_manager.check_embargo_status(
                    symbol, datetime.now()
                )
                if is_embargoed:
                    embargoed_symbols.append({
                        'symbol': symbol,
                        'reasons': reasons
                    })
                    available_symbols.remove(symbol)
                    self.embargo_violations += 1
        
        if not available_symbols:
            return {
                "opportunities": [],
                "metadata": {
                    "analysis_time": time.time() - start_time,
                    "symbols_analyzed": 0,
                    "embargoed_symbols": embargoed_symbols,
                    "enhanced_features": {
                        "embargo_check": True,
                        "lob_features": False,
                        "ensemble_prediction": False
                    }
                }
            }
        
        # Step 2: Get market data for available symbols
        market_data = await self._get_market_data(
            available_symbols, 
            analysis_payload.timeframes,
            analysis_payload.lookback_periods
        )
        
        # Step 3: Run technical analysis
        all_opportunities = []
        valid_strategies = [s for s in analysis_payload.strategies if s in self.strategies]
        if not valid_strategies:
            valid_strategies = ["imbalance", "trend"]
        
        for symbol in available_symbols:
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                continue
            
            # Step 4: Get LOB features if enabled
            lob_features = {}
            if use_lob_features and self.lob_extractor:
                try:
                    lob_features = await self._get_lob_features(symbol)
                    self.lob_analyses += 1
                except Exception as e:
                    lob_features = {"error": str(e)}
            
            for strategy_name in valid_strategies:
                strategy = self.strategies[strategy_name]
                opportunities = strategy.analyze(symbol_data, symbol, analysis_payload.timeframes)
                
                # Filter by confidence score
                filtered_opportunities = [
                    opp for opp in opportunities 
                    if opp.confidence_score >= analysis_payload.min_score
                ]
                
                # Step 5: Enhance opportunities with LOB features
                for opp in filtered_opportunities:
                    opp.lob_features = lob_features
                    
                    # Step 6: Generate ensemble prediction if enabled
                    if use_ensemble and self.hierarchical_ensemble:
                        try:
                            ensemble_result = await self._get_ensemble_prediction(opp, lob_features)
                            opp.ensemble_prediction = ensemble_result
                            self.ensemble_predictions += 1
                        except Exception as e:
                            opp.ensemble_prediction = {"error": str(e)}
                
                all_opportunities.extend(filtered_opportunities)
        
        # Step 7: Apply risk filtering
        risk_filtered_opportunities = self._apply_risk_filter(
            all_opportunities, 
            analysis_payload.max_risk
        )
        
        # Step 8: Convert to serializable format
        serialized_opportunities = []
        for opp in risk_filtered_opportunities:
            opp_dict = {
                "id": opp.id,
                "symbol": opp.symbol,
                "strategy": opp.strategy,
                "direction": opp.direction.value,
                "entry_price": opp.entry_price,
                "stop_loss": opp.stop_loss,
                "take_profit": opp.take_profit,
                "confidence_score": opp.confidence_score,
                "timeframe": opp.timeframe,
                "timestamp": opp.timestamp.isoformat(),
                "metadata": opp.metadata,
                "lob_features": opp.lob_features,
                "ensemble_prediction": opp.ensemble_prediction
            }
            serialized_opportunities.append(opp_dict)
        
        # Step 9: Create enhanced metadata
        enhanced_metadata = {
            "analysis_time": time.time() - start_time,
            "symbols_analyzed": len(available_symbols),
            "opportunities_found": len(serialized_opportunities),
            "embargoed_symbols": embargoed_symbols,
            "enhanced_features": {
                "embargo_check": use_embargo,
                "lob_features": use_lob_features,
                "ensemble_prediction": use_ensemble
            },
            "performance_metrics": {
                "embargo_violations": self.embargo_violations,
                "lob_analyses": self.lob_analyses,
                "ensemble_predictions": self.ensemble_predictions
            }
        }
        
        # Store analysis history
        self.analysis_history.append({
            "timestamp": datetime.now(),
            "symbols_analyzed": len(available_symbols),
            "opportunities_found": len(serialized_opportunities),
            "embargoed_count": len(embargoed_symbols)
        })
        
        return {
            "opportunities": serialized_opportunities,
            "metadata": enhanced_metadata
        }
    
    async def _get_lob_features(self, symbol: str) -> Dict[str, Any]:
        """Get LOB features for a symbol"""
        try:
            # In production, this would fetch real LOB data
            # For now, create sample data
            from agents.flow.lob_features import OrderBookSnapshot, OrderBookLevel, OrderSide
            
            now = datetime.now()
            
            # Sample order book levels
            bids = [
                OrderBookLevel(price=150.00 - i*0.05, size=1000 + i*500, 
                              side=OrderSide.BID, timestamp=now, venue="NASDAQ")
                for i in range(10)
            ]
            
            asks = [
                OrderBookLevel(price=150.05 + i*0.05, size=1200 + i*300, 
                              side=OrderSide.ASK, timestamp=now, venue="NASDAQ")
                for i in range(10)
            ]
            
            order_book = OrderBookSnapshot(
                symbol=symbol,
                timestamp=now,
                bids=bids,
                asks=asks,
                last_trade_price=150.02,
                last_trade_size=500
            )
            
            # Extract features
            features = await self.lob_extractor.extract_lob_features(order_book)
            
            return {
                "order_imbalance": features.get("order_imbalance", 0),
                "spread_bps": features.get("spread_bps", 0),
                "kyle_lambda": features.get("kyle_lambda", 0),
                "buy_impact_10000": features.get("buy_impact_10000", 0),
                "sell_impact_10000": features.get("sell_impact_10000", 0),
                "total_depth_3": features.get("total_depth_3", 0),
                "large_orders_total": features.get("large_orders_total", 0)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_ensemble_prediction(self, opportunity: TechnicalOpportunity, 
                                     lob_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get ensemble prediction for an opportunity"""
        try:
            # Create feature vector for ensemble
            features = {
                "technical_score": opportunity.confidence_score,
                "strategy_imbalance": 1.0 if opportunity.strategy == "imbalance" else 0.0,
                "strategy_trend": 1.0 if opportunity.strategy == "trend" else 0.0,
                "strategy_breakout": 1.0 if opportunity.strategy == "breakout" else 0.0,
                "direction_long": 1.0 if opportunity.direction == Direction.LONG else 0.0,
                "direction_short": 1.0 if opportunity.direction == Direction.SHORT else 0.0,
                "lob_imbalance": lob_features.get("order_imbalance", 0),
                "spread_bps": lob_features.get("spread_bps", 0),
                "kyle_lambda": lob_features.get("kyle_lambda", 0),
                "buy_impact": lob_features.get("buy_impact_10000", 0),
                "sell_impact": lob_features.get("sell_impact_10000", 0),
                "liquidity_depth": lob_features.get("total_depth_3", 0),
                "large_orders": lob_features.get("large_orders_total", 0)
            }
            
            # Make prediction
            import pandas as pd
            X = pd.DataFrame([features])
            
            predictions, uncertainties, intervals = await self.hierarchical_ensemble.predict_with_uncertainty(X)
            
            return {
                "prediction": float(predictions[0]),
                "uncertainty": float(uncertainties[0]),
                "confidence_interval": {
                    "lower": float(intervals[0][0]),
                    "upper": float(intervals[0][1])
                },
                "features_used": list(features.keys())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_market_data(self, symbols: List[str], timeframes: List[str], 
                              lookback_periods: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get market data for symbols and timeframes"""
        # This would integrate with your data adapter
        # For now, return empty data structure
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = {}
            for timeframe in timeframes:
                # Create sample data
                dates = pd.date_range(end=datetime.now(), periods=lookback_periods, freq='1H')
                market_data[symbol][timeframe] = pd.DataFrame({
                    'open': np.random.randn(lookback_periods).cumsum() + 100,
                    'high': np.random.randn(lookback_periods).cumsum() + 102,
                    'low': np.random.randn(lookback_periods).cumsum() + 98,
                    'close': np.random.randn(lookback_periods).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, lookback_periods)
                }, index=dates)
        
        return market_data
    
    def _apply_risk_filter(self, opportunities: List[TechnicalOpportunity], 
                          max_risk: float) -> List[TechnicalOpportunity]:
        """Apply risk filtering to opportunities"""
        filtered_opportunities = []
        
        for opp in opportunities:
            # Calculate risk as percentage of entry price
            if opp.entry_price > 0:
                risk_pct = abs(opp.stop_loss - opp.entry_price) / opp.entry_price
                if risk_pct <= max_risk:
                    filtered_opportunities.append(opp)
        
        return filtered_opportunities
    
    async def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced agent status"""
        return {
            "embargo_manager": {
                "active": self.embargo_manager is not None,
                "violations": self.embargo_violations
            },
            "lob_extractor": {
                "active": self.lob_extractor is not None,
                "analyses": self.lob_analyses
            },
            "hierarchical_ensemble": {
                "active": self.hierarchical_ensemble is not None,
                "predictions": self.ensemble_predictions
            },
            "analysis_history": {
                "total_analyses": len(self.analysis_history),
                "recent_analyses": len([h for h in self.analysis_history 
                                      if (datetime.now() - h["timestamp"]).days <= 1])
            }
        }
    
    async def add_embargo_event(self, symbol: str, event_type: str, 
                               event_date: datetime, embargo_horizon: int = 7,
                               embargo_duration: int = 3) -> bool:
        """Add embargo event for a symbol"""
        try:
            if not self.embargo_manager:
                await self.initialize_enhanced_features()
            
            from common.feature_store.embargo import EmbargoEvent, EmbargoType
            
            event = EmbargoEvent(
                event_id=f"{symbol}_{event_type}_{event_date.strftime('%Y%m%d')}",
                event_type=EmbargoType(event_type),
                symbol=symbol,
                event_date=event_date,
                embargo_start=event_date - timedelta(days=embargo_horizon),
                embargo_end=event_date + timedelta(days=embargo_duration),
                embargo_horizon=embargo_horizon,
                embargo_duration=embargo_duration,
                confidence=0.9,
                source="technical_agent"
            )
            
            return await self.embargo_manager.add_embargo_event(event)
            
        except Exception as e:
            print(f"Error adding embargo event: {e}")
            return False


# Factory function for easy integration
async def create_enhanced_technical_agent(config: Optional[Dict[str, Any]] = None) -> EnhancedTechnicalAgent:
    """Create and initialize enhanced technical agent"""
    agent = EnhancedTechnicalAgent(config=config)
    await agent.initialize_enhanced_features()
    return agent
