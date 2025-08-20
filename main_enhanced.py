#!/usr/bin/env python3
"""
Enhanced Trading Intelligence System - Main API Server

Integrates all major enhancements:
- Multi-event embargo system with universe drift tracking
- Advanced LOB and microstructure features
- Hierarchical meta-ensemble with uncertainty-aware stacking
- Options surface analysis for insider detection
"""

import os
import sys
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core system imports
from agents.technical.agent import TechnicalAgent
from agents.flow.agent import FlowAgent
from agents.sentiment.agent import SentimentAgent
from agents.macro.agent import MacroAgent
from agents.undervalued.agent_real_data import RealDataUndervaluedAgent
from agents.insider.agent import InsiderAgent
from agents.learning.agent import LearningAgent

# Enhanced components
from common.feature_store.embargo import create_embargo_manager, EmbargoEvent, EmbargoType
from agents.flow.lob_features import create_lob_extractor, OrderBookSnapshot, OrderBookLevel, OrderSide
from ml_models.hierarchical_meta_ensemble import create_hierarchical_ensemble
from agents.insider.options_surface import create_options_analyzer, OptionsSurface, OptionContract, OptionType

# Core system components
from common.scoring.unified_score import UnifiedScorer
from common.event_bus.bus import EventBus, EventType
from common.data_adapters.polygon_adapter import PolygonAdapter

class TradingIntelligenceSystem:
    """Main trading intelligence system that integrates all components"""
    
    def __init__(self, config=None):
        self.config = config or {
            'use_enhanced_features': True,
            'embargo_horizon': 30,
            'include_options_analysis': True,
            'polygon_api_key': 'demo_key'
        }
        
        # Initialize agents
        self.technical_agent = TechnicalAgent()
        self.flow_agent = FlowAgent()
        self.sentiment_agent = SentimentAgent()
        self.macro_agent = MacroAgent()
        self.undervalued_agent = RealDataUndervaluedAgent({"polygon_api_key": self.config['polygon_api_key']})
        self.insider_agent = InsiderAgent()
        self.learning_agent = LearningAgent()
        
        # Initialize enhanced components
        self.embargo_manager = None
        self.lob_extractor = None
        self.hierarchical_ensemble = None
        self.options_analyzer = None
        self.polygon_adapter = PolygonAdapter({"polygon_api_key": self.config['polygon_api_key']})
        
        # Initialize core components
        self.unified_scorer = UnifiedScorer()
        self.event_bus = EventBus(persist_events=True)
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the trading intelligence system"""
        try:
            print("🚀 Initializing Trading Intelligence System...")
            
            # Initialize enhanced components if enabled
            if self.config['use_enhanced_features']:
                self.embargo_manager = create_embargo_manager()
                self.lob_extractor = create_lob_extractor()
                self.hierarchical_ensemble = create_hierarchical_ensemble()
                
                if self.config['include_options_analysis']:
                    self.options_analyzer = create_options_analyzer()
            
            # Initialize data adapter
            await self.polygon_adapter.connect()
            
            self.is_initialized = True
            print("✅ Trading Intelligence System initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Trading Intelligence System: {e}")
            return False
    
    async def analyze_symbols(self, symbols: List[str], analysis_types: List[str] = None) -> Dict[str, Any]:
        """Analyze symbols using specified analysis types"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if analysis_types is None:
                analysis_types = ["technical", "flow", "sentiment", "macro", "undervalued", "insider"]
            
            results = {}
            
            for symbol in symbols:
                symbol_results = {}
                
                for analysis_type in analysis_types:
                    if analysis_type == "technical":
                        result = await self.technical_agent.analyze(symbol)
                        symbol_results["technical"] = result
                    elif analysis_type == "flow":
                        result = await self.flow_agent.analyze(symbol)
                        symbol_results["flow"] = result
                    elif analysis_type == "sentiment":
                        result = await self.sentiment_agent.analyze(symbol)
                        symbol_results["sentiment"] = result
                    elif analysis_type == "macro":
                        result = await self.macro_agent.analyze(symbol)
                        symbol_results["macro"] = result
                    elif analysis_type == "undervalued":
                        result = await self.undervalued_agent.analyze(symbol)
                        symbol_results["undervalued"] = result
                    elif analysis_type == "insider":
                        result = await self.insider_agent.analyze(symbol)
                        symbol_results["insider"] = result
                
                results[symbol] = symbol_results
            
            return {
                'success': True,
                'results': results,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health"""
        try:
            return {
                'system_name': 'Trading Intelligence System',
                'version': '2.0.0',
                'is_initialized': self.is_initialized,
                'agents': {
                    'technical': self.technical_agent is not None,
                    'flow': self.flow_agent is not None,
                    'sentiment': self.sentiment_agent is not None,
                    'macro': self.macro_agent is not None,
                    'undervalued': self.undervalued_agent is not None,
                    'insider': self.insider_agent is not None,
                    'learning': self.learning_agent is not None
                },
                'enhanced_features': {
                    'embargo_manager': self.embargo_manager is not None,
                    'lob_extractor': self.lob_extractor is not None,
                    'hierarchical_ensemble': self.hierarchical_ensemble is not None,
                    'options_analyzer': self.options_analyzer is not None
                },
                'data_adapter': {
                    'polygon_adapter': self.polygon_adapter.is_connected if hasattr(self.polygon_adapter, 'is_connected') else False
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def shutdown(self):
        """Shutdown the trading intelligence system"""
        try:
            if self.polygon_adapter:
                await self.polygon_adapter.disconnect()
            
            self.is_initialized = False
            print("🛑 Trading Intelligence System shutdown")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Trading Intelligence System",
    description="Best-in-class multi-agent trading intelligence system with advanced features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize enhanced global components
technical_agent = TechnicalAgent()
flow_agent = FlowAgent()
sentiment_agent = SentimentAgent()
macro_agent = MacroAgent()
undervalued_agent = RealDataUndervaluedAgent({"polygon_api_key": "demo_key"})
insider_agent = InsiderAgent()
learning_agent = LearningAgent()

# Enhanced components
embargo_manager = None
lob_extractor = None
hierarchical_ensemble = None
options_analyzer = None
polygon_adapter = PolygonAdapter({"polygon_api_key": "demo_key"})

# Core system components
unified_scorer = UnifiedScorer()
event_bus = EventBus(persist_events=True)

# Request/Response models
class TechnicalAnalysisRequest(BaseModel):
    symbols: List[str]
    timeframes: List[str] = ["15m", "1h", "4h"]
    strategies: List[str] = ["imbalance", "trend"]
    min_score: float = 0.6
    max_risk: float = 0.02
    lookback_periods: int = 200
    use_embargo: bool = True
    use_lob_features: bool = True

class EnhancedAnalysisRequest(BaseModel):
    symbols: List[str]
    analysis_types: List[str] = ["technical", "flow", "sentiment", "macro", "undervalued", "insider"]
    timeframes: List[str] = ["15m", "1h", "4h"]
    use_enhanced_features: bool = True
    embargo_horizon: int = 30
    include_options_analysis: bool = True
    include_lob_analysis: bool = True

class LOBAnalysisRequest(BaseModel):
    symbol: str
    timestamp: Optional[datetime] = None
    levels: int = 10

class OptionsAnalysisRequest(BaseModel):
    symbol: str
    include_greeks: bool = True
    include_flow_analysis: bool = True
    include_insider_detection: bool = True

class EnsemblePredictionRequest(BaseModel):
    features: Dict[str, float]
    include_uncertainty: bool = True
    confidence_level: float = 0.95

class EmbargoManagementRequest(BaseModel):
    event_type: str
    symbol: str
    event_date: datetime
    embargo_horizon: int = 7
    embargo_duration: int = 3
    confidence: float = 0.9

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    agents: Dict[str, str]
    enhanced_features: Dict[str, str]

class SystemStatsResponse(BaseModel):
    events_processed: int
    opportunities_scored: int
    embargo_checks: int
    lob_analyses: int
    options_analyses: int
    ensemble_predictions: int
    uptime_seconds: float
    memory_usage_mb: float

# Global stats
app_start_time = datetime.now()
stats = {
    "events_processed": 0,
    "opportunities_scored": 0,
    "embargo_checks": 0,
    "lob_analyses": 0,
    "options_analyses": 0,
    "ensemble_predictions": 0
}

@app.on_event("startup")
async def startup_event():
    """Initialize enhanced system on startup"""
    global embargo_manager, lob_extractor, hierarchical_ensemble, options_analyzer
    
    print("🚀 Initializing Enhanced Trading Intelligence System...")
    
    # Initialize enhanced components
    print("📊 Initializing embargo manager...")
    embargo_manager = await create_embargo_manager()
    
    print("📈 Initializing LOB feature extractor...")
    lob_extractor = await create_lob_extractor()
    
    print("🧠 Initializing hierarchical meta-ensemble...")
    hierarchical_ensemble = await create_hierarchical_ensemble({
        'n_base_models': 10,
        'n_meta_models': 3,
        'uncertainty_method': 'bootstrap',
        'calibration_window': 500,
        'drift_threshold': 0.1
    })
    
    print("📊 Initializing options surface analyzer...")
    options_analyzer = await create_options_analyzer()
    
    # Start event bus
    await event_bus.start()
    
    print("✅ Enhanced Trading Intelligence System started successfully!")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/health")
    print("🎯 Enhanced demo: http://localhost:8000/demo/enhanced_analysis")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await event_bus.stop()
    print("🛑 Enhanced Trading Intelligence System stopped")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Trading Intelligence System API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "enhanced_features": "embargo,lob,ensemble,options"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        agents={
            "technical": "active",
            "flow": "active",
            "sentiment": "active",
            "macro": "active",
            "undervalued": "active",
            "insider": "active",
            "learning": "active",
            "scoring": "active",
            "event_bus": "active"
        },
        enhanced_features={
            "embargo_system": "active",
            "lob_features": "active",
            "hierarchical_ensemble": "active",
            "options_analysis": "active"
        }
    )

@app.get("/stats", response_model=SystemStatsResponse)
async def get_stats():
    """Get enhanced system statistics"""
    import psutil
    process = psutil.Process()
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return SystemStatsResponse(
        events_processed=stats["events_processed"],
        opportunities_scored=stats["opportunities_scored"],
        embargo_checks=stats["embargo_checks"],
        lob_analyses=stats["lob_analyses"],
        options_analyses=stats["options_analyses"],
        ensemble_predictions=stats["ensemble_predictions"],
        uptime_seconds=uptime,
        memory_usage_mb=memory_mb
    )

@app.post("/embargo/check")
async def check_embargo_status(symbol: str, timestamp: Optional[datetime] = None):
    """Check embargo status for a symbol"""
    try:
        if timestamp is None:
            timestamp = datetime.now()
        
        is_embargoed, reasons = await embargo_manager.check_embargo_status(symbol, timestamp)
        stats["embargo_checks"] += 1
        
        return {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "is_embargoed": is_embargoed,
            "reasons": reasons
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embargo check failed: {str(e)}")

@app.post("/embargo/add_event")
async def add_embargo_event(request: EmbargoManagementRequest):
    """Add a new embargo event"""
    try:
        event = EmbargoEvent(
            event_id=f"{request.symbol}_{request.event_type}_{request.event_date.strftime('%Y%m%d')}",
            event_type=EmbargoType(request.event_type),
            symbol=request.symbol,
            event_date=request.event_date,
            embargo_start=request.event_date - timedelta(days=request.embargo_horizon),
            embargo_end=request.event_date + timedelta(days=request.embargo_duration),
            embargo_horizon=request.embargo_horizon,
            embargo_duration=request.embargo_duration,
            confidence=request.confidence,
            source="api"
        )
        
        success = await embargo_manager.add_embargo_event(event)
        
        return {
            "success": success,
            "event_id": event.event_id,
            "embargo_start": event.embargo_start.isoformat(),
            "embargo_end": event.embargo_end.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add embargo event: {str(e)}")

@app.post("/lob/analyze")
async def analyze_lob(request: LOBAnalysisRequest):
    """Analyze limit order book for a symbol"""
    try:
        # Create sample order book data (in production, this would come from real-time feed)
        timestamp = request.timestamp or datetime.now()
        
        # Sample order book levels
        bids = [
            OrderBookLevel(price=150.00 - i*0.05, size=1000 + i*500, 
                          side=OrderSide.BID, timestamp=timestamp, venue="NASDAQ")
            for i in range(request.levels)
        ]
        
        asks = [
            OrderBookLevel(price=150.05 + i*0.05, size=1200 + i*300, 
                          side=OrderSide.ASK, timestamp=timestamp, venue="NASDAQ")
            for i in range(request.levels)
        ]
        
        order_book = OrderBookSnapshot(
            symbol=request.symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            last_trade_price=150.02,
            last_trade_size=500
        )
        
        # Extract LOB features
        features = await lob_extractor.extract_lob_features(order_book)
        stats["lob_analyses"] += 1
        
        return {
            "symbol": request.symbol,
            "timestamp": timestamp.isoformat(),
            "features": features,
            "order_book_summary": {
                "best_bid": features.get("best_bid", 0),
                "best_ask": features.get("best_ask", 0),
                "spread_bps": features.get("spread_bps", 0),
                "order_imbalance": features.get("order_imbalance", 0),
                "kyle_lambda": features.get("kyle_lambda", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LOB analysis failed: {str(e)}")

@app.post("/options/analyze")
async def analyze_options_surface(request: OptionsAnalysisRequest):
    """Analyze options surface for a symbol"""
    try:
        # Create sample options data (in production, this would come from real-time feed)
        now = datetime.now()
        underlying_price = 150.00
        
        # Sample call options
        calls = [
            OptionContract(
                symbol=f"{request.symbol}240315C{strike}",
                strike=strike,
                expiry=now + timedelta(days=30),
                option_type=OptionType.CALL,
                last_price=8.50 - (strike - 145) * 0.5,
                bid=8.45 - (strike - 145) * 0.5,
                ask=8.55 - (strike - 145) * 0.5,
                volume=1500,
                open_interest=5000,
                implied_volatility=0.25 + (strike - 145) * 0.01,
                delta=0.65 - (strike - 145) * 0.1,
                gamma=0.02,
                theta=-0.15,
                vega=0.12,
                rho=0.08,
                timestamp=now
            )
            for strike in [145, 150, 155]
        ]
        
        # Sample put options
        puts = [
            OptionContract(
                symbol=f"{request.symbol}240315P{strike}",
                strike=strike,
                expiry=now + timedelta(days=30),
                option_type=OptionType.PUT,
                last_price=3.80 + (strike - 145) * 0.5,
                bid=3.75 + (strike - 145) * 0.5,
                ask=3.85 + (strike - 145) * 0.5,
                volume=1800,
                open_interest=6000,
                implied_volatility=0.30 + (strike - 145) * 0.01,
                delta=-0.35 - (strike - 145) * 0.1,
                gamma=0.02,
                theta=-0.12,
                vega=0.10,
                rho=-0.04,
                timestamp=now
            )
            for strike in [145, 150, 155]
        ]
        
        options_surface = OptionsSurface(
            symbol=request.symbol,
            underlying_price=underlying_price,
            timestamp=now,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            calls=calls,
            puts=puts
        )
        
        # Analyze options surface
        features = await options_analyzer.analyze_options_surface(options_surface)
        stats["options_analyses"] += 1
        
        return {
            "symbol": request.symbol,
            "timestamp": now.isoformat(),
            "underlying_price": underlying_price,
            "features": features,
            "summary": {
                "put_call_ratio": features.get("put_call_volume_ratio", 0),
                "iv_skew": features.get("iv_skew", 0),
                "vw_delta": features.get("vw_delta", 0),
                "vw_gamma": features.get("vw_gamma", 0),
                "volume_anomaly": features.get("volume_anomaly", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Options analysis failed: {str(e)}")

@app.post("/ensemble/predict")
async def make_ensemble_prediction(request: EnsemblePredictionRequest):
    """Make prediction using hierarchical meta-ensemble"""
    try:
        # Convert features to DataFrame
        import pandas as pd
        X = pd.DataFrame([request.features])
        
        # Make prediction with uncertainty
        predictions, uncertainties, intervals = await hierarchical_ensemble.predict_with_uncertainty(X)
        
        stats["ensemble_predictions"] += 1
        
        result = {
            "prediction": float(predictions[0]),
            "uncertainty": float(uncertainties[0]),
            "confidence_level": request.confidence_level
        }
        
        if request.include_uncertainty:
            result["prediction_interval"] = {
                "lower": float(intervals[0][0]),
                "upper": float(intervals[0][1])
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")

@app.post("/technical/enhanced_find_opportunities")
async def enhanced_find_technical_opportunities(request: TechnicalAnalysisRequest):
    """Find technical opportunities with enhanced features"""
    try:
        # Check embargo status if enabled
        embargoed_symbols = []
        if request.use_embargo:
            for symbol in request.symbols:
                is_embargoed, _ = await embargo_manager.check_embargo_status(symbol, datetime.now())
                if is_embargoed:
                    embargoed_symbols.append(symbol)
        
        # Filter out embargoed symbols
        available_symbols = [s for s in request.symbols if s not in embargoed_symbols]
        
        if not available_symbols:
            return {
                "message": "All symbols are under embargo",
                "embargoed_symbols": embargoed_symbols,
                "opportunities": []
            }
        
        # Convert request to dict
        payload = request.model_dump()
        payload["symbols"] = available_symbols
        
        # Call technical agent
        result = await technical_agent.find_opportunities(payload)
        
        # Add LOB features if enabled
        if request.use_lob_features:
            for opportunity in result["opportunities"]:
                try:
                    lob_result = await analyze_lob(LOBAnalysisRequest(symbol=opportunity["symbol"]))
                    opportunity["lob_features"] = lob_result["features"]
                except Exception as e:
                    opportunity["lob_features"] = {"error": str(e)}
        
        # Publish event
        await event_bus.publish_agent_signal(
            source="api",
            agent_name="technical_enhanced",
            signal_type="enhanced_analysis_complete",
            confidence=1.0,
            additional_data={
                "opportunities_found": len(result["opportunities"]),
                "symbols_analyzed": len(available_symbols),
                "embargoed_symbols": embargoed_symbols,
                "lob_features_included": request.use_lob_features
            }
        )
        
        stats["events_processed"] += 1
        
        return {
            **result,
            "embargoed_symbols": embargoed_symbols,
            "enhanced_features": {
                "embargo_check": request.use_embargo,
                "lob_features": request.use_lob_features
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced technical analysis failed: {str(e)}")

@app.post("/demo/enhanced_analysis")
async def demo_enhanced_analysis():
    """Enhanced demo endpoint showing full analysis pipeline with all features"""
    try:
        print("🔄 Running enhanced analysis demo...")
        
        # Step 1: Enhanced Technical Analysis
        technical_request = TechnicalAnalysisRequest(
            symbols=["AAPL", "TSLA", "MSFT"],
            timeframes=["1h", "4h"],
            strategies=["imbalance", "trend"],
            min_score=0.01,
            use_embargo=True,
            use_lob_features=True
        )
        
        technical_result = await enhanced_find_technical_opportunities(technical_request)
        opportunities = technical_result["opportunities"]
        
        print(f"📈 Found {len(opportunities)} technical opportunities")
        print(f"🚫 Embargoed symbols: {technical_result['embargoed_symbols']}")
        
        # Step 2: Options Analysis for each symbol
        options_results = {}
        for symbol in ["AAPL", "TSLA", "MSFT"]:
            try:
                options_result = await analyze_options_surface(
                    OptionsAnalysisRequest(symbol=symbol)
                )
                options_results[symbol] = options_result
            except Exception as e:
                options_results[symbol] = {"error": str(e)}
        
        # Step 3: LOB Analysis
        lob_results = {}
        for symbol in ["AAPL", "TSLA", "MSFT"]:
            try:
                lob_result = await analyze_lob(LOBAnalysisRequest(symbol=symbol))
                lob_results[symbol] = lob_result
            except Exception as e:
                lob_results[symbol] = {"error": str(e)}
        
        # Step 4: Ensemble Predictions
        ensemble_predictions = {}
        for i, opp in enumerate(opportunities[:3]):  # Test with first 3 opportunities
            try:
                # Create sample features for prediction
                features = {
                    "technical_score": opp.get("confidence_score", 0.5),
                    "lob_imbalance": opp.get("lob_features", {}).get("order_imbalance", 0),
                    "spread_bps": opp.get("lob_features", {}).get("spread_bps", 0),
                    "volume": 1000000,  # Mock volume
                    "volatility": 0.2,  # Mock volatility
                }
                
                prediction_result = await make_ensemble_prediction(
                    EnsemblePredictionRequest(features=features, include_uncertainty=True)
                )
                ensemble_predictions[f"opportunity_{i+1}"] = prediction_result
            except Exception as e:
                ensemble_predictions[f"opportunity_{i+1}"] = {"error": str(e)}
        
        # Step 5: Get embargo summary
        embargo_summary = await embargo_manager.get_embargo_summary()
        
        print(f"✅ Enhanced analysis complete!")
        
        return {
            "message": "Enhanced analysis complete",
            "technical_analysis": technical_result,
            "options_analysis": options_results,
            "lob_analysis": lob_results,
            "ensemble_predictions": ensemble_predictions,
            "embargo_summary": embargo_summary,
            "system_stats": {
                "events_processed": stats["events_processed"],
                "embargo_checks": stats["embargo_checks"],
                "lob_analyses": stats["lob_analyses"],
                "options_analyses": stats["options_analyses"],
                "ensemble_predictions": stats["ensemble_predictions"],
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced demo analysis failed: {str(e)}")

@app.get("/enhanced/status")
async def get_enhanced_status():
    """Get status of all enhanced features"""
    try:
        return {
            "embargo_system": {
                "status": "active" if embargo_manager else "inactive",
                "active_embargos": len(embargo_manager.active_embargos) if embargo_manager else 0,
                "total_checks": embargo_manager.total_checks if embargo_manager else 0
            },
            "lob_features": {
                "status": "active" if lob_extractor else "inactive",
                "analyses_performed": stats["lob_analyses"]
            },
            "hierarchical_ensemble": {
                "status": "active" if hierarchical_ensemble else "inactive",
                "predictions_made": stats["ensemble_predictions"],
                "base_models": len(hierarchical_ensemble.base_models) if hierarchical_ensemble else 0,
                "meta_models": len(hierarchical_ensemble.meta_models) if hierarchical_ensemble else 0
            },
            "options_analysis": {
                "status": "active" if options_analyzer else "inactive",
                "analyses_performed": stats["options_analyses"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced status: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/docs", "/health", "/stats", "/enhanced/status"]}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": str(exc)}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced Trading Intelligence System API Server...")
    print("📊 Access the API documentation at: http://localhost:8000/docs")
    print("🏥 Health check at: http://localhost:8000/health")
    print("🎯 Enhanced demo at: http://localhost:8000/demo/enhanced_analysis")
    print("📈 Enhanced status at: http://localhost:8000/enhanced/status")
    
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
