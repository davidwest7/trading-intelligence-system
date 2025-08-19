#!/usr/bin/env python3
"""
Trading Intelligence System - Main API Server

A simple FastAPI server to demonstrate the multi-agent trading intelligence system.
"""

import os
import sys
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.technical.agent import TechnicalAgent
from common.scoring.unified_score import UnifiedScorer
from common.event_bus.bus import EventBus, EventType

# Initialize FastAPI app
app = FastAPI(
    title="Trading Intelligence System",
    description="Multi-agent trading intelligence system for research-grade analysis",
    version="1.0.0",
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

# Initialize global components
technical_agent = TechnicalAgent()
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

class ScoringRequest(BaseModel):
    opportunities: List[Dict[str, Any]]
    calibration_method: str = "isotonic"
    regime_aware: bool = True
    diversification_penalty: float = 0.1

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    agents: Dict[str, str]

class SystemStatsResponse(BaseModel):
    events_processed: int
    opportunities_scored: int
    uptime_seconds: float
    memory_usage_mb: float

# Global stats
app_start_time = datetime.now()
stats = {
    "events_processed": 0,
    "opportunities_scored": 0
}

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await event_bus.start()
    print("🚀 Trading Intelligence System started")
    print("📊 API Documentation: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await event_bus.stop()
    print("🛑 Trading Intelligence System stopped")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Trading Intelligence System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        agents={
            "technical": "active",
            "sentiment": "stub",
            "flow": "stub",
            "macro": "stub",
            "scoring": "active",
            "event_bus": "active"
        }
    )

@app.get("/stats", response_model=SystemStatsResponse)
async def get_stats():
    """Get system statistics"""
    import psutil
    process = psutil.Process()
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return SystemStatsResponse(
        events_processed=stats["events_processed"],
        opportunities_scored=stats["opportunities_scored"],
        uptime_seconds=uptime,
        memory_usage_mb=memory_mb
    )

@app.post("/technical/find_opportunities")
async def find_technical_opportunities(request: TechnicalAnalysisRequest):
    """Find technical trading opportunities"""
    try:
        # Convert request to dict
        payload = request.model_dump()
        
        # Call technical agent
        result = await technical_agent.find_opportunities(payload)
        
        # Publish event
        await event_bus.publish_agent_signal(
            source="api",
            agent_name="technical",
            signal_type="analysis_complete",
            confidence=1.0,
            additional_data={
                "opportunities_found": len(result["opportunities"]),
                "symbols": request.symbols
            }
        )
        
        stats["events_processed"] += 1
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

@app.post("/scoring/score_opportunities")
async def score_opportunities(request: ScoringRequest):
    """Score trading opportunities using unified scoring system"""
    try:
        # Score opportunities
        scored_opportunities = unified_scorer.score_opportunities(
            opportunities=request.opportunities,
            scoring_config={
                "calibration_method": request.calibration_method,
                "regime_aware": request.regime_aware,
                "diversification_penalty": request.diversification_penalty
            }
        )
        
        # Convert to JSON serializable format
        result = {
            "ranked_opportunities": [
                {
                    "id": opp.id,
                    "symbol": opp.symbol,
                    "strategy": opp.strategy,
                    "rank": opp.rank,
                    "unified_score": opp.unified_score,
                    "calibrated_probability": opp.calibrated_probability,
                    "percentile_rank": opp.percentile_rank,
                    "confidence_interval": {
                        "lower": opp.confidence_interval[0],
                        "upper": opp.confidence_interval[1]
                    }
                }
                for opp in scored_opportunities
            ],
            "portfolio_metrics": {
                "total_opportunities": len(scored_opportunities),
                "avg_score": sum(opp.unified_score for opp in scored_opportunities) / len(scored_opportunities) if scored_opportunities else 0,
                "top_score": max(opp.unified_score for opp in scored_opportunities) if scored_opportunities else 0
            }
        }
        
        stats["opportunities_scored"] += len(scored_opportunities)
        
        # Publish event
        await event_bus.publish_agent_signal(
            source="api",
            agent_name="scorer",
            signal_type="scoring_complete",
            confidence=1.0,
            additional_data={
                "opportunities_scored": len(scored_opportunities),
                "avg_score": result["portfolio_metrics"]["avg_score"]
            }
        )
        
        stats["events_processed"] += 1
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.get("/events/history")
async def get_event_history(event_type: Optional[str] = None, limit: int = 100):
    """Get event history"""
    try:
        event_type_enum = EventType(event_type) if event_type else None
        history = event_bus.get_event_history(event_type_enum, limit)
        
        # Convert events to JSON serializable format
        return {
            "events": [event.to_dict() for event in history],
            "total": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get event history: {str(e)}")

@app.post("/demo/full_analysis")
async def demo_full_analysis():
    """Demo endpoint showing full analysis pipeline"""
    try:
        print("🔄 Running full analysis demo...")
        
        # Step 1: Technical Analysis
        technical_request = TechnicalAnalysisRequest(
            symbols=["AAPL", "TSLA", "EURUSD"],
            timeframes=["1h", "4h"],
            strategies=["imbalance", "trend"],
            min_score=0.01  # Lower threshold for demo
        )
        
        technical_result = await find_technical_opportunities(technical_request)
        opportunities = technical_result["opportunities"]
        
        print(f"📈 Found {len(opportunities)} technical opportunities")
        
        if not opportunities:
            return {
                "message": "No opportunities found",
                "technical_analysis": technical_result,
                "scored_opportunities": {"ranked_opportunities": [], "portfolio_metrics": {}}
            }
        
        # Step 2: Add mock raw signals to opportunities for scoring
        for i, opp in enumerate(opportunities):
            # Add ID if missing
            if "id" not in opp:
                opp["id"] = f"opp_{i+1}_{opp['symbol']}_{opp['strategy']}"
            
            opp["raw_signals"] = {
                "likelihood": opp["confidence_score"],
                "expected_return": 0.02 + (i * 0.01),  # Mock expected returns
                "risk": abs(opp["stop_loss"] - opp["entry_price"]) / opp["entry_price"],
                "liquidity": 0.9,
                "conviction": opp["confidence_score"],
                "recency": 1.0,
                "regime_fit": 0.7
            }
            opp["metadata"] = {"asset_class": "equities" if opp["symbol"] in ["AAPL", "TSLA"] else "fx"}
        
        # Step 3: Score opportunities
        scoring_request = ScoringRequest(
            opportunities=opportunities,
            calibration_method="isotonic",
            regime_aware=True
        )
        
        scoring_result = await score_opportunities(scoring_request)
        
        print(f"🏆 Scored {len(scoring_result['ranked_opportunities'])} opportunities")
        
        # Step 4: Return combined results
        return {
            "message": "Full analysis complete",
            "technical_analysis": technical_result,
            "scored_opportunities": scoring_result,
            "summary": {
                "total_symbols_analyzed": len(technical_request.symbols),
                "opportunities_found": len(opportunities),
                "top_opportunity": scoring_result["ranked_opportunities"][0] if scoring_result["ranked_opportunities"] else None,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo analysis failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/docs", "/health", "/stats"]}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": str(exc)}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Trading Intelligence System API Server...")
    print("📊 Access the API documentation at: http://localhost:8000/docs")
    print("🏥 Health check at: http://localhost:8000/health")
    print("🎯 Demo analysis at: http://localhost:8000/demo/full_analysis")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
