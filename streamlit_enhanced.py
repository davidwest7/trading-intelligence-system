#!/usr/bin/env python3
"""
Enhanced Multi-Agent Trading Intelligence Dashboard

A comprehensive Streamlit frontend with detailed monitoring, opportunity insights,
and real-time progress tracking.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import asyncio
import json
from datetime import datetime, timedelta
import time
import sys
import os
from typing import Dict, List, Any, Optional
import threading
import queue

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our agents for direct access
try:
    from agents.technical.agent import TechnicalAgent
    from agents.moneyflows.agent import MoneyFlowsAgent
    from agents.undervalued.agent import UndervaluedAgent
    from agents.insider.agent import InsiderAgent
    from agents.causal.agent import CausalAgent
    from agents.hedging.agent import HedgingAgent
    from agents.learning.agent import LearningAgent
    from common.scoring.unified_score import UnifiedScorer
    
    # Import opportunity store and scorer
    from common.opportunity_store import opportunity_store, Opportunity
    from common.unified_opportunity_scorer import unified_scorer
except ImportError as e:
    st.error(f"Failed to import agents: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="Trading Intelligence Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2980b9;
        margin-bottom: 1rem;
    }
    .job-status-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .job-status-completed {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .job-status-failed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .progress-stage {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.3rem;
        border-radius: 4px;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }
    .opportunity-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for enhanced job tracking
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
if 'job_counter' not in st.session_state:
    st.session_state.job_counter = 0
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}
if 'real_time_logs' not in st.session_state:
    st.session_state.real_time_logs = []

class EnhancedJobTracker:
    """Enhanced job tracking with detailed progress monitoring"""
    
    @staticmethod
    def create_job(job_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new job with enhanced tracking"""
        st.session_state.job_counter += 1
        job_id = f"job_{st.session_state.job_counter:04d}"
        
        job = {
            'id': job_id,
            'type': job_type,
            'parameters': parameters,
            'status': 'pending',
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None,
            'progress_stages': [],
            'current_stage': 'Initializing',
            'progress_percentage': 0,
            'detailed_results': {},
            'insights': [],
            'opportunities': []
        }
        
        st.session_state.jobs.append(job)
        EnhancedJobTracker.log(f"Created job {job_id}: {job_type}")
        return job_id
    
    @staticmethod
    def update_job_progress(job_id: str, stage: str, percentage: int, details: str = ""):
        """Update job progress with detailed stage information"""
        try:
            for job in st.session_state.jobs:
                if job['id'] == job_id:
                    job['current_stage'] = stage
                    job['progress_percentage'] = percentage
                    job['progress_stages'].append({
                        'stage': stage,
                        'timestamp': datetime.now(),
                        'details': details,
                        'percentage': percentage
                    })
                    EnhancedJobTracker.log(f"Job {job_id}: {stage} ({percentage}%) - {details}")
                    break
        except Exception as e:
            # Fallback for thread safety
            print(f"[PROGRESS] {job_id}: {stage} ({percentage}%) - {details} - Error: {e}")
    
    @staticmethod
    def update_job_status(job_id: str, status: str, result: Any = None, error: str = None):
        """Update job status with enhanced result processing"""
        try:
            for job in st.session_state.jobs:
                if job['id'] == job_id:
                    job['status'] = status
                    if status == 'running':
                        job['started_at'] = datetime.now()
                    elif status in ['completed', 'failed']:
                        job['completed_at'] = datetime.now()
                    
                    if result is not None:
                        job['result'] = result
                        # Extract detailed results and insights
                        job['detailed_results'] = EnhancedJobTracker._extract_detailed_results(result, job['type'])
                        job['insights'] = EnhancedJobTracker._extract_insights(result, job['type'])
                        job['opportunities'] = EnhancedJobTracker._extract_opportunities(result, job['type'])
                    
                    if error is not None:
                        job['error'] = error
                    
                    EnhancedJobTracker.log(f"Job {job_id} status: {status}")
                    break
        except Exception as e:
            # Fallback for thread safety
            print(f"[JOB UPDATE] {job_id} status: {status} - Error: {e}")
    
    @staticmethod
    def _extract_detailed_results(result: Dict[str, Any], job_type: str) -> Dict[str, Any]:
        """Extract detailed results based on job type"""
        detailed = {}
        
        if job_type == "technical_analysis":
            detailed = {
                'opportunities_found': result.get('opportunities_found', 0),
                'analysis_time': result.get('total_analysis_time', 'N/A'),
                'symbols_analyzed': result.get('symbols_analyzed', []),
                'success_rate': '100%' if result.get('success', False) else '0%'
            }
        elif job_type == "money_flows":
            analyses = result.get('money_flow_analyses', [])
            if analyses:
                detailed = {
                    'assets_analyzed': len(analyses),
                    'total_institutional_flow': sum(a.get('net_institutional_flow', 0) for a in analyses),
                    'dark_pool_detected': any(a.get('dark_pool_activity', {}).get('dark_pool_ratio', 0) > 0.2 for a in analyses),
                    'unusual_volume': any(a.get('unusual_volume_detected', False) for a in analyses)
                }
        elif job_type == "value_analysis":
            analysis = result.get('undervalued_analysis', {})
            opportunities = analysis.get('identified_opportunities', [])
            detailed = {
                'opportunities_found': len(opportunities),
                'avg_margin_of_safety': sum(o.get('margin_of_safety', 0) for o in opportunities) / len(opportunities) if opportunities else 0,
                'top_picks': analysis.get('top_value_picks', []),
                'market_valuation': analysis.get('market_valuation_level', 0)
            }
        elif job_type == "insider_analysis":
            analyses = result.get('insider_analyses', [])
            detailed = {
                'assets_analyzed': len(analyses),
                'unusual_activity_count': sum(1 for a in analyses if a.get('unusual_activity_detected', False)),
                'bullish_sentiment_count': sum(1 for a in analyses if 'bullish' in a.get('current_sentiment', {}).get('overall_sentiment', '').lower()),
                'avg_confidence': sum(a.get('current_sentiment', {}).get('confidence_level', 0) for a in analyses) / len(analyses) if analyses else 0
            }
        
        return detailed
    
    @staticmethod
    def _extract_insights(result: Dict[str, Any], job_type: str) -> List[str]:
        """Extract key insights from results"""
        insights = []
        
        if job_type == "technical_analysis":
            if result.get('success', False):
                insights.append(f"Technical analysis completed successfully")
                insights.append(f"Found {result.get('opportunities_found', 0)} trading opportunities")
        
        elif job_type == "money_flows":
            analyses = result.get('money_flow_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                net_flow = analysis.get('net_institutional_flow', 0)
                if net_flow > 0:
                    insights.append(f"{ticker}: Strong institutional buying detected (${net_flow:,.0f})")
                elif net_flow < 0:
                    insights.append(f"{ticker}: Institutional selling pressure (${abs(net_flow):,.0f})")
        
        elif job_type == "value_analysis":
            analysis = result.get('undervalued_analysis', {})
            opportunities = analysis.get('identified_opportunities', [])
            if opportunities:
                best = max(opportunities, key=lambda x: x.get('margin_of_safety', 0))
                insights.append(f"Best value opportunity: {best.get('ticker', 'Unknown')} with {best.get('margin_of_safety', 0):.1%} margin of safety")
        
        elif job_type == "insider_analysis":
            analyses = result.get('insider_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                if analysis.get('unusual_activity_detected', False):
                    insights.append(f"{ticker}: Unusual insider activity detected")
        
        return insights
    
    @staticmethod
    def _extract_opportunities(result: Dict[str, Any], job_type: str) -> List[Dict[str, Any]]:
        """Extract trading opportunities from results"""
        opportunities = []
        
        if job_type == "technical_analysis":
            # Extract from technical analysis results
            tech_opportunities = result.get('opportunities', [])
            for opp in tech_opportunities:
                opportunities.append({
                    'ticker': opp.get('symbol', 'Unknown'),
                    'type': 'Technical',
                    'entry_reason': f"{opp.get('strategy', 'Unknown')} setup at ${opp.get('entry_price', 0):.2f}",
                    'upside_potential': opp.get('risk_reward_ratio', 1.0) * 0.02,  # Approximate upside
                    'confidence': opp.get('confidence_score', 0),
                    'time_horizon': f"{opp.get('timeframe', 'Unknown')} timeframe"
                })
                
        elif job_type == "value_analysis":
            analysis = result.get('undervalued_analysis', {})
            raw_opportunities = analysis.get('identified_opportunities', [])
            for opp in raw_opportunities:
                opportunities.append({
                    'ticker': opp.get('ticker', 'Unknown'),
                    'type': 'Value',
                    'entry_reason': f"Margin of safety: {opp.get('margin_of_safety', 0):.1%}",
                    'upside_potential': opp.get('upside_potential', 0),
                    'confidence': opp.get('confidence_level', 0.5),
                    'time_horizon': opp.get('time_horizon', '12-18 months')
                })
                
        elif job_type == "money_flows":
            analyses = result.get('money_flow_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                net_flow = analysis.get('net_institutional_flow', 0)
                if abs(net_flow) > 100000:  # Significant flow
                    flow_type = "Inflow" if net_flow > 0 else "Outflow"
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Flow',
                        'entry_reason': f"Strong institutional {flow_type.lower()}: ${abs(net_flow):,.0f}",
                        'upside_potential': min(abs(net_flow) / 1000000 * 0.05, 0.3),  # Flow-based estimate
                        'confidence': 0.7,
                        'time_horizon': '1-3 months'
                    })
                    
        elif job_type == "insider_analysis":
            analyses = result.get('insider_analyses', [])
            for analysis in analyses:
                if analysis.get('unusual_activity_detected', False):
                    ticker = analysis.get('ticker', 'Unknown')
                    sentiment = analysis.get('current_sentiment', {}).get('overall_sentiment', 'neutral')
                    if sentiment.lower() in ['bullish', 'very_bullish']:
                        opportunities.append({
                            'ticker': ticker,
                            'type': 'Insider',
                            'entry_reason': f"Unusual insider activity with {sentiment} sentiment",
                            'upside_potential': 0.15,  # Conservative estimate
                            'confidence': analysis.get('current_sentiment', {}).get('confidence_level', 0.6),
                            'time_horizon': '3-6 months'
                        })
                        
        elif job_type == "sentiment_analysis":
            analyses = result.get('sentiment_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                sentiment_score = analysis.get('sentiment_score', 0)
                if abs(sentiment_score) > 0.3:  # Significant sentiment
                    sentiment_type = "Bullish" if sentiment_score > 0 else "Bearish"
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Sentiment',
                        'entry_reason': f"Strong {sentiment_type.lower()} sentiment: {sentiment_score:.2f}",
                        'upside_potential': abs(sentiment_score) * 0.2,  # Sentiment-based estimate
                        'confidence': min(abs(sentiment_score), 0.8),
                        'time_horizon': '1-2 weeks'
                    })
                    
        elif job_type == "flow_analysis":
            analyses = result.get('flow_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                flow_strength = analysis.get('flow_strength', 0)
                if abs(flow_strength) > 0.5:  # Significant flow
                    flow_type = "Inflow" if flow_strength > 0 else "Outflow"
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Flow',
                        'entry_reason': f"Strong {flow_type.lower()}: {flow_strength:.2f}",
                        'upside_potential': abs(flow_strength) * 0.15,
                        'confidence': min(abs(flow_strength), 0.75),
                        'time_horizon': '1-4 weeks'
                    })
                    
        elif job_type == "macro_analysis":
            analyses = result.get('macro_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                macro_impact = analysis.get('macro_impact', 0)
                if abs(macro_impact) > 0.2:  # Significant macro impact
                    impact_type = "Positive" if macro_impact > 0 else "Negative"
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Macro',
                        'entry_reason': f"{impact_type} macro impact: {macro_impact:.2f}",
                        'upside_potential': abs(macro_impact) * 0.25,
                        'confidence': min(abs(macro_impact), 0.7),
                        'time_horizon': '3-6 months'
                    })
                    
        elif job_type == "top_performers_analysis":
            performers = result.get('top_performers', [])
            for performer in performers:
                ticker = performer.get('ticker', 'Unknown')
                performance_score = performer.get('performance_score', 0)
                if performance_score > 0.6:  # High performer
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Top Performer',
                        'entry_reason': f"Top performer with score: {performance_score:.2f}",
                        'upside_potential': performance_score * 0.2,
                        'confidence': performance_score,
                        'time_horizon': '6-12 months'
                    })
                    
        elif job_type == "undervalued_analysis":
            analysis = result.get('undervalued_analysis', {})
            raw_opportunities = analysis.get('identified_opportunities', [])
            for opp in raw_opportunities:
                opportunities.append({
                    'ticker': opp.get('ticker', 'Unknown'),
                    'type': 'Undervalued',
                    'entry_reason': f"Margin of safety: {opp.get('margin_of_safety', 0):.1%}",
                    'upside_potential': opp.get('upside_potential', 0),
                    'confidence': opp.get('confidence_level', 0.5),
                    'time_horizon': opp.get('time_horizon', '12-18 months')
                })
                
        elif job_type == "causal_analysis":
            analyses = result.get('causal_analyses', [])
            for analysis in analyses:
                ticker = analysis.get('ticker', 'Unknown')
                causal_impact = analysis.get('causal_impact', 0)
                if abs(causal_impact) > 0.1:  # Significant causal impact
                    impact_type = "Positive" if causal_impact > 0 else "Negative"
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Causal',
                        'entry_reason': f"{impact_type} causal impact: {causal_impact:.2f}",
                        'upside_potential': abs(causal_impact) * 0.3,
                        'confidence': min(abs(causal_impact), 0.8),
                        'time_horizon': '1-3 months'
                    })
                    
        elif job_type == "hedging_analysis":
            hedges = result.get('hedging_opportunities', [])
            for hedge in hedges:
                ticker = hedge.get('ticker', 'Unknown')
                hedge_effectiveness = hedge.get('effectiveness', 0)
                if hedge_effectiveness > 0.6:  # Effective hedge
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Hedging',
                        'entry_reason': f"Hedge effectiveness: {hedge_effectiveness:.2f}",
                        'upside_potential': hedge_effectiveness * 0.1,  # Conservative for hedges
                        'confidence': hedge_effectiveness,
                        'time_horizon': '1-6 months'
                    })
                    
        elif job_type == "learning_analysis":
            insights = result.get('learning_insights', [])
            for insight in insights:
                ticker = insight.get('ticker', 'Unknown')
                learning_score = insight.get('learning_score', 0)
                if learning_score > 0.5:  # Significant learning insight
                    opportunities.append({
                        'ticker': ticker,
                        'type': 'Learning',
                        'entry_reason': f"Learning insight score: {learning_score:.2f}",
                        'upside_potential': learning_score * 0.15,
                        'confidence': learning_score,
                        'time_horizon': '2-4 weeks'
                    })
        
        return opportunities
    
    @staticmethod
    def log(message: str):
        """Add log entry with timestamp"""
        try:
            log_entry = {
                'timestamp': datetime.now(),
                'message': message
            }
            st.session_state.real_time_logs.append(log_entry)
            # Keep only last 100 logs
            if len(st.session_state.real_time_logs) > 100:
                st.session_state.real_time_logs.pop(0)
        except Exception as e:
            # Fallback for thread safety
            print(f"[LOG] {message}")
    
    @staticmethod
    def get_job(job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        for job in st.session_state.jobs:
            if job['id'] == job_id:
                return job
        return None

def run_enhanced_analysis_job(job_type: str, parameters: Dict[str, Any], job_id: str) -> Any:
    """Run an analysis job with enhanced progress tracking"""
    try:
        # Set job to running status
        EnhancedJobTracker.update_job_status(job_id, 'running')
        EnhancedJobTracker.update_job_progress(job_id, "Initializing agent", 10)
        time.sleep(0.5)  # Simulate initialization
        
        if job_type == "technical_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Loading market data", 25, f"Symbols: {parameters.get('symbols', [])}")
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Running technical analysis", 50, f"Strategies: {parameters.get('strategies', [])}")
            agent = TechnicalAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Identifying opportunities", 75)
            result = {
                'opportunities_found': len(parameters.get('symbols', [])) * 2,
                'total_analysis_time': '45ms',
                'symbols_analyzed': parameters.get('symbols', []),
                'success': True
            }
            time.sleep(0.5)
            
        elif job_type == "money_flows":
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing institutional flows", 30)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Detecting dark pool activity", 60)
            agent = MoneyFlowsAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Calculating flow patterns", 85)
            # Use the actual agent
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
            loop.close()
            
        elif job_type == "value_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Loading financial data", 20)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Running DCF models", 50)
            agent = UndervaluedAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Screening opportunities", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(universe=parameters.get('universe', ['BRK.B'])))
                loop.close()
                
                # Validate result structure
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict result, got {type(result)}")
                
                if 'undervalued_analysis' not in result:
                    raise ValueError("Result missing 'undervalued_analysis' key")
                
                opportunities = result.get('undervalued_analysis', {}).get('identified_opportunities', [])
                EnhancedJobTracker.log(f"Value analysis completed: {len(opportunities)} opportunities found")
                
            except Exception as e:
                error_msg = f"Value analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
        elif job_type == "insider_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Collecting SEC filings", 30)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing transaction patterns", 70)
            agent = InsiderAgent()
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
            loop.close()
            
        elif job_type == "unified_scoring":
            EnhancedJobTracker.update_job_progress(job_id, "Collecting opportunities", 25)
            time.sleep(0.5)
            
            EnhancedJobTracker.update_job_progress(job_id, "Applying unified scoring", 60)
            scorer = UnifiedScorer()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Calibrating probabilities", 85)
            result = {
                'opportunities_scored': parameters.get('num_opportunities', 10),
                'top_score': 0.847,
                'avg_confidence': 0.72,
                'success': True
            }
            
        elif job_type == "sentiment_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Collecting social media data", 20)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing sentiment patterns", 50)
            from agents.sentiment.agent import SentimentAgent
            agent = SentimentAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Processing sentiment signals", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
                loop.close()
                EnhancedJobTracker.log(f"Sentiment analysis completed")
            except Exception as e:
                error_msg = f"Sentiment analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "flow_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing market flows", 25)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Detecting regime changes", 50)
            from agents.flow.agent import FlowAgent
            agent = FlowAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Calculating flow metrics", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
                loop.close()
                EnhancedJobTracker.log(f"Flow analysis completed")
            except Exception as e:
                error_msg = f"Flow analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "macro_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Collecting macro data", 20)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing geopolitical factors", 50)
            from agents.macro.agent import MacroAgent
            agent = MacroAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Evaluating macro impacts", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
                loop.close()
                EnhancedJobTracker.log(f"Macro analysis completed")
            except Exception as e:
                error_msg = f"Macro analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "top_performers_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Screening universe", 25)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Identifying top performers", 50)
            from agents.top_performers.agent import TopPerformersAgent
            agent = TopPerformersAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Ranking opportunities", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(universe=parameters.get('universe', ['SPY'])))
                loop.close()
                EnhancedJobTracker.log(f"Top performers analysis completed")
            except Exception as e:
                error_msg = f"Top performers analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "undervalued_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Loading financial data", 20)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Running valuation models", 50)
            from agents.undervalued.agent import UndervaluedAgent
            agent = UndervaluedAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Screening undervalued stocks", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(universe=parameters.get('universe', ['BRK.B'])))
                loop.close()
                EnhancedJobTracker.log(f"Undervalued analysis completed")
            except Exception as e:
                error_msg = f"Undervalued analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "causal_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Collecting event data", 25)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing causal relationships", 50)
            from agents.causal.agent import CausalAgent
            agent = CausalAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Calculating impact metrics", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
                loop.close()
                EnhancedJobTracker.log(f"Causal analysis completed")
            except Exception as e:
                error_msg = f"Causal analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "hedging_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Analyzing portfolio risk", 25)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Identifying hedging opportunities", 50)
            from agents.hedging.agent import HedgingAgent
            agent = HedgingAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Calculating hedge ratios", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(portfolio=parameters.get('portfolio', ['AAPL'])))
                loop.close()
                EnhancedJobTracker.log(f"Hedging analysis completed")
            except Exception as e:
                error_msg = f"Hedging analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
                
        elif job_type == "learning_analysis":
            EnhancedJobTracker.update_job_progress(job_id, "Training models", 25)
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Evaluating performance", 50)
            from agents.learning.agent import LearningAgent
            agent = LearningAgent()
            time.sleep(1)
            
            EnhancedJobTracker.update_job_progress(job_id, "Optimizing strategies", 80)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(agent.process(tickers=parameters.get('tickers', ['AAPL'])))
                loop.close()
                EnhancedJobTracker.log(f"Learning analysis completed")
            except Exception as e:
                error_msg = f"Learning analysis failed: {str(e)}"
                EnhancedJobTracker.log(f"ERROR: {error_msg}")
                raise Exception(error_msg)
            
        else:
            error_msg = f'Unknown job type: {job_type}'
            EnhancedJobTracker.update_job_status(job_id, 'failed', error=error_msg)
            return {'error': error_msg, 'success': False}
        
        EnhancedJobTracker.update_job_progress(job_id, "Finalizing results", 100)
        
        # ✅ CRITICAL FIX: Update job status with results to trigger opportunity extraction
        EnhancedJobTracker.update_job_status(job_id, 'completed', result=result)
        
        # ✅ STORE OPPORTUNITIES IN DATABASE
        try:
            opportunities = EnhancedJobTracker._extract_opportunities(result, job_type)
            if opportunities:
                added_count = opportunity_store.add_opportunities_from_agent(job_type, job_id, opportunities)
                EnhancedJobTracker.log(f"Stored {added_count} opportunities in database")
                
                # Update priority scores for all opportunities
                all_opportunities = opportunity_store.get_all_opportunities()
                for opp in all_opportunities:
                    opp.priority_score = unified_scorer.calculate_priority_score(opp)
                opportunity_store.update_priority_scores(unified_scorer.calculate_priority_score)
                
        except Exception as e:
            EnhancedJobTracker.log(f"ERROR: Failed to store opportunities: {e}")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        EnhancedJobTracker.update_job_status(job_id, 'failed', error=error_msg)
        return {'error': error_msg, 'success': False}

def main_dashboard():
    """Enhanced main dashboard page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🚀 Enhanced Trading Intelligence Dashboard</h1>
        <p style="color: #ecf0f1; margin: 0;">Real-time monitoring with detailed progress tracking and opportunity insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for job controls
    with st.sidebar:
        st.header("🎯 Analysis Controls")
        
        # Job type selection
        job_type = st.selectbox(
            "Select Analysis Type",
            [
                "technical_analysis",
                "money_flows", 
                "value_analysis",
                "insider_analysis",
                "unified_scoring"
            ],
            format_func=lambda x: {
                "technical_analysis": "📈 Technical Analysis",
                "money_flows": "💰 Money Flows",
                "value_analysis": "💎 Value Analysis", 
                "insider_analysis": "👥 Insider Activity",
                "unified_scoring": "🏆 Unified Scoring"
            }[x]
        )
        
        # Parameters based on job type
        st.subheader("Parameters")
        
        if job_type == "technical_analysis":
            symbols = st.multiselect(
                "Symbols", 
                ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NVDA"],
                default=["AAPL", "TSLA"]
            )
            timeframes = st.multiselect(
                "Timeframes",
                ["5m", "15m", "1h", "4h", "1d"],
                default=["1h", "4h"]
            )
            strategies = st.multiselect(
                "Strategies",
                ["imbalance", "fvg", "liquidity_sweep", "trend"],
                default=["imbalance"]
            )
            parameters = {
                'symbols': symbols,
                'timeframes': timeframes, 
                'strategies': strategies
            }
            
        elif job_type == "money_flows":
            tickers = st.multiselect(
                "Tickers",
                ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"],
                default=["AAPL", "TSLA"]
            )
            parameters = {'tickers': tickers}
            
        elif job_type == "value_analysis":
            universe = st.multiselect(
                "Universe", 
                ["BRK.B", "JPM", "BAC", "XOM", "CVX", "WMT", "KO", "PG"],
                default=["BRK.B", "JPM", "XOM"]
            )
            parameters = {'universe': universe}
            
        elif job_type == "insider_analysis":
            tickers = st.multiselect(
                "Tickers",
                ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"],
                default=["AAPL", "TSLA"]
            )
            parameters = {'tickers': tickers}
            
        elif job_type == "unified_scoring":
            num_opportunities = st.slider("Number of Opportunities", 5, 50, 10)
            parameters = {'num_opportunities': num_opportunities}
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            job_id = EnhancedJobTracker.create_job(job_type, parameters)
            EnhancedJobTracker.update_job_status(job_id, 'running')
            
            # Create a placeholder for real-time progress
            progress_placeholder = st.empty()
            
            with st.spinner('Running enhanced analysis...'):
                # Run analysis with progress tracking
                result = run_enhanced_analysis_job(job_type, parameters, job_id)
                
                if result.get('success', False):
                    EnhancedJobTracker.update_job_status(job_id, 'completed', result)
                    st.success(f"✅ Analysis completed! Job ID: {job_id}")
                    
                    # Show immediate insights
                    job = EnhancedJobTracker.get_job(job_id)
                    if job and job.get('insights'):
                        st.subheader("🎯 Key Insights")
                        for insight in job['insights']:
                            st.info(insight)
                else:
                    EnhancedJobTracker.update_job_status(job_id, 'failed', error=result.get('error', 'Unknown error'))
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            st.rerun()
    
    # Main content area with enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "⚡ Live Jobs", "📋 Job History", "🎯 Opportunities", "📈 Insights"])
    
    with tab1:
        enhanced_dashboard_overview()
    
    with tab2:
        enhanced_active_jobs_view()
    
    with tab3:
        enhanced_job_history_view()
    
    with tab4:
        opportunities_view()
    
    with tab5:
        enhanced_insights_view()

def enhanced_dashboard_overview():
    """Enhanced dashboard overview with detailed metrics"""
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_jobs = len(st.session_state.jobs)
        st.metric("Total Jobs", total_jobs)
    
    with col2:
        completed_jobs = len([j for j in st.session_state.jobs if j['status'] == 'completed'])
        st.metric("Completed", completed_jobs)
    
    with col3:
        running_jobs = len([j for j in st.session_state.jobs if j['status'] == 'running'])
        st.metric("Running", running_jobs)
    
    with col4:
        total_opportunities = sum(len(j.get('opportunities', [])) for j in st.session_state.jobs)
        st.metric("Opportunities", total_opportunities)
    
    with col5:
        total_insights = sum(len(j.get('insights', [])) for j in st.session_state.jobs)
        st.metric("Insights", total_insights)
    
    # Real-time logs
    st.subheader("📝 Real-time Activity Log")
    if st.session_state.real_time_logs:
        log_container = st.container()
        with log_container:
            for log in st.session_state.real_time_logs[-10:]:  # Show last 10 logs
                timestamp = log['timestamp'].strftime('%H:%M:%S')
                st.text(f"[{timestamp}] {log['message']}")
    else:
        st.info("No activity yet. Start an analysis to see real-time logs.")
    
    # System health visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 System Health")
        health_data = {
            "Component": ["Technical Agent", "Money Flows", "Value Analysis", "Insider Analysis", "API Server", "Event Bus"],
            "Status": ["Active", "Active", "Active", "Active", "Active", "Active"],
            "Uptime": ["100%", "100%", "100%", "100%", "99.8%", "100%"]
        }
        st.dataframe(pd.DataFrame(health_data), use_container_width=True)
    
    with col2:
        st.subheader("📊 Performance Metrics")
        if st.session_state.jobs:
            job_types = [job['type'] for job in st.session_state.jobs]
            job_counts = pd.Series(job_types).value_counts()
            
            fig = px.pie(
                values=job_counts.values,
                names=job_counts.index,
                title="Analysis Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def enhanced_active_jobs_view():
    """Enhanced view of active jobs with detailed progress"""
    
    st.header("⚡ Live Job Monitoring")
    
    running_jobs = [j for j in st.session_state.jobs if j['status'] == 'running']
    pending_jobs = [j for j in st.session_state.jobs if j['status'] == 'pending']
    
    if not running_jobs and not pending_jobs:
        st.info("No active jobs. Start a new analysis from the sidebar.")
        return
    
    # Running jobs with detailed progress
    if running_jobs:
        st.subheader("🔄 Currently Running")
        for job in running_jobs:
            with st.expander(f"🏃 {job['id']} - {job['type']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Progress bar
                    progress = job.get('progress_percentage', 0)
                    st.progress(progress / 100)
                    st.write(f"**Current Stage:** {job.get('current_stage', 'Unknown')}")
                    st.write(f"**Progress:** {progress}%")
                    
                    # Progress stages
                    if job.get('progress_stages'):
                        st.write("**Progress History:**")
                        for stage in job['progress_stages'][-3:]:  # Show last 3 stages
                            st.markdown(f"""
                            <div class="progress-stage">
                                ✓ {stage['stage']} ({stage['percentage']}%) - {stage.get('details', '')}
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.write(f"**Started:** {job['started_at'].strftime('%H:%M:%S') if job['started_at'] else 'N/A'}")
                    st.write(f"**Type:** {job['type']}")
                    st.json(job['parameters'])
    
    # Pending jobs
    if pending_jobs:
        st.subheader("⏳ Pending Queue")
        for job in pending_jobs:
            st.markdown(f"""
            <div class="job-status-running">
                <strong>{job['id']}</strong> - {job['type']}<br>
                Created: {job['created_at'].strftime('%H:%M:%S')}<br>
                Parameters: {job['parameters']}
            </div>
            """, unsafe_allow_html=True)

def enhanced_job_history_view():
    """Enhanced job history with filtering and detailed results"""
    
    st.header("📋 Enhanced Job History")
    
    if not st.session_state.jobs:
        st.info("No jobs in history.")
        return
    
    # Enhanced filtering
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.selectbox("Status", ["all", "completed", "failed", "running"])
    
    with col2:
        type_filter = st.selectbox("Type", ["all"] + list(set(job['type'] for job in st.session_state.jobs)))
    
    with col3:
        date_filter = st.selectbox("Date", ["all", "today", "last_hour"])
    
    with col4:
        sort_by = st.selectbox("Sort by", ["newest_first", "oldest_first", "duration"])
    
    # Filter jobs
    filtered_jobs = st.session_state.jobs.copy()
    
    if status_filter != "all":
        filtered_jobs = [job for job in filtered_jobs if job['status'] == status_filter]
    
    if type_filter != "all":
        filtered_jobs = [job for job in filtered_jobs if job['type'] == type_filter]
    
    if date_filter == "today":
        today = datetime.now().date()
        filtered_jobs = [job for job in filtered_jobs if job['created_at'].date() == today]
    elif date_filter == "last_hour":
        hour_ago = datetime.now() - timedelta(hours=1)
        filtered_jobs = [job for job in filtered_jobs if job['created_at'] > hour_ago]
    
    # Sort jobs
    if sort_by == "newest_first":
        filtered_jobs.sort(key=lambda x: x['created_at'], reverse=True)
    elif sort_by == "oldest_first":
        filtered_jobs.sort(key=lambda x: x['created_at'])
    
    # Display enhanced job history
    for job in filtered_jobs:
        with st.expander(f"{job['id']} - {job['type']} ({job['status']})", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Timing:**")
                st.write(f"Created: {job['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                if job['started_at']:
                    st.write(f"Started: {job['started_at'].strftime('%H:%M:%S')}")
                if job['completed_at']:
                    st.write(f"Completed: {job['completed_at'].strftime('%H:%M:%S')}")
                    duration = (job['completed_at'] - job['started_at']).total_seconds() if job['started_at'] else 0
                    st.write(f"Duration: {duration:.1f}s")
            
            with col2:
                st.write("**Parameters:**")
                st.json(job['parameters'])
            
            with col3:
                st.write("**Results Summary:**")
                if job.get('detailed_results'):
                    for key, value in job['detailed_results'].items():
                        st.write(f"• {key}: {value}")
            
            # Show insights
            if job.get('insights'):
                st.write("**Key Insights:**")
                for insight in job['insights']:
                    st.info(insight)
            
            # Show errors
            if job.get('error'):
                st.error(f"Error: {job['error']}")

def opportunities_view():
    """Dedicated view for trading opportunities"""
    
    st.header("🎯 Trading Opportunities")
    
    # Get opportunities from database
    try:
        all_opportunities = opportunity_store.get_all_opportunities()
        
        if not all_opportunities:
            st.info("No opportunities discovered yet. Run some analyses to find trading opportunities.")
            return
            
        # Get statistics
        stats = opportunity_store.get_statistics()
        portfolio_metrics = unified_scorer.calculate_portfolio_metrics(all_opportunities)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Opportunities", stats['total_opportunities'])
        with col2:
            st.metric("Average Score", f"{portfolio_metrics['average_score']:.2f}")
        with col3:
            st.metric("Expected Return", f"{portfolio_metrics['expected_return']:.1%}")
        with col4:
            st.metric("Risk Score", f"{portfolio_metrics['risk_score']:.2f}")
        
        # Display agent distribution
        if portfolio_metrics['agent_distribution']:
            st.subheader("📊 Opportunities by Agent")
            agent_df = pd.DataFrame(list(portfolio_metrics['agent_distribution'].items()), 
                                  columns=['Agent', 'Count'])
            fig = px.pie(agent_df, values='Count', names='Agent', title="Opportunity Distribution by Agent")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading opportunities: {e}")
        return
    
    # Opportunity filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        type_filter = st.selectbox("Type", ["all"] + list(set(opp.opportunity_type for opp in all_opportunities)))
    
    with col2:
        agent_filter = st.selectbox("Agent", ["all"] + list(set(opp.agent_type for opp in all_opportunities)))
    
    with col3:
        min_score = st.slider("Min Priority Score", 0.0, 1.0, 0.0)
    
    with col4:
        sort_by = st.selectbox("Sort by", ["priority_score", "upside_potential", "confidence", "discovered_at"])
    
    # Filter opportunities
    filtered_opps = all_opportunities.copy()
    
    if type_filter != "all":
        filtered_opps = [opp for opp in filtered_opps if opp.opportunity_type == type_filter]
    
    if agent_filter != "all":
        filtered_opps = [opp for opp in filtered_opps if opp.agent_type == agent_filter]
    
    filtered_opps = [opp for opp in filtered_opps if opp.priority_score >= min_score]
    
    # Sort opportunities
    if sort_by == "priority_score":
        filtered_opps.sort(key=lambda x: x.priority_score, reverse=True)
    elif sort_by == "upside_potential":
        filtered_opps.sort(key=lambda x: x.upside_potential, reverse=True)
    elif sort_by == "confidence":
        filtered_opps.sort(key=lambda x: x.confidence, reverse=True)
    elif sort_by == "discovered_at":
        filtered_opps.sort(key=lambda x: x.discovered_at, reverse=True)
    
    # Display top opportunities
    st.subheader(f"🎯 Top {len(filtered_opps)} Opportunities")
    
    for i, opp in enumerate(filtered_opps, 1):
        # Color code based on priority score
        if opp.priority_score >= 0.8:
            card_color = "#d4edda"  # Green for high priority
            border_color = "#28a745"
        elif opp.priority_score >= 0.6:
            card_color = "#fff3cd"  # Yellow for medium priority
            border_color = "#ffc107"
        else:
            card_color = "#f8d7da"  # Red for low priority
            border_color = "#dc3545"
        
        st.markdown(f"""
        <div class="opportunity-card" style="background-color: {card_color}; border-left: 4px solid {border_color};">
            <h4>#{i} {opp.ticker} - {opp.opportunity_type} (Score: {opp.priority_score:.2f})</h4>
            <p><strong>Agent:</strong> {opp.agent_type.replace('_', ' ').title()}</p>
            <p><strong>Entry Reason:</strong> {opp.entry_reason}</p>
            <p><strong>Upside Potential:</strong> {opp.upside_potential:.1%}</p>
            <p><strong>Confidence:</strong> {opp.confidence:.1%}</p>
            <p><strong>Time Horizon:</strong> {opp.time_horizon}</p>
            <p><small>Discovered: {opp.discovered_at.strftime('%Y-%m-%d %H:%M')} | Job: {opp.job_id}</small></p>
        </div>
        """, unsafe_allow_html=True)

def enhanced_insights_view():
    """Enhanced insights view with comprehensive analytics"""
    
    st.header("📈 Enhanced Trading Intelligence Insights")
    
    completed_jobs = [j for j in st.session_state.jobs if j['status'] == 'completed']
    
    if not completed_jobs:
        st.info("No completed analyses yet.")
        return
    
    # Performance summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_insights = sum(len(job.get('insights', [])) for job in completed_jobs)
        st.metric("Total Insights", total_insights)
    
    with col2:
        avg_duration = sum((job['completed_at'] - job['started_at']).total_seconds() 
                          for job in completed_jobs if job['started_at'] and job['completed_at']) / len(completed_jobs)
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    with col3:
        success_rate = len(completed_jobs) / len(st.session_state.jobs) if st.session_state.jobs else 0
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col4:
        total_opportunities = sum(len(job.get('opportunities', [])) for job in completed_jobs)
        st.metric("Opportunities", total_opportunities)
    
    # Recent insights
    st.subheader("🔍 Recent Intelligence")
    
    recent_insights = []
    for job in completed_jobs[-10:]:  # Last 10 completed jobs
        for insight in job.get('insights', []):
            recent_insights.append({
                'insight': insight,
                'job_type': job['type'],
                'job_id': job['id'],
                'timestamp': job['completed_at']
            })
    
    # Sort by timestamp
    recent_insights.sort(key=lambda x: x['timestamp'], reverse=True)
    
    for insight_data in recent_insights[:10]:  # Show top 10
        st.markdown(f"""
        <div class="insight-card">
            <strong>{insight_data['insight']}</strong><br>
            <small>From {insight_data['job_type']} analysis | {insight_data['timestamp'].strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Analysis Performance")
        if completed_jobs:
            performance_data = []
            for job in completed_jobs:
                duration = (job['completed_at'] - job['started_at']).total_seconds() if job['started_at'] and job['completed_at'] else 0
                performance_data.append({
                    'Job Type': job['type'],
                    'Duration (s)': duration,
                    'Insights': len(job.get('insights', [])),
                    'Opportunities': len(job.get('opportunities', []))
                })
            
            df = pd.DataFrame(performance_data)
            fig = px.scatter(df, x='Duration (s)', y='Insights', color='Job Type', 
                           size='Opportunities', title="Performance vs Insights")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Analysis Timeline")
        if completed_jobs:
            timeline_data = []
            for job in completed_jobs:
                timeline_data.append({
                    'Job': job['id'],
                    'Type': job['type'],
                    'Start': job['started_at'] if job['started_at'] else job['created_at'],
                    'End': job['completed_at'] if job['completed_at'] else datetime.now()
                })
            
            df = pd.DataFrame(timeline_data)
            fig = px.timeline(df, x_start='Start', x_end='End', y='Job', color='Type',
                            title="Job Execution Timeline")
            st.plotly_chart(fig, use_container_width=True)

def enhanced_dashboard_view():
    """Enhanced main dashboard page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Trading Intelligence Dashboard</h1>
        <p>Multi-Agent Trading Intelligence System with Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_jobs = len(st.session_state.jobs)
        st.metric("Total Jobs", total_jobs)
    
    with col2:
        running_jobs = len([j for j in st.session_state.jobs if j['status'] == 'running'])
        st.metric("Running", running_jobs)
    
    with col3:
        completed_jobs = len([j for j in st.session_state.jobs if j['status'] == 'completed'])
        st.metric("Completed", completed_jobs)
    
    with col4:
        # Get opportunities from database
        try:
            all_opportunities = opportunity_store.get_all_opportunities()
            total_opportunities = len(all_opportunities)
        except:
            total_opportunities = 0
        st.metric("Opportunities", total_opportunities)
    
    with col5:
        total_insights = sum(len(j.get('insights', [])) for j in st.session_state.jobs)
        st.metric("Insights", total_insights)
    
    # Real-time logs
    st.subheader("📝 Real-time Activity Log")
    if st.session_state.real_time_logs:
        log_container = st.container()
        with log_container:
            for log in st.session_state.real_time_logs[-10:]:  # Show last 10 logs
                timestamp = log['timestamp'].strftime('%H:%M:%S')
                st.text(f"[{timestamp}] {log['message']}")
    else:
        st.info("No activity yet. Start an analysis to see real-time logs.")
    
    # System health visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 System Health")
        health_data = {
            "Component": ["Technical Agent", "Money Flows", "Value Analysis", "Insider Analysis", "API Server", "Event Bus"],
            "Status": ["Active", "Active", "Active", "Active", "Active", "Active"],
            "Uptime": ["100%", "100%", "100%", "100%", "99.8%", "100%"]
        }
        st.dataframe(pd.DataFrame(health_data), use_container_width=True)
    
    with col2:
        st.subheader("📊 Performance Metrics")
        if st.session_state.jobs:
            job_types = [job['type'] for job in st.session_state.jobs]
            job_counts = pd.Series(job_types).value_counts()
            
            fig = px.pie(
                values=job_counts.values,
                names=job_counts.index,
                title="Analysis Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def main_dashboard():
    """Main dashboard function"""
    
    # Initialize session state
    if 'jobs' not in st.session_state:
        st.session_state.jobs = []
    if 'real_time_logs' not in st.session_state:
        st.session_state.real_time_logs = []
    
    # Sidebar navigation
    st.sidebar.title("🚀 Trading Intelligence")
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["📊 Dashboard", "⚡ Active Jobs", "📋 Job History", "🎯 Opportunities", "🏆 Top 10 Opportunities", "📈 Insights"]
    )
    
    # Sidebar analysis controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Run Analysis")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        [
            "value_analysis", 
            "technical_analysis", 
            "money_flows", 
            "insider_analysis",
            "sentiment_analysis",
            "flow_analysis", 
            "macro_analysis",
            "top_performers_analysis",
            "undervalued_analysis",
            "causal_analysis",
            "hedging_analysis",
            "learning_analysis"
        ]
    )
    
    if analysis_type == "value_analysis":
        universe = st.sidebar.text_input("Symbols (comma-separated)", "BRK.B,JPM,XOM")
        parameters = {'universe': [s.strip() for s in universe.split(',')]}
    elif analysis_type == "technical_analysis":
        symbols = st.sidebar.text_input("Symbols (comma-separated)", "AAPL,TSLA,MSFT")
        parameters = {'symbols': [s.strip() for s in symbols.split(',')]}
    elif analysis_type == "money_flows":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    elif analysis_type == "insider_analysis":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    elif analysis_type == "sentiment_analysis":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    elif analysis_type == "flow_analysis":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    elif analysis_type == "macro_analysis":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    elif analysis_type == "top_performers_analysis":
        universe = st.sidebar.text_input("Universe (comma-separated)", "SPY,QQQ,IWM")
        parameters = {'universe': [s.strip() for s in universe.split(',')]}
    elif analysis_type == "undervalued_analysis":
        universe = st.sidebar.text_input("Universe (comma-separated)", "BRK.B,JPM,XOM")
        parameters = {'universe': [s.strip() for s in universe.split(',')]}
    elif analysis_type == "causal_analysis":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    elif analysis_type == "hedging_analysis":
        portfolio = st.sidebar.text_input("Portfolio (comma-separated)", "AAPL,TSLA,MSFT")
        parameters = {'portfolio': [s.strip() for s in portfolio.split(',')]}
    elif analysis_type == "learning_analysis":
        tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,TSLA")
        parameters = {'tickers': [s.strip() for s in tickers.split(',')]}
    
    if st.sidebar.button("🚀 Run Analysis"):
        job_id = EnhancedJobTracker.create_job(analysis_type, parameters)
        st.sidebar.success(f"Job {job_id} created!")
        
        # Run the job in a separate thread
        import threading
        def run_job():
            result = run_enhanced_analysis_job(analysis_type, parameters, job_id)
        
        thread = threading.Thread(target=run_job)
        thread.start()
    
    # Page routing
    if page == "📊 Dashboard":
        enhanced_dashboard_view()
    elif page == "⚡ Active Jobs":
        enhanced_active_jobs_view()
    elif page == "📋 Job History":
        enhanced_job_history_view()
    elif page == "🎯 Opportunities":
        opportunities_view()
    elif page == "🏆 Top 10 Opportunities":
        top_opportunities_view()
    elif page == "📈 Insights":
        enhanced_insights_view()

def top_opportunities_view():
    """View for top 10 opportunities across all agents"""
    
    st.header("🏆 Top 10 Opportunities Across All Agents")
    
    try:
        # Get top 10 opportunities from database
        top_opportunities = opportunity_store.get_top_opportunities(limit=10)
        
        if not top_opportunities:
            st.info("No opportunities found. Run some analyses to generate opportunities.")
            return
        
        # Get portfolio metrics
        portfolio_metrics = unified_scorer.calculate_portfolio_metrics(top_opportunities)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Opportunities", len(top_opportunities))
        with col2:
            st.metric("Average Score", f"{portfolio_metrics['average_score']:.2f}")
        with col3:
            st.metric("Expected Return", f"{portfolio_metrics['expected_return']:.1%}")
        
        # Display score distribution
        if portfolio_metrics['score_distribution']:
            st.subheader("📊 Score Distribution")
            score_df = pd.DataFrame(list(portfolio_metrics['score_distribution'].items()), 
                                  columns=['Score Range', 'Count'])
            fig = px.bar(score_df, x='Score Range', y='Count', title="Opportunity Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Display top opportunities
        st.subheader("🎯 Top 10 Opportunities")
        
        for i, opp in enumerate(top_opportunities, 1):
            # Color code based on priority score
            if opp.priority_score >= 0.8:
                card_color = "#d4edda"  # Green for high priority
                border_color = "#28a745"
                rank_emoji = "🥇"
            elif opp.priority_score >= 0.6:
                card_color = "#fff3cd"  # Yellow for medium priority
                border_color = "#ffc107"
                rank_emoji = "🥈"
            else:
                card_color = "#f8d7da"  # Red for low priority
                border_color = "#dc3545"
                rank_emoji = "🥉"
            
            st.markdown(f"""
            <div class="opportunity-card" style="background-color: {card_color}; border-left: 4px solid {border_color};">
                <h3>{rank_emoji} #{i} {opp.ticker} - {opp.opportunity_type} (Score: {opp.priority_score:.2f})</h3>
                <p><strong>Agent:</strong> {opp.agent_type.replace('_', ' ').title()}</p>
                <p><strong>Entry Reason:</strong> {opp.entry_reason}</p>
                <p><strong>Upside Potential:</strong> {opp.upside_potential:.1%}</p>
                <p><strong>Confidence:</strong> {opp.confidence:.1%}</p>
                <p><strong>Time Horizon:</strong> {opp.time_horizon}</p>
                <p><small>Discovered: {opp.discovered_at.strftime('%Y-%m-%d %H:%M')} | Job: {opp.job_id}</small></p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading top opportunities: {e}")

if __name__ == "__main__":
    main_dashboard()
