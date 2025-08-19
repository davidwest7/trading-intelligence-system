"""
Fixed Enhanced Technical Analysis Agent with Realistic Market Data
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

from .models import TechnicalOpportunity, Direction, VolatilityRegime
from .strategies_fixed import FixedEnhancedTechnicalStrategies, ImbalanceLevel, TrendDirection
from common.opportunity_store import OpportunityStore
from common.unified_opportunity_scorer_enhanced import EnhancedUnifiedOpportunityScorer


class FixedEnhancedTechnicalAgent:
    """
    Fixed Enhanced Technical Analysis Agent with realistic market data and improved algorithms
    """
    
    def __init__(self):
        self.strategies = FixedEnhancedTechnicalStrategies()
        self.opportunity_store = OpportunityStore()
        self.scorer = EnhancedUnifiedOpportunityScorer()
        self.min_confidence = 0.4  # Minimum confidence threshold
        self.max_opportunities_per_symbol = 3  # Maximum opportunities per symbol
        
    async def find_opportunities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find technical trading opportunities with enhanced algorithms
        """
        try:
            start_time = time.time()
            
            symbols = payload.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])
            timeframes = payload.get('timeframes', ['1h', '4h'])
            strategies = payload.get('strategies', ['imbalance', 'trend', 'liquidity'])
            
            print(f"🔍 Analyzing {len(symbols)} symbols with {len(strategies)} strategies...")
            
            all_opportunities = []
            successful_analyses = 0
            
            for symbol in symbols:
                symbol_opportunities = []
                
                # Analyze each strategy
                for strategy in strategies:
                    try:
                        opportunity = await self._analyze_symbol_strategy(symbol, strategy, timeframes)
                        if opportunity and opportunity.confidence_score >= self.min_confidence:
                            symbol_opportunities.append(opportunity)
                    except Exception as e:
                        print(f"Error analyzing {symbol} with {strategy}: {e}")
                        continue
                
                # Limit opportunities per symbol
                if symbol_opportunities:
                    # Sort by confidence and take top opportunities
                    symbol_opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
                    symbol_opportunities = symbol_opportunities[:self.max_opportunities_per_symbol]
                    
                    all_opportunities.extend(symbol_opportunities)
                    successful_analyses += 1
                
                # Rate limiting to avoid overwhelming APIs
                await asyncio.sleep(0.1)
            
            # Calculate priority scores and store opportunities
            for opportunity in all_opportunities:
                opportunity.priority_score = self.scorer.calculate_priority_score(opportunity)
                self.opportunity_store.add_opportunity(opportunity)
            
            # Sort by priority score
            all_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Calculate metadata
            analysis_time = time.time() - start_time
            avg_confidence = np.mean([opp.confidence_score for opp in all_opportunities]) if all_opportunities else 0
            avg_priority = np.mean([opp.priority_score for opp in all_opportunities]) if all_opportunities else 0
            
            metadata = {
                'analysis_time': analysis_time,
                'symbols_analyzed': len(symbols),
                'successful_analyses': successful_analyses,
                'opportunities_found': len(all_opportunities),
                'average_confidence': avg_confidence,
                'average_priority_score': avg_priority,
                'strategies_used': strategies,
                'timeframes_used': timeframes
            }
            
            print(f"✅ Found {len(all_opportunities)} opportunities in {analysis_time:.2f}s")
            print(f"📊 Average confidence: {avg_confidence:.2%}, Priority: {avg_priority:.2%}")
            
            return {
                'opportunities': [self._opportunity_to_dict(opp) for opp in all_opportunities],
                'metadata': metadata,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in find_opportunities: {e}")
            return {
                'opportunities': [],
                'metadata': {'error': str(e)},
                'success': False
            }
    
    async def _analyze_symbol_strategy(self, symbol: str, strategy: str, 
                                     timeframes: List[str]) -> Optional[TechnicalOpportunity]:
        """
        Analyze a specific symbol with a specific strategy
        """
        try:
            if strategy == 'imbalance':
                return await self._analyze_imbalances(symbol, timeframes)
            elif strategy == 'trend':
                return await self._analyze_trends(symbol, timeframes)
            elif strategy == 'liquidity':
                return await self._analyze_liquidity_sweeps(symbol, timeframes)
            else:
                print(f"Unknown strategy: {strategy}")
                return None
                
        except Exception as e:
            print(f"Error in _analyze_symbol_strategy for {symbol} {strategy}: {e}")
            return None
    
    async def _analyze_imbalances(self, symbol: str, timeframes: List[str]) -> Optional[TechnicalOpportunity]:
        """
        Analyze imbalances for a symbol
        """
        try:
            # Analyze multiple timeframes
            all_imbalances = []
            for timeframe in timeframes:
                imbalances = await self.strategies.find_imbalances(symbol, timeframe, 5)
                all_imbalances.extend(imbalances)
            
            if not all_imbalances:
                return None
            
            # Find the strongest imbalance
            strongest_imbalance = max(all_imbalances, key=lambda x: x.strength)
            
            # Create opportunity
            opportunity = await self.strategies.create_opportunity(
                symbol=symbol,
                strategy='imbalance',
                imbalance=strongest_imbalance
            )
            
            return opportunity
            
        except Exception as e:
            print(f"Error analyzing imbalances for {symbol}: {e}")
            return None
    
    async def _analyze_trends(self, symbol: str, timeframes: List[str]) -> Optional[TechnicalOpportunity]:
        """
        Analyze trends for a symbol
        """
        try:
            # Analyze primary timeframe
            primary_timeframe = timeframes[0]
            trend = await self.strategies.detect_trends(symbol, primary_timeframe, 10)
            
            if trend.strength < 0.6:  # Minimum trend strength
                return None
            
            # Create opportunity
            opportunity = await self.strategies.create_opportunity(
                symbol=symbol,
                strategy='trend',
                trend=trend
            )
            
            return opportunity
            
        except Exception as e:
            print(f"Error analyzing trends for {symbol}: {e}")
            return None
    
    async def _analyze_liquidity_sweeps(self, symbol: str, timeframes: List[str]) -> Optional[TechnicalOpportunity]:
        """
        Analyze liquidity sweeps for a symbol
        """
        try:
            # Analyze multiple timeframes
            all_sweeps = []
            for timeframe in timeframes:
                sweeps = await self.strategies.find_liquidity_sweeps(symbol, timeframe, 3)
                all_sweeps.extend(sweeps)
            
            if not all_sweeps:
                return None
            
            # Find the strongest sweep
            strongest_sweep = max(all_sweeps, key=lambda x: x['strength'])
            
            # Create opportunity based on sweep
            if strongest_sweep['strength'] > 0.5:
                opportunity = await self.strategies.create_opportunity(
                    symbol=symbol,
                    strategy='liquidity',
                    sweeps=[strongest_sweep]
                )
                return opportunity
            
            return None
            
        except Exception as e:
            print(f"Error analyzing liquidity sweeps for {symbol}: {e}")
            return None
    
    def _opportunity_to_dict(self, opportunity: TechnicalOpportunity) -> Dict[str, Any]:
        """
        Convert opportunity to dictionary for API response
        """
        return {
            'symbol': opportunity.symbol,
            'strategy': opportunity.strategy,
            'direction': opportunity.direction.value,
            'entry_price': opportunity.entry_price,
            'stop_loss': opportunity.stop_loss,
            'take_profit': opportunity.take_profit,
            'risk_reward_ratio': opportunity.risk_reward_ratio,
            'confidence_score': opportunity.confidence_score,
            'priority_score': opportunity.priority_score,
            'timestamp': opportunity.timestamp.isoformat(),
            'timeframe_alignment': {
                'primary': opportunity.timeframe_alignment.primary,
                'confirmation': opportunity.timeframe_alignment.confirmation,
                'alignment_score': opportunity.timeframe_alignment.alignment_score
            },
            'technical_features': {
                'trend_strength': opportunity.technical_features.trend_strength,
                'volatility_regime': opportunity.technical_features.volatility_regime.value,
                'imbalance_zones': len(opportunity.technical_features.imbalance_zones),
                'liquidity_levels': len(opportunity.technical_features.liquidity_levels)
            },
            'risk_metrics': {
                'max_loss': opportunity.risk_metrics.max_loss,
                'position_size': opportunity.risk_metrics.position_size,
                'sharpe_ratio': opportunity.risk_metrics.sharpe_ratio,
                'max_drawdown': opportunity.risk_metrics.max_drawdown
            }
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics
        """
        try:
            # Get opportunities from store
            opportunities = self.opportunity_store.get_all_opportunities()
            tech_opportunities = [opp for opp in opportunities if opp.agent_type == 'technical']
            
            if not tech_opportunities:
                return {
                    'total_opportunities': 0,
                    'average_confidence': 0,
                    'average_priority_score': 0,
                    'success_rate': 0
                }
            
            avg_confidence = np.mean([opp.confidence for opp in tech_opportunities])
            avg_priority = np.mean([opp.priority_score for opp in tech_opportunities])
            
            return {
                'total_opportunities': len(tech_opportunities),
                'average_confidence': avg_confidence,
                'average_priority_score': avg_priority,
                'success_rate': len([opp for opp in tech_opportunities if opp.confidence > 0.6]) / len(tech_opportunities)
            }
            
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
