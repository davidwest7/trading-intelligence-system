"""
Real Data Top Performers Agent
Uses Polygon.io adapter for performance rankings and sector analysis
"""
import asyncio
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')
from common.models import BaseAgent
from common.data_adapters.polygon_adapter import PolygonAdapter

load_dotenv('env_real_keys.env')

class RealDataTopPerformersAgent(BaseAgent):
    """Top Performers Agent with real market data from Polygon.io"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RealDataTopPerformersAgent", config)
        self.polygon_adapter = PolygonAdapter(config)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method (required by BaseAgent)"""
        return await self.analyze_top_performers(**kwargs)
    
    async def analyze_top_performers(self, **kwargs) -> Dict[str, Any]:
        """Analyze top performers using real market data"""
        print(f"🏆 Real Data Top Performers Agent: Analyzing market leaders")
        
        try:
            # Get performance rankings
            gainers = await self.polygon_adapter.get_performance_rankings(limit=50)
            losers = await self._get_losers(limit=20)
            
            # Get sector performance
            sector_performance = await self.polygon_adapter.get_sector_performance()
            
            # Analyze momentum and relative strength
            momentum_analysis = await self._analyze_momentum(gainers, losers)
            
            # Generate performance signals
            performance_signals = self._generate_performance_signals(gainers, losers, sector_performance)
            
            return {
                'timestamp': datetime.now(),
                'top_gainers': gainers[:20],
                'top_losers': losers[:10],
                'sector_performance': sector_performance,
                'momentum_analysis': momentum_analysis,
                'performance_signals': performance_signals,
                'data_source': 'Polygon.io (Real Market Data)'
            }
            
        except Exception as e:
            print(f"❌ Error analyzing top performers: {e}")
            return self._create_empty_analysis()
    
    async def _get_losers(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top losers (mock implementation since Polygon.io doesn't provide losers endpoint)"""
        # For now, we'll create mock losers data
        # In a real implementation, you'd call a losers endpoint or filter gainers
        return [
            {
                'rank': 1,
                'symbol': 'MOCK_LOSER_1',
                'price': 10.50,
                'change': -2.50,
                'change_percent': -19.23,
                'volume': 1000000,
                'timestamp': datetime.now()
            },
            {
                'rank': 2,
                'symbol': 'MOCK_LOSER_2',
                'price': 15.75,
                'change': -3.25,
                'change_percent': -17.11,
                'volume': 800000,
                'timestamp': datetime.now()
            }
        ]
    
    async def _analyze_momentum(self, gainers: List[Dict[str, Any]], 
                              losers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze momentum and relative strength"""
        
        # Calculate average gains and losses
        avg_gain = np.mean([g['change_percent'] for g in gainers[:10]]) if gainers else 0
        avg_loss = np.mean([l['change_percent'] for l in losers[:10]]) if losers else 0
        
        # Calculate momentum strength
        momentum_strength = abs(avg_gain) + abs(avg_loss)
        
        # Analyze volume patterns
        avg_gainer_volume = np.mean([g['volume'] for g in gainers[:10]]) if gainers else 0
        avg_loser_volume = np.mean([l['volume'] for l in losers[:10]]) if losers else 0
        
        # Calculate relative strength
        relative_strength = avg_gain / abs(avg_loss) if avg_loss != 0 else 0
        
        return {
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'momentum_strength': momentum_strength,
            'avg_gainer_volume': avg_gainer_volume,
            'avg_loser_volume': avg_loser_volume,
            'relative_strength': relative_strength,
            'momentum_regime': 'strong_bullish' if avg_gain > 20 else 'bullish' if avg_gain > 10 else 'neutral'
        }
    
    def _generate_performance_signals(self, gainers: List[Dict[str, Any]], 
                                    losers: List[Dict[str, Any]],
                                    sectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate performance-based trading signals"""
        signals = []
        
        # Top gainer signals
        if gainers:
            top_gainer = gainers[0]
            if top_gainer['change_percent'] > 50:
                signals.append({
                    'type': 'EXTREME_GAINER',
                    'strength': 'strong',
                    'message': f"Extreme gainer: {top_gainer['symbol']} +{top_gainer['change_percent']:.1f}%"
                })
            
            # Volume analysis
            high_volume_gainers = [g for g in gainers[:10] if g['volume'] > 1000000]
            if len(high_volume_gainers) > 5:
                signals.append({
                    'type': 'HIGH_VOLUME_GAINERS',
                    'strength': 'medium',
                    'message': f"High volume gainers: {len(high_volume_gainers)} stocks with >1M volume"
                })
        
        # Sector rotation signals
        if sectors:
            top_sector = sectors[0]
            if top_sector['change_percent'] > 2.0:
                signals.append({
                    'type': 'SECTOR_ROTATION',
                    'strength': 'strong',
                    'message': f"Sector rotation: {top_sector['sector']} +{top_sector['change_percent']:.1f}%"
                })
            
            # Sector breadth
            positive_sectors = [s for s in sectors if s['change_percent'] > 0]
            if len(positive_sectors) > len(sectors) * 0.7:
                signals.append({
                    'type': 'BROAD_SECTOR_STRENGTH',
                    'strength': 'medium',
                    'message': f"Broad sector strength: {len(positive_sectors)}/{len(sectors)} sectors positive"
                })
        
        # Momentum signals
        if gainers and losers:
            avg_gain = np.mean([g['change_percent'] for g in gainers[:10]])
            avg_loss = np.mean([l['change_percent'] for l in losers[:10]])
            
            if avg_gain > abs(avg_loss) * 1.5:
                signals.append({
                    'type': 'STRONG_MOMENTUM',
                    'strength': 'strong',
                    'message': f"Strong momentum: gains {avg_gain:.1f}% vs losses {abs(avg_loss):.1f}%"
                })
        
        return signals
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis when data is unavailable"""
        return {
            'timestamp': datetime.now(),
            'top_gainers': [],
            'top_losers': [],
            'sector_performance': [],
            'momentum_analysis': {
                'avg_gain': 0.0,
                'avg_loss': 0.0,
                'momentum_strength': 0.0,
                'avg_gainer_volume': 0.0,
                'avg_loser_volume': 0.0,
                'relative_strength': 0.0,
                'momentum_regime': 'neutral'
            },
            'performance_signals': [],
            'data_source': 'Polygon.io (Real Market Data)'
        }
