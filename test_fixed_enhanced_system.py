"""
Comprehensive Test of Fixed Enhanced Trading Intelligence System
"""

import asyncio
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append('.')

from agents.technical.agent_fixed import FixedEnhancedTechnicalAgent
from common.data_adapters.yfinance_adapter_fixed import FixedYFinanceAdapter
from common.unified_opportunity_scorer_enhanced import EnhancedUnifiedOpportunityScorer
from common.opportunity_store import OpportunityStore


async def test_fixed_enhanced_system():
    """Test the fixed enhanced trading intelligence system"""
    
    print("🚀 FIXED ENHANCED TRADING INTELLIGENCE SYSTEM TEST")
    print("=" * 65)
    
    # Initialize components
    print("\n1. Initializing Fixed Enhanced Components...")
    
    # Test Fixed YFinance Adapter
    print("   Testing Fixed YFinance Adapter...")
    yf_adapter = FixedYFinanceAdapter({})
    connected = await yf_adapter.connect()
    print(f"   ✓ YFinance Connection: {'SUCCESS' if connected else 'FAILED'}")
    
    # Test Enhanced Scorer
    print("   Testing Enhanced Scorer...")
    enhanced_scorer = EnhancedUnifiedOpportunityScorer()
    print(f"   ✓ Enhanced Scorer: {len(enhanced_scorer.agent_weights)} agents supported")
    
    # Test Opportunity Store
    print("   Testing Opportunity Store...")
    store = OpportunityStore()
    stats = store.get_statistics()
    print(f"   ✓ Opportunity Store: {stats['total_opportunities']} opportunities")
    
    # Test Fixed Enhanced Technical Agent
    print("   Testing Fixed Enhanced Technical Agent...")
    tech_agent = FixedEnhancedTechnicalAgent()
    print(f"   ✓ Fixed Enhanced Technical Agent: Initialized")
    
    print("\n2. Testing Market Data Integration...")
    
    # Test market data fetching
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    for symbol in symbols:
        try:
            quote = await yf_adapter.get_quote(symbol)
            print(f"   ✓ {symbol}: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
        except Exception as e:
            print(f"   ✗ {symbol}: Error - {e}")
    
    print("\n3. Testing Fixed Enhanced Technical Analysis...")
    
    # Test technical analysis with realistic data
    start_time = time.time()
    
    result = await tech_agent.find_opportunities({
        'symbols': symbols,
        'timeframes': ['1h', '4h'],
        'strategies': ['imbalance', 'trend', 'liquidity']
    })
    
    analysis_time = time.time() - start_time
    
    print(f"   Analysis completed in {analysis_time:.2f} seconds")
    print(f"   Opportunities found: {len(result['opportunities'])}")
    
    if result['opportunities']:
        print("\n   Top Opportunities:")
        for i, opp in enumerate(result['opportunities'][:5]):
            print(f"   {i+1}. {opp['symbol']} - {opp['strategy']} - "
                  f"Confidence: {opp['confidence_score']:.2%} - "
                  f"Priority: {opp['priority_score']:.2%}")
    
    print("\n4. Testing Enhanced Scoring System...")
    
    # Get all opportunities and test enhanced scoring
    all_opportunities = store.get_all_opportunities()
    
    if all_opportunities:
        # Calculate enhanced scores
        for opp in all_opportunities:
            opp.priority_score = enhanced_scorer.calculate_priority_score(opp)
        
        # Get top opportunities
        top_opportunities = enhanced_scorer.get_top_opportunities(all_opportunities, 10)
        
        print(f"   Total opportunities: {len(all_opportunities)}")
        print(f"   Top 10 average score: {np.mean([opp.priority_score for opp in top_opportunities]):.2%}")
        
        # Calculate portfolio metrics
        portfolio_metrics = enhanced_scorer.calculate_portfolio_metrics(all_opportunities)
        
        print(f"   Portfolio average score: {portfolio_metrics['average_score']:.2%}")
        print(f"   High confidence ratio: {portfolio_metrics['risk_metrics']['high_confidence_ratio']:.2%}")
        
        print("\n   Score Distribution:")
        for range_name, count in portfolio_metrics['score_distribution'].items():
            if count > 0:
                print(f"     {range_name}: {count} opportunities")
    
    print("\n5. Performance Metrics...")
    
    # Get technical agent performance
    tech_performance = await tech_agent.get_performance_metrics()
    
    print(f"   Technical Agent Performance:")
    print(f"     Total opportunities: {tech_performance['total_opportunities']}")
    print(f"     Average confidence: {tech_performance['average_confidence']:.2%}")
    print(f"     Average priority score: {tech_performance['average_priority_score']:.2%}")
    print(f"     Success rate: {tech_performance['success_rate']:.2%}")
    
    print("\n6. System Health Check...")
    
    # Overall system health
    total_opportunities = store.get_statistics()['total_opportunities']
    avg_priority = store.get_statistics()['average_priority_score']
    
    print(f"   Total opportunities in system: {total_opportunities}")
    print(f"   Average priority score: {avg_priority:.2%}")
    print(f"   Analysis time: {analysis_time:.2f}s")
    
    # Calculate improvement metrics
    improvement_metrics = {
        'opportunity_generation': len(result['opportunities']) > 5,  # More than 5 opportunities
        'priority_score_quality': avg_priority > 0.5,  # Average score > 50%
        'analysis_speed': analysis_time < 15,  # Analysis < 15 seconds
        'data_quality': connected  # YFinance connection successful
    }
    
    print(f"\n   Improvement Status:")
    for metric, status in improvement_metrics.items():
        status_icon = "✅" if status else "❌"
        print(f"     {status_icon} {metric.replace('_', ' ').title()}")
    
    success_rate = sum(improvement_metrics.values()) / len(improvement_metrics)
    print(f"\n   Overall Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.75:
        print("   �� FIXED ENHANCED SYSTEM: EXCELLENT PERFORMANCE")
    elif success_rate >= 0.5:
        print("   ✅ FIXED ENHANCED SYSTEM: GOOD PERFORMANCE")
    else:
        print("   ⚠️  FIXED ENHANCED SYSTEM: NEEDS IMPROVEMENT")
    
    print("\n" + "=" * 65)
    print("🏁 FIXED ENHANCED SYSTEM TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_fixed_enhanced_system())
