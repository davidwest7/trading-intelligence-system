#!/usr/bin/env python3
"""
Small-Scale Test - Enhanced Multi-Timeframe Technical Agent

Tests the enhanced Technical Agent with multi-timeframe analysis and liquidity gap detection.
"""

import asyncio
import time
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

# Load environment variables
load_dotenv('env_real_keys.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTechnicalAgentTest:
    """Test class for the enhanced multi-timeframe Technical Agent"""
    
    def __init__(self):
        self.config = {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
            'twitter_api_key': os.getenv('TWITTER_API_KEY'),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID')
        }
        
        # Test tickers (small scale)
        self.test_tickers = ['AAPL', 'TSLA', 'SPY']
        
    async def run_enhanced_technical_test(self):
        """Run enhanced technical agent test"""
        print("🔧 **ENHANCED MULTI-TIMEFRAME TECHNICAL AGENT TEST**")
        print("=" * 60)
        print(f"Testing enhanced Technical Agent with multi-timeframe analysis")
        print(f"Test Tickers: {', '.join(self.test_tickers)}")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Import and initialize enhanced agent
            from agents.technical.agent_enhanced_multi_timeframe import EnhancedMultiTimeframeTechnicalAgent
            enhanced_agent = EnhancedMultiTimeframeTechnicalAgent(self.config)
            print("✅ Enhanced Multi-Timeframe Technical Agent initialized")
            
            # Run analysis
            print(f"\n🔍 **ANALYZING {len(self.test_tickers)} TICKERS**")
            print("-" * 40)
            
            results = await enhanced_agent.analyze_multi_timeframe_technical_indicators(self.test_tickers)
            
            # Display results
            await self._display_enhanced_results(results)
            
            total_time = time.time() - start_time
            
            # Summary
            print(f"\n" + "=" * 60)
            print("📊 **ENHANCED TECHNICAL AGENT TEST SUMMARY**")
            print("=" * 60)
            print(f"✅ Tickers Analyzed: {results['tickers_analyzed']}")
            print(f"⏱️  Processing Time: {total_time:.2f} seconds")
            print(f"🌐 Data Source: {results['data_source']}")
            print(f"📈 Overall Sentiment: {results['overall_signals']['overall_sentiment']}")
            print(f"🎯 Total Signals: {results['overall_signals']['total_signals']}")
            print(f"🔄 Multi-Timeframe Alignment: {results['overall_signals']['multi_timeframe_alignment']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Enhanced Technical Agent test failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _display_enhanced_results(self, results):
        """Display enhanced technical analysis results"""
        
        for ticker, analysis in results['technical_analysis'].items():
            print(f"\n📊 **{ticker} ANALYSIS**")
            print("-" * 30)
            
            # Basic info
            print(f"💰 Current Price: ${analysis['current_price']:.2f}")
            print(f"📈 Change: {analysis['change_percent']:.2f}%")
            print(f"📊 Volume: {analysis['volume']:,}")
            
            # Multi-timeframe consensus
            consensus = analysis.get('multi_timeframe', {}).get('consensus', {})
            if consensus:
                print(f"\n🔄 **MULTI-TIMEFRAME CONSENSUS:**")
                print(f"   Weighted RSI: {consensus.get('weighted_rsi', 0):.1f}")
                print(f"   Weighted Trend: {consensus.get('weighted_trend', 'neutral')}")
                print(f"   Trend Agreement: {consensus.get('trend_agreement', 0):.1%}")
                print(f"   Signal Strength: {consensus.get('signal_strength', 0):.1%}")
            
            # Liquidity gaps
            liquidity = analysis.get('liquidity_gaps', {})
            if liquidity:
                print(f"\n🌊 **LIQUIDITY GAPS:**")
                
                # Order book imbalance
                imbalance = liquidity.get('order_book_imbalance', {})
                if imbalance:
                    print(f"   Order Book Pressure: {imbalance.get('pressure', 'neutral')}")
                    print(f"   Imbalance Ratio: {imbalance.get('imbalance_ratio', 0):.2f}")
                
                # Price gaps
                price_gaps = liquidity.get('price_gaps', [])
                if price_gaps:
                    print(f"   Price Gaps Found: {len(price_gaps)}")
                    for gap in price_gaps[:2]:  # Show first 2 gaps
                        print(f"     {gap.get('gap_type', 'unknown')} gap: {gap.get('gap_percentage', 0):.1f}% {'(filled)' if gap.get('filled') else '(unfilled)'}")
                
                # Volume gaps
                volume_gaps = liquidity.get('volume_gaps', [])
                if volume_gaps:
                    print(f"   Volume Anomalies: {len(volume_gaps)}")
                    for vgap in volume_gaps[:2]:  # Show first 2 anomalies
                        print(f"     {vgap.get('anomaly_type', 'unknown')}: {vgap.get('volume_ratio', 0):.1f}x (z-score: {vgap.get('z_score', 0):.1f})")
            
            # Volume profile
            volume_profile = analysis.get('volume_profile', {})
            if volume_profile:
                print(f"\n📈 **VOLUME PROFILE:**")
                
                # Volume distribution
                dist = volume_profile.get('volume_distribution', {})
                if dist:
                    print(f"   Mean Volume: {dist.get('mean_volume', 0):,.0f}")
                    print(f"   Volume Volatility: {dist.get('volume_volatility', 0):.2f}")
                
                # Price-volume relationship
                pv_rel = volume_profile.get('price_volume_relationship', {})
                if pv_rel:
                    print(f"   Price-Volume Correlation: {pv_rel.get('correlation', 0):.2f}")
                    print(f"   Relationship: {pv_rel.get('relationship', 'neutral')}")
                
                # Volume trend
                v_trend = volume_profile.get('volume_trend', {})
                if v_trend:
                    print(f"   Volume Trend: {v_trend.get('trend', 'neutral')}")
                    print(f"   Trend Strength: {v_trend.get('strength', 0):.2f}")
            
            # Consolidated signals
            signals = analysis.get('consolidated_signals', [])
            if signals:
                print(f"\n🎯 **CONSOLIDATED SIGNALS ({len(signals)}):**")
                for signal in signals[:3]:  # Show first 3 signals
                    strength_emoji = "🔴" if signal.get('strength') == 'strong' else "🟡" if signal.get('strength') == 'medium' else "🟢"
                    print(f"   {strength_emoji} {signal.get('type', 'unknown')}: {signal.get('message', 'No message')}")
            
            # Timeframe breakdown
            timeframes = analysis.get('multi_timeframe', {}).get('timeframes', {})
            if timeframes:
                print(f"\n⏰ **TIMEFRAME BREAKDOWN:**")
                for tf_name, tf_data in timeframes.items():
                    if tf_data.get('data_points', 0) > 0:
                        indicators = tf_data.get('indicators', {})
                        print(f"   {tf_name}: RSI {indicators.get('rsi', 0):.1f}, Trend {indicators.get('trend', 'neutral')}, {len(indicators.get('timeframe_signals', []))} signals")


async def main():
    """Main test function"""
    print("🚀 **ENHANCED TECHNICAL AGENT SMALL-SCALE TEST**")
    print("=" * 60)
    print("Testing Enhanced Multi-Timeframe Technical Agent")
    print("=" * 60)
    
    # Initialize test
    test = EnhancedTechnicalAgentTest()
    
    # Run enhanced technical test
    results = await test.run_enhanced_technical_test()
    
    if results:
        print(f"\n🎉 Enhanced Technical Agent test completed successfully!")
        print(f"📊 Results available in test results")
    else:
        print(f"\n❌ Enhanced Technical Agent test failed!")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
