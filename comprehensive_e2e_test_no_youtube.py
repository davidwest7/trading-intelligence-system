#!/usr/bin/env python3
"""
Comprehensive End-to-End Test - No YouTube (Quota Exceeded)
Tests all working APIs: NewsAPI, Finnhub, SEC Filings
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from comprehensive_data_integration_phase4 import ComprehensiveDataIntegrationPhase4

class ComprehensiveE2ETest:
    def __init__(self):
        self.integration = ComprehensiveDataIntegrationPhase4()
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        self.results = {}
        
    async def test_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Test a single symbol with all available APIs"""
        print(f"\n🔍 Testing {symbol}...")
        print("=" * 50)
        
        try:
            # Get comprehensive data
            start_time = time.time()
            data = await self.integration.get_comprehensive_data(symbol)
            test_time = time.time() - start_time
            
            # Analyze results
            result = {
                'symbol': symbol,
                'test_time': round(test_time, 2),
                'status': data.get('status', 'UNKNOWN'),
                'collection_time': data.get('collection_time', 0),
                'sources': {},
                'summary': data.get('summary', {}),
                'errors': []
            }
            
            # Check each source
            sources = data.get('sources', {})
            for source_name, source_data in sources.items():
                source_status = source_data.get('status', 'UNKNOWN')
                result['sources'][source_name] = source_status
                
                if source_status != 'WORKING':
                    error_msg = source_data.get('error', 'Unknown error')
                    result['errors'].append(f"{source_name}: {error_msg}")
            
            # Print results
            print(f"⏱️ Test Time: {test_time:.2f}s")
            print(f"📊 Collection Time: {data.get('collection_time', 0):.2f}s")
            print(f"📋 Status: {data.get('status', 'UNKNOWN')}")
            
            # Print source status
            print(f"\n📡 SOURCE STATUS:")
            for source_name, source_status in result['sources'].items():
                status_emoji = {
                    'WORKING': '✅',
                    'ERROR': '❌',
                    'NO_API_KEY': '🔑',
                    'RATE_LIMITED': '⏱️',
                    'NO_DATA': '📭'
                }.get(source_status, '❓')
                print(f"   {status_emoji} {source_name.upper()}: {source_status}")
            
            # Print summary if available
            summary = data.get('summary', {})
            if summary:
                print(f"\n📊 SUMMARY:")
                coverage = summary.get('data_coverage', {})
                print(f"   Data Coverage: {coverage.get('coverage_percentage', 0):.1f}%")
                
                sentiment = summary.get('sentiment_analysis', {})
                if sentiment:
                    print(f"   Sentiment: {sentiment.get('trend', 'unknown')} ({sentiment.get('overall_compound', 0):.3f})")
                
                market_data = summary.get('market_data', {})
                if market_data:
                    print(f"   Market Data: {market_data.get('success_rate', 0):.1f}% success")
                
                print(f"   Overall Score: {summary.get('overall_score', 0):.1f}/100")
            
            # Print errors if any
            if result['errors']:
                print(f"\n❌ ERRORS:")
                for error in result['errors']:
                    print(f"   • {error}")
            
            return result
            
        except Exception as e:
            print(f"❌ Exception testing {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'test_time': 0,
                'status': 'EXCEPTION',
                'sources': {},
                'errors': [f"Exception: {str(e)}"]
            }
    
    async def test_all_symbols(self) -> Dict[str, Any]:
        """Test all symbols and generate comprehensive report"""
        print("🚀 Comprehensive E2E Test - No YouTube")
        print("=" * 60)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Testing Symbols: {', '.join(self.test_symbols)}")
        print(f"📡 APIs: NewsAPI, Finnhub, SEC Filings")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test each symbol
        for symbol in self.test_symbols:
            result = await self.test_single_symbol(symbol)
            self.results[symbol] = result
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(total_time)
        
        return report
    
    def _generate_comprehensive_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print(f"\n📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Calculate statistics
        total_symbols = len(self.test_symbols)
        successful_symbols = sum(1 for r in self.results.values() if r['status'] not in ['ERROR', 'EXCEPTION'])
        failed_symbols = total_symbols - successful_symbols
        
        # Source statistics
        source_stats = {}
        for result in self.results.values():
            for source_name, source_status in result['sources'].items():
                if source_name not in source_stats:
                    source_stats[source_name] = {'working': 0, 'total': 0}
                source_stats[source_name]['total'] += 1
                if source_status == 'WORKING':
                    source_stats[source_name]['working'] += 1
        
        # Calculate success rates
        for source_name, stats in source_stats.items():
            stats['success_rate'] = (stats['working'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Overall statistics
        avg_test_time = sum(r['test_time'] for r in self.results.values()) / total_symbols
        avg_collection_time = sum(r['collection_time'] for r in self.results.values()) / total_symbols
        
        # Print summary
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total Symbols Tested: {total_symbols}")
        print(f"   Successful Tests: {successful_symbols}")
        print(f"   Failed Tests: {failed_symbols}")
        print(f"   Success Rate: {(successful_symbols/total_symbols*100):.1f}%")
        print(f"   Total Test Time: {total_time:.2f}s")
        print(f"   Average Test Time: {avg_test_time:.2f}s")
        print(f"   Average Collection Time: {avg_collection_time:.2f}s")
        
        print(f"\n📡 API SUCCESS RATES:")
        for source_name, stats in source_stats.items():
            status_emoji = '✅' if stats['success_rate'] >= 80 else '⚠️' if stats['success_rate'] >= 50 else '❌'
            print(f"   {status_emoji} {source_name.upper()}: {stats['success_rate']:.1f}% ({stats['working']}/{stats['total']})")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for symbol, result in self.results.items():
            status_emoji = '✅' if result['status'] not in ['ERROR', 'EXCEPTION'] else '❌'
            print(f"   {status_emoji} {symbol}: {result['status']} ({result['test_time']:.2f}s)")
            
            if result['errors']:
                for error in result['errors']:
                    print(f"      • {error}")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        collection_times = [r['collection_time'] for r in self.results.values()]
        if collection_times:
            print(f"   Fastest Collection: {min(collection_times):.2f}s")
            print(f"   Slowest Collection: {max(collection_times):.2f}s")
            print(f"   Average Collection: {sum(collection_times)/len(collection_times):.2f}s")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if successful_symbols == total_symbols:
            print("   ✅ All tests passed! System is working perfectly.")
        elif successful_symbols >= total_symbols * 0.8:
            print("   ⚠️ Most tests passed. Minor issues to address.")
        else:
            print("   ❌ Many tests failed. Significant issues to resolve.")
        
        # Check for common issues
        common_errors = {}
        for result in self.results.values():
            for error in result['errors']:
                common_errors[error] = common_errors.get(error, 0) + 1
        
        if common_errors:
            print(f"\n🔍 COMMON ISSUES:")
            for error, count in sorted(common_errors.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {error} (occurred {count} times)")
        
        # Create report object
        report = {
            'test_date': datetime.now().isoformat(),
            'total_symbols': total_symbols,
            'successful_symbols': successful_symbols,
            'failed_symbols': failed_symbols,
            'success_rate': (successful_symbols/total_symbols*100),
            'total_time': total_time,
            'avg_test_time': avg_test_time,
            'avg_collection_time': avg_collection_time,
            'source_stats': source_stats,
            'detailed_results': self.results,
            'common_errors': common_errors,
            'recommendations': self._generate_recommendations(successful_symbols, total_symbols, source_stats)
        }
        
        return report
    
    def _generate_recommendations(self, successful: int, total: int, source_stats: Dict) -> List[str]:
        """Generate specific recommendations based on test results"""
        recommendations = []
        
        if successful == total:
            recommendations.append("✅ All systems operational - ready for production use")
        elif successful >= total * 0.8:
            recommendations.append("⚠️ Minor issues detected - review error logs")
        else:
            recommendations.append("❌ Significant issues - requires immediate attention")
        
        # Check individual API performance
        for source_name, stats in source_stats.items():
            if stats['success_rate'] < 50:
                recommendations.append(f"❌ {source_name.upper()} API needs immediate attention")
            elif stats['success_rate'] < 80:
                recommendations.append(f"⚠️ {source_name.upper()} API has reliability issues")
        
        return recommendations
    
    async def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"e2e_test_report_no_youtube_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save report: {str(e)}")

async def main():
    """Run comprehensive E2E test"""
    print("🚀 Starting Comprehensive E2E Test (No YouTube)")
    print("=" * 60)
    
    # Create test instance
    test = ComprehensiveE2ETest()
    
    # Run tests
    report = await test.test_all_symbols()
    
    # Save report
    await test.save_report(report)
    
    # Final summary
    print(f"\n🎉 E2E TEST COMPLETE!")
    print(f"📊 Success Rate: {report['success_rate']:.1f}%")
    print(f"⏱️ Total Time: {report['total_time']:.2f}s")
    
    if report['success_rate'] >= 80:
        print("✅ System is ready for production use!")
    elif report['success_rate'] >= 50:
        print("⚠️ System has minor issues but is mostly functional")
    else:
        print("❌ System has significant issues that need attention")

if __name__ == "__main__":
    asyncio.run(main())
