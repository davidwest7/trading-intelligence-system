#!/usr/bin/env python3
"""
Comprehensive End-to-End Test - Phase 3
Tests all integrations including YouTube Live News and Defeat Beta API
"""

import asyncio
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

class ComprehensiveE2ETestPhase3:
    """Comprehensive end-to-end test for Phase 3 integrations"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Test configuration
        self.config = {
            'timeout': 30,  # seconds per test
            'max_retries': 3,
            'min_success_rate': 0.8  # 80% success rate required
        }
    
    def log_test_result(self, test_name: str, status: str, message: str, data: Any = None):
        """Log test result"""
        result = {
            'test_name': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self.test_results.append(result)
        
        # Print result
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARNING': '⚠️',
            'SKIP': '⏭️'
        }.get(status, '❓')
        
        print(f"{status_emoji} {test_name}: {message}")
    
    async def test_api_keys_availability(self):
        """Test API key availability"""
        print("\n🔑 Testing API Key Availability...")
        
        required_keys = {
            'NEWS_API_KEY': 'NewsAPI',
            'YOUTUBE_API_KEY': 'YouTube API',
            'POLYGON_API_KEY': 'Polygon.io',
            'FMP_API_KEY': 'Financial Modeling Prep',
            'FRED_API_KEY': 'FRED API'
        }
        
        available_keys = 0
        total_keys = len(required_keys)
        
        for key_name, service_name in required_keys.items():
            key_value = os.getenv(key_name, '')
            if key_value and key_value != f'your_{key_name.lower()}_here':
                self.log_test_result(
                    f"{service_name} API Key",
                    "PASS",
                    f"API key available for {service_name}"
                )
                available_keys += 1
            else:
                self.log_test_result(
                    f"{service_name} API Key",
                    "FAIL",
                    f"API key missing for {service_name}"
                )
        
        success_rate = available_keys / total_keys
        if success_rate >= 0.8:
            self.log_test_result(
                "API Key Availability",
                "PASS",
                f"{available_keys}/{total_keys} API keys available ({success_rate:.1%})"
            )
        else:
            self.log_test_result(
                "API Key Availability",
                "FAIL",
                f"Only {available_keys}/{total_keys} API keys available ({success_rate:.1%})"
            )
    
    async def test_newsapi_integration(self):
        """Test NewsAPI integration"""
        print("\n📰 Testing NewsAPI Integration...")
        
        try:
            from comprehensive_data_integration_phase3 import ComprehensiveDataIntegrationPhase3
            integration = ComprehensiveDataIntegrationPhase3()
            
            # Test with AAPL
            news_data = await integration.get_news_sentiment('AAPL')
            
            if news_data.get('status') == 'WORKING':
                articles_count = len(news_data.get('sources', {}).get('newsapi', {}).get('articles', []))
                sentiment = news_data.get('aggregated_sentiment', {})
                
                self.log_test_result(
                    "NewsAPI Integration",
                    "PASS",
                    f"Found {articles_count} articles, sentiment: {sentiment.get('overall_compound', 0.0):.3f}",
                    {
                        'articles_count': articles_count,
                        'sentiment_compound': sentiment.get('overall_compound', 0.0),
                        'confidence': sentiment.get('confidence', 0.0)
                    }
                )
            else:
                self.log_test_result(
                    "NewsAPI Integration",
                    "FAIL",
                    f"NewsAPI status: {news_data.get('status')}"
                )
                
        except Exception as e:
            self.log_test_result(
                "NewsAPI Integration",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    async def test_youtube_live_news_integration(self):
        """Test YouTube Live News integration"""
        print("\n📺 Testing YouTube Live News Integration...")
        
        try:
            from youtube_live_news_integration import YouTubeLiveNewsIntegration
            youtube = YouTubeLiveNewsIntegration()
            
            # Test earnings coverage
            coverage = await youtube.get_earnings_announcement_coverage('AAPL')
            
            videos_count = len(coverage.get('videos', []))
            live_streams_count = len(coverage.get('live_streams', []))
            sentiment_score = coverage.get('sentiment', {}).get('overall_sentiment', 0.0)
            
            if videos_count > 0 or live_streams_count > 0:
                self.log_test_result(
                    "YouTube Live News Integration",
                    "PASS",
                    f"Found {videos_count} videos, {live_streams_count} live streams, sentiment: {sentiment_score:.3f}",
                    {
                        'videos_count': videos_count,
                        'live_streams_count': live_streams_count,
                        'sentiment_score': sentiment_score,
                        'channels_covered': coverage.get('coverage_summary', {}).get('channels_covered', 0)
                    }
                )
            else:
                self.log_test_result(
                    "YouTube Live News Integration",
                    "WARNING",
                    f"No videos or live streams found, but API is working"
                )
                
        except Exception as e:
            self.log_test_result(
                "YouTube Live News Integration",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    async def test_defeatbeta_api_integration(self):
        """Test Defeat Beta API integration"""
        print("\n🎯 Testing Defeat Beta API Integration...")
        
        try:
            from defeatbeta_api_integration import DefeatBetaAPIIntegration
            defeatbeta = DefeatBetaAPIIntegration()
            
            if not defeatbeta.installed:
                self.log_test_result(
                    "Defeat Beta API Installation",
                    "FAIL",
                    "Defeat Beta API not installed"
                )
                return
            
            # Test comprehensive data
            comprehensive_data = await defeatbeta.get_comprehensive_data('AAPL')
            
            summary = comprehensive_data.get('summary', {})
            working_sources = summary.get('working_sources', 0)
            total_sources = summary.get('total_sources', 0)
            data_points = summary.get('data_points', 0)
            success_rate = summary.get('success_rate', 0.0)
            
            if success_rate >= 0.8:
                self.log_test_result(
                    "Defeat Beta API Integration",
                    "PASS",
                    f"{working_sources}/{total_sources} sources working, {data_points} data points, {success_rate:.1%} success rate",
                    {
                        'working_sources': working_sources,
                        'total_sources': total_sources,
                        'data_points': data_points,
                        'success_rate': success_rate
                    }
                )
            else:
                self.log_test_result(
                    "Defeat Beta API Integration",
                    "WARNING",
                    f"Only {working_sources}/{total_sources} sources working ({success_rate:.1%} success rate)"
                )
                
        except Exception as e:
            self.log_test_result(
                "Defeat Beta API Integration",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    async def test_sec_filings_integration(self):
        """Test SEC filings integration"""
        print("\n📋 Testing SEC Filings Integration...")
        
        try:
            from sec_filings_integration import SECFilingsIntegration
            sec = SECFilingsIntegration()
            
            # Test comprehensive SEC data
            sec_data = await sec.get_comprehensive_sec_data('AAPL')
            
            if 'error' not in sec_data:
                insights = sec_data.get('insights', {})
                insider_activity = insights.get('insider_activity', 'unknown')
                institutional_interest = insights.get('institutional_interest', 'unknown')
                recent_events = insights.get('recent_events', 0)
                
                self.log_test_result(
                    "SEC Filings Integration",
                    "PASS",
                    f"Insider activity: {insider_activity}, institutional interest: {institutional_interest}, {recent_events} recent events",
                    {
                        'insider_activity': insider_activity,
                        'institutional_interest': institutional_interest,
                        'recent_events': recent_events,
                        'company_name': sec_data.get('company_info', {}).get('name', 'Unknown')
                    }
                )
            else:
                self.log_test_result(
                    "SEC Filings Integration",
                    "FAIL",
                    f"SEC data error: {sec_data.get('error')}"
                )
                
        except Exception as e:
            self.log_test_result(
                "SEC Filings Integration",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    async def test_comprehensive_data_integration(self):
        """Test comprehensive data integration"""
        print("\n🎯 Testing Comprehensive Data Integration...")
        
        try:
            from comprehensive_data_integration_phase3 import ComprehensiveDataIntegrationPhase3
            integration = ComprehensiveDataIntegrationPhase3()
            
            # Test comprehensive data collection
            comprehensive_data = await integration.get_comprehensive_data('AAPL')
            
            collection_time = comprehensive_data.get('collection_time', 0)
            summary = comprehensive_data.get('summary', {})
            data_coverage = summary.get('data_coverage', {})
            overall_score = summary.get('overall_score', 0.0)
            
            working_sources = data_coverage.get('working_sources', 0)
            total_sources = data_coverage.get('total_sources', 0)
            coverage_percentage = data_coverage.get('coverage_percentage', 0.0)
            
            if coverage_percentage >= 0.8:
                self.log_test_result(
                    "Comprehensive Data Integration",
                    "PASS",
                    f"{working_sources}/{total_sources} sources working ({coverage_percentage:.1%}), score: {overall_score:.2f}, time: {collection_time}s",
                    {
                        'working_sources': working_sources,
                        'total_sources': total_sources,
                        'coverage_percentage': coverage_percentage,
                        'overall_score': overall_score,
                        'collection_time': collection_time
                    }
                )
            else:
                self.log_test_result(
                    "Comprehensive Data Integration",
                    "WARNING",
                    f"Only {working_sources}/{total_sources} sources working ({coverage_percentage:.1%})"
                )
                
        except Exception as e:
            self.log_test_result(
                "Comprehensive Data Integration",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    async def test_multi_symbol_processing(self):
        """Test processing multiple symbols"""
        print("\n📊 Testing Multi-Symbol Processing...")
        
        try:
            from comprehensive_data_integration_phase3 import ComprehensiveDataIntegrationPhase3
            integration = ComprehensiveDataIntegrationPhase3()
            
            results = {}
            total_time = 0
            
            for symbol in self.test_symbols:
                start_time = time.time()
                data = await integration.get_comprehensive_data(symbol)
                end_time = time.time()
                
                symbol_time = end_time - start_time
                total_time += symbol_time
                
                summary = data.get('summary', {})
                coverage = summary.get('data_coverage', {})
                working_sources = coverage.get('working_sources', 0)
                total_sources = coverage.get('total_sources', 0)
                
                results[symbol] = {
                    'working_sources': working_sources,
                    'total_sources': total_sources,
                    'collection_time': symbol_time,
                    'overall_score': summary.get('overall_score', 0.0)
                }
            
            # Calculate success rate
            successful_symbols = sum(1 for r in results.values() if r['working_sources'] >= 3)
            success_rate = successful_symbols / len(self.test_symbols)
            
            if success_rate >= 0.8:
                self.log_test_result(
                    "Multi-Symbol Processing",
                    "PASS",
                    f"{successful_symbols}/{len(self.test_symbols)} symbols processed successfully, avg time: {total_time/len(self.test_symbols):.2f}s",
                    {
                        'successful_symbols': successful_symbols,
                        'total_symbols': len(self.test_symbols),
                        'success_rate': success_rate,
                        'total_time': total_time,
                        'average_time': total_time / len(self.test_symbols),
                        'results': results
                    }
                )
            else:
                self.log_test_result(
                    "Multi-Symbol Processing",
                    "WARNING",
                    f"Only {successful_symbols}/{len(self.test_symbols)} symbols processed successfully"
                )
                
        except Exception as e:
            self.log_test_result(
                "Multi-Symbol Processing",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    async def test_error_handling(self):
        """Test error handling and fallbacks with improved validation"""
        print("\n🛡️ Testing Error Handling...")
        
        try:
            from comprehensive_data_integration_phase3 import ComprehensiveDataIntegrationPhase3
            integration = ComprehensiveDataIntegrationPhase3()
            
            # Test with invalid symbol
            invalid_data = await integration.get_comprehensive_data('INVALID_SYMBOL_12345')
            
            # Check if system properly detects invalid symbols
            if invalid_data.get('status') == 'INVALID_SYMBOL':
                validation = invalid_data.get('validation', {})
                errors = validation.get('errors', [])
                
                self.log_test_result(
                    "Error Handling",
                    "PASS",
                    f"System properly detected invalid symbol with {len(errors)} validation errors",
                    {
                        'validation_errors': errors,
                        'symbol': invalid_data.get('symbol'),
                        'status': invalid_data.get('status')
                    }
                )
            else:
                # Check if system handles invalid symbols gracefully
                sources = invalid_data.get('sources', {})
                error_count = sum(1 for source in sources.values() if source.get('status') == 'ERROR')
                
                if error_count > 0:
                    self.log_test_result(
                        "Error Handling",
                        "PASS",
                        f"System properly handled {error_count} errors for invalid symbol",
                        {
                            'error_count': error_count,
                            'total_sources': len(sources),
                            'error_rate': error_count / len(sources) if sources else 0
                        }
                    )
                else:
                    self.log_test_result(
                        "Error Handling",
                        "WARNING",
                        "No errors detected for invalid symbol (may indicate issues)"
                    )
                
        except Exception as e:
            self.log_test_result(
                "Error Handling",
                "FAIL",
                f"Exception during error handling test: {str(e)}"
            )
    
    async def test_performance_metrics(self):
        """Test performance metrics"""
        print("\n⚡ Testing Performance Metrics...")
        
        try:
            from comprehensive_data_integration_phase3 import ComprehensiveDataIntegrationPhase3
            integration = ComprehensiveDataIntegrationPhase3()
            
            # Test performance with multiple symbols
            start_time = time.time()
            
            tasks = []
            for symbol in self.test_symbols:
                task = integration.get_comprehensive_data(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate performance metrics
            successful_results = [r for r in results if not isinstance(r, Exception)]
            success_rate = len(successful_results) / len(results)
            
            avg_collection_time = 0
            if successful_results:
                collection_times = [r.get('collection_time', 0) for r in successful_results]
                avg_collection_time = sum(collection_times) / len(collection_times)
            
            # Performance thresholds
            max_total_time = 60  # 60 seconds for all symbols
            max_avg_time = 20    # 20 seconds per symbol
            
            if total_time <= max_total_time and avg_collection_time <= max_avg_time:
                self.log_test_result(
                    "Performance Metrics",
                    "PASS",
                    f"Total time: {total_time:.2f}s, avg per symbol: {avg_collection_time:.2f}s, success rate: {success_rate:.1%}",
                    {
                        'total_time': total_time,
                        'average_time_per_symbol': avg_collection_time,
                        'success_rate': success_rate,
                        'symbols_processed': len(results)
                    }
                )
            else:
                self.log_test_result(
                    "Performance Metrics",
                    "WARNING",
                    f"Performance below threshold - Total: {total_time:.2f}s, Avg: {avg_collection_time:.2f}s"
                )
                
        except Exception as e:
            self.log_test_result(
                "Performance Metrics",
                "FAIL",
                f"Exception during performance test: {str(e)}"
            )
    
    async def test_data_quality(self):
        """Test data quality and consistency"""
        print("\n🔍 Testing Data Quality...")
        
        try:
            from comprehensive_data_integration_phase3 import ComprehensiveDataIntegrationPhase3
            integration = ComprehensiveDataIntegrationPhase3()
            
            # Test data quality for a single symbol
            data = await integration.get_comprehensive_data('AAPL')
            
            quality_metrics = {
                'has_news_data': False,
                'has_youtube_data': False,
                'has_defeatbeta_data': False,
                'has_sec_data': False,
                'sentiment_consistency': False,
                'data_freshness': False
            }
            
            sources = data.get('sources', {})
            
            # Check data availability
            if sources.get('news', {}).get('status') == 'WORKING':
                quality_metrics['has_news_data'] = True
            
            if sources.get('youtube', {}).get('status') == 'WORKING':
                quality_metrics['has_youtube_data'] = True
            
            if sources.get('defeatbeta', {}).get('status') == 'WORKING':
                quality_metrics['has_defeatbeta_data'] = True
            
            if sources.get('sec', {}).get('status') == 'WORKING':
                quality_metrics['has_sec_data'] = True
            
            # Check sentiment consistency
            news_sentiment = sources.get('news', {}).get('aggregated_sentiment', {})
            youtube_sentiment = sources.get('youtube', {}).get('sentiment', {})
            
            if news_sentiment and youtube_sentiment:
                news_compound = news_sentiment.get('overall_compound', 0.0)
                youtube_compound = youtube_sentiment.get('overall_sentiment', 0.0)
                
                # Check if sentiments are within reasonable range
                sentiment_diff = abs(news_compound - youtube_compound)
                quality_metrics['sentiment_consistency'] = sentiment_diff <= 0.5
            
            # Check data freshness
            timestamp = data.get('timestamp', '')
            if timestamp:
                try:
                    data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    current_time = datetime.now()
                    time_diff = (current_time - data_time).total_seconds()
                    quality_metrics['data_freshness'] = time_diff <= 300  # 5 minutes
                except:
                    pass
            
            # Calculate quality score
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            if quality_score >= 0.7:
                self.log_test_result(
                    "Data Quality",
                    "PASS",
                    f"Quality score: {quality_score:.1%}, {sum(quality_metrics.values())}/{len(quality_metrics)} metrics passed",
                    {
                        'quality_score': quality_score,
                        'metrics': quality_metrics,
                        'sentiment_consistency': sentiment_diff if 'sentiment_consistency' in quality_metrics else None
                    }
                )
            else:
                self.log_test_result(
                    "Data Quality",
                    "WARNING",
                    f"Quality score: {quality_score:.1%}, some data quality issues detected"
                )
                
        except Exception as e:
            self.log_test_result(
                "Data Quality",
                "FAIL",
                f"Exception during data quality test: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all end-to-end tests"""
        print("🚀 Comprehensive End-to-End Test - Phase 3")
        print("="*60)
        print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        test_methods = [
            self.test_api_keys_availability,
            self.test_newsapi_integration,
            self.test_youtube_live_news_integration,
            self.test_defeatbeta_api_integration,
            self.test_sec_filings_integration,
            self.test_comprehensive_data_integration,
            self.test_multi_symbol_processing,
            self.test_error_handling,
            self.test_performance_metrics,
            self.test_data_quality
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.log_test_result(
                    test_method.__name__,
                    "FAIL",
                    f"Test method failed: {str(e)}"
                )
        
        # Generate final report
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate final test report"""
        print("\n" + "="*60)
        print("📊 FINAL TEST REPORT")
        print("="*60)
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        warning_tests = sum(1 for r in self.test_results if r['status'] == 'WARNING')
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        total_time = time.time() - self.start_time
        
        # Print summary
        print(f"⏱️ Total Test Time: {total_time:.2f} seconds")
        print(f"📊 Test Results: {passed_tests} PASS, {warning_tests} WARNING, {failed_tests} FAIL")
        print(f"🎯 Success Rate: {success_rate:.1%}")
        
        # Overall status
        if success_rate >= 0.8:
            overall_status = "✅ PASS"
        elif success_rate >= 0.6:
            overall_status = "⚠️ WARNING"
        else:
            overall_status = "❌ FAIL"
        
        print(f"🏆 Overall Status: {overall_status}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status_emoji = {
                'PASS': '✅',
                'FAIL': '❌',
                'WARNING': '⚠️',
                'SKIP': '⏭️'
            }.get(result['status'], '❓')
            
            print(f"   {status_emoji} {result['test_name']}: {result['message']}")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'success_rate': success_rate,
            'overall_status': overall_status,
            'test_results': self.test_results,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'warning_tests': warning_tests
            }
        }
        
        report_filename = f"e2e_test_report_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_filename}")
        
        # Print recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if success_rate >= 0.9:
            print("   🎉 Excellent! System is production-ready")
        elif success_rate >= 0.8:
            print("   ✅ Good! System is ready with minor improvements")
        elif success_rate >= 0.6:
            print("   ⚠️ Fair! Some issues need attention before production")
        else:
            print("   ❌ Poor! Significant issues need to be resolved")
        
        if failed_tests > 0:
            print("   🔧 Review failed tests and fix critical issues")
        
        if warning_tests > 0:
            print("   📝 Address warnings to improve system reliability")
        
        print(f"\n🎉 End-to-end test completed!")

async def main():
    """Run the comprehensive end-to-end test"""
    test_suite = ComprehensiveE2ETestPhase3()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
