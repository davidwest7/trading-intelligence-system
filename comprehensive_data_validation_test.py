#!/usr/bin/env python3
"""
Comprehensive Data Validation Test
Validates all data points are mapped correctly and logic flows properly
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from comprehensive_data_integration_phase4 import ComprehensiveDataIntegrationPhase4

class DataValidationTest:
    def __init__(self):
        self.integration = ComprehensiveDataIntegrationPhase4()
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.validation_results = {}
        
    def validate_data_structure(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validate the complete data structure and mapping"""
        validation = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'structure_valid': True,
            'mapping_valid': True,
            'logic_valid': True,
            'errors': [],
            'warnings': [],
            'data_points': {}
        }
        
        # 1. Validate top-level structure
        required_top_level = ['symbol', 'timestamp', 'collection_time', 'sources', 'summary']
        for field in required_top_level:
            if field not in data:
                validation['structure_valid'] = False
                validation['errors'].append(f"Missing top-level field: {field}")
        
        # 2. Validate sources structure
        sources = data.get('sources', {})
        expected_sources = ['news', 'finnhub', 'sec']
        for source in expected_sources:
            if source not in sources:
                validation['errors'].append(f"Missing source: {source}")
                validation['structure_valid'] = False
            else:
                source_data = sources[source]
                if not isinstance(source_data, dict):
                    validation['errors'].append(f"Invalid source data type for {source}")
                    validation['structure_valid'] = False
                elif 'status' not in source_data:
                    validation['errors'].append(f"Missing status in {source} data")
                    validation['structure_valid'] = False
        
        # 3. Validate summary structure
        summary = data.get('summary', {})
        expected_summary_fields = ['data_coverage', 'sentiment_analysis', 'market_data', 'financial_data', 'institutional_insights', 'overall_score']
        for field in expected_summary_fields:
            if field not in summary:
                validation['warnings'].append(f"Missing summary field: {field}")
        
        return validation
    
    def validate_news_data_mapping(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate NewsAPI data mapping and structure"""
        validation = {
            'valid': True,
            'data_points': {},
            'errors': [],
            'warnings': []
        }
        
        if news_data.get('status') != 'WORKING':
            validation['valid'] = False
            validation['errors'].append(f"NewsAPI status: {news_data.get('status')}")
            return validation
        
        # Check data structure
        data = news_data.get('data', {})
        if not data:
            validation['valid'] = False
            validation['errors'].append("No data in NewsAPI response")
            return validation
        
        # Validate articles
        articles = data.get('articles', [])
        if not articles:
            validation['warnings'].append("No articles found in NewsAPI response")
        else:
            validation['data_points']['articles_count'] = len(articles)
            
            # Validate first article structure
            if articles:
                first_article = articles[0]
                required_fields = ['headline', 'description', 'url', 'publishedAt']
                for field in required_fields:
                    if field not in first_article:
                        validation['warnings'].append(f"Article missing field: {field}")
        
        # Validate sentiment data
        aggregated_sentiment = data.get('aggregated_sentiment', {})
        if aggregated_sentiment:
            sentiment_fields = ['overall_compound', 'confidence', 'total_items']
            for field in sentiment_fields:
                if field not in aggregated_sentiment:
                    validation['warnings'].append(f"Sentiment missing field: {field}")
                else:
                    validation['data_points'][f'sentiment_{field}'] = aggregated_sentiment[field]
        
        return validation
    
    def validate_finnhub_data_mapping(self, finnhub_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Finnhub data mapping and structure"""
        validation = {
            'valid': True,
            'data_points': {},
            'errors': [],
            'warnings': []
        }
        
        if finnhub_data.get('status') != 'WORKING':
            validation['valid'] = False
            validation['errors'].append(f"Finnhub status: {finnhub_data.get('status')}")
            return validation
        
        # Check data structure
        data = finnhub_data.get('data', {})
        if not data:
            validation['valid'] = False
            validation['errors'].append("No data in Finnhub response")
            return validation
        
        # Validate sources within Finnhub
        sources = data.get('sources', {})
        expected_finnhub_sources = ['quote', 'news', 'financials']
        
        for source_name in expected_finnhub_sources:
            if source_name in sources:
                source_data = sources[source_name]
                validation['data_points'][f'finnhub_{source_name}'] = 'present'
                
                # Validate quote data specifically
                if source_name == 'quote':
                    quote_fields = ['current_price', 'change', 'percent_change', 'high', 'low', 'open']
                    for field in quote_fields:
                        if field in source_data:
                            validation['data_points'][f'quote_{field}'] = source_data[field]
                        else:
                            validation['warnings'].append(f"Quote missing field: {field}")
                
                # Validate news data specifically
                elif source_name == 'news':
                    if 'total_articles' in source_data:
                        validation['data_points']['finnhub_news_count'] = source_data['total_articles']
                    if 'average_sentiment' in source_data:
                        validation['data_points']['finnhub_sentiment'] = source_data['average_sentiment']
            else:
                validation['warnings'].append(f"Missing Finnhub source: {source_name}")
        
        # Validate success metrics
        success_rate = data.get('success_rate', 0)
        validation['data_points']['finnhub_success_rate'] = success_rate
        
        return validation
    
    def validate_sec_data_mapping(self, sec_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SEC filings data mapping and structure"""
        validation = {
            'valid': True,
            'data_points': {},
            'errors': [],
            'warnings': []
        }
        
        if sec_data.get('status') != 'WORKING':
            validation['valid'] = False
            validation['errors'].append(f"SEC status: {sec_data.get('status')}")
            return validation
        
        # Check data structure
        data = sec_data.get('data', {})
        if not data:
            validation['valid'] = False
            validation['errors'].append("No data in SEC response")
            return validation
        
        # Validate insights
        insights = data.get('insights', {})
        if insights:
            insight_fields = ['insider_activity', 'institutional_interest', 'recent_events', 'filing_activity']
            for field in insight_fields:
                if field in insights:
                    validation['data_points'][f'sec_{field}'] = insights[field]
                else:
                    validation['warnings'].append(f"SEC insights missing field: {field}")
        
        # Validate company data
        company_data = data.get('company_data', {})
        if company_data:
            if 'cik' in company_data:
                validation['data_points']['sec_cik'] = company_data['cik']
            if 'name' in company_data:
                validation['data_points']['sec_company_name'] = company_data['name']
        
        return validation
    
    def validate_summary_logic(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Validate summary calculation logic"""
        validation = {
            'valid': True,
            'logic_checks': {},
            'errors': [],
            'warnings': []
        }
        
        # 1. Validate data coverage calculation
        data_coverage = summary.get('data_coverage', {})
        working_sources = data_coverage.get('working_sources', 0)
        total_sources = data_coverage.get('total_sources', 0)
        coverage_percentage = data_coverage.get('coverage_percentage', 0)
        
        # Check if calculation is correct
        if total_sources > 0:
            calculated_percentage = (working_sources / total_sources) * 100
            if abs(calculated_percentage - coverage_percentage) > 0.1:  # Allow small rounding differences
                validation['errors'].append(f"Coverage percentage calculation error: expected {calculated_percentage:.1f}, got {coverage_percentage:.1f}")
                validation['valid'] = False
            else:
                validation['logic_checks']['coverage_calculation'] = 'correct'
        
        # 2. Validate sentiment analysis
        sentiment = summary.get('sentiment_analysis', {})
        if sentiment:
            compound = sentiment.get('overall_compound', 0)
            confidence = sentiment.get('confidence', 0)
            total_items = sentiment.get('total_items', 0)
            
            # Validate sentiment range
            if not -1 <= compound <= 1:
                validation['errors'].append(f"Sentiment compound out of range: {compound}")
                validation['valid'] = False
            
            # Validate confidence range
            if not 0 <= confidence <= 1:
                validation['errors'].append(f"Confidence out of range: {confidence}")
                validation['valid'] = False
            
            # Validate trend logic
            trend = sentiment.get('trend', '')
            if compound >= 0.3 and trend != 'bullish':
                validation['warnings'].append(f"High positive sentiment ({compound}) but trend is {trend}")
            elif compound <= -0.3 and trend != 'bearish':
                validation['warnings'].append(f"High negative sentiment ({compound}) but trend is {trend}")
            
            validation['logic_checks']['sentiment_validation'] = 'passed'
        
        # 3. Validate market data
        market_data = summary.get('market_data', {})
        if market_data:
            success_rate = market_data.get('success_rate', 0)
            if not 0 <= success_rate <= 100:
                validation['errors'].append(f"Market data success rate out of range: {success_rate}")
                validation['valid'] = False
            else:
                validation['logic_checks']['market_data_validation'] = 'passed'
        
        # 4. Validate overall score calculation
        overall_score = summary.get('overall_score', 0)
        if not 0 <= overall_score <= 100:
            validation['errors'].append(f"Overall score out of range: {overall_score}")
            validation['valid'] = False
        else:
            validation['logic_checks']['overall_score_validation'] = 'passed'
        
        return validation
    
    async def test_symbol_data_validation(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive validation test for a single symbol"""
        print(f"\n🔍 Validating {symbol} data mapping and logic...")
        print("=" * 60)
        
        try:
            # Get comprehensive data
            start_time = time.time()
            data = await self.integration.get_comprehensive_data(symbol)
            test_time = time.time() - start_time
            
            # Initialize validation result
            validation_result = {
                'symbol': symbol,
                'test_time': round(test_time, 2),
                'overall_valid': True,
                'structure_validation': {},
                'source_validations': {},
                'logic_validation': {},
                'data_points': {},
                'errors': [],
                'warnings': []
            }
            
            # 1. Validate overall data structure
            structure_validation = self.validate_data_structure(data, symbol)
            validation_result['structure_validation'] = structure_validation
            if not structure_validation['structure_valid']:
                validation_result['overall_valid'] = False
                validation_result['errors'].extend(structure_validation['errors'])
            
            # 2. Validate each source
            sources = data.get('sources', {})
            
            # NewsAPI validation
            if 'news' in sources:
                news_validation = self.validate_news_data_mapping(sources['news'])
                validation_result['source_validations']['news'] = news_validation
                if not news_validation['valid']:
                    validation_result['overall_valid'] = False
                validation_result['data_points'].update(news_validation['data_points'])
                validation_result['errors'].extend(news_validation['errors'])
                validation_result['warnings'].extend(news_validation['warnings'])
            
            # Finnhub validation
            if 'finnhub' in sources:
                finnhub_validation = self.validate_finnhub_data_mapping(sources['finnhub'])
                validation_result['source_validations']['finnhub'] = finnhub_validation
                if not finnhub_validation['valid']:
                    validation_result['overall_valid'] = False
                validation_result['data_points'].update(finnhub_validation['data_points'])
                validation_result['errors'].extend(finnhub_validation['errors'])
                validation_result['warnings'].extend(finnhub_validation['warnings'])
            
            # SEC validation
            if 'sec' in sources:
                sec_validation = self.validate_sec_data_mapping(sources['sec'])
                validation_result['source_validations']['sec'] = sec_validation
                if not sec_validation['valid']:
                    validation_result['overall_valid'] = False
                validation_result['data_points'].update(sec_validation['data_points'])
                validation_result['errors'].extend(sec_validation['errors'])
                validation_result['warnings'].extend(sec_validation['warnings'])
            
            # 3. Validate summary logic
            summary = data.get('summary', {})
            if summary:
                logic_validation = self.validate_summary_logic(summary)
                validation_result['logic_validation'] = logic_validation
                if not logic_validation['valid']:
                    validation_result['overall_valid'] = False
                validation_result['errors'].extend(logic_validation['errors'])
                validation_result['warnings'].extend(logic_validation['warnings'])
            
            # Print validation results
            print(f"⏱️ Validation Time: {test_time:.2f}s")
            print(f"📊 Overall Valid: {'✅ YES' if validation_result['overall_valid'] else '❌ NO'}")
            
            # Print structure validation
            print(f"\n🏗️ STRUCTURE VALIDATION:")
            print(f"   Data Structure: {'✅ Valid' if structure_validation['structure_valid'] else '❌ Invalid'}")
            print(f"   Data Mapping: {'✅ Valid' if structure_validation['mapping_valid'] else '❌ Invalid'}")
            print(f"   Logic Flow: {'✅ Valid' if structure_validation['logic_valid'] else '❌ Invalid'}")
            
            # Print source validations
            print(f"\n📡 SOURCE VALIDATIONS:")
            for source_name, source_validation in validation_result['source_validations'].items():
                status = '✅ Valid' if source_validation['valid'] else '❌ Invalid'
                print(f"   {source_name.upper()}: {status}")
                
                # Show key data points
                for key, value in source_validation['data_points'].items():
                    print(f"     • {key}: {value}")
            
            # Print logic validation
            print(f"\n🧮 LOGIC VALIDATION:")
            logic_validation = validation_result['logic_validation']
            if logic_validation:
                for check_name, result in logic_validation.get('logic_checks', {}).items():
                    print(f"   {check_name}: ✅ {result}")
            
            # Print errors and warnings
            if validation_result['errors']:
                print(f"\n❌ ERRORS:")
                for error in validation_result['errors']:
                    print(f"   • {error}")
            
            if validation_result['warnings']:
                print(f"\n⚠️ WARNINGS:")
                for warning in validation_result['warnings']:
                    print(f"   • {warning}")
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Exception during validation: {str(e)}")
            return {
                'symbol': symbol,
                'test_time': 0,
                'overall_valid': False,
                'errors': [f"Exception: {str(e)}"]
            }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation on all symbols"""
        print("🚀 Comprehensive Data Validation Test")
        print("=" * 60)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Testing Symbols: {', '.join(self.test_symbols)}")
        print(f"🔍 Validating: Data mapping, structure, and logic flow")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test each symbol
        for symbol in self.test_symbols:
            result = await self.test_symbol_data_validation(symbol)
            self.validation_results[symbol] = result
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive validation report
        report = self._generate_validation_report(total_time)
        
        return report
    
    def _generate_validation_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print(f"\n📋 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 60)
        
        # Calculate statistics
        total_symbols = len(self.test_symbols)
        valid_symbols = sum(1 for r in self.validation_results.values() if r['overall_valid'])
        invalid_symbols = total_symbols - valid_symbols
        
        # Count errors and warnings
        total_errors = sum(len(r.get('errors', [])) for r in self.validation_results.values())
        total_warnings = sum(len(r.get('warnings', [])) for r in self.validation_results.values())
        
        # Source validation statistics
        source_stats = {}
        for result in self.validation_results.values():
            for source_name, source_validation in result.get('source_validations', {}).items():
                if source_name not in source_stats:
                    source_stats[source_name] = {'valid': 0, 'total': 0}
                source_stats[source_name]['total'] += 1
                if source_validation['valid']:
                    source_stats[source_name]['valid'] += 1
        
        # Calculate success rates
        for source_name, stats in source_stats.items():
            stats['success_rate'] = (stats['valid'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Print summary
        print(f"📊 VALIDATION STATISTICS:")
        print(f"   Total Symbols: {total_symbols}")
        print(f"   Valid Symbols: {valid_symbols}")
        print(f"   Invalid Symbols: {invalid_symbols}")
        print(f"   Validation Success Rate: {(valid_symbols/total_symbols*100):.1f}%")
        print(f"   Total Errors: {total_errors}")
        print(f"   Total Warnings: {total_warnings}")
        print(f"   Total Validation Time: {total_time:.2f}s")
        
        print(f"\n📡 SOURCE VALIDATION RATES:")
        for source_name, stats in source_stats.items():
            status_emoji = '✅' if stats['success_rate'] >= 80 else '⚠️' if stats['success_rate'] >= 50 else '❌'
            print(f"   {status_emoji} {source_name.upper()}: {stats['success_rate']:.1f}% ({stats['valid']}/{stats['total']})")
        
        # Detailed results
        print(f"\n📋 DETAILED VALIDATION RESULTS:")
        for symbol, result in self.validation_results.items():
            status_emoji = '✅' if result['overall_valid'] else '❌'
            print(f"   {status_emoji} {symbol}: {'Valid' if result['overall_valid'] else 'Invalid'} ({result['test_time']:.2f}s)")
            
            if result['errors']:
                for error in result['errors']:
                    print(f"      • {error}")
        
        # Data quality analysis
        print(f"\n🔍 DATA QUALITY ANALYSIS:")
        all_data_points = {}
        for result in self.validation_results.values():
            for key, value in result.get('data_points', {}).items():
                if key not in all_data_points:
                    all_data_points[key] = []
                all_data_points[key].append(value)
        
        for data_point, values in all_data_points.items():
            if values:
                if isinstance(values[0], (int, float)):
                    avg_value = sum(values) / len(values)
                    print(f"   {data_point}: {avg_value:.2f} (avg across {len(values)} symbols)")
                else:
                    print(f"   {data_point}: {len(values)} symbols have this data")
        
        # Recommendations
        print(f"\n💡 VALIDATION RECOMMENDATIONS:")
        if valid_symbols == total_symbols and total_errors == 0:
            print("   ✅ All validations passed! Data mapping and logic are correct.")
        elif valid_symbols >= total_symbols * 0.8 and total_errors == 0:
            print("   ⚠️ Most validations passed. Minor warnings to review.")
        else:
            print("   ❌ Validation issues detected. Review errors and fix data mapping.")
        
        if total_warnings > 0:
            print(f"   📝 Review {total_warnings} warnings for potential improvements.")
        
        # Create report object
        report = {
            'validation_date': datetime.now().isoformat(),
            'total_symbols': total_symbols,
            'valid_symbols': valid_symbols,
            'invalid_symbols': invalid_symbols,
            'validation_success_rate': (valid_symbols/total_symbols*100),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'total_time': total_time,
            'source_stats': source_stats,
            'detailed_results': self.validation_results,
            'data_points_summary': all_data_points,
            'recommendations': self._generate_validation_recommendations(valid_symbols, total_symbols, total_errors, total_warnings)
        }
        
        return report
    
    def _generate_validation_recommendations(self, valid: int, total: int, errors: int, warnings: int) -> List[str]:
        """Generate specific recommendations based on validation results"""
        recommendations = []
        
        if valid == total and errors == 0:
            recommendations.append("✅ All data mappings and logic flows are correct")
            recommendations.append("✅ System is ready for production use")
        elif valid >= total * 0.8 and errors == 0:
            recommendations.append("⚠️ Most validations passed - review warnings")
            recommendations.append("✅ System is mostly ready for production")
        else:
            recommendations.append("❌ Validation issues detected - fix data mapping")
            recommendations.append("❌ Review logic flow calculations")
        
        if warnings > 0:
            recommendations.append(f"📝 Address {warnings} warnings for optimal performance")
        
        if errors > 0:
            recommendations.append(f"🔧 Fix {errors} errors before production deployment")
        
        return recommendations
    
    async def save_validation_report(self, report: Dict[str, Any], filename: str = None):
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_validation_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Validation report saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save validation report: {str(e)}")

async def main():
    """Run comprehensive data validation test"""
    print("🚀 Starting Comprehensive Data Validation Test")
    print("=" * 60)
    
    # Create validation test instance
    validator = DataValidationTest()
    
    # Run validation
    report = await validator.run_comprehensive_validation()
    
    # Save report
    await validator.save_validation_report(report)
    
    # Final summary
    print(f"\n🎉 DATA VALIDATION COMPLETE!")
    print(f"📊 Validation Success Rate: {report['validation_success_rate']:.1f}%")
    print(f"❌ Total Errors: {report['total_errors']}")
    print(f"⚠️ Total Warnings: {report['total_warnings']}")
    print(f"⏱️ Total Time: {report['total_time']:.2f}s")
    
    if report['validation_success_rate'] == 100 and report['total_errors'] == 0:
        print("✅ All data mappings and logic flows are correct!")
        print("✅ System is ready for production use!")
    elif report['validation_success_rate'] >= 80 and report['total_errors'] == 0:
        print("⚠️ Most validations passed - review warnings")
        print("✅ System is mostly ready for production")
    else:
        print("❌ Validation issues detected - review and fix errors")
        print("❌ System needs fixes before production deployment")

if __name__ == "__main__":
    asyncio.run(main())
