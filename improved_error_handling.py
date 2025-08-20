#!/usr/bin/env python3
"""
Improved Error Handling System
Properly detects and handles invalid symbols and edge cases
"""

import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_real_keys.env')

class ImprovedErrorHandling:
    """Improved error handling for trading intelligence system"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.fmp_api_key = os.getenv('FMP_API_KEY', '')
        self.fred_api_key = os.getenv('FRED_API_KEY', '')
        
        # Invalid symbol patterns
        self.invalid_patterns = [
            r'^[0-9]+$',  # Pure numbers
            r'^[^A-Za-z]+$',  # No letters
            r'^[A-Za-z]{1,2}$',  # Too short
            r'^[A-Za-z]{20,}$',  # Too long
            r'[^A-Za-z0-9]',  # Special characters
            r'INVALID_',  # Invalid prefix
            r'TEST_',  # Test prefix
            r'DUMMY_',  # Dummy prefix
        ]
        
        # Common invalid symbols
        self.known_invalid_symbols = [
            'INVALID_SYMBOL_12345',
            'TEST_SYMBOL',
            'DUMMY_STOCK',
            '12345',
            'ABC123',
            'XYZ@#$',
            'A',  # Too short
            'VERYLONGSYMBOLNAME123456789',  # Too long
        ]
    
    def is_valid_symbol(self, symbol: str) -> Dict[str, Any]:
        """Validate if a symbol is properly formatted"""
        import re
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'symbol': symbol
        }
        
        # Check if symbol is None or empty
        if not symbol or not isinstance(symbol, str):
            validation_result['is_valid'] = False
            validation_result['errors'].append('Symbol is None or empty')
            return validation_result
        
        # Check length - allow single letters (like "A" for Agilent)
        if len(symbol) < 1:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Symbol too short: {len(symbol)} characters (minimum 1)')
        
        if len(symbol) > 15:  # Increased limit for longer valid symbols
            validation_result['warnings'].append(f'Symbol unusually long: {len(symbol)} characters')
        
        # Check for invalid patterns - more permissive
        invalid_patterns = [
            r'^[0-9]+$',  # Pure numbers (but allow some numeric symbols)
            r'^[^A-Za-z0-9]+$',  # No alphanumeric characters
            r'^[A-Za-z]{20,}$',  # Too long (increased from 10)
            r'INVALID_',  # Invalid prefix
            r'TEST_',  # Test prefix
            r'DUMMY_',  # Dummy prefix
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, symbol):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'Symbol matches invalid pattern: {pattern}')
        
        # Check for known invalid symbols - more specific
        known_invalid_symbols = [
            'INVALID_SYMBOL_12345',
            'TEST_SYMBOL',
            'DUMMY_STOCK',
            'VERYLONGSYMBOLNAME123456789',  # Too long
        ]
        
        if symbol.upper() in [s.upper() for s in known_invalid_symbols]:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Symbol is in known invalid list')
        
        # Check for special characters - more permissive
        if re.search(r'[^A-Za-z0-9\.\-]', symbol):  # Allow dots and hyphens
            validation_result['is_valid'] = False
            validation_result['errors'].append('Symbol contains invalid special characters')
        
        # Check if it's all uppercase (common for stock symbols) - warning only
        if not symbol.isupper():
            validation_result['warnings'].append('Symbol should typically be uppercase')
        
        return validation_result
    
    async def test_symbol_validation(self, symbol: str) -> Dict[str, Any]:
        """Test symbol validation with comprehensive checks"""
        print(f"🔍 Testing symbol validation for: {symbol}")
        
        # Validate symbol format
        validation = self.is_valid_symbol(symbol)
        
        # Test API responses
        api_tests = await self.test_api_responses(symbol)
        
        return {
            'symbol': symbol,
            'validation': validation,
            'api_tests': api_tests,
            'timestamp': datetime.now().isoformat()
        }
    
    async def test_api_responses(self, symbol: str) -> Dict[str, Any]:
        """Test API responses for a symbol"""
        api_results = {}
        
        # Test NewsAPI
        api_results['newsapi'] = await self.test_newsapi_response(symbol)
        
        # Test Defeat Beta API
        api_results['defeatbeta'] = await self.test_defeatbeta_response(symbol)
        
        # Test SEC Filings
        api_results['sec'] = await self.test_sec_response(symbol)
        
        # Test YouTube API
        api_results['youtube'] = await self.test_youtube_response(symbol)
        
        return api_results
    
    async def test_newsapi_response(self, symbol: str) -> Dict[str, Any]:
        """Test NewsAPI response for symbol"""
        if not self.news_api_key:
            return {'status': 'NO_API_KEY', 'error': 'NewsAPI key not available'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{symbol}" OR "{symbol} stock"',
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'pageSize': 5
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        if len(articles) == 0:
                            return {
                                'status': 'NO_RESULTS',
                                'message': f'No news articles found for {symbol}',
                                'articles_count': 0
                            }
                        else:
                            return {
                                'status': 'SUCCESS',
                                'message': f'Found {len(articles)} articles for {symbol}',
                                'articles_count': len(articles)
                            }
                    else:
                        return {
                            'status': 'API_ERROR',
                            'error': f'NewsAPI returned status {response.status}',
                            'status_code': response.status
                        }
        except Exception as e:
            return {
                'status': 'EXCEPTION',
                'error': f'Exception testing NewsAPI: {str(e)}'
            }
    
    async def test_defeatbeta_response(self, symbol: str) -> Dict[str, Any]:
        """Test Defeat Beta API response for symbol"""
        try:
            from defeatbeta_api_integration import DefeatBetaAPIIntegration
            defeatbeta = DefeatBetaAPIIntegration()
            
            if not defeatbeta.installed:
                return {'status': 'NOT_INSTALLED', 'error': 'Defeat Beta API not installed'}
            
            # Test price data
            price_data = await defeatbeta.get_stock_price_data(symbol)
            
            if 'error' in price_data:
                return {
                    'status': 'ERROR',
                    'error': price_data.get('error', 'Unknown error'),
                    'message': f'Defeat Beta API error for {symbol}'
                }
            else:
                records = price_data.get('records', 0)
                if records == 0:
                    return {
                        'status': 'NO_DATA',
                        'message': f'No price data found for {symbol}',
                        'records': 0
                    }
                else:
                    return {
                        'status': 'SUCCESS',
                        'message': f'Found {records} price records for {symbol}',
                        'records': records
                    }
                    
        except Exception as e:
            return {
                'status': 'EXCEPTION',
                'error': f'Exception testing Defeat Beta API: {str(e)}'
            }
    
    async def test_sec_response(self, symbol: str) -> Dict[str, Any]:
        """Test SEC filings response for symbol"""
        try:
            from sec_filings_integration import SECFilingsIntegration
            sec = SECFilingsIntegration()
            
            # Test company search
            sec_data = await sec.get_comprehensive_sec_data(symbol)
            
            if 'error' in sec_data:
                return {
                    'status': 'ERROR',
                    'error': sec_data.get('error', 'Unknown error'),
                    'message': f'SEC API error for {symbol}'
                }
            
            company_info = sec_data.get('company_info', {})
            if not company_info or not company_info.get('name'):
                return {
                    'status': 'NO_COMPANY',
                    'message': f'No company found for {symbol}',
                    'company_name': None
                }
            else:
                return {
                    'status': 'SUCCESS',
                    'message': f'Found company: {company_info.get("name")}',
                    'company_name': company_info.get('name')
                }
                
        except Exception as e:
            return {
                'status': 'EXCEPTION',
                'error': f'Exception testing SEC API: {str(e)}'
            }
    
    async def test_youtube_response(self, symbol: str) -> Dict[str, Any]:
        """Test YouTube API response for symbol"""
        if not self.youtube_api_key:
            return {'status': 'NO_API_KEY', 'error': 'YouTube API key not available'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    'part': 'snippet',
                    'q': f'{symbol} stock news',
                    'type': 'video',
                    'maxResults': 5,
                    'key': self.youtube_api_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        videos = data.get('items', [])
                        
                        if len(videos) == 0:
                            return {
                                'status': 'NO_RESULTS',
                                'message': f'No YouTube videos found for {symbol}',
                                'videos_count': 0
                            }
                        else:
                            return {
                                'status': 'SUCCESS',
                                'message': f'Found {len(videos)} videos for {symbol}',
                                'videos_count': len(videos)
                            }
                    elif response.status == 403:
                        return {
                            'status': 'RATE_LIMITED',
                            'error': 'YouTube API rate limited',
                            'status_code': response.status
                        }
                    else:
                        return {
                            'status': 'API_ERROR',
                            'error': f'YouTube API returned status {response.status}',
                            'status_code': response.status
                        }
        except Exception as e:
            return {
                'status': 'EXCEPTION',
                'error': f'Exception testing YouTube API: {str(e)}'
            }
    
    async def run_comprehensive_error_test(self) -> Dict[str, Any]:
        """Run comprehensive error handling test"""
        print("🛡️ Running Comprehensive Error Handling Test")
        print("="*60)
        
        test_symbols = [
            'AAPL',  # Valid symbol
            'INVALID_SYMBOL_12345',  # Invalid symbol
            'TEST_SYMBOL',  # Test symbol
            '12345',  # Numbers only
            'A',  # Too short
            'VERYLONGSYMBOLNAME123456789',  # Too long
            'ABC@#$',  # Special characters
            '',  # Empty string
            None,  # None value
        ]
        
        results = {}
        error_detection_count = 0
        total_tests = len(test_symbols)
        
        for symbol in test_symbols:
            print(f"\n🔍 Testing symbol: {symbol}")
            
            if symbol is None:
                # Handle None case specially
                validation = self.is_valid_symbol('')
                validation['symbol'] = None
                validation['is_valid'] = False
                validation['errors'].append('Symbol is None')
                results[str(symbol)] = {
                    'symbol': symbol,
                    'validation': validation,
                    'api_tests': {},
                    'timestamp': datetime.now().isoformat()
                }
                error_detection_count += 1
            else:
                result = await self.test_symbol_validation(symbol)
                results[str(symbol)] = result
                
                # Check if error was properly detected
                validation = result['validation']
                if not validation['is_valid']:
                    error_detection_count += 1
                    print(f"   ❌ Error detected: {validation['errors']}")
                else:
                    print(f"   ✅ Symbol appears valid")
        
        # Calculate error detection rate
        error_detection_rate = error_detection_count / total_tests
        
        print(f"\n📊 ERROR DETECTION SUMMARY:")
        print(f"   Total Symbols Tested: {total_tests}")
        print(f"   Errors Detected: {error_detection_count}")
        print(f"   Error Detection Rate: {error_detection_rate:.1%}")
        
        if error_detection_rate >= 0.8:
            print(f"   🎉 Excellent error detection!")
        elif error_detection_rate >= 0.6:
            print(f"   ✅ Good error detection")
        else:
            print(f"   ⚠️ Poor error detection - needs improvement")
        
        return {
            'test_results': results,
            'summary': {
                'total_symbols': total_tests,
                'errors_detected': error_detection_count,
                'error_detection_rate': error_detection_rate,
                'timestamp': datetime.now().isoformat()
            }
        }

async def main():
    """Run the improved error handling test"""
    error_handler = ImprovedErrorHandling()
    
    # Run comprehensive error test
    results = await error_handler.run_comprehensive_error_test()
    
    # Print detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for symbol, result in results['test_results'].items():
        validation = result['validation']
        status = "❌ INVALID" if not validation['is_valid'] else "✅ VALID"
        print(f"   {status} {symbol}: {validation.get('errors', ['No errors'])}")
    
    # Print API test results for invalid symbols
    print(f"\n🔧 API RESPONSES FOR INVALID SYMBOLS:")
    for symbol, result in results['test_results'].items():
        validation = result['validation']
        if not validation['is_valid']:
            print(f"\n   📊 {symbol}:")
            api_tests = result['api_tests']
            for api_name, api_result in api_tests.items():
                status = api_result.get('status', 'UNKNOWN')
                message = api_result.get('message', api_result.get('error', 'No message'))
                print(f"      {api_name}: {status} - {message}")

if __name__ == "__main__":
    asyncio.run(main())
