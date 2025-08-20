#!/usr/bin/env python3
"""
Finnhub API Integration
Comprehensive integration for real-time financial data
"""
import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv('env_real_keys.env')

class FinnhubIntegration:
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
        self.base_url = "https://finnhub.io/api/v1"
        self.rate_limits = {
            'calls': 0,
            'limit': 60,  # 60 calls per minute
            'reset_time': time.time() + 60
        }
        
    def _check_rate_limit(self):
        """Check and manage rate limits"""
        current_time = time.time()
        if current_time > self.rate_limits['reset_time']:
            self.rate_limits['calls'] = 0
            self.rate_limits['reset_time'] = current_time + 60
        
        if self.rate_limits['calls'] >= self.rate_limits['limit']:
            sleep_time = self.rate_limits['reset_time'] - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.rate_limits['calls'] = 0
                self.rate_limits['reset_time'] = time.time() + 60
        
        self.rate_limits['calls'] += 1
    
    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make API request with rate limiting"""
        self._check_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    print(f"❌ Finnhub API: Invalid API key")
                    return None
                elif response.status == 429:
                    print(f"⚠️ Finnhub API: Rate limit exceeded")
                    return None
                else:
                    print(f"❌ Finnhub API: Request failed with status {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Finnhub API: Request error: {str(e)}")
            return None
    
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote"""
        try:
            async with aiohttp.ClientSession() as session:
                data = await self._make_request(session, "quote", {'symbol': symbol})
                
                if data:
                    return {
                        'status': 'WORKING',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'data': {
                            'current_price': data.get('c', 0),
                            'change': data.get('d', 0),
                            'percent_change': data.get('dp', 0),
                            'high': data.get('h', 0),
                            'low': data.get('l', 0),
                            'open': data.get('o', 0),
                            'previous_close': data.get('pc', 0)
                        }
                    }
                else:
                    return {'status': 'ERROR', 'error': 'Failed to get quote data'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Quote error: {str(e)}'}
    
    async def get_company_news(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """Get company-specific news with sentiment"""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            async with aiohttp.ClientSession() as session:
                data = await self._make_request(session, "company-news", {
                    'symbol': symbol,
                    'from': from_date,
                    'to': to_date
                })
                
                if data:
                    # Analyze sentiment for each article
                    articles_with_sentiment = []
                    total_sentiment = 0
                    
                    for article in data[:20]:  # Limit to 20 articles
                        sentiment_score = article.get('sentiment', 0)
                        total_sentiment += sentiment_score
                        
                        articles_with_sentiment.append({
                            'headline': article.get('headline', ''),
                            'summary': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'datetime': article.get('datetime', ''),
                            'sentiment': sentiment_score,
                            'source': article.get('source', '')
                        })
                    
                    avg_sentiment = total_sentiment / len(articles_with_sentiment) if articles_with_sentiment else 0
                    
                    return {
                        'status': 'WORKING',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'data': {
                            'articles': articles_with_sentiment,
                            'total_articles': len(data),
                            'analyzed_articles': len(articles_with_sentiment),
                            'average_sentiment': round(avg_sentiment, 4),
                            'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
                        }
                    }
                else:
                    return {'status': 'ERROR', 'error': 'Failed to get news data'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'News error: {str(e)}'}
    
    async def get_financial_statements(self, symbol: str) -> Dict[str, Any]:
        """Get financial statements data"""
        try:
            async with aiohttp.ClientSession() as session:
                data = await self._make_request(session, "stock/financials-reported", {'symbol': symbol})
                
                if data:
                    return {
                        'status': 'WORKING',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'data': {
                            'financials': data,
                            'available_statements': list(data.keys()) if isinstance(data, dict) else [],
                            'data_points': len(data) if isinstance(data, list) else 0
                        }
                    }
                else:
                    return {'status': 'ERROR', 'error': 'Failed to get financial data'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Financials error: {str(e)}'}
    
    async def get_insider_transactions(self, symbol: str) -> Dict[str, Any]:
        """Get insider trading data"""
        try:
            async with aiohttp.ClientSession() as session:
                data = await self._make_request(session, "stock/insider-transactions", {'symbol': symbol})
                
                if data:
                    return {
                        'status': 'WORKING',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'data': {
                            'transactions': data,
                            'total_transactions': len(data),
                            'recent_activity': len([t for t in data if t.get('transactionDate', '') > (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')])
                        }
                    }
                else:
                    return {'status': 'ERROR', 'error': 'Failed to get insider data'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Insider error: {str(e)}'}
    
    async def get_institutional_holdings(self, symbol: str) -> Dict[str, Any]:
        """Get institutional holdings data"""
        try:
            async with aiohttp.ClientSession() as session:
                data = await self._make_request(session, "stock/institutional-holdings", {'symbol': symbol})
                
                if data:
                    return {
                        'status': 'WORKING',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'data': {
                            'holdings': data,
                            'total_holders': len(data),
                            'total_shares': sum(h.get('shares', 0) for h in data),
                            'total_value': sum(h.get('value', 0) for h in data)
                        }
                    }
                else:
                    return {'status': 'ERROR', 'error': 'Failed to get institutional data'}
        except Exception as e:
            return {'status': 'ERROR', 'error': f'Institutional error: {str(e)}'}
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data from all Finnhub endpoints"""
        print(f"🎯 Getting comprehensive Finnhub data for {symbol}...")
        
        start_time = time.time()
        
        # Collect data from all endpoints
        tasks = [
            self.get_stock_quote(symbol),
            self.get_company_news(symbol),
            self.get_financial_statements(symbol),
            self.get_insider_transactions(symbol),
            self.get_institutional_holdings(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_sources = {}
        working_sources = 0
        total_sources = len(results)
        
        source_names = ['quote', 'news', 'financials', 'insider', 'institutional']
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Error in {source_names[i]}: {result}")
                data_sources[source_names[i]] = {'status': 'ERROR', 'error': str(result)}
            else:
                data_sources[source_names[i]] = result
                if result.get('status') == 'WORKING':
                    working_sources += 1
        
        success_rate = (working_sources / total_sources * 100) if total_sources > 0 else 0
        
        return {
            'status': 'WORKING' if working_sources > 0 else 'ERROR',
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'collection_time': round(time.time() - start_time, 2),
            'summary': f'Finnhub: {working_sources}/{total_sources} sources working',
            'data': {
                'working_sources': working_sources,
                'total_sources': total_sources,
                'success_rate': success_rate,
                'available_data': list(data_sources.keys()),
                'sources': data_sources
            }
        }

async def main():
    """Demo the Finnhub integration"""
    print("🚀 Finnhub API Integration Demo")
    print("=" * 50)
    
    finnhub = FinnhubIntegration()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting comprehensive data for {symbol}...")
    
    comprehensive_data = await finnhub.get_comprehensive_data(symbol)
    
    # Print results
    print(f"\n✅ Data collection complete!")
    print(f"⏱️ Collection time: {comprehensive_data['collection_time']} seconds")
    print(f"📈 Success rate: {comprehensive_data['data']['success_rate']:.1f}%")
    
    # Print detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for source_name, source_data in comprehensive_data['data']['sources'].items():
        status = source_data.get('status', 'UNKNOWN')
        emoji = '✅' if status == 'WORKING' else '❌'
        print(f"   {emoji} {source_name.upper()}: {status}")
        
        if status == 'WORKING' and 'data' in source_data:
            data = source_data['data']
            if source_name == 'quote':
                print(f"      💰 Price: ${data.get('current_price', 0):.2f} ({data.get('percent_change', 0):+.2f}%)")
            elif source_name == 'news':
                print(f"      📰 Articles: {data.get('total_articles', 0)} (Sentiment: {data.get('average_sentiment', 0):.3f})")
            elif source_name == 'financials':
                print(f"      📊 Statements: {data.get('data_points', 0)} data points")
            elif source_name == 'insider':
                print(f"      👥 Transactions: {data.get('total_transactions', 0)}")
            elif source_name == 'institutional':
                print(f"      🏢 Holders: {data.get('total_holders', 0)}")

if __name__ == "__main__":
    asyncio.run(main())
