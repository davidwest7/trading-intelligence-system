#!/usr/bin/env python3
"""
SEC Filings Integration
Real SEC filings data for institutional insights and insider trading
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

class SECFilingsIntegration:
    """SEC filings integration for institutional data"""
    
    def __init__(self):
        # SEC EDGAR API (free, no key required)
        self.base_url = "https://data.sec.gov"
        self.user_agent = "TradingSentimentBot/1.0 (your-email@domain.com)"
        
        # Rate limiting (SEC requires 10 requests per second max)
        self.rate_limits = {
            'calls': 0,
            'limit': 10,
            'reset_time': time.time() + 1  # 1 second window
        }
        
        # Important filing types
        self.filing_types = {
            '4': 'Insider Trading',
            '13F-HR': 'Institutional Holdings',
            '13F-HR/A': 'Institutional Holdings Amendment',
            '8-K': 'Current Report',
            '10-K': 'Annual Report',
            '10-Q': 'Quarterly Report',
            'DEF 14A': 'Proxy Statement',
            'SC 13G': 'Beneficial Ownership',
            'SC 13D': 'Beneficial Ownership (5%+)'
        }
    
    def _check_rate_limit(self):
        """Check SEC rate limits"""
        if time.time() > self.rate_limits['reset_time']:
            self.rate_limits['calls'] = 0
            self.rate_limits['reset_time'] = time.time() + 1
        
        if self.rate_limits['calls'] >= self.rate_limits['limit']:
            return False
        
        self.rate_limits['calls'] += 1
        return True
    
    async def _make_sec_request(self, session: aiohttp.ClientSession, url: str, 
                               headers: dict = None) -> Optional[dict]:
        """Make SEC API request with proper headers"""
        if not self._check_rate_limit():
            print("⚠️ SEC rate limit reached, waiting...")
            await asyncio.sleep(1)
            return None
        
        default_headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json'
        }
        
        if headers:
            default_headers.update(headers)
        
        try:
            async with session.get(url, headers=default_headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    print("⚠️ SEC rate limited, waiting...")
                    await asyncio.sleep(2)
                    return None
                else:
                    print(f"⚠️ SEC request failed: {url}, status: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error making SEC request to {url}: {e}")
            return None
    
    async def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """Get company facts and financial data"""
        print(f"📊 Getting company facts for CIK: {cik}")
        
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_sec_request(session, url)
            
            if data:
                return {
                    'cik': cik,
                    'company_name': data.get('entityName', ''),
                    'facts': data.get('facts', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'cik': cik,
                    'error': 'Failed to fetch company facts',
                    'timestamp': datetime.now().isoformat()
                }
    
    async def get_recent_filings(self, cik: str, filing_type: str = None, 
                                limit: int = 10) -> List[Dict]:
        """Get recent filings for a company"""
        print(f"📄 Getting recent filings for CIK: {cik}")
        
        # Get company submissions
        url = f"{self.base_url}/submissions/CIK{cik.zfill(10)}.json"
        
        async with aiohttp.ClientSession() as session:
            data = await self._make_sec_request(session, url)
            
            if not data or 'filings' not in data:
                return []
            
            filings = data['filings']['recent']
            recent_filings = []
            
            for i in range(min(len(filings['accessionNumber']), limit)):
                filing = {
                    'accessionNumber': filings['accessionNumber'][i],
                    'filingDate': filings['filingDate'][i],
                    'form': filings['form'][i],
                    'description': filings.get('primaryDocument', [''])[i] if 'primaryDocument' in filings else '',
                    'fileNumber': filings.get('fileNumber', [''])[i] if 'fileNumber' in filings else ''
                }
                
                # Filter by filing type if specified
                if filing_type and filing['form'] != filing_type:
                    continue
                
                recent_filings.append(filing)
            
            return recent_filings
    
    async def get_insider_trading(self, cik: str, days_back: int = 30) -> List[Dict]:
        """Get recent insider trading activity (Form 4 filings)"""
        print(f"👥 Getting insider trading for CIK: {cik}")
        
        filings = await self.get_recent_filings(cik, filing_type='4', limit=50)
        
        insider_data = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for filing in filings:
            try:
                filing_date = datetime.strptime(filing['filingDate'], '%Y-%m-%d')
                if filing_date >= cutoff_date:
                    insider_data.append({
                        'filing_date': filing['filingDate'],
                        'accession_number': filing['accessionNumber'],
                        'form': filing['form'],
                        'description': filing['description']
                    })
            except Exception as e:
                print(f"⚠️ Error parsing filing date: {e}")
                continue
        
        return insider_data
    
    async def get_institutional_holdings(self, cik: str) -> List[Dict]:
        """Get institutional holdings (13F filings)"""
        print(f"🏢 Getting institutional holdings for CIK: {cik}")
        
        # Get 13F-HR filings
        filings = await self.get_recent_filings(cik, filing_type='13F-HR', limit=5)
        
        institutional_data = []
        
        for filing in filings:
            institutional_data.append({
                'filing_date': filing['filingDate'],
                'accession_number': filing['accessionNumber'],
                'form': filing['form'],
                'description': filing['description']
            })
        
        return institutional_data
    
    async def get_company_events(self, cik: str, days_back: int = 30) -> List[Dict]:
        """Get recent company events (8-K filings)"""
        print(f"📢 Getting company events for CIK: {cik}")
        
        filings = await self.get_recent_filings(cik, filing_type='8-K', limit=20)
        
        events_data = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for filing in filings:
            try:
                filing_date = datetime.strptime(filing['filingDate'], '%Y-%m-%d')
                if filing_date >= cutoff_date:
                    events_data.append({
                        'filing_date': filing['filingDate'],
                        'accession_number': filing['accessionNumber'],
                        'form': filing['form'],
                        'description': filing['description']
                    })
            except Exception as e:
                print(f"⚠️ Error parsing filing date: {e}")
                continue
        
        return events_data
    
    async def search_companies(self, query: str) -> List[Dict]:
        """Search for companies by name or ticker"""
        print(f"🔍 Searching for companies: {query}")
        
        # SEC doesn't have a direct search API, so we'll use a common approach
        # For now, return common companies that match the query
        common_companies = {
            'AAPL': {'cik': '0000320193', 'name': 'Apple Inc.'},
            'MSFT': {'cik': '0000789019', 'name': 'Microsoft Corporation'},
            'GOOGL': {'cik': '0001652044', 'name': 'Alphabet Inc.'},
            'AMZN': {'cik': '0001018724', 'name': 'Amazon.com Inc.'},
            'TSLA': {'cik': '0001318605', 'name': 'Tesla Inc.'},
            'META': {'cik': '0001326801', 'name': 'Meta Platforms Inc.'},
            'NVDA': {'cik': '0001045810', 'name': 'NVIDIA Corporation'},
            'NFLX': {'cik': '0001065280', 'name': 'Netflix Inc.'},
            'JPM': {'cik': '0000019617', 'name': 'JPMorgan Chase & Co.'},
            'JNJ': {'cik': '0000200404', 'name': 'Johnson & Johnson'}
        }
        
        results = []
        query_upper = query.upper()
        
        for ticker, company in common_companies.items():
            if (query_upper in ticker or 
                query_upper in company['name'].upper() or
                query_upper in company['cik']):
                results.append({
                    'ticker': ticker,
                    'cik': company['cik'],
                    'name': company['name']
                })
        
        return results
    
    async def get_comprehensive_sec_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive SEC data for a symbol"""
        print(f"📋 Getting comprehensive SEC data for {symbol}...")
        
        # Search for company
        companies = await self.search_companies(symbol)
        
        if not companies:
            return {
                'symbol': symbol,
                'error': 'Company not found',
                'timestamp': datetime.now().isoformat()
            }
        
        company = companies[0]  # Use first match
        cik = company['cik']
        
        # Collect all SEC data
        tasks = [
            self.get_company_facts(cik),
            self.get_insider_trading(cik),
            self.get_institutional_holdings(cik),
            self.get_company_events(cik)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        sec_data = {
            'symbol': symbol,
            'company_info': company,
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        # Company facts
        if not isinstance(results[0], Exception):
            sec_data['data']['company_facts'] = results[0]
        else:
            sec_data['data']['company_facts'] = {'error': str(results[0])}
        
        # Insider trading
        if not isinstance(results[1], Exception):
            sec_data['data']['insider_trading'] = results[1]
        else:
            sec_data['data']['insider_trading'] = {'error': str(results[1])}
        
        # Institutional holdings
        if not isinstance(results[2], Exception):
            sec_data['data']['institutional_holdings'] = results[2]
        else:
            sec_data['data']['institutional_holdings'] = {'error': str(results[2])}
        
        # Company events
        if not isinstance(results[3], Exception):
            sec_data['data']['company_events'] = results[3]
        else:
            sec_data['data']['company_events'] = {'error': str(results[3])}
        
        # Calculate insights
        sec_data['insights'] = self._calculate_sec_insights(sec_data['data'])
        
        return sec_data
    
    def _calculate_sec_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate insights from SEC data"""
        insights = {
            'insider_activity': 'neutral',
            'institutional_interest': 'neutral',
            'recent_events': 0,
            'filing_activity': 'normal'
        }
        
        # Analyze insider trading
        insider_data = data.get('insider_trading', [])
        if isinstance(insider_data, list):
            insights['insider_activity'] = 'high' if len(insider_data) > 5 else 'normal' if len(insider_data) > 0 else 'low'
        
        # Analyze institutional holdings
        institutional_data = data.get('institutional_holdings', [])
        if isinstance(institutional_data, list):
            insights['institutional_interest'] = 'high' if len(institutional_data) > 2 else 'normal' if len(institutional_data) > 0 else 'low'
        
        # Count recent events
        events_data = data.get('company_events', [])
        if isinstance(events_data, list):
            insights['recent_events'] = len(events_data)
        
        # Determine filing activity
        total_filings = 0
        for key in ['insider_trading', 'institutional_holdings', 'company_events']:
            if isinstance(data.get(key), list):
                total_filings += len(data[key])
        
        if total_filings > 10:
            insights['filing_activity'] = 'high'
        elif total_filings > 5:
            insights['filing_activity'] = 'normal'
        else:
            insights['filing_activity'] = 'low'
        
        return insights

async def main():
    """Test SEC filings integration"""
    print("📋 SEC Filings Integration Test")
    print("="*50)
    
    sec = SECFilingsIntegration()
    
    # Test with AAPL
    symbol = 'AAPL'
    print(f"\n📊 Getting comprehensive SEC data for {symbol}...")
    
    sec_data = await sec.get_comprehensive_sec_data(symbol)
    
    # Display results
    print(f"\n📋 RESULTS:")
    print(f"   Symbol: {sec_data['symbol']}")
    print(f"   Company: {sec_data['company_info']['name']}")
    print(f"   CIK: {sec_data['company_info']['cik']}")
    
    # Display insights
    insights = sec_data.get('insights', {})
    print(f"\n🎯 INSIGHTS:")
    print(f"   Insider Activity: {insights.get('insider_activity', 'unknown')}")
    print(f"   Institutional Interest: {insights.get('institutional_interest', 'unknown')}")
    print(f"   Recent Events: {insights.get('recent_events', 0)}")
    print(f"   Filing Activity: {insights.get('filing_activity', 'unknown')}")
    
    # Display data summary
    data = sec_data.get('data', {})
    print(f"\n📊 DATA SUMMARY:")
    
    for key, value in data.items():
        if isinstance(value, list):
            print(f"   {key.replace('_', ' ').title()}: {len(value)} items")
        elif isinstance(value, dict):
            if 'error' in value:
                print(f"   {key.replace('_', ' ').title()}: Error - {value['error']}")
            else:
                print(f"   {key.replace('_', ' ').title()}: Available")
    
    # Show sample insider trading
    insider_data = data.get('insider_trading', [])
    if isinstance(insider_data, list) and insider_data:
        print(f"\n👥 SAMPLE INSIDER TRADING:")
        for i, insider in enumerate(insider_data[:3]):
            print(f"   {i+1}. {insider['filing_date']}: {insider['form']}")
    
    # Show sample company events
    events_data = data.get('company_events', [])
    if isinstance(events_data, list) and events_data:
        print(f"\n📢 SAMPLE COMPANY EVENTS:")
        for i, event in enumerate(events_data[:3]):
            print(f"   {i+1}. {event['filing_date']}: {event['form']}")
    
    print(f"\n🎉 SEC filings integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())
