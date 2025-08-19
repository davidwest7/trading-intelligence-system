#!/usr/bin/env python3
"""
Debug Polygon.io Data Retrieval
Test different timeframes and see what data is actually returned
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add current directory to path
sys.path.append('.')

# Load environment variables
load_dotenv('env_real_keys.env')

from common.data_adapters.polygon_adapter import PolygonAdapter

async def debug_polygon_data():
    """Debug Polygon.io data retrieval"""
    print("🔍 **POLYGON.IO DATA DEBUG**")
    print("=" * 50)
    
    config = {
        'polygon_api_key': os.getenv('POLYGON_API_KEY')
    }
    
    adapter = PolygonAdapter(config)
    test_symbol = 'AAPL'
    
    # Test different timeframes
    timeframes = [
        ('1m', '1', 100),
        ('5m', '5', 100),
        ('15m', '15', 100),
        ('1h', '60', 100),
        ('1d', 'D', 50)
    ]
    
    print(f"Testing symbol: {test_symbol}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    for tf_name, interval, limit in timeframes:
        print(f"\n📊 **TESTING {tf_name} TIMEFRAME**")
        print("-" * 30)
        
        try:
            # Get data
            data = await adapter.get_intraday_data(test_symbol, interval=interval, limit=limit)
            
            print(f"✅ Data retrieved successfully")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            
            if not data.empty:
                print(f"   First row: {data.iloc[0].to_dict()}")
                print(f"   Last row: {data.iloc[-1].to_dict()}")
                print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # Check for valid data
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in data.columns:
                        print(f"   {col}: min={data[col].min():.2f}, max={data[col].max():.2f}, mean={data[col].mean():.2f}")
                
                # Check for any zero or negative values
                zero_prices = (data[['open', 'high', 'low', 'close']] <= 0).any().any()
                zero_volume = (data['volume'] <= 0).any()
                
                if zero_prices:
                    print(f"   ⚠️ WARNING: Zero or negative prices found")
                if zero_volume:
                    print(f"   ⚠️ WARNING: Zero or negative volume found")
                
            else:
                print(f"   ❌ Empty DataFrame returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test Level 2 data
    print(f"\n📊 **TESTING LEVEL 2 DATA**")
    print("-" * 30)
    
    try:
        level2_data = await adapter.get_level2_data(test_symbol)
        print(f"✅ Level 2 data retrieved")
        print(f"   Data: {level2_data}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test real-time quote
    print(f"\n📊 **TESTING REAL-TIME QUOTE**")
    print("-" * 30)
    
    try:
        quote = await adapter.get_real_time_quote(test_symbol)
        print(f"✅ Real-time quote retrieved")
        print(f"   Data: {quote}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_polygon_data())
