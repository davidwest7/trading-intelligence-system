#!/bin/bash

# Real Backtest Setup Script
# This script helps set up and run a real backtest with Polygon data

echo "🚀 Setting up Real Backtest with Polygon Data"
echo "============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if Polygon API key is set
if [ -z "$POLYGON_API_KEY" ]; then
    echo "❌ POLYGON_API_KEY environment variable not set!"
    echo ""
    echo "📋 To get a Polygon API key:"
    echo "1. Go to https://polygon.io/"
    echo "2. Sign up for a free account"
    echo "3. Get your API key from the dashboard"
    echo ""
    echo "🔑 Set your API key:"
    echo "export POLYGON_API_KEY='your_api_key_here'"
    echo ""
    echo "💡 Or add it to your .env file:"
    echo "echo 'POLYGON_API_KEY=your_api_key_here' >> .env"
    echo ""
    echo "⚠️  Please set your Polygon API key and run this script again."
    exit 1
fi

echo "✅ Polygon API key found: ${POLYGON_API_KEY:0:8}..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p real_data
mkdir -p real_backtest_results
mkdir -p logs

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python3 -c "import pandas, numpy, requests, pyarrow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements_backtesting.txt
fi

# Test Polygon API connection
echo "🔗 Testing Polygon API connection..."
python3 -c "
import os
import requests

api_key = os.getenv('POLYGON_API_KEY')
if api_key:
    url = f'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02?apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        print('✅ Polygon API connection successful!')
    else:
        print(f'❌ Polygon API connection failed: {response.status_code}')
        print('Please check your API key and internet connection.')
else:
    print('❌ No API key found')
"

# Run the real backtest
echo ""
echo "🎯 Starting Real Backtest..."
echo "This will download 3+ years of real market data and run comprehensive strategies."
echo "Estimated time: 5-10 minutes (depending on data download speed)"
echo ""

python3 real_backtest.py

echo ""
echo "🎉 Real backtest completed!"
echo "📁 Check the results in: real_backtest_results/"
echo "📋 View the log file: real_backtest.log"
echo ""
echo "📊 To view results:"
echo "cat real_backtest_results/strategy_comparison.csv"
echo ""
echo "Happy trading! 🚀"
