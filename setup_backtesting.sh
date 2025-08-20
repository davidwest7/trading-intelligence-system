#!/bin/bash

# Backtesting System Setup Script
# This script sets up the comprehensive backtesting system

echo "🚀 Setting up Comprehensive Backtesting System"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements_backtesting.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p backtest_results
mkdir -p logs

# Set up environment variables template
if [ ! -f ".env" ]; then
    echo "🔑 Creating environment template..."
    cat > .env << EOF
# Polygon API Configuration
POLYGON_API_KEY=your_polygon_api_key_here

# S3 Configuration (Optional)
S3_BUCKET=your_s3_bucket_here
S3_PREFIX=polygon

# Local Storage Configuration
LOCAL_DATA_PATH=data

# Logging Configuration
LOG_LEVEL=INFO
EOF
    echo "📝 Created .env template. Please edit with your API keys."
fi

# Run basic tests
echo "🧪 Running basic tests..."
python test_backtesting_system.py

# Run demo
echo "🎯 Running demo..."
python demo_backtesting_system.py

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "1. Edit .env file with your Polygon API key"
echo "2. Run: python demo_backtesting_system.py"
echo "3. Check backtest_results/ for output"
echo ""
echo "📚 Documentation:"
echo "- README.md - Quick start guide"
echo "- BACKTESTING_SYSTEM_SUMMARY.md - Comprehensive documentation"
echo "- demo_backtesting_system.py - Example usage"
echo ""
echo "🔧 Configuration:"
echo "- config/backtest_config.yaml - System configuration"
echo "- .env - Environment variables"
echo ""
echo "Happy backtesting! 🚀"
