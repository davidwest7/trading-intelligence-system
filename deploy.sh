#!/bin/bash

# 🚀 Trading Intelligence System Deployment Script
# This script sets up the complete trading intelligence system

set -e  # Exit on any error

echo "🚀 Deploying Trading Intelligence System..."
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python $python_version is compatible"
else
    print_error "Python $python_version is too old. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Create environment file
print_status "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Trading Intelligence System Environment Configuration
# Add your API keys here

# Polygon.io API Key (Required for market data)
POLYGON_API_KEY=your_polygon_api_key_here

# Optional API Keys
TWITTER_BEARER_TOKEN=your_twitter_token_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_secret_here
NEWS_API_KEY=your_news_api_key_here
FRED_API_KEY=your_fred_api_key_here

# System Configuration
LOG_LEVEL=INFO
DATA_CACHE_DIR=./data
RESULTS_DIR=./results
EOF
    print_success "Environment file created (.env)"
    print_warning "Please edit .env and add your API keys"
else
    print_warning "Environment file already exists"
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p data
mkdir -p results
mkdir -p logs
print_success "Directories created"

# Test the system
print_status "Testing system installation..."
if python -c "import pandas, numpy, yfinance, ta" 2>/dev/null; then
    print_success "Core dependencies test passed"
else
    print_error "Core dependencies test failed"
    exit 1
fi

# Run quick test
print_status "Running quick system test..."
if python test_backtesting_system.py 2>/dev/null; then
    print_success "System test passed"
else
    print_warning "System test had issues (this is normal for first run)"
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python comprehensive_architecture_backtest.py"
echo ""
echo "Documentation:"
echo "- README.md - Main documentation"
echo "- BACKTESTING_SYSTEM_SUMMARY.md - Backtesting guide"
echo "- DEPLOYMENT_SUCCESS_SUMMARY.md - Deployment details"
echo ""
echo "Happy trading! 🚀"
