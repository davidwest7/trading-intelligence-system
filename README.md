# Trading Intelligence System

A production-ready, multi-agent trading intelligence system built for research-grade analysis across multiple asset classes including FX, equities, futures, crypto, and fixed income.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd trading-intelligence-system

# Quick start (includes setup, build, and run)
make quickstart

# Or step by step:
make setup
make docker-build
make docker-up
```

**Access Points:**
- 📊 **Dashboard**: http://localhost:3000 (admin/admin)
- 🔌 **API**: http://localhost:8080
- 📓 **Analytics**: http://localhost:8888
- 📈 **Monitoring**: http://localhost:9090

## 🏗️ Architecture

### BMAD Structure

The system follows a **Builder-Manager-Agent-Data** (BMAD) architecture:

- **Builder**: FastAPI services and MCP tools
- **Manager**: Unified scoring and ranking system
- **Agent**: Specialized trading intelligence agents
- **Data**: DuckDB/Parquet feature store with vector embeddings

### Core Components

```
├── agents/                 # Trading intelligence agents
│   ├── technical/         # Technical analysis (imbalance, FVG, liquidity sweeps)
│   ├── sentiment/         # Multi-source sentiment analysis
│   ├── flow/              # Market regime and flow analysis
│   ├── macro/             # Economic calendar and geopolitical events
│   ├── top_performers/    # Cross-sectional momentum ranking
│   ├── undervalued/       # Fundamental and technical undervaluation
│   ├── insider/           # Form 4 and insider trading analysis
│   ├── causal/            # DoWhy synthetic controls and impact analysis
│   ├── hedging/           # Linear overlays and volatility targeting
│   └── learning/          # Contextual bandit strategy weights
├── common/                # Shared utilities and infrastructure
│   ├── data_adapters/     # IBKR, TradingView, and other data sources
│   ├── feature_store/     # DuckDB/Parquet feature storage
│   ├── event_bus/         # Inter-agent event communication
│   └── scoring/           # Unified scoring and ranking system
├── schemas/               # MCP tool contracts (JSON)
└── config/                # Configuration and deployment
```

## 🤖 Trading Agents

### 1. Technical Strategy Agent
**Status**: ✅ **Implemented**
- Imbalance/Fair Value Gap (FVG) detection
- Liquidity sweep identification
- Institutional Dealing Range/Point (IDFP) analysis
- Multi-timeframe alignment
- Purged cross-validation backtesting

### 2. Sentiment Analysis Agent
**Status**: 🔧 **Stub + TODOs**
- Multi-source sentiment (Twitter/X, Reddit, News, Telegram, Discord)
- Bot detection and deduplication
- Entity resolution and stance analysis
- Velocity and dispersion metrics

### 3. Direction-of-Flow Agent
**Status**: 🔧 **Stub + TODOs**
- Market breadth indicators
- Volatility term structure analysis
- Hidden Markov Model regime detection
- Cross-asset correlation analysis

### 4. Macro/Geopolitical Agent
**Status**: 🔧 **Stub + TODOs**
- Economic calendar integration (FRED, Trading Economics)
- Central bank communication analysis
- Election and policy tracking
- Scenario mapping and stress testing

### 5. Other Agents
All other agents have stubs with comprehensive TODO lists:
- **Top Performers**: Cross-sectional momentum models
- **Undervalued**: DCF, multiples, and technical oversold analysis
- **Insider Trading**: Form 4 parsing and alpha decay curves
- **Causal Impact**: DoWhy synthetic controls
- **Hedging**: Linear overlays and vol targeting
- **Learning**: Contextual bandit over strategy weights

## 📊 Unified Scoring System

**Formula**:
```
UnifiedScore = w1×Likelihood + w2×ExpectedReturn – w3×Risk + w4×Liquidity + w5×Conviction + w6×Recency + w7×RegimeFit
```

**Features**:
- Asset class-specific weight configurations
- Isotonic/Platt calibration for probability estimates
- Regime-aware scoring adjustments
- Diversification penalties
- Confidence intervals

**Default Weights by Asset Class**:
```yaml
equities:
  likelihood: 0.25
  expected_return: 0.20
  risk: 0.20
  liquidity: 0.10
  conviction: 0.10
  recency: 0.10
  regime_fit: 0.05

fx:
  likelihood: 0.30
  expected_return: 0.15
  risk: 0.25
  liquidity: 0.15
  conviction: 0.10
  recency: 0.05
  regime_fit: 0.0
```

## 🔌 MCP Tool Contracts

All 13 agent tools have complete JSON schemas in `/schemas/`:

1. **market-data.get_ohlcv** - OHLCV data retrieval
2. **news.search** - Financial news search
3. **sentiment.stream** - Real-time sentiment analysis
4. **technical.find_opportunities** - Technical trading opportunities
5. **top_performers.rank** - Asset performance ranking
6. **undervalued.scan** - Undervaluation analysis
7. **flow.regime_map** - Market regime detection
8. **macro.timeline** - Economic event timeline
9. **moneyflows.rotate** - Flow rotation analysis
10. **causal.estimate** - Causal impact estimation
11. **ranker.score** - Unified opportunity scoring
12. **hedger.plan** - Portfolio hedging plans
13. **learning.bandit** - Adaptive strategy weights

## 💾 Data Architecture

### Feature Store (DuckDB + Parquet)
```python
# Write features
await feature_store.write_features('technical_indicators', features_df)

# Read with point-in-time correctness
features = await feature_store.get_point_in_time_features(
    timestamp=datetime.now(),
    symbols=['AAPL', 'TSLA'],
    feature_groups=['technical_indicators', 'sentiment_scores']
)
```

### Event Bus
```python
# Publish events
await event_bus.publish_market_tick('ibkr', 'AAPL', 150.0, 1000)
await event_bus.publish_agent_signal('technical', 'buy_signal', 0.8)

# Subscribe to events
event_bus.subscribe(EventType.MARKET_TICK, handle_market_data)
```

## 📈 Data Sources

### Market Data
- **Interactive Brokers** (primary) - Live quotes and paper trading
- **TradingView** - Charts and OHLCV data
- **Yahoo Finance** - Free backup data source
- **Alpha Vantage** - Alternative data provider

### News & Sentiment
- **Twitter/X API** - Social sentiment
- **Reddit API** - Community sentiment
- **Financial News APIs** - Professional news sources
- **Telegram/Discord** - Alternative social sources

### Economic Data
- **FRED** - Federal Reserve economic data
- **Trading Economics** - Global economic indicators
- **EDGAR** - SEC filings and insider trading

### Geographic Coverage
**Prioritized regions**: US, EU, UK, JPY, China, Korea, India, UAE, Saudi

## 🚦 Compliance & Safety

### Research-Grade Only
- **Default**: `ENABLE_EXECUTION=false`
- All strategies are research and backtesting only
- No automatic trade execution unless explicitly enabled
- Paper trading mode for IBKR integration

### Risk Management
- Position sizing and risk limits
- Maximum drawdown controls
- Volatility targeting
- Diversification constraints

## 🛠️ Development

### Environment Setup
```bash
# Create virtual environment
make setup-venv
source venv/bin/activate

# Install dependencies
make install-deps

# Setup environment
cp env.template .env
# Edit .env with your API keys
```

### Running Tests
```bash
# Full test suite
make test

# Individual components
make test-agents
make test-integration

# Code quality
make lint
make type-check
make format
```

### Docker Development
```bash
# Development environment
make dev

# Production-like environment
make docker-up

# View logs
make docker-logs

# Health check
make health-check
```

## 📋 Configuration

### Environment Variables
Copy `env.template` to `.env` and configure:

**Critical Settings**:
```bash
# Safety first
ENABLE_EXECUTION=false

# Data sources
IBKR_HOST=127.0.0.1
IBKR_PAPER_TRADING=true

# APIs (optional for basic functionality)
TWITTER_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
ALPHA_VANTAGE_API_KEY=your_key
```

### Scoring Configuration
```bash
# Generate default config
python -c "from common.scoring.unified_score import UnifiedScorer; UnifiedScorer().save_default_config('config/scoring_weights.yaml')"

# Edit weights per asset class
vim config/scoring_weights.yaml
```

## 🚀 Deployment

### Local Development
```bash
make quickstart
```

### Staging/Production
```bash
# AWS ECS/Fargate (placeholder)
make deploy-aws

# Google Cloud Run (placeholder) 
make deploy-gcp

# Manual production deployment
cp .env.template .env.prod
# Configure production settings
make prod
```

### Infrastructure
- **Local**: Docker Compose
- **Staging/Prod**: AWS ECS/Fargate or Google Cloud Run
- **Storage**: S3/GCS for artifacts, Secrets Manager for keys
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions

## 📊 Monitoring

### Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Docs**: http://localhost:8080/docs

### Key Metrics
- Agent performance and latency
- Data source availability
- Scoring system calibration
- System resource usage

### Alerts
- Data source failures
- Agent errors
- Performance degradation
- Security events

## 🧪 Testing Strategy

### Test Coverage
- **Unit Tests**: Individual agent logic
- **Integration Tests**: End-to-end workflows
- **Backtests**: Historical performance validation
- **Load Tests**: System performance under load

### Purged Cross-Validation
```python
from agents.technical.backtest import PurgedCrossValidationBacktester

backtester = PurgedCrossValidationBacktester(
    purge_pct=0.02,    # 2% purge to avoid data leakage
    embargo_pct=0.01,  # 1% embargo period
    n_splits=5         # 5-fold CV
)

results = await backtester.run_backtest(strategy, data, start_date, end_date)
```

## 🔐 Security

### API Security
- JWT authentication
- Rate limiting
- CORS configuration
- Input validation

### Data Security
- Secrets management
- Encrypted communications
- Access logging
- Regular security scans

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Example API Calls
```bash
# Find technical opportunities
curl -X POST "http://localhost:8080/technical/find_opportunities" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["EURUSD", "GBPUSD"],
    "timeframes": ["15m", "1h", "4h"],
    "strategies": ["imbalance", "trend"],
    "min_score": 0.7
  }'

# Get sentiment stream
curl -X POST "http://localhost:8080/sentiment/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "TSLA"],
    "window": "1h",
    "sources": ["twitter", "reddit"]
  }'
```

## 🤝 Contributing

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards
- **Python**: PEP 8 with Black formatting
- **Type Hints**: Required for all functions
- **Documentation**: Docstrings for all classes and functions
- **Tests**: Minimum 80% coverage for new code

### TODO Priorities
1. **High Priority**: Complete sentiment and flow agents
2. **Medium Priority**: Implement causal analysis and insider tracking
3. **Low Priority**: Advanced ML features and optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for research and educational purposes only. It is not financial advice and should not be used for live trading without proper risk management and compliance review. Past performance does not guarantee future results.

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: `/docs` folder (when generated)

---

**Built with ❤️ for quantitative researchers and algorithmic traders**
