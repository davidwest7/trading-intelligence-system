# Contributing to Trading Intelligence System

Thank you for your interest in contributing to the Trading Intelligence System! This document provides guidelines and information for contributors.

## 🎯 Contributing Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Keep discussions on-topic and professional

### Getting Started
1. Fork the repository
2. Set up your development environment
3. Read the architecture documentation
4. Start with a good first issue

## 🏗️ Development Setup

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- Git
- Make (for automation)

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/trading-intelligence-system.git
cd trading-intelligence-system

# Setup development environment
make setup

# Activate virtual environment
source venv/bin/activate

# Start development services
make dev
```

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code ...

# Run tests and quality checks
make test
make lint
make type-check

# Commit your changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

## 📋 Code Standards

### Python Code Style
- **Formatter**: Black with 100 character line length
- **Import Sorting**: isort with Black profile
- **Linting**: flake8 + pylint
- **Type Checking**: mypy

### Code Quality Requirements
```bash
# Format code
make format

# Check code quality
make lint           # Must pass
make type-check     # Must pass
make test           # Coverage >= 80%
```

### Documentation Standards
- **Docstrings**: Required for all public classes and functions
- **Type Hints**: Required for all function parameters and returns
- **Comments**: Explain complex business logic
- **README**: Update if adding new features

### Example Code Style
```python
"""
Module docstring explaining the purpose
"""

from typing import Dict, List, Optional
import asyncio


class TradingAgent:
    """
    Base class for trading agents
    
    Args:
        name: Agent identifier
        config: Configuration dictionary
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
    
    async def process_signal(self, signal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process incoming trading signal
        
        Args:
            signal_data: Raw signal data from source
            
        Returns:
            List of processed opportunities
            
        Raises:
            ValueError: If signal_data is invalid
        """
        if not signal_data:
            raise ValueError("Signal data cannot be empty")
            
        # Process the signal...
        return []
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/              # Unit tests for individual components
│   ├── agents/
│   ├── common/
│   └── conftest.py
├── integration/       # Integration tests for workflows
├── load/              # Load and performance tests
└── fixtures/          # Test data and fixtures
```

### Writing Tests
```python
import pytest
from unittest.mock import AsyncMock, patch

from agents.technical.agent import TechnicalAgent


class TestTechnicalAgent:
    """Test suite for Technical Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent instance"""
        return TechnicalAgent(config={'test': True})
    
    @pytest.mark.asyncio
    async def test_find_opportunities_success(self, agent):
        """Test successful opportunity finding"""
        # Arrange
        payload = {
            'symbols': ['EURUSD'],
            'timeframes': ['1h'],
            'strategies': ['imbalance']
        }
        
        # Act
        result = await agent.find_opportunities(payload)
        
        # Assert
        assert 'opportunities' in result
        assert 'metadata' in result
        assert isinstance(result['opportunities'], list)
    
    @pytest.mark.asyncio
    async def test_find_opportunities_invalid_input(self, agent):
        """Test error handling for invalid input"""
        with pytest.raises(ValueError):
            await agent.find_opportunities({})
```

### Test Data Management
- Use fixtures for reusable test data
- Mock external API calls
- Create realistic but deterministic test data
- Clean up test data after tests

### Coverage Requirements
- **Minimum**: 80% overall coverage
- **New Features**: 90% coverage
- **Critical Paths**: 95% coverage

## 🏗️ Architecture Guidelines

### Agent Development
When creating new agents:

1. **Inherit from BaseAgent**:
```python
from common.models import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("my_agent", config)
```

2. **Implement Required Methods**:
```python
async def process(self, *args, **kwargs) -> Dict[str, Any]:
    """Main processing method"""
    pass
```

3. **Add Comprehensive TODOs**:
```python
"""
My Agent for specific trading analysis

TODO Items:
1. Implement core algorithm
2. Add data source integration
3. Implement error handling
4. Add performance monitoring
5. Create unit tests
"""
```

### Data Integration
- Use the feature store for persistent features
- Implement proper data adapters
- Handle data quality and validation
- Implement proper error handling

### Event-Driven Architecture
```python
# Publishing events
await event_bus.publish_agent_signal(
    source="my_agent",
    agent_name="my_agent", 
    signal_type="buy_signal",
    confidence=0.8,
    additional_data={"symbol": "AAPL", "price": 150.0}
)

# Subscribing to events
async def handle_market_data(event: Event):
    # Process market data event
    pass

event_bus.subscribe(EventType.MARKET_TICK, handle_market_data)
```

## 📊 Performance Guidelines

### Optimization Principles
- **Async First**: Use async/await for I/O operations
- **Batch Processing**: Process multiple items together
- **Caching**: Cache expensive computations
- **Monitoring**: Add performance metrics

### Memory Management
- Use generators for large datasets
- Implement proper cleanup in finally blocks
- Monitor memory usage in long-running processes
- Use connection pooling for databases

### Example Optimization
```python
import asyncio
from typing import List

async def process_symbols_batch(symbols: List[str]) -> List[Dict[str, Any]]:
    """Process symbols in parallel batches"""
    batch_size = 10
    results = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        
        # Process batch in parallel
        tasks = [process_single_symbol(symbol) for symbol in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error processing symbol: {result}")
            else:
                results.append(result)
    
    return results
```

## 🔐 Security Guidelines

### Data Protection
- Never commit API keys or passwords
- Use environment variables for secrets
- Implement proper input validation
- Log security events

### Code Security
```python
# Good: Input validation
def validate_symbol(symbol: str) -> str:
    if not symbol or not symbol.isalnum():
        raise ValueError("Invalid symbol format")
    return symbol.upper()

# Good: SQL injection prevention
query = "SELECT * FROM prices WHERE symbol = %s"
cursor.execute(query, (symbol,))

# Bad: String concatenation
query = f"SELECT * FROM prices WHERE symbol = '{symbol}'"  # DON'T DO THIS
```

### API Security
- Implement rate limiting
- Use JWT tokens for authentication
- Validate all inputs
- Log security events

## 📈 Financial Compliance

### Research-Grade Development
- Default to `ENABLE_EXECUTION=false`
- Implement proper risk controls
- Add compliance warnings
- Document regulatory considerations

### Risk Management
```python
class RiskManager:
    """Risk management utilities"""
    
    def validate_position_size(self, symbol: str, size: float) -> bool:
        """Validate position size against limits"""
        max_position = self.get_max_position(symbol)
        return size <= max_position
    
    def check_drawdown_limit(self, portfolio_value: float) -> bool:
        """Check if drawdown exceeds limits"""
        max_drawdown = self.calculate_max_drawdown()
        return max_drawdown <= self.config.max_drawdown_limit
```

## 🚀 Deployment Guidelines

### Docker Best Practices
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Configuration Management
- Use environment variables
- Implement configuration validation
- Support different environments (dev/staging/prod)
- Document all configuration options

## 📚 Documentation

### Code Documentation
- Docstrings for all public APIs
- Type hints for all functions
- Inline comments for complex logic
- Architecture decision records (ADRs)

### API Documentation
- OpenAPI/Swagger specs
- Request/response examples
- Error code documentation
- Rate limiting information

### User Documentation
- Setup and installation guides
- Configuration examples
- Troubleshooting guides
- FAQ sections

## 🎯 Contribution Areas

### High Priority
1. **Complete Sentiment Agent**
   - Twitter/Reddit integration
   - Bot detection algorithms
   - Entity resolution

2. **Complete Flow Agent**
   - HMM regime detection
   - Breadth indicators
   - Cross-asset analysis

3. **Improve Technical Agent**
   - Additional strategies
   - Better backtesting
   - Risk metrics

### Medium Priority
1. **Causal Analysis Agent**
   - DoWhy integration
   - Synthetic controls
   - Impact estimation

2. **Data Source Integration**
   - Additional APIs
   - Error handling
   - Rate limiting

3. **Performance Optimization**
   - Caching strategies
   - Async improvements
   - Memory optimization

### Low Priority
1. **Advanced Features**
   - ML model integration
   - Advanced visualizations
   - Mobile API

2. **Deployment Automation**
   - AWS/GCP deployment
   - Monitoring setup
   - Scaling automation

## 🔍 Code Review Process

### Pull Request Requirements
- [ ] Tests pass
- [ ] Code coverage >= 80%
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] CHANGELOG updated (for features)

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Logic is correct and efficient
- [ ] Error handling is appropriate
- [ ] Security considerations addressed
- [ ] Performance implications considered
- [ ] Documentation is clear and complete

### Review Process
1. **Automated Checks**: CI/CD pipeline runs
2. **Code Review**: Maintainer reviews code
3. **Testing**: Manual testing if needed
4. **Approval**: Approval from maintainer
5. **Merge**: Squash and merge to main

## 🆘 Getting Help

### Resources
- **Documentation**: README.md and docs/
- **Issues**: GitHub Issues for bugs
- **Discussions**: GitHub Discussions for questions
- **Examples**: Look at existing agent implementations

### Common Issues
1. **Setup Problems**: Check Python version and dependencies
2. **Docker Issues**: Ensure Docker is running and has sufficient resources
3. **API Errors**: Check environment variables and API keys
4. **Test Failures**: Run tests locally and check logs

### Contact
- Create GitHub Issue for bugs
- Start GitHub Discussion for questions
- Tag maintainers for urgent issues

## 📄 License and Legal

### Contribution License
By contributing to this project, you agree that your contributions will be licensed under the MIT License.

### Financial Disclaimer
This software is for research and educational purposes only. Contributors must ensure their code includes appropriate disclaimers and risk warnings.

### Data Rights Checklist
- [ ] No proprietary data included
- [ ] API usage complies with terms of service
- [ ] Personal data is properly protected
- [ ] Licensing is compatible with MIT

---

Thank you for contributing to the Trading Intelligence System! 🚀
