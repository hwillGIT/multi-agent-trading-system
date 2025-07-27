# Multi-Agent Trading System

A comprehensive trading system using multiple specialized agents for market analysis, signal generation, and risk management.

## Architecture Overview

The system consists of 6 main layers with specialized agents:

### 1. Data/Universe & Preprocessing Layer
- **Data Universe Agent**: Defines investment universe, fetches and cleans multi-frequency data
- Supports equities, ETFs, futures, options, FX, crypto, and cross-asset data
- Handles data validation, outlier filtering, and timestamp harmonization

### 2. Feature Engineering & Model Signal Generation
- **Technical Analysis Agent**: Extracts classic and exotic indicators, patterns, and statistical features
- **ML Ensemble Agent**: Runs custom ML/AI models including LLMs, ensemble methods, and uncertainty quantification

### 3. Strategy Modules (Explicit, Separable)
- **Momentum Strategy Agent**: Multi-factor momentum, sector rotation, breakouts
- **Statistical Arbitrage Agent**: Pairs trading, mean reversion, cointegration
- **Event-Driven Agent**: Earnings, macro events, news sentiment analysis
- **Options Strategy Agent**: Multi-leg strategies, volatility analysis
- **Cross-Asset Agent**: Portfolio context, correlation analysis

### 4. Multi-Layer Synthesis, Validation, and Risk
- **Signal Synthesis Agent**: Combines and arbitrates between strategy signals
- **Risk Management Agent**: Advanced risk modeling, scenario analysis, stress testing
- **Ranking & Filtering Agent**: Multi-metric ranking with user preferences

### 5. Output/Justification, Logging & Audit
- **Recommendation Agent**: Full rationale with chain-of-thought explanations
- **Logging Agent**: Comprehensive audit trails for compliance and improvement

## Features

- **Multi-Asset Support**: Equities, ETFs, options, futures, crypto
- **Multi-Frequency Analysis**: Daily, weekly, monthly data harmonization
- **Advanced Technical Analysis**: 20+ indicators, pattern recognition
- **Machine Learning Integration**: Ensemble methods, LLM integration
- **Comprehensive Risk Management**: VaR, stress testing, scenario analysis
- **Full Audit Trail**: Complete decision tracking for compliance
- **Modular Architecture**: Easily extensible and maintainable

## Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- API keys for data providers (Alpha Vantage recommended)

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run the system:
```bash
python main.py
```

### Docker Installation

1. Clone and configure:
```bash
git clone <repository-url>
cd trading_system
cp .env.example .env
# Edit .env with your configuration
```

2. Start with Docker Compose:
```bash
docker-compose up -d
```

This will start:
- Trading System application
- PostgreSQL database
- Redis cache
- Jupyter Lab (port 8888)
- Prometheus monitoring (port 9090)
- Grafana dashboards (port 3000)

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key_here
QUANDL_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379/0

# LLM APIs (optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

### Configuration File

Modify `config/config.yaml` to customize:
- Asset universe criteria
- Risk management parameters
- Strategy weights and thresholds
- Output preferences

## Usage

### Basic Usage

```python
from datetime import datetime, timedelta
from main import TradingSystem

# Initialize system
system = TradingSystem()

# Define analysis period
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Run analysis
results = await system.run_analysis(
    start_date=start_date,
    end_date=end_date,
    asset_classes=["equities", "etfs"],
    exchanges=["NYSE", "NASDAQ"],
    custom_symbols=["AAPL", "GOOGL", "MSFT"]
)
```

### Advanced Configuration

```python
# Custom universe configuration
universe_config = {
    "equities": {
        "market_cap_min": 5e9,  # $5B minimum
        "volume_min": 2e6,      # 2M daily volume
        "sectors": ["Technology", "Healthcare"]
    },
    "risk_management": {
        "max_portfolio_risk": 0.015,  # 1.5% daily VaR
        "max_single_position": 0.03   # 3% max position
    }
}
```

## Output Format

The system provides structured recommendations:

```json
{
  "rank": 1,
  "strategy_name": "AAPL Momentum Breakout",
  "signal": "BUY",
  "strategies_used": ["Momentum", "Technical", "ML"],
  "expected_return": 0.08,
  "volatility": 0.25,
  "sharpe_ratio": 1.45,
  "max_drawdown": 0.12,
  "confirming_factors": [
    "Technical breakout confirmed",
    "ML model 85% confidence",
    "Positive sentiment spike"
  ],
  "rationale": "Strong momentum with technical confirmation...",
  "warnings": ["High volatility period", "Earnings next week"]
}
```

## API Documentation

### Core APIs

- **Market Data API**: Multi-provider market data fetching
- **Fundamental Data API**: Financial statements and ratios
- **News Sentiment API**: News analysis and sentiment scoring
- **Options Data API**: Options chains and Greeks calculation

### Agent APIs

Each agent exposes a consistent interface:

```python
# Agent execution
result = await agent.safe_execute(inputs)

# Result structure
{
    "agent_name": "DataUniverseAgent",
    "timestamp": "2024-01-15T10:30:00Z",
    "data": { ... },
    "metadata": { ... },
    "success": true,
    "execution_time_ms": 1250
}
```

## Development

### Project Structure

```
trading_system/
├── agents/                 # Specialized trading agents
│   ├── data_universe/     # Data fetching and cleaning
│   ├── feature_engineering/ # Technical analysis
│   ├── ml_ensemble/       # ML model ensemble
│   ├── strategies/        # Trading strategies
│   ├── synthesis/         # Signal combination
│   ├── risk_management/   # Risk analysis
│   └── output/           # Recommendations
├── core/                  # Core framework
│   ├── base/             # Base classes
│   ├── utils/            # Utilities
│   └── apis/             # External APIs
├── config/               # Configuration files
├── data/                 # Data storage
├── logs/                 # Log files
├── tests/                # Unit tests
└── docs/                 # Documentation
```

### Adding New Agents

1. Create agent class inheriting from `BaseAgent`:

```python
from core.base.agent import BaseAgent, AgentOutput

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyCustomAgent", "my_config_section")
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        # Implementation here
        return AgentOutput(
            agent_name=self.name,
            data={"result": "my_result"},
            success=True
        )
```

2. Register in main system orchestrator
3. Add configuration section to `config.yaml`
4. Add tests

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading_system

# Run specific test
pytest tests/test_data_universe.py
```

## Monitoring

### Logs

- Application logs: `logs/trading_system.log`
- Audit logs: `logs/audit.log`
- Structured logging with JSON format

### Metrics

- Performance metrics via Prometheus
- Custom dashboards in Grafana
- Real-time monitoring of agent execution

### Health Checks

```bash
# Check system status
curl http://localhost:8000/health

# Check agent status
curl http://localhost:8000/agents/status
```

## Security

- API key management via environment variables
- Input validation and sanitization
- Audit logging for compliance
- Rate limiting on external API calls

## Performance

- Asynchronous agent execution
- Concurrent data fetching
- Redis caching for expensive operations
- Optimized database queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[License information here]

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the configuration examples

## Roadmap

- [ ] Real-time data streaming
- [ ] Advanced ML models (transformers, reinforcement learning)
- [ ] Portfolio optimization algorithms
- [ ] Risk management enhancements
- [ ] Web-based dashboard
- [ ] Mobile notifications
- [ ] Additional asset classes
- [ ] Enhanced backtesting framework