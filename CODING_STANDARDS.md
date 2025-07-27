# Trading System - Global Coding Best Practices & Standards

## Table of Contents
1. [Code Style & Formatting](#code-style--formatting)
2. [Architecture & Design Principles](#architecture--design-principles)
3. [Documentation Standards](#documentation-standards)
4. [Error Handling & Logging](#error-handling--logging)
5. [Testing Standards](#testing-standards)
6. [Security Best Practices](#security-best-practices)
7. [Performance Guidelines](#performance-guidelines)
8. [Git & Version Control](#git--version-control)
9. [Code Review Guidelines](#code-review-guidelines)
10. [Financial Domain Specifics](#financial-domain-specifics)

## Code Style & Formatting

### Python Style Guide
- **Follow PEP 8** with the following adjustments:
  - Line length: 88 characters (Black default)
  - Use double quotes for strings
  - Use trailing commas in multi-line structures

### Formatting Tools
```bash
# Auto-formatting
black .
isort .

# Linting
flake8 .
mypy .
```

### Naming Conventions
```python
# Classes: PascalCase
class DataUniverseAgent:
    pass

# Functions/methods: snake_case
def calculate_sharpe_ratio():
    pass

# Variables: snake_case
market_data = get_data()

# Constants: UPPER_SNAKE_CASE
MAX_PORTFOLIO_RISK = 0.02

# Private methods: _snake_case
def _internal_helper():
    pass

# Financial terms: Keep standard abbreviations
var_95 = calculate_var()  # Value at Risk
pe_ratio = calculate_pe()  # Price-to-Earnings
```

### Type Hints (Mandatory)
```python
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

async def get_market_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d"
) -> pd.DataFrame:
    """Always include type hints for better code clarity."""
    pass

# Complex types
MarketData = Dict[str, pd.DataFrame]
RiskMetrics = Dict[str, float]
SignalOutput = Dict[str, Union[float, str, List[str]]]
```

## Architecture & Design Principles

### SOLID Principles

#### 1. Single Responsibility Principle
```python
# Good: Each agent has one clear purpose
class DataUniverseAgent(BaseAgent):
    """Responsible only for data fetching and cleaning."""
    pass

class RiskManagementAgent(BaseAgent):
    """Responsible only for risk calculations."""
    pass

# Bad: Agent doing multiple unrelated tasks
class EverythingAgent(BaseAgent):
    """DON'T: Handles data, strategies, and risk."""
    pass
```

#### 2. Open/Closed Principle
```python
# Good: Extensible without modification
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self) -> float:
        pass

class MomentumStrategy(BaseStrategy):
    def generate_signal(self) -> float:
        # Implementation
        pass
```

#### 3. Liskov Substitution Principle
```python
# All agents must be substitutable
def execute_agent(agent: BaseAgent, inputs: Dict) -> AgentOutput:
    return await agent.safe_execute(inputs)
```

#### 4. Interface Segregation
```python
# Separate interfaces for different capabilities
class PriceDataProvider(Protocol):
    async def get_prices(self, symbol: str) -> pd.DataFrame:
        pass

class FundamentalDataProvider(Protocol):
    async def get_fundamentals(self, symbol: str) -> Dict:
        pass
```

#### 5. Dependency Inversion
```python
# Depend on abstractions, not concretions
class TradingStrategy:
    def __init__(self, data_provider: PriceDataProvider):
        self.data_provider = data_provider  # Interface, not concrete class
```

### Agent Design Pattern

Every agent follows this structure:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.base.agent import BaseAgent, AgentOutput

class YourAgent(BaseAgent):
    """
    Clear, one-line description of agent purpose.
    
    This agent handles [specific responsibility] by [method].
    Inputs: [what it needs]
    Outputs: [what it produces]
    """
    
    def __init__(self):
        super().__init__("YourAgent", "your_config_section")
        # Initialize dependencies
        self._setup_dependencies()
    
    def _setup_dependencies(self) -> None:
        """Initialize external dependencies and validate configuration."""
        pass
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Main execution logic.
        
        Args:
            inputs: Required input parameters
            
        Returns:
            AgentOutput with processed results
            
        Raises:
            ValidationError: If inputs are invalid
            ProcessingError: If processing fails
        """
        # 1. Validate inputs
        self._validate_inputs(inputs)
        
        # 2. Process data
        results = await self._process_data(inputs)
        
        # 3. Validate outputs
        self._validate_outputs(results)
        
        # 4. Return structured output
        return AgentOutput(
            agent_name=self.name,
            data=results,
            metadata=self._generate_metadata(inputs, results)
        )
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["key1", "key2"]
        self.validate_inputs(inputs, required_keys)
        
        # Domain-specific validation
        if inputs["some_value"] < 0:
            raise ValidationError("Value must be positive")
    
    async def _process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Core processing logic."""
        # Implement main algorithm
        pass
    
    def _validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """Validate output data quality."""
        # Check for NaN, inf, empty results
        pass
    
    def _generate_metadata(self, inputs: Dict, outputs: Dict) -> Dict[str, Any]:
        """Generate metadata about the processing."""
        return {
            "processing_time": datetime.utcnow(),
            "input_summary": self._summarize_inputs(inputs),
            "output_summary": self._summarize_outputs(outputs)
        }
```

## Documentation Standards

### Docstring Format (Google Style)
```python
def calculate_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio performance metrics.
    
    This function computes key risk-adjusted performance metrics including
    Sharpe ratio, Sortino ratio, maximum drawdown, and beta coefficient.
    
    Args:
        returns: Portfolio return series with datetime index
        benchmark_returns: Benchmark return series for comparison
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
            Defaults to 2% (0.02).
    
    Returns:
        Dictionary containing calculated metrics:
            - sharpe_ratio: Risk-adjusted return metric
            - sortino_ratio: Downside deviation adjusted return
            - max_drawdown: Maximum peak-to-trough decline
            - beta: Sensitivity to benchmark movements
            - alpha: Excess return over expected return
    
    Raises:
        ValidationError: If return series are empty or misaligned
        ValueError: If risk_free_rate is negative
    
    Example:
        >>> portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> benchmark_returns = pd.Series([0.008, 0.015, -0.005, 0.025])
        >>> metrics = calculate_portfolio_metrics(portfolio_returns, benchmark_returns)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        Sharpe Ratio: 1.45
    
    Note:
        All ratios are annualized assuming 252 trading days per year.
        Returns should be in decimal format (0.01 for 1%).
    """
    pass
```

### Code Comments
```python
# Good: Explain WHY, not WHAT
def calculate_options_gamma(spot: float, strike: float, time_to_expiry: float) -> float:
    # Use Black-Scholes formula for European options
    # Gamma measures the rate of change of delta with respect to underlying price
    d1 = self._calculate_d1(spot, strike, time_to_expiry)
    
    # Standard normal probability density function
    # Higher gamma indicates greater delta sensitivity near the strike
    gamma = stats.norm.pdf(d1) / (spot * self.volatility * math.sqrt(time_to_expiry))
    
    return gamma

# Bad: Comments that restate the code
def calculate_returns(prices):
    # Calculate returns  # <- This doesn't add value
    returns = prices.pct_change()  # Calculate percentage change
    return returns  # Return the returns
```

### README Structure
Each module should have:
1. Purpose and scope
2. Key components
3. Usage examples
4. Configuration options
5. Dependencies
6. Performance considerations

## Error Handling & Logging

### Exception Hierarchy
```python
# Custom exception hierarchy
class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.utcnow()

class DataError(TradingSystemError):
    """Data quality or availability issues."""
    pass

class ValidationError(TradingSystemError):
    """Input or output validation failures."""
    pass

class ModelError(TradingSystemError):
    """ML model training or prediction issues."""
    pass

class RiskError(TradingSystemError):
    """Risk limit violations or calculation failures."""
    pass
```

### Error Handling Patterns
```python
# Pattern 1: Graceful degradation
async def get_market_data_with_fallback(symbol: str) -> pd.DataFrame:
    """Get market data with provider fallback."""
    providers = ['primary', 'secondary', 'tertiary']
    
    for provider in providers:
        try:
            data = await self._fetch_from_provider(provider, symbol)
            self.logger.info(f"Successfully fetched {symbol} from {provider}")
            return data
        except APIError as e:
            self.logger.warning(f"Provider {provider} failed for {symbol}: {e}")
            continue
    
    raise DataError(f"All providers failed for {symbol}")

# Pattern 2: Retry with exponential backoff
import asyncio
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except APIError as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator

# Pattern 3: Context managers for resources
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_transaction():
    """Ensure database transactions are properly handled."""
    transaction = await db.begin()
    try:
        yield transaction
        await transaction.commit()
    except Exception:
        await transaction.rollback()
        raise
    finally:
        await transaction.close()
```

### Logging Best Practices
```python
from loguru import logger

# Structured logging with context
logger = logger.bind(
    agent="MomentumStrategy",
    symbol="AAPL",
    strategy_id="momentum_001"
)

# Log levels and usage
logger.debug("Detailed diagnostic information")  # Development only
logger.info("General operational information")   # Normal flow
logger.warning("Unexpected but recoverable condition")  # Degraded performance
logger.error("Error that prevents operation")    # Failed operation
logger.critical("System-threatening condition")  # Immediate attention

# Financial-specific logging
def log_trade_signal(symbol: str, signal: float, confidence: float, reasons: List[str]):
    """Log trading signals with full context for audit trail."""
    logger.bind(
        event_type="trade_signal",
        symbol=symbol,
        signal_strength=signal,
        confidence_level=confidence,
        reasoning=reasons
    ).info(f"Generated {signal:.2f} signal for {symbol}")

# Performance logging
import time
from functools import wraps

def log_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper
```

## Testing Standards

### Test Structure
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from datetime import datetime, timedelta

class TestMomentumStrategy:
    """Test suite for MomentumStrategy agent."""
    
    @pytest.fixture
    def momentum_strategy(self):
        """Create strategy instance for testing."""
        return MomentumStrategy()
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = pd.Series(range(100, 200), index=dates)
        return pd.DataFrame({
            "close": prices,
            "volume": [1000000] * 100
        })
    
    # Unit tests
    def test_calculate_momentum_signal(self, momentum_strategy, sample_price_data):
        """Test momentum signal calculation."""
        signal = momentum_strategy._calculate_momentum_signal(sample_price_data)
        
        assert isinstance(signal, float)
        assert -1 <= signal <= 1
        assert not math.isnan(signal)
    
    # Integration tests
    @pytest.mark.asyncio
    async def test_full_execution_flow(self, momentum_strategy):
        """Test complete agent execution."""
        inputs = {
            "symbol": "AAPL",
            "lookback_period": 20,
            "price_data": self.sample_price_data
        }
        
        result = await momentum_strategy.safe_execute(inputs)
        
        assert result.success
        assert "momentum_signal" in result.data
        assert result.execution_time_ms > 0
    
    # Edge cases
    def test_empty_data_handling(self, momentum_strategy):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValidationError):
            momentum_strategy._calculate_momentum_signal(empty_data)
    
    # Mock external dependencies
    @patch('agents.momentum.MarketDataAPI')
    async def test_with_mocked_api(self, mock_api, momentum_strategy):
        """Test with mocked external API."""
        mock_api.return_value.get_historical_data.return_value = self.sample_price_data
        
        # Test logic without actual API calls
        pass

# Financial-specific test utilities
class FinancialTestUtils:
    """Utilities for financial testing."""
    
    @staticmethod
    def create_price_series(
        start_price: float = 100,
        periods: int = 252,
        volatility: float = 0.2,
        drift: float = 0.1
    ) -> pd.Series:
        """Generate realistic price series using geometric Brownian motion."""
        import numpy as np
        
        dt = 1/252  # Daily time step
        dates = pd.date_range("2024-01-01", periods=periods, freq="D")
        
        # Generate random walks
        random_shocks = np.random.normal(0, 1, periods)
        returns = drift * dt + volatility * np.sqrt(dt) * random_shocks
        
        # Calculate cumulative prices
        prices = start_price * np.exp(np.cumsum(returns))
        
        return pd.Series(prices, index=dates)
    
    @staticmethod
    def assert_valid_returns(returns: pd.Series):
        """Assert return series meets basic validity criteria."""
        assert not returns.empty
        assert not returns.isnull().all()
        assert returns.abs().max() < 1.0  # No 100%+ daily returns
        assert not np.isinf(returns).any()
```

## Security Best Practices

### Secrets Management
```python
# Good: Use environment variables
import os
from core.base.config import config

API_KEY = config.alpha_vantage_api_key  # From environment
DATABASE_URL = config.database_url

# Bad: Hardcoded secrets
API_KEY = "abc123"  # DON'T DO THIS

# Secure configuration access
class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self):
        self._validate_required_secrets()
    
    def _validate_required_secrets(self):
        required_vars = [
            "ALPHA_VANTAGE_API_KEY",
            "DATABASE_URL",
            "SECRET_KEY"
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ConfigError(f"Missing required environment variables: {missing}")
```

### Input Validation
```python
from typing import Any
import re

def validate_symbol(symbol: str) -> str:
    """Validate stock symbol format."""
    if not isinstance(symbol, str):
        raise ValidationError("Symbol must be a string")
    
    symbol = symbol.upper().strip()
    
    # Basic symbol validation
    if not re.match(r'^[A-Z]{1,5}(-USD)?$', symbol):
        raise ValidationError(f"Invalid symbol format: {symbol}")
    
    return symbol

def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """Validate date range for data requests."""
    if start_date >= end_date:
        raise ValidationError("Start date must be before end date")
    
    if (end_date - start_date).days > 3650:  # 10 years
        raise ValidationError("Date range too large (max 10 years)")
    
    if start_date < datetime(1900, 1, 1):
        raise ValidationError("Start date too early")

def sanitize_user_input(input_data: Any) -> Any:
    """Sanitize user inputs to prevent injection attacks."""
    if isinstance(input_data, str):
        # Remove potentially dangerous characters
        input_data = re.sub(r'[<>"\';]', '', input_data)
        input_data = input_data.strip()
    
    return input_data
```

### API Security
```python
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """Rate limiting for external API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.call_times = defaultdict(list)
    
    async def acquire(self, api_name: str) -> None:
        """Acquire permission to make an API call."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old calls
        self.call_times[api_name] = [
            call_time for call_time in self.call_times[api_name]
            if call_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.call_times[api_name]) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[api_name][0]).total_seconds()
            await asyncio.sleep(max(0, sleep_time))
        
        self.call_times[api_name].append(now)
```

## Performance Guidelines

### Async/Await Best Practices
```python
import asyncio
from typing import List, Dict

# Good: Concurrent execution
async def fetch_multiple_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols concurrently."""
    tasks = [
        self.market_data_api.get_historical_data(symbol, start_date, end_date)
        for symbol in symbols
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        symbol: result for symbol, result in zip(symbols, results)
        if not isinstance(result, Exception)
    }

# Bad: Sequential execution
async def fetch_multiple_symbols_slow(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """DON'T: Sequential fetching is slow."""
    results = {}
    for symbol in symbols:
        results[symbol] = await self.market_data_api.get_historical_data(
            symbol, start_date, end_date
        )
    return results
```

### Memory Management
```python
import gc
from functools import lru_cache

# Use caching for expensive calculations
@lru_cache(maxsize=128)
def calculate_technical_indicator(prices_hash: int, period: int) -> pd.Series:
    """Cache expensive technical indicator calculations."""
    # Implementation
    pass

# Clean up large objects
def process_large_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Process large datasets with memory management."""
    try:
        # Process in chunks to manage memory
        chunk_size = 10000
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size].copy()
            processed_chunk = self._process_chunk(chunk)
            results.append(processed_chunk)
            
            # Clean up intermediate objects
            del chunk
        
        return pd.concat(results, ignore_index=True)
    
    finally:
        # Force garbage collection for large objects
        gc.collect()
```

### Database Optimization
```python
from sqlalchemy import text
import asyncpg

# Good: Use batch operations
async def insert_price_data_batch(price_data: List[Dict]) -> None:
    """Insert price data in batches for better performance."""
    batch_size = 1000
    
    for i in range(0, len(price_data), batch_size):
        batch = price_data[i:i+batch_size]
        
        query = """
        INSERT INTO price_data (symbol, date, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        await self.db.executemany(query, [
            (item['symbol'], item['date'], item['open'], 
             item['high'], item['low'], item['close'], item['volume'])
            for item in batch
        ])

# Use connection pooling
class DatabaseManager:
    """Manage database connections efficiently."""
    
    def __init__(self, database_url: str, pool_size: int = 10):
        self.database_url = database_url
        self.pool_size = pool_size
        self.pool = None
    
    async def initialize_pool(self):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=self.pool_size,
            command_timeout=60
        )
```

## Git & Version Control

### Commit Message Format
```
type(scope): short description

Longer description if needed.

Closes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Build/maintenance tasks

Examples:
```
feat(momentum): add RSI-based momentum signals

Implement relative strength index calculation for momentum
strategy with configurable periods and thresholds.

Closes #45

fix(data): handle missing price data gracefully

Add validation and interpolation for gaps in price series
to prevent downstream calculation errors.

Closes #67
```

### Branch Strategy
```
main                    # Production-ready code
├── develop            # Integration branch
├── feature/agent-ml   # Feature branches
├── feature/options-greeks
├── hotfix/data-validation  # Critical fixes
└── release/v1.1.0     # Release preparation
```

## Code Review Guidelines

### Pre-Review Checklist
- [ ] Code follows style guidelines (Black, isort, flake8)
- [ ] All functions have type hints and docstrings
- [ ] Tests added for new functionality
- [ ] No secrets or sensitive data in code
- [ ] Error handling implemented
- [ ] Performance considered for data-heavy operations
- [ ] Financial calculations verified with test cases

### Review Focus Areas

#### 1. Financial Accuracy
```python
# Reviewer should verify:
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    # Is the risk-free rate properly annualized?
    # Are returns in the correct format (decimal vs percentage)?
    # Is the calculation period (252 days) correct?
    pass
```

#### 2. Data Quality
```python
# Check for proper validation:
def process_price_data(prices):
    # Are outliers handled?
    # Is missing data addressed?
    # Are data types correct?
    # Is the index properly formatted?
    pass
```

#### 3. Risk Management
```python
# Ensure risk controls:
def generate_portfolio_weights(signals):
    # Are position sizes reasonable?
    # Is total exposure controlled?
    # Are correlations considered?
    pass
```

## Financial Domain Specifics

### Precision & Rounding
```python
from decimal import Decimal, ROUND_HALF_UP

# Use Decimal for financial calculations
def calculate_pnl(quantity: Decimal, entry_price: Decimal, exit_price: Decimal) -> Decimal:
    """Calculate P&L with proper decimal precision."""
    price_diff = exit_price - entry_price
    pnl = quantity * price_diff
    
    # Round to 2 decimal places for currency
    return pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

# Handle different asset classes properly
def normalize_price(price: float, asset_class: str) -> float:
    """Normalize price based on asset class conventions."""
    if asset_class == "bond":
        return round(price, 4)  # Bonds to 4 decimal places
    elif asset_class == "fx":
        return round(price, 5)  # FX to 5 decimal places
    else:
        return round(price, 2)  # Equities to 2 decimal places
```

### Time Handling
```python
from datetime import datetime, timezone
import pandas as pd

# Always use timezone-aware datetimes for financial data
def get_market_close_time(date: datetime, exchange: str = "NYSE") -> datetime:
    """Get market close time for a specific date and exchange."""
    market_tz = {
        "NYSE": "America/New_York",
        "LSE": "Europe/London",
        "TSE": "Asia/Tokyo"
    }
    
    tz = timezone(market_tz[exchange])
    close_time = date.replace(hour=16, minute=0, second=0, microsecond=0)
    return close_time.replace(tzinfo=tz)

# Handle market holidays properly
def is_trading_day(date: datetime, exchange: str = "NYSE") -> bool:
    """Check if a date is a trading day for the exchange."""
    # Use proper holiday calendars
    from pandas.tseries.holiday import USFederalHolidayCalendar
    
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=date, end=date)
    
    return date.weekday() < 5 and date not in holidays
```

### Validation Rules
```python
def validate_financial_data(data: pd.DataFrame, data_type: str) -> None:
    """Validate financial data based on type."""
    
    if data_type == "prices":
        # Price validation rules
        assert (data['high'] >= data['low']).all(), "High must be >= Low"
        assert (data['high'] >= data['open']).all(), "High must be >= Open"
        assert (data['high'] >= data['close']).all(), "High must be >= Close"
        assert (data['low'] <= data['open']).all(), "Low must be <= Open"
        assert (data['low'] <= data['close']).all(), "Low must be <= Close"
        assert (data['volume'] >= 0).all(), "Volume must be non-negative"
    
    elif data_type == "returns":
        # Return validation rules
        assert data.abs().max() < 1.0, "Daily returns should not exceed 100%"
        assert not data.isnull().all(), "Cannot have all null returns"
        
    elif data_type == "weights":
        # Portfolio weight validation
        assert abs(data.sum() - 1.0) < 0.01, "Weights must sum to approximately 1"
        assert (data >= -1).all() and (data <= 1).all(), "Weights must be between -1 and 1"
```

## Implementation Standards Summary

1. **Every agent** must inherit from `BaseAgent` and follow the standard pattern
2. **All functions** must have type hints and comprehensive docstrings
3. **Financial calculations** must use appropriate precision and validation
4. **Error handling** must be comprehensive with proper exception hierarchy
5. **Testing** must cover unit, integration, and edge cases
6. **Logging** must provide audit trails for compliance
7. **Performance** must be considered for large-scale financial data
8. **Security** must protect sensitive financial information

These standards ensure the trading system is robust, maintainable, auditable, and suitable for production financial environments.