"""
Custom exceptions for the trading system.
"""

class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    pass

class DataError(TradingSystemError):
    """Raised when there are issues with data processing or retrieval."""
    pass

class ModelError(TradingSystemError):
    """Raised when there are issues with ML models or predictions."""
    pass

class RiskError(TradingSystemError):
    """Raised when risk limits are exceeded or risk calculations fail."""
    pass

class ConfigError(TradingSystemError):
    """Raised when there are configuration issues."""
    pass

class APIError(TradingSystemError):
    """Raised when external API calls fail."""
    pass

class ValidationError(TradingSystemError):
    """Raised when data validation fails."""
    pass