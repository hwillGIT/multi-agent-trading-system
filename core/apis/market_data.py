"""
Market data API interface for multiple providers.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
from loguru import logger

from ..base.config import config
from ..base.exceptions import APIError, DataError


class BaseMarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str) -> pd.DataFrame:
        """Get historical OHLCV data."""
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote data."""
        pass
    
    @abstractmethod
    async def get_intraday_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get intraday data."""
        pass


class YahooFinanceProvider(BaseMarketDataProvider):
    """Yahoo Finance data provider."""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.rate_limit = config.get("market_data.yahoo_finance.rate_limit", 2000)
        self.logger = logger.bind(provider="yahoo_finance")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise DataError(f"No data found for symbol {symbol}")
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            data.index.name = 'timestamp'
            
            self.logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            raise APIError(f"Yahoo Finance API error for {symbol}: {str(e)}")
    
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'timestamp': datetime.utcnow()
            }
            
            return quote
            
        except Exception as e:
            raise APIError(f"Yahoo Finance quote error for {symbol}: {str(e)}")
    
    async def get_intraday_data(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Get intraday data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval=interval)
            
            if data.empty:
                raise DataError(f"No intraday data found for symbol {symbol}")
            
            data.columns = data.columns.str.lower()
            data.index.name = 'timestamp'
            
            return data
            
        except Exception as e:
            raise APIError(f"Yahoo Finance intraday error for {symbol}: {str(e)}")


class AlphaVantageProvider(BaseMarketDataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self):
        self.name = "Alpha Vantage"
        self.api_key = config.alpha_vantage_api_key
        self.base_url = config.get("market_data.alpha_vantage.base_url")
        self.rate_limit = config.get("market_data.alpha_vantage.rate_limit", 5)
        self.logger = logger.bind(provider="alpha_vantage")
        
        if not self.api_key:
            raise APIError("Alpha Vantage API key not configured")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "daily") -> pd.DataFrame:
        """Get historical data from Alpha Vantage."""
        try:
            # Map interval to Alpha Vantage function
            function_map = {
                "daily": "TIME_SERIES_DAILY",
                "weekly": "TIME_SERIES_WEEKLY",
                "monthly": "TIME_SERIES_MONTHLY"
            }
            
            function = function_map.get(interval, "TIME_SERIES_DAILY")
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                raise DataError(f"No time series data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            self.logger.info(f"Retrieved {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            raise APIError(f"Alpha Vantage API error for {symbol}: {str(e)}")
    
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'Global Quote' not in data:
                raise DataError(f"No quote data found for {symbol}")
            
            quote_data = data['Global Quote']
            
            quote = {
                'symbol': symbol,
                'price': float(quote_data.get('05. price', 0)),
                'change': float(quote_data.get('09. change', 0)),
                'change_percent': float(quote_data.get('10. change percent', '0').rstrip('%')),
                'volume': int(quote_data.get('06. volume', 0)),
                'timestamp': datetime.utcnow()
            }
            
            return quote
            
        except Exception as e:
            raise APIError(f"Alpha Vantage quote error for {symbol}: {str(e)}")
    
    async def get_intraday_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get intraday data from Alpha Vantage."""
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            # Find the time series key
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                raise DataError(f"No intraday data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            raise APIError(f"Alpha Vantage intraday error for {symbol}: {str(e)}")


class MarketDataAPI:
    """
    Unified market data API that aggregates multiple providers.
    """
    
    def __init__(self):
        self.providers = {
            'yahoo': YahooFinanceProvider(),
            'alpha_vantage': AlphaVantageProvider() if config.alpha_vantage_api_key else None
        }
        
        # Remove None providers
        self.providers = {k: v for k, v in self.providers.items() if v is not None}
        
        self.primary_provider = config.get("market_data.primary_provider", "yahoo")
        self.logger = logger.bind(service="market_data_api")
        
        if not self.providers:
            raise APIError("No market data providers configured")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "1d",
                                provider: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical data with provider fallback.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            provider: Specific provider to use
            
        Returns:
            DataFrame with OHLCV data
        """
        provider_name = provider or self.primary_provider
        
        if provider_name not in self.providers:
            raise APIError(f"Provider {provider_name} not available")
        
        try:
            return await self.providers[provider_name].get_historical_data(
                symbol, start_date, end_date, interval
            )
        except Exception as e:
            self.logger.error(f"Primary provider {provider_name} failed: {e}")
            
            # Try fallback providers
            for fallback_name, fallback_provider in self.providers.items():
                if fallback_name != provider_name:
                    try:
                        self.logger.info(f"Trying fallback provider: {fallback_name}")
                        return await fallback_provider.get_historical_data(
                            symbol, start_date, end_date, interval
                        )
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback provider {fallback_name} failed: {fallback_error}")
                        continue
            
            raise APIError(f"All providers failed for symbol {symbol}")
    
    async def get_multiple_symbols(self, symbols: List[str], start_date: datetime,
                                 end_date: datetime, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols concurrently.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        tasks = []
        for symbol in symbols:
            task = self.get_historical_data(symbol, start_date, end_date, interval)
            tasks.append((symbol, task))
        
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (symbol, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to get data for {symbol}: {result}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
            else:
                results[symbol] = result
        
        return results
    
    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time quotes for multiple symbols."""
        provider = self.providers[self.primary_provider]
        
        tasks = []
        for symbol in symbols:
            task = provider.get_real_time_quote(symbol)
            tasks.append((symbol, task))
        
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (symbol, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to get quote for {symbol}: {result}")
                results[symbol] = {}
            else:
                results[symbol] = result
        
        return results
    
    def get_supported_symbols(self, exchange: str = "NYSE") -> List[str]:
        """
        Get list of supported symbols for an exchange.
        Note: This is a simplified implementation. 
        In production, you'd fetch this from the exchange or a reference data provider.
        """
        # Sample symbols - in production, fetch from exchange listings
        sample_symbols = {
            "NYSE": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"],
            "NASDAQ": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM"],
        }
        
        return sample_symbols.get(exchange.upper(), [])