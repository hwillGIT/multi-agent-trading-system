"""
Data Universe Agent - Defines universe, fetches and cleans multi-frequency data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from ...core.base.agent import BaseAgent, AgentOutput
from ...core.base.exceptions import DataError, ValidationError
from ...core.apis.market_data import MarketDataAPI
from ...core.utils.data_validation import DataValidator
from ...core.utils.time_utils import TimeUtils


class DataUniverseAgent(BaseAgent):
    """
    Agent responsible for defining the investment universe and fetching/cleaning data.
    
    This agent:
    1. Defines the universe of assets to analyze
    2. Fetches multi-frequency market data
    3. Cleans and harmonizes timestamps
    4. Calculates forward returns for supervised learning
    5. Maps asset metadata (sector, style, factors)
    """
    
    def __init__(self):
        super().__init__("DataUniverseAgent", "universe")
        self.market_data_api = MarketDataAPI()
        self.universe_config = self.get_config_value("", {})
        
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute the data universe agent.
        
        Args:
            inputs: Dictionary containing:
                - start_date: Start date for data collection
                - end_date: End date for data collection
                - asset_classes: List of asset classes to include
                - exchanges: List of exchanges to include
                - custom_symbols: Optional list of specific symbols
                
        Returns:
            AgentOutput with cleaned, joined, timestamp-aligned multi-asset feature matrix
        """
        self.validate_inputs(inputs, ["start_date", "end_date"])
        
        start_date = inputs["start_date"]
        end_date = inputs["end_date"]
        asset_classes = inputs.get("asset_classes", ["equities"])
        exchanges = inputs.get("exchanges", ["NYSE", "NASDAQ"])
        custom_symbols = inputs.get("custom_symbols", [])
        
        try:
            # Step 1: Define universe
            universe = await self._define_universe(asset_classes, exchanges, custom_symbols)
            self.logger.info(f"Defined universe with {len(universe)} symbols")
            
            # Step 2: Fetch multi-frequency data
            raw_data = await self._fetch_multi_frequency_data(universe, start_date, end_date)
            self.logger.info(f"Fetched data for {len(raw_data)} symbols")
            
            # Step 3: Clean and harmonize data
            cleaned_data = await self._clean_and_harmonize_data(raw_data)
            self.logger.info("Data cleaning and harmonization completed")
            
            # Step 4: Calculate forward returns
            data_with_returns = await self._calculate_forward_returns(cleaned_data)
            self.logger.info("Forward returns calculated")
            
            # Step 5: Add asset metadata
            final_data = await self._add_asset_metadata(data_with_returns, universe)
            self.logger.info("Asset metadata added")
            
            # Step 6: Create joined feature matrix
            feature_matrix = await self._create_feature_matrix(final_data)
            self.logger.info(f"Feature matrix created with shape {feature_matrix.shape}")
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "feature_matrix": feature_matrix,
                    "universe": universe,
                    "data_quality_report": self._generate_data_quality_report(final_data),
                    "metadata": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "symbols_count": len(universe),
                        "data_points": len(feature_matrix)
                    }
                },
                metadata={
                    "universe_size": len(universe),
                    "date_range": f"{start_date} to {end_date}",
                    "asset_classes": asset_classes
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in data universe agent: {str(e)}")
            raise DataError(f"Data universe processing failed: {str(e)}")
    
    async def _define_universe(self, asset_classes: List[str], exchanges: List[str], 
                             custom_symbols: List[str]) -> List[str]:
        """
        Define the investment universe based on criteria.
        
        Args:
            asset_classes: Asset classes to include
            exchanges: Exchanges to include
            custom_symbols: Custom symbols to include
            
        Returns:
            List of symbols in the universe
        """
        universe = set()
        
        # Add custom symbols if provided
        if custom_symbols:
            universe.update(custom_symbols)
        
        # Add symbols based on asset classes and exchanges
        for asset_class in asset_classes:
            if asset_class == "equities":
                for exchange in exchanges:
                    symbols = self._get_equity_universe(exchange)
                    universe.update(symbols)
            elif asset_class == "etfs":
                etf_symbols = self._get_etf_universe()
                universe.update(etf_symbols)
            elif asset_class == "crypto":
                crypto_symbols = self._get_crypto_universe()
                universe.update(crypto_symbols)
        
        # Apply universe filters
        filtered_universe = await self._apply_universe_filters(list(universe))
        
        return filtered_universe
    
    def _get_equity_universe(self, exchange: str) -> List[str]:
        """Get equity symbols for a specific exchange."""
        # In production, this would fetch from a reference data provider
        equity_symbols = {
            "NYSE": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
                "WMT", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "NFLX", "CRM",
                "XOM", "VZ", "KO", "PFE", "NKE", "MRK", "ABT", "TMO", "COST", "AVGO"
            ],
            "NASDAQ": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "PYPL", "ADBE", "NFLX",
                "CRM", "INTC", "CSCO", "PEP", "CMCSA", "TXN", "QCOM", "AMGN", "HON", "SBUX",
                "GILD", "MDLZ", "ADP", "INTU", "ISRG", "BKNG", "MU", "LRCX", "REGN", "ATVI"
            ]
        }
        
        return equity_symbols.get(exchange.upper(), [])
    
    def _get_etf_universe(self) -> List[str]:
        """Get ETF symbols."""
        return [
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "LQD",
            "HYG", "EMB", "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI"
        ]
    
    def _get_crypto_universe(self) -> List[str]:
        """Get cryptocurrency symbols."""
        return [
            "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOT-USD", 
            "AVAX-USD", "LINK-USD", "MATIC-USD", "UNI-USD", "LTC-USD"
        ]
    
    async def _apply_universe_filters(self, symbols: List[str]) -> List[str]:
        """
        Apply filters to the universe (market cap, volume, etc.).
        
        Args:
            symbols: List of symbols to filter
            
        Returns:
            Filtered list of symbols
        """
        # Get filter criteria from config
        min_market_cap = self.get_config_value("equities.market_cap_min", 1e9)
        min_volume = self.get_config_value("equities.volume_min", 1e6)
        allowed_sectors = self.get_config_value("equities.sectors", [])
        
        filtered_symbols = []
        
        # In production, you would fetch actual market cap and volume data
        # For now, we'll use a simplified filter
        for symbol in symbols:
            # Simple filter: exclude penny stocks and low-volume stocks
            if not symbol.endswith("-USD"):  # Not crypto
                # Add basic filters here
                pass
            
            filtered_symbols.append(symbol)
        
        return filtered_symbols[:50]  # Limit for demo purposes
    
    async def _fetch_multi_frequency_data(self, universe: List[str], 
                                        start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data at multiple frequencies for all symbols.
        
        Args:
            universe: List of symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with symbol -> frequency -> DataFrame mapping
        """
        frequencies = ["1d", "1wk", "1mo"]  # Daily, Weekly, Monthly
        all_data = {}
        
        for frequency in frequencies:
            self.logger.info(f"Fetching {frequency} data for {len(universe)} symbols")
            
            # Fetch data for all symbols at this frequency
            frequency_data = await self.market_data_api.get_multiple_symbols(
                universe, start_date, end_date, frequency
            )
            
            for symbol, data in frequency_data.items():
                if symbol not in all_data:
                    all_data[symbol] = {}
                all_data[symbol][frequency] = data
        
        return all_data
    
    async def _clean_and_harmonize_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Clean and harmonize timestamp data across all symbols and frequencies.
        
        Args:
            raw_data: Raw data dictionary
            
        Returns:
            Cleaned data dictionary
        """
        cleaned_data = {}
        
        for symbol, frequency_data in raw_data.items():
            cleaned_data[symbol] = {}
            
            for frequency, df in frequency_data.items():
                try:
                    # Skip empty DataFrames
                    if df.empty:
                        self.logger.warning(f"Empty data for {symbol} at {frequency}")
                        continue
                    
                    # Basic data validation
                    DataValidator.validate_price_data(df)
                    
                    # Clean the data
                    cleaned_df = self._clean_price_data(df)
                    
                    # Harmonize timestamps
                    cleaned_df = self._harmonize_timestamps(cleaned_df, frequency)
                    
                    cleaned_data[symbol][frequency] = cleaned_df
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning data for {symbol} at {frequency}: {e}")
                    continue
        
        return cleaned_data
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean individual price DataFrame.
        
        Args:
            df: Raw price DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()
        
        # Remove rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Handle outliers (prices that are too extreme)
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned_df.columns:
                # Remove prices that are more than 10x the median (likely errors)
                median_price = cleaned_df[col].median()
                outlier_mask = (cleaned_df[col] > median_price * 10) | (cleaned_df[col] < median_price * 0.1)
                cleaned_df.loc[outlier_mask, col] = np.nan
        
        # Forward fill missing values (up to 5 periods)
        cleaned_df = cleaned_df.fillna(method='ffill', limit=5)
        
        # Remove remaining rows with critical missing data
        critical_columns = ['close']  # At minimum, we need close prices
        cleaned_df = cleaned_df.dropna(subset=critical_columns)
        
        # Ensure OHLC consistency after cleaning
        if all(col in cleaned_df.columns for col in ['open', 'high', 'low', 'close']):
            # Fix OHLC inconsistencies
            cleaned_df['high'] = cleaned_df[['open', 'high', 'low', 'close']].max(axis=1)
            cleaned_df['low'] = cleaned_df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return cleaned_df
    
    def _harmonize_timestamps(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Harmonize timestamps for consistent alignment.
        
        Args:
            df: DataFrame with datetime index
            frequency: Data frequency
            
        Returns:
            DataFrame with harmonized timestamps
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # For daily data, align to market close time
        if frequency == "1d":
            df.index = df.index.normalize() + pd.Timedelta(hours=16)  # 4 PM EST
        
        return df
    
    async def _calculate_forward_returns(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate forward returns for supervised learning.
        
        Args:
            data: Cleaned data dictionary
            
        Returns:
            Data with forward returns added
        """
        return_periods = [1, 5, 10, 20]  # 1-day, 1-week, 2-week, 1-month forward returns
        
        for symbol, frequency_data in data.items():
            for frequency, df in frequency_data.items():
                if df.empty or 'close' not in df.columns:
                    continue
                
                # Calculate returns for different forward periods
                for period in return_periods:
                    col_name = f"forward_return_{period}d"
                    df[col_name] = df['close'].pct_change(periods=-period).shift(-period)
                
                # Calculate log returns as well
                for period in return_periods:
                    col_name = f"forward_log_return_{period}d"
                    df[col_name] = np.log(df['close'] / df['close'].shift(-period)).shift(-period)
                
                data[symbol][frequency] = df
        
        return data
    
    async def _add_asset_metadata(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                                universe: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Add asset metadata (sector, style, factors, geography).
        
        Args:
            data: Data with returns
            universe: List of symbols
            
        Returns:
            Data with metadata added
        """
        # In production, this would fetch from a reference data provider
        metadata_map = self._get_asset_metadata_map(universe)
        
        for symbol, frequency_data in data.items():
            if symbol not in metadata_map:
                continue
                
            metadata = metadata_map[symbol]
            
            for frequency, df in frequency_data.items():
                if df.empty:
                    continue
                
                # Add metadata as columns
                for key, value in metadata.items():
                    df[f"metadata_{key}"] = value
                
                data[symbol][frequency] = df
        
        return data
    
    def _get_asset_metadata_map(self, universe: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata mapping for assets.
        
        Args:
            universe: List of symbols
            
        Returns:
            Dictionary mapping symbols to metadata
        """
        # Simplified metadata - in production, fetch from reference data
        metadata_map = {}
        
        for symbol in universe:
            metadata = {
                "sector": self._infer_sector(symbol),
                "asset_class": self._infer_asset_class(symbol),
                "geography": self._infer_geography(symbol),
                "market_cap_category": self._infer_market_cap_category(symbol),
                "style": self._infer_style(symbol)
            }
            metadata_map[symbol] = metadata
        
        return metadata_map
    
    def _infer_sector(self, symbol: str) -> str:
        """Infer sector from symbol (simplified)."""
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ADBE", "CRM", "INTC", "CSCO"]
        finance_stocks = ["JPM", "V", "MA", "GS", "MS", "BAC", "WFC", "C"]
        healthcare_stocks = ["JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "AMGN", "GILD"]
        
        if symbol in tech_stocks:
            return "Technology"
        elif symbol in finance_stocks:
            return "Financials"
        elif symbol in healthcare_stocks:
            return "Healthcare"
        else:
            return "Other"
    
    def _infer_asset_class(self, symbol: str) -> str:
        """Infer asset class from symbol."""
        if symbol.endswith("-USD"):
            return "Cryptocurrency"
        elif symbol in ["SPY", "QQQ", "IWM", "VTI", "VOO"]:
            return "ETF"
        else:
            return "Equity"
    
    def _infer_geography(self, symbol: str) -> str:
        """Infer geography from symbol."""
        return "US"  # Simplified
    
    def _infer_market_cap_category(self, symbol: str) -> str:
        """Infer market cap category."""
        large_cap = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        if symbol in large_cap:
            return "Large Cap"
        else:
            return "Mid Cap"
    
    def _infer_style(self, symbol: str) -> str:
        """Infer investment style."""
        growth_stocks = ["TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM"]
        if symbol in growth_stocks:
            return "Growth"
        else:
            return "Value"
    
    async def _create_feature_matrix(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Create a unified feature matrix from all symbols and frequencies.
        
        Args:
            data: Data with metadata and returns
            
        Returns:
            Unified feature matrix
        """
        all_features = []
        
        for symbol, frequency_data in data.items():
            for frequency, df in frequency_data.items():
                if df.empty:
                    continue
                
                # Add symbol and frequency identifiers
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy['frequency'] = frequency
                df_copy['timestamp'] = df_copy.index
                
                all_features.append(df_copy)
        
        if not all_features:
            return pd.DataFrame()
        
        # Combine all DataFrames
        feature_matrix = pd.concat(all_features, ignore_index=True, sort=False)
        
        # Sort by timestamp for time series consistency
        feature_matrix = feature_matrix.sort_values(['timestamp', 'symbol', 'frequency'])
        
        return feature_matrix
    
    def _generate_data_quality_report(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Generate a data quality report.
        
        Args:
            data: Processed data
            
        Returns:
            Data quality report
        """
        report = {
            "total_symbols": len(data),
            "symbols_with_data": 0,
            "total_data_points": 0,
            "missing_data_percentage": 0,
            "date_range_coverage": {},
            "frequency_coverage": {}
        }
        
        total_missing = 0
        total_possible = 0
        
        for symbol, frequency_data in data.items():
            has_data = False
            for frequency, df in frequency_data.items():
                if not df.empty:
                    has_data = True
                    report["total_data_points"] += len(df)
                    
                    # Count missing values
                    missing_count = df.isnull().sum().sum()
                    total_missing += missing_count
                    total_possible += df.size
                    
                    # Track frequency coverage
                    if frequency not in report["frequency_coverage"]:
                        report["frequency_coverage"][frequency] = 0
                    report["frequency_coverage"][frequency] += 1
            
            if has_data:
                report["symbols_with_data"] += 1
        
        if total_possible > 0:
            report["missing_data_percentage"] = (total_missing / total_possible) * 100
        
        return report