"""
Tests for the Data Universe Agent.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from agents.data_universe import DataUniverseAgent
from core.base.exceptions import DataError


@pytest.fixture
def data_universe_agent():
    """Create a DataUniverseAgent instance for testing."""
    return DataUniverseAgent()


@pytest.fixture
def sample_inputs():
    """Sample inputs for testing."""
    return {
        "start_date": datetime.now() - timedelta(days=30),
        "end_date": datetime.now(),
        "asset_classes": ["equities"],
        "exchanges": ["NYSE"],
        "custom_symbols": ["AAPL", "GOOGL"]
    }


class TestDataUniverseAgent:
    """Test cases for DataUniverseAgent."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, data_universe_agent):
        """Test agent initialization."""
        assert data_universe_agent.name == "DataUniverseAgent"
        assert data_universe_agent.config_section == "universe"
        assert hasattr(data_universe_agent, 'market_data_api')
    
    @pytest.mark.asyncio
    async def test_define_universe(self, data_universe_agent):
        """Test universe definition."""
        universe = await data_universe_agent._define_universe(
            asset_classes=["equities"],
            exchanges=["NYSE"],
            custom_symbols=["AAPL", "GOOGL"]
        )
        
        assert isinstance(universe, list)
        assert len(universe) > 0
        assert "AAPL" in universe
        assert "GOOGL" in universe
    
    @pytest.mark.asyncio
    async def test_get_equity_universe(self, data_universe_agent):
        """Test equity universe retrieval."""
        nyse_symbols = data_universe_agent._get_equity_universe("NYSE")
        nasdaq_symbols = data_universe_agent._get_equity_universe("NASDAQ")
        
        assert isinstance(nyse_symbols, list)
        assert isinstance(nasdaq_symbols, list)
        assert len(nyse_symbols) > 0
        assert len(nasdaq_symbols) > 0
        assert "AAPL" in nyse_symbols
    
    def test_get_etf_universe(self, data_universe_agent):
        """Test ETF universe retrieval."""
        etf_symbols = data_universe_agent._get_etf_universe()
        
        assert isinstance(etf_symbols, list)
        assert len(etf_symbols) > 0
        assert "SPY" in etf_symbols
        assert "QQQ" in etf_symbols
    
    def test_get_crypto_universe(self, data_universe_agent):
        """Test crypto universe retrieval."""
        crypto_symbols = data_universe_agent._get_crypto_universe()
        
        assert isinstance(crypto_symbols, list)
        assert len(crypto_symbols) > 0
        assert "BTC-USD" in crypto_symbols
        assert "ETH-USD" in crypto_symbols
    
    def test_clean_price_data(self, data_universe_agent):
        """Test price data cleaning."""
        # Create sample data with issues
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        sample_data = pd.DataFrame({
            "open": [100, 101, 102, 1000, 104, None, 106, 107, 108, 109],  # Outlier and missing
            "high": [102, 103, 104, 1005, 106, None, 108, 109, 110, 111],
            "low": [99, 100, 101, 999, 103, None, 105, 106, 107, 108],
            "close": [101, 102, 103, 1002, 105, None, 107, 108, 109, 110],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
        
        cleaned_data = data_universe_agent._clean_price_data(sample_data)
        
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) <= len(sample_data)  # Some rows may be removed
        assert 'close' in cleaned_data.columns
    
    def test_harmonize_timestamps(self, data_universe_agent):
        """Test timestamp harmonization."""
        # Create sample data with datetime index
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        sample_data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104]
        }, index=dates)
        
        harmonized_data = data_universe_agent._harmonize_timestamps(sample_data, "1d")
        
        assert isinstance(harmonized_data.index, pd.DatetimeIndex)
        assert len(harmonized_data) == len(sample_data)
    
    def test_get_asset_metadata_map(self, data_universe_agent):
        """Test asset metadata retrieval."""
        symbols = ["AAPL", "GOOGL", "JPM", "SPY"]
        metadata_map = data_universe_agent._get_asset_metadata_map(symbols)
        
        assert isinstance(metadata_map, dict)
        assert len(metadata_map) == len(symbols)
        
        for symbol in symbols:
            assert symbol in metadata_map
            metadata = metadata_map[symbol]
            assert "sector" in metadata
            assert "asset_class" in metadata
            assert "geography" in metadata
    
    def test_infer_sector(self, data_universe_agent):
        """Test sector inference."""
        assert data_universe_agent._infer_sector("AAPL") == "Technology"
        assert data_universe_agent._infer_sector("JPM") == "Financials"
        assert data_universe_agent._infer_sector("JNJ") == "Healthcare"
        assert data_universe_agent._infer_sector("UNKNOWN") == "Other"
    
    def test_infer_asset_class(self, data_universe_agent):
        """Test asset class inference."""
        assert data_universe_agent._infer_asset_class("AAPL") == "Equity"
        assert data_universe_agent._infer_asset_class("SPY") == "ETF"
        assert data_universe_agent._infer_asset_class("BTC-USD") == "Cryptocurrency"
    
    @pytest.mark.asyncio
    async def test_execute_missing_inputs(self, data_universe_agent):
        """Test execute with missing required inputs."""
        incomplete_inputs = {
            "start_date": datetime.now() - timedelta(days=30)
            # Missing end_date
        }
        
        result = await data_universe_agent.safe_execute(incomplete_inputs)
        assert not result.success
        assert "Missing required inputs" in result.error_message
    
    @pytest.mark.asyncio
    async def test_data_quality_report_generation(self, data_universe_agent):
        """Test data quality report generation."""
        # Create sample data structure
        sample_data = {
            "AAPL": {
                "1d": pd.DataFrame({
                    "close": [150, 151, 152],
                    "volume": [1000000, 1100000, 1200000]
                })
            },
            "GOOGL": {
                "1d": pd.DataFrame()  # Empty DataFrame
            }
        }
        
        report = data_universe_agent._generate_data_quality_report(sample_data)
        
        assert isinstance(report, dict)
        assert "total_symbols" in report
        assert "symbols_with_data" in report
        assert "missing_data_percentage" in report
        assert report["total_symbols"] == 2
        assert report["symbols_with_data"] == 1  # Only AAPL has data


@pytest.mark.integration
class TestDataUniverseIntegration:
    """Integration tests for DataUniverseAgent."""
    
    @pytest.mark.asyncio
    async def test_full_execution_flow(self, data_universe_agent, sample_inputs):
        """Test the complete execution flow."""
        # This test requires actual market data APIs to be available
        # Skip if no API keys are configured
        
        result = await data_universe_agent.safe_execute(sample_inputs)
        
        # The result should either succeed or fail gracefully
        assert isinstance(result.success, bool)
        assert hasattr(result, 'agent_name')
        assert result.agent_name == "DataUniverseAgent"
        
        if result.success:
            assert 'feature_matrix' in result.data
            assert 'universe' in result.data
            assert 'data_quality_report' in result.data
        else:
            # If it fails, it should have an error message
            assert result.error_message is not None


if __name__ == "__main__":
    pytest.main([__file__])