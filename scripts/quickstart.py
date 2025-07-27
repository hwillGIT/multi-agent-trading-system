#!/usr/bin/env python3
"""
Quick start script for the Multi-Agent Trading System.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading_system directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import TradingSystem
from core.utils.logging_setup import setup_logging


async def quickstart_demo():
    """
    Run a quick demonstration of the trading system.
    """
    print("=" * 80)
    print("MULTI-AGENT TRADING SYSTEM - QUICK START DEMO")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    
    # Check environment
    print("\n1. Checking Environment...")
    env_checks = {
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "DATABASE_URL": os.getenv("DATABASE_URL", "Not configured"),
        "REDIS_URL": os.getenv("REDIS_URL", "Not configured")
    }
    
    for key, value in env_checks.items():
        status = "✓" if value and value != "Not configured" else "✗"
        print(f"  {status} {key}: {'Configured' if value and value != 'Not configured' else 'Not configured'}")
    
    # Initialize system
    print("\n2. Initializing Trading System...")
    try:
        system = TradingSystem()
        print("  ✓ Trading system initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize trading system: {e}")
        print("\nTip: Make sure you have:")
        print("  - Created .env file with API keys")
        print("  - Installed all dependencies (pip install -r requirements.txt)")
        return
    
    # Define analysis parameters
    print("\n3. Setting up Analysis Parameters...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30 days of data
    
    analysis_params = {
        "start_date": start_date,
        "end_date": end_date,
        "asset_classes": ["equities"],
        "exchanges": ["NYSE", "NASDAQ"],
        "custom_symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    }
    
    print(f"  ✓ Analysis period: {start_date.date()} to {end_date.date()}")
    print(f"  ✓ Asset classes: {analysis_params['asset_classes']}")
    print(f"  ✓ Custom symbols: {analysis_params['custom_symbols']}")
    
    # Run analysis
    print("\n4. Running Trading System Analysis...")
    print("  This may take a few minutes depending on your internet connection...")
    
    try:
        results = await system.run_analysis(**analysis_params)
        print("  ✓ Analysis completed successfully!")
        
        # Display results
        print("\n5. Analysis Results:")
        print("-" * 50)
        
        if 'data_universe' in results:
            universe_data = results['data_universe'].data
            metadata = universe_data['metadata']
            quality_report = universe_data['data_quality_report']
            
            print(f"Universe Size: {metadata['symbols_count']} symbols")
            print(f"Data Points: {metadata['data_points']:,}")
            print(f"Date Range: {metadata['start_date'].date()} to {metadata['end_date'].date()}")
            print(f"Data Quality: {quality_report['symbols_with_data']}/{quality_report['total_symbols']} symbols have data")
            print(f"Missing Data: {quality_report['missing_data_percentage']:.2f}%")
            
            # Show feature matrix info
            feature_matrix = universe_data['feature_matrix']
            if not feature_matrix.empty:
                print(f"Feature Matrix Shape: {feature_matrix.shape}")
                print(f"Available Columns: {list(feature_matrix.columns)[:10]}...")  # Show first 10 columns
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Total Execution Time: {summary['total_execution_time_ms']:.0f}ms")
            print(f"Successful Agents: {summary['successful_agents']}")
        
        # Show next steps
        print("\n6. Next Steps:")
        print("-" * 50)
        print("  📊 View detailed results in Jupyter notebook")
        print("  📈 Explore feature matrix and data quality")
        print("  🔧 Configure strategies in config/config.yaml")
        print("  🚀 Run full analysis with more agents")
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Verify API keys are valid")
        print("  - Ensure sufficient API rate limits")
        print("  - Check logs for detailed error information")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'loguru', 'pydantic', 
        'aiohttp', 'asyncio', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {missing_packages}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    return True


def setup_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data/raw",
        "data/processed", 
        "data/features",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("Setting up directories...")
    setup_directories()
    
    print("Starting quickstart demo...")
    asyncio.run(quickstart_demo())