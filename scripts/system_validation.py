#!/usr/bin/env python3
"""
System validation script to test all implemented agents.
"""

import asyncio
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading_system directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.logging_setup import setup_logging
from agents.data_universe import DataUniverseAgent
from agents.feature_engineering import TechnicalAnalysisAgent
from agents.ml_ensemble import MLEnsembleAgent
from agents.strategies.momentum import MomentumAgent


async def test_data_universe_agent():
    """Test the Data Universe Agent."""
    print("\n" + "="*60)
    print("TESTING DATA UNIVERSE AGENT")
    print("="*60)
    
    agent = DataUniverseAgent()
    
    # Test inputs
    inputs = {
        "start_date": datetime.now() - timedelta(days=30),
        "end_date": datetime.now(),
        "asset_classes": ["equities"],
        "exchanges": ["NYSE"],
        "custom_symbols": ["AAPL", "GOOGL", "MSFT"]
    }
    
    try:
        result = await agent.safe_execute(inputs)
        
        if result.success:
            print("✅ Data Universe Agent: SUCCESS")
            print(f"   - Symbols processed: {result.metadata.get('universe_size', 0)}")
            print(f"   - Data points: {result.data.get('metadata', {}).get('data_points', 0)}")
            print(f"   - Execution time: {result.execution_time_ms:.0f}ms")
            return result
        else:
            print(f"❌ Data Universe Agent: FAILED - {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Data Universe Agent: ERROR - {str(e)}")
        traceback.print_exc()
        return None


async def test_technical_analysis_agent(universe_result):
    """Test the Technical Analysis Agent."""
    print("\n" + "="*60)
    print("TESTING TECHNICAL ANALYSIS AGENT")
    print("="*60)
    
    if not universe_result:
        print("❌ Skipping Technical Analysis - No universe data")
        return None
    
    agent = TechnicalAnalysisAgent()
    
    inputs = {
        "feature_matrix": universe_result.data["feature_matrix"],
        "symbols": list(universe_result.data["universe"])[:3],  # Test with 3 symbols
        "add_patterns": True,
        "add_microstructure": False
    }
    
    try:
        result = await agent.safe_execute(inputs)
        
        if result.success:
            print("✅ Technical Analysis Agent: SUCCESS")
            print(f"   - Features added: {result.data.get('feature_summary', {}).get('total_features_added', 0)}")
            print(f"   - Symbols processed: {result.metadata.get('symbols_processed', 0)}")
            print(f"   - Execution time: {result.execution_time_ms:.0f}ms")
            return result
        else:
            print(f"❌ Technical Analysis Agent: FAILED - {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Technical Analysis Agent: ERROR - {str(e)}")
        traceback.print_exc()
        return None


async def test_ml_ensemble_agent(technical_result):
    """Test the ML Ensemble Agent."""
    print("\n" + "="*60)
    print("TESTING ML ENSEMBLE AGENT")
    print("="*60)
    
    if not technical_result:
        print("❌ Skipping ML Ensemble - No technical analysis data")
        return None
    
    agent = MLEnsembleAgent()
    
    inputs = {
        "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
        "target_variable": "forward_return_1d",
        "train_models": True,
        "prediction_mode": "regression"
    }
    
    try:
        result = await agent.safe_execute(inputs)
        
        if result.success:
            print("✅ ML Ensemble Agent: SUCCESS")
            print(f"   - Models trained: {result.metadata.get('models_trained', 0)}")
            print(f"   - Symbols processed: {result.metadata.get('symbols_processed', 0)}")
            print(f"   - Execution time: {result.execution_time_ms:.0f}ms")
            
            # Show model performance if available
            performance = result.data.get('model_performance', {})
            if performance:
                print(f"   - Model R²: {performance.get('test_r2', 'N/A')}")
                print(f"   - Directional Accuracy: {performance.get('directional_accuracy', 'N/A')}")
            
            return result
        else:
            print(f"❌ ML Ensemble Agent: FAILED - {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ ML Ensemble Agent: ERROR - {str(e)}")
        traceback.print_exc()
        return None


async def test_momentum_strategy_agent(technical_result, ml_result):
    """Test the Momentum Strategy Agent."""
    print("\n" + "="*60)
    print("TESTING MOMENTUM STRATEGY AGENT")
    print("="*60)
    
    if not technical_result:
        print("❌ Skipping Momentum Strategy - No technical analysis data")
        return None
    
    agent = MomentumAgent()
    
    inputs = {
        "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
        "ml_predictions": ml_result.data if ml_result else {},
        "symbols": technical_result.data.get("enhanced_feature_matrix", {}).get("symbol", []).unique()[:3] if hasattr(technical_result.data.get("enhanced_feature_matrix", {}), "get") else [],
        "current_positions": {}
    }
    
    try:
        result = await agent.safe_execute(inputs)
        
        if result.success:
            print("✅ Momentum Strategy Agent: SUCCESS")
            print(f"   - Signals generated: {result.metadata.get('signals_generated', 0)}")
            print(f"   - Recommendations: {result.metadata.get('recommendations_count', 0)}")
            print(f"   - Execution time: {result.execution_time_ms:.0f}ms")
            
            # Show sample recommendations
            recommendations = result.data.get('recommendations', [])
            if recommendations:
                print(f"   - Sample recommendation: {recommendations[0].get('symbol', 'N/A')} - {recommendations[0].get('action', 'N/A')}")
            
            return result
        else:
            print(f"❌ Momentum Strategy Agent: FAILED - {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Momentum Strategy Agent: ERROR - {str(e)}")
        traceback.print_exc()
        return None


def print_system_summary(results):
    """Print overall system validation summary."""
    print("\n" + "="*80)
    print("SYSTEM VALIDATION SUMMARY")
    print("="*80)
    
    total_agents = 4
    successful_agents = len([r for r in results.values() if r is not None])
    
    print(f"Agents Tested: {total_agents}")
    print(f"Successful: {successful_agents}")
    print(f"Failed: {total_agents - successful_agents}")
    print(f"Success Rate: {(successful_agents/total_agents)*100:.1f}%")
    
    if successful_agents == total_agents:
        print("\n🎉 ALL AGENTS WORKING CORRECTLY!")
        print("   The trading system core is fully functional.")
    elif successful_agents >= 2:
        print("\n⚠️  PARTIAL SUCCESS")
        print("   Core functionality is working, some agents may need attention.")
    else:
        print("\n❌ SYSTEM NEEDS ATTENTION")
        print("   Multiple agents are failing, check configuration and dependencies.")
    
    # Total execution time
    total_time = sum([
        r.execution_time_ms for r in results.values() 
        if r is not None and hasattr(r, 'execution_time_ms') and r.execution_time_ms
    ])
    print(f"\nTotal Execution Time: {total_time:.0f}ms")
    
    print("\n" + "="*80)


async def run_system_validation():
    """Run complete system validation."""
    print("MULTI-AGENT TRADING SYSTEM - VALIDATION SUITE")
    print("="*80)
    print("Testing all implemented agents...")
    
    # Setup logging
    setup_logging()
    
    results = {}
    
    # Test each agent in sequence
    results['data_universe'] = await test_data_universe_agent()
    results['technical_analysis'] = await test_technical_analysis_agent(results['data_universe'])
    results['ml_ensemble'] = await test_ml_ensemble_agent(results['technical_analysis'])
    results['momentum_strategy'] = await test_momentum_strategy_agent(
        results['technical_analysis'], 
        results['ml_ensemble']
    )
    
    # Print summary
    print_system_summary(results)
    
    return results


if __name__ == "__main__":
    print("Starting system validation...")
    print("This may take a few minutes depending on your internet connection and API keys.")
    print("-" * 80)
    
    try:
        results = asyncio.run(run_system_validation())
        
        # Save results if successful
        if any(r is not None for r in results.values()):
            print(f"\nValidation completed at {datetime.now()}")
            print("Check logs/ directory for detailed execution logs.")
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
    except Exception as e:
        print(f"\n\nValidation failed with error: {e}")
        traceback.print_exc()