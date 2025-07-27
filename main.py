"""
Main entry point for the Multi-Agent Trading System.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from loguru import logger
from core.utils.logging_setup import setup_logging
from core.base.config import config
from agents.data_universe import DataUniverseAgent
from agents.feature_engineering import TechnicalAnalysisAgent
from agents.ml_ensemble import MLEnsembleAgent
from agents.strategies.momentum import MomentumAgent
from agents.strategies.stat_arb import StatisticalArbitrageAgent
from agents.strategies.event_driven import EventDrivenAgent
from agents.strategies.options import OptionsAgent
from agents.strategies.cross_asset import CrossAssetAgent
from agents.synthesis import SignalSynthesisAgent
from agents.risk_management import RiskModelingAgent
from agents.output import RecommendationAgent


class TradingSystem:
    """
    Main trading system orchestrator that coordinates all agents.
    """
    
    def __init__(self):
        self.agents = {
            'data_universe': DataUniverseAgent(),
            'technical_analysis': TechnicalAnalysisAgent(),
            'ml_ensemble': MLEnsembleAgent(),
            'momentum_strategy': MomentumAgent(),
            'stat_arb': StatisticalArbitrageAgent(),
            'event_driven': EventDrivenAgent(),
            'options_strategy': OptionsAgent(),
            'cross_asset': CrossAssetAgent(),
            'signal_synthesis': SignalSynthesisAgent(),
            'risk_modeling': RiskModelingAgent(),
            'recommendation': RecommendationAgent(),
        }
        
        setup_logging()
        self.logger = logger.bind(system="trading_system")
        
    async def run_analysis(self, 
                          start_date: datetime,
                          end_date: datetime,
                          asset_classes: list = None,
                          exchanges: list = None,
                          custom_symbols: list = None) -> Dict[str, Any]:
        """
        Run the complete trading system analysis.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            asset_classes: Asset classes to analyze
            exchanges: Exchanges to include
            custom_symbols: Custom symbols to analyze
            
        Returns:
            Complete analysis results
        """
        if asset_classes is None:
            asset_classes = ["equities", "etfs"]
        if exchanges is None:
            exchanges = ["NYSE", "NASDAQ"]
        if custom_symbols is None:
            custom_symbols = []
            
        self.logger.info(f"Starting trading system analysis from {start_date} to {end_date}")
        
        results = {}
        
        try:
            # Step 1: Data Universe & Preprocessing
            self.logger.info("Step 1: Data Universe & Preprocessing")
            universe_inputs = {
                "start_date": start_date,
                "end_date": end_date,
                "asset_classes": asset_classes,
                "exchanges": exchanges,
                "custom_symbols": custom_symbols
            }
            
            universe_result = await self.agents['data_universe'].safe_execute(universe_inputs)
            if not universe_result.success:
                raise Exception(f"Data universe failed: {universe_result.error_message}")
            
            results['data_universe'] = universe_result
            self.logger.info(f"Universe processing completed with {universe_result.data['metadata']['symbols_count']} symbols")
            
            # Step 2: Feature Engineering & Technical Analysis
            self.logger.info("Step 2: Feature Engineering & Technical Analysis")
            technical_inputs = {
                "feature_matrix": universe_result.data["feature_matrix"],
                "symbols": list(universe_result.data["universe"]),
                "add_patterns": True,
                "add_microstructure": False
            }
            
            technical_result = await self.agents['technical_analysis'].safe_execute(technical_inputs)
            if not technical_result.success:
                self.logger.warning(f"Technical analysis failed: {technical_result.error_message}")
            else:
                results['technical_analysis'] = technical_result
                self.logger.info(f"Technical analysis completed with {technical_result.data['feature_summary']['total_features_added']} features added")
            
            # Step 3: ML Model Ensemble
            if technical_result.success:
                self.logger.info("Step 3: ML Model Ensemble")
                ml_inputs = {
                    "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
                    "target_variable": "forward_return_1d",
                    "train_models": True,
                    "prediction_mode": "regression",
                    "symbols": list(universe_result.data["universe"])
                }
                
                ml_result = await self.agents['ml_ensemble'].safe_execute(ml_inputs)
                if not ml_result.success:
                    self.logger.warning(f"ML ensemble failed: {ml_result.error_message}")
                else:
                    results['ml_ensemble'] = ml_result
                    self.logger.info(f"ML ensemble completed with {ml_result.metadata['models_trained']} models trained")
            
            # Step 4A: Momentum Strategy
            if technical_result.success:
                self.logger.info("Step 4A: Momentum Strategy Analysis")
                momentum_inputs = {
                    "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
                    "ml_predictions": results.get('ml_ensemble', {}).data if 'ml_ensemble' in results else {},
                    "symbols": list(universe_result.data["universe"]),
                    "current_positions": {}
                }
                
                momentum_result = await self.agents['momentum_strategy'].safe_execute(momentum_inputs)
                if not momentum_result.success:
                    self.logger.warning(f"Momentum strategy failed: {momentum_result.error_message}")
                else:
                    results['momentum_strategy'] = momentum_result
                    self.logger.info(f"Momentum strategy completed with {momentum_result.metadata['recommendations_count']} recommendations")
            
            # Step 4B: Statistical Arbitrage Strategy
            if technical_result.success:
                self.logger.info("Step 4B: Statistical Arbitrage Analysis")
                stat_arb_inputs = {
                    "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
                    "symbols": list(universe_result.data["universe"])[:20],  # Limit for pairs analysis
                    "current_positions": {},
                    "lookback_window": 252
                }
                
                stat_arb_result = await self.agents['stat_arb'].safe_execute(stat_arb_inputs)
                if not stat_arb_result.success:
                    self.logger.warning(f"Statistical arbitrage failed: {stat_arb_result.error_message}")
                else:
                    results['stat_arb'] = stat_arb_result
                    pairs_count = stat_arb_result.metadata.get('pairs_identified', 0)
                    self.logger.info(f"Statistical arbitrage completed with {pairs_count} pairs identified")
            
            # Step 4C: Event-Driven Strategy (simplified inputs for demonstration)
            self.logger.info("Step 4C: Event-Driven Analysis")
            event_driven_inputs = {
                "symbols": list(universe_result.data["universe"])[:10],  # Limit for news analysis
                "news_data": [],  # Would be populated with real news data
                "earnings_calendar": {},  # Would be populated with earnings data
                "corporate_events": []  # Would be populated with corporate events
            }
            
            event_driven_result = await self.agents['event_driven'].safe_execute(event_driven_inputs)
            if not event_driven_result.success:
                self.logger.warning(f"Event-driven strategy failed: {event_driven_result.error_message}")
            else:
                results['event_driven'] = event_driven_result
                events_count = event_driven_result.metadata.get('earnings_events', 0) + event_driven_result.metadata.get('corporate_events', 0)
                self.logger.info(f"Event-driven analysis completed with {events_count} events analyzed")
            
            # Step 4D: Options Strategy (simplified inputs for demonstration)
            self.logger.info("Step 4D: Options Strategy Analysis")
            options_inputs = {
                "symbols": list(universe_result.data["universe"])[:5],  # Limit for options analysis
                "options_chain": [],  # Would be populated with real options data
                "market_view": {"direction": "neutral", "volatility_expectation": "stable"},
                "risk_constraints": {"max_loss_per_trade": 1000}
            }
            
            options_result = await self.agents['options_strategy'].safe_execute(options_inputs)
            if not options_result.success:
                self.logger.warning(f"Options strategy failed: {options_result.error_message}")
            else:
                results['options_strategy'] = options_result
                strategies_count = options_result.metadata.get('strategies_generated', 0)
                self.logger.info(f"Options strategy completed with {strategies_count} strategies generated")
            
            # Step 4E: Cross-Asset Analysis
            if technical_result.success:
                self.logger.info("Step 4E: Cross-Asset Analysis")
                cross_asset_inputs = {
                    "price_data": universe_result.data["feature_matrix"][list(universe_result.data["universe"])],
                    "symbols": list(universe_result.data["universe"])[:15],  # Limit for cross-asset analysis
                    "current_portfolio": {},
                    "risk_constraints": {"risk_tolerance": "moderate"}
                }
                
                cross_asset_result = await self.agents['cross_asset'].safe_execute(cross_asset_inputs)
                if not cross_asset_result.success:
                    self.logger.warning(f"Cross-asset analysis failed: {cross_asset_result.error_message}")
                else:
                    results['cross_asset'] = cross_asset_result
                    assets_analyzed = cross_asset_result.metadata.get('assets_analyzed', 0)
                    self.logger.info(f"Cross-asset analysis completed with {assets_analyzed} assets analyzed")
            
            # Step 5: Signal Synthesis & Consensus Validation
            if 'momentum_strategy' in results and results['momentum_strategy'].success:
                self.logger.info("Step 5: Signal Synthesis & Consensus Validation")
                
                # Collect all strategy signals for synthesis
                strategy_signals = {}
                if 'momentum_strategy' in results:
                    strategy_signals['momentum'] = results['momentum_strategy'].data
                
                # Add ML predictions if available
                if 'ml_ensemble' in results:
                    strategy_signals['ml_ensemble'] = results['ml_ensemble'].data
                
                synthesis_inputs = {
                    "strategy_signals": strategy_signals,
                    "market_data": {
                        "volatility_indicators": {
                            "current_volatility": 0.2,  # Placeholder - would extract from technical analysis
                            "historical_volatility": 0.18
                        },
                        "trend_indicators": {
                            "trend_strength": 0.6,  # Placeholder - would extract from technical analysis
                            "trend_direction": 1
                        }
                    },
                    "risk_constraints": {
                        "max_single_position": 0.1,
                        "max_portfolio_risk": 0.02,
                        "max_sector_exposure": 0.20,
                        "max_total_leverage": 1.0
                    },
                    "portfolio_context": {
                        "positions": {},
                        "total_capital": 1000000
                    }
                }
                
                synthesis_result = await self.agents['signal_synthesis'].safe_execute(synthesis_inputs)
                if not synthesis_result.success:
                    self.logger.warning(f"Signal synthesis failed: {synthesis_result.error_message}")
                else:
                    results['signal_synthesis'] = synthesis_result
                    final_recs = synthesis_result.data.get('final_recommendations', [])
                    self.logger.info(f"Signal synthesis completed with {len(final_recs)} consensus-validated recommendations")
                    
                    # Step 6: Advanced Risk Modeling & Validation
                    if final_recs:
                        self.logger.info("Step 6: Advanced Risk Modeling & Multi-layer Validation")
                        
                        risk_inputs = {
                            "recommendations": final_recs,
                            "portfolio_data": {
                                "positions": {},
                                "weights": {rec["symbol"]: rec.get("position_size", 0) for rec in final_recs},
                                "total_capital": 1000000
                            },
                            "market_data": {
                                "volatility_data": {"current_volatility": 0.2, "historical_volatility": 0.18},
                                "correlation_data": {"average_correlation": 0.5}
                            },
                            "risk_constraints": {
                                "max_var_95": -0.03,
                                "max_volatility": 0.15,
                                "max_single_position": 0.1,
                                "max_sector_exposure": 0.25,
                                "max_portfolio_risk": 0.02
                            }
                        }
                        
                        risk_result = await self.agents['risk_modeling'].safe_execute(risk_inputs)
                        if not risk_result.success:
                            self.logger.warning(f"Risk modeling failed: {risk_result.error_message}")
                        else:
                            results['risk_modeling'] = risk_result
                            risk_validated_recs = risk_result.data.get('risk_validated_recommendations', [])
                            consensus_achieved = risk_result.metadata.get('consensus_achieved', False)
                            self.logger.info(f"Risk modeling completed: {len(risk_validated_recs)} risk-validated recommendations, consensus: {consensus_achieved}")
            
            # Step 7: Final Recommendations with Cross-Validation
            self.logger.info("Step 7: Final Recommendations with Cross-Validation and Chain-of-Thought")
            
            # Collect all agent outputs for final recommendation generation
            agent_outputs_for_recommendation = {}
            
            # Only include successful agent outputs
            for agent_name in ['data_universe', 'technical_analysis', 'ml_ensemble', 
                             'momentum_strategy', 'signal_synthesis', 'risk_modeling']:
                if agent_name in results and hasattr(results[agent_name], 'success') and results[agent_name].success:
                    agent_outputs_for_recommendation[agent_name] = results[agent_name]
            
            if len(agent_outputs_for_recommendation) >= 2:  # Minimum for cross-validation
                recommendation_inputs = {
                    "agent_outputs": agent_outputs_for_recommendation,
                    "user_preferences": {
                        "ranking_method": "confidence",
                        "min_confidence": 0.6,
                        "preferred_sectors": [],
                        "excluded_sectors": []
                    },
                    "output_format": "comprehensive"
                }
                
                recommendation_result = await self.agents['recommendation'].safe_execute(recommendation_inputs)
                if not recommendation_result.success:
                    self.logger.warning(f"Recommendation generation failed: {recommendation_result.error_message}")
                else:
                    results['recommendation'] = recommendation_result
                    final_recommendations = recommendation_result.data.get('final_recommendations', {}).get('ranked_recommendations', [])
                    self.logger.info(f"Final recommendations generated: {len(final_recommendations)} high-confidence trades with full rationale")
            else:
                self.logger.warning("Insufficient successful agents for cross-validation")
            
            # Step 8: Additional Strategy Modules (placeholder)
            self.logger.info("Step 8: Additional Strategy Modules - Statistical Arbitrage, Event-Driven, Options (Not yet implemented)")
            
            results['summary'] = {
                'total_execution_time_ms': sum([
                    result.execution_time_ms for result in results.values() 
                    if hasattr(result, 'execution_time_ms') and result.execution_time_ms
                ]),
                'successful_agents': len([r for r in results.values() if hasattr(r, 'success') and r.success]),
                'universe_size': universe_result.data['metadata']['symbols_count'],
                'analysis_period': f"{start_date.date()} to {end_date.date()}"
            }
            
            self.logger.info("Trading system analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Trading system analysis failed: {str(e)}")
            raise


async def main():
    """Main function to run the trading system."""
    
    # Example usage
    system = TradingSystem()
    
    # Define analysis parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Run analysis
    results = await system.run_analysis(
        start_date=start_date,
        end_date=end_date,
        asset_classes=["equities", "etfs"],
        exchanges=["NYSE", "NASDAQ"],
        custom_symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ"]
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TRADING SYSTEM ANALYSIS RESULTS")
    print("="*80)
    
    if 'data_universe' in results:
        universe_data = results['data_universe'].data
        print(f"Universe Size: {universe_data['metadata']['symbols_count']} symbols")
        print(f"Data Points: {universe_data['metadata']['data_points']:,}")
        print(f"Date Range: {universe_data['metadata']['start_date'].date()} to {universe_data['metadata']['end_date'].date()}")
        
        quality_report = universe_data['data_quality_report']
        print(f"Data Quality: {quality_report['symbols_with_data']}/{quality_report['total_symbols']} symbols have data")
        print(f"Missing Data: {quality_report['missing_data_percentage']:.2f}%")
    
    if 'summary' in results:
        summary = results['summary']
        print(f"Total Execution Time: {summary['total_execution_time_ms']:.0f}ms")
        print(f"Successful Agents: {summary['successful_agents']}")
    
    print("="*80)
    
    # Save results (optional)
    # import json
    # with open(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    #     json.dump(results, f, default=str, indent=2)


if __name__ == "__main__":
    asyncio.run(main())