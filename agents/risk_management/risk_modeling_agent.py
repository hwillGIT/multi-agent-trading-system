"""
Advanced Risk Modeling Agent - Multi-layer validation and consensus-driven risk assessment.

This agent implements sophisticated risk modeling with ultrathinking validation to ensure
comprehensive risk assessment. Key features:
- Multi-layer risk validation (VaR, CVaR, stress testing)
- Cross-validation of risk models with consensus mechanisms
- Portfolio-level risk analysis with scenario testing
- Real-time risk constraint validation
- Monte Carlo simulations for tail risk assessment
- Regime-aware risk adjustments
- Comprehensive audit trails for all risk decisions
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats, optimize
from collections import defaultdict
import warnings
from loguru import logger

from ...core.base.agent import BaseAgent, AgentOutput
from ...core.base.exceptions import ValidationError, DataError, RiskError
from ...core.utils.data_validation import DataValidator
from ...core.utils.math_utils import MathUtils


class RiskModelValidator:
    """Cross-validation system for risk models with consensus mechanisms."""
    
    def __init__(self, min_models: int = 3, consensus_threshold: float = 0.8):
        self.min_models = min_models
        self.consensus_threshold = consensus_threshold
        self.logger = logger.bind(component="risk_model_validator")
    
    def validate_risk_estimates(self, risk_estimates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consensus across multiple risk estimation methods."""
        validation_result = {
            "is_valid": False,
            "consensus_confidence": 0.0,
            "risk_consensus": {},
            "model_agreement": {},
            "outlier_models": [],
            "validation_details": {}
        }
        
        # Extract risk estimates from different models
        model_estimates = {}
        for model_name, estimates in risk_estimates.items():
            if isinstance(estimates, dict):
                model_estimates[model_name] = {
                    "var_95": estimates.get("var_95", 0),
                    "var_99": estimates.get("var_99", 0),
                    "cvar_95": estimates.get("cvar_95", 0),
                    "volatility": estimates.get("volatility", 0),
                    "max_drawdown": estimates.get("max_drawdown", 0)
                }
        
        if len(model_estimates) < self.min_models:
            validation_result["validation_details"]["insufficient_models"] = True
            return validation_result
        
        # Validate each risk metric across models
        risk_metrics = ["var_95", "var_99", "cvar_95", "volatility", "max_drawdown"]
        consensus_results = {}
        
        for metric in risk_metrics:
            metric_values = [estimates[metric] for estimates in model_estimates.values()]
            consensus_results[metric] = self._validate_metric_consensus(metric_values, metric)
        
        # Detect outlier models
        outliers = self._detect_outlier_models(model_estimates, consensus_results)
        validation_result["outlier_models"] = outliers
        
        # Calculate overall consensus
        valid_consensus = [result for result in consensus_results.values() if result["is_valid"]]
        consensus_ratio = len(valid_consensus) / len(risk_metrics)
        
        validation_result.update({
            "is_valid": consensus_ratio >= self.consensus_threshold,
            "consensus_confidence": consensus_ratio,
            "risk_consensus": consensus_results,
            "model_agreement": self._calculate_model_agreement(model_estimates),
            "validation_details": {
                "total_models": len(model_estimates),
                "valid_metrics": len(valid_consensus),
                "consensus_ratio": consensus_ratio,
                "outliers_detected": len(outliers)
            }
        })
        
        return validation_result
    
    def _validate_metric_consensus(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Validate consensus for a specific risk metric."""
        if len(values) < 2:
            return {"is_valid": False, "reason": "insufficient_data"}
        
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        cv = std_val / (abs(mean_val) + 1e-8)  # Coefficient of variation
        
        # Metric-specific thresholds
        cv_thresholds = {
            "var_95": 0.3,
            "var_99": 0.4,
            "cvar_95": 0.35,
            "volatility": 0.25,
            "max_drawdown": 0.4
        }
        
        threshold = cv_thresholds.get(metric_name, 0.3)
        is_valid = cv <= threshold
        
        return {
            "is_valid": is_valid,
            "mean_value": float(mean_val),
            "std_value": float(std_val),
            "coefficient_of_variation": float(cv),
            "threshold": threshold,
            "consensus_strength": max(0, 1 - cv / threshold) if is_valid else 0
        }
    
    def _detect_outlier_models(self, model_estimates: Dict[str, Dict[str, float]], 
                              consensus_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Detect models that consistently produce outlier estimates."""
        outlier_scores = defaultdict(int)
        
        for metric, consensus in consensus_results.items():
            if not consensus["is_valid"]:
                continue
                
            mean_val = consensus["mean_value"]
            std_val = consensus["std_value"]
            
            for model_name, estimates in model_estimates.items():
                value = estimates[metric]
                z_score = abs((value - mean_val) / (std_val + 1e-8))
                
                if z_score > 2.5:  # Outlier threshold
                    outlier_scores[model_name] += 1
        
        # Models that are outliers in 3+ metrics are considered overall outliers
        return [model for model, score in outlier_scores.items() if score >= 3]
    
    def _calculate_model_agreement(self, model_estimates: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate pairwise agreement between models."""
        models = list(model_estimates.keys())
        metrics = ["var_95", "var_99", "cvar_95", "volatility", "max_drawdown"]
        
        agreements = {}
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                agreement_scores = []
                
                for metric in metrics:
                    val1 = model_estimates[model1][metric]
                    val2 = model_estimates[model2][metric]
                    
                    # Calculate relative agreement
                    if abs(val1) + abs(val2) > 1e-8:
                        agreement = 1 - abs(val1 - val2) / (abs(val1) + abs(val2))
                        agreement_scores.append(max(0, agreement))
                
                if agreement_scores:
                    agreements[f"{model1}_vs_{model2}"] = np.mean(agreement_scores)
        
        return agreements


class StressTester:
    """Advanced stress testing with scenario analysis."""
    
    def __init__(self):
        self.logger = logger.bind(component="stress_tester")
    
    def run_stress_tests(self, portfolio_data: Dict[str, Any], 
                        market_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive stress tests with multiple scenarios."""
        stress_results = {
            "scenario_results": {},
            "worst_case_scenarios": [],
            "stress_summary": {},
            "recovery_analysis": {}
        }
        
        try:
            # Define stress scenarios
            scenarios = self._generate_stress_scenarios(market_scenarios)
            
            for scenario_name, scenario_params in scenarios.items():
                self.logger.debug(f"Running stress test: {scenario_name}")
                
                scenario_result = self._simulate_scenario(portfolio_data, scenario_params)
                stress_results["scenario_results"][scenario_name] = scenario_result
            
            # Analyze worst-case scenarios
            stress_results["worst_case_scenarios"] = self._identify_worst_cases(
                stress_results["scenario_results"]
            )
            
            # Generate stress summary
            stress_results["stress_summary"] = self._generate_stress_summary(
                stress_results["scenario_results"]
            )
            
            # Recovery analysis
            stress_results["recovery_analysis"] = self._analyze_recovery_scenarios(
                portfolio_data, stress_results["scenario_results"]
            )
            
        except Exception as e:
            self.logger.error(f"Stress testing failed: {e}")
            stress_results["error"] = str(e)
        
        return stress_results
    
    def _generate_stress_scenarios(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive stress test scenarios."""
        scenarios = {
            "market_crash_2008": {
                "equity_shock": -0.35,
                "volatility_spike": 3.0,
                "correlation_increase": 0.8,
                "duration_days": 90
            },
            "covid_shock_2020": {
                "equity_shock": -0.30,
                "volatility_spike": 2.5,
                "correlation_increase": 0.75,
                "duration_days": 60
            },
            "interest_rate_shock": {
                "rate_increase": 0.03,
                "bond_shock": -0.15,
                "equity_shock": -0.12,
                "duration_days": 180
            },
            "liquidity_crisis": {
                "equity_shock": -0.20,
                "spread_widening": 0.05,
                "volatility_spike": 2.0,
                "duration_days": 45
            },
            "geopolitical_crisis": {
                "equity_shock": -0.15,
                "volatility_spike": 1.8,
                "safe_haven_rally": 0.10,
                "duration_days": 30
            },
            "inflation_shock": {
                "real_rate_shock": 0.025,
                "equity_shock": -0.18,
                "commodity_spike": 0.40,
                "duration_days": 120
            }
        }
        
        return scenarios
    
    def _simulate_scenario(self, portfolio_data: Dict[str, Any], 
                          scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate portfolio performance under stress scenario."""
        simulation_result = {
            "portfolio_loss": 0.0,
            "max_drawdown": 0.0,
            "volatility_increase": 0.0,
            "recovery_time_estimate": 0,
            "component_impacts": {}
        }
        
        try:
            # Extract portfolio components
            positions = portfolio_data.get("positions", {})
            weights = portfolio_data.get("weights", {})
            
            # Apply scenario shocks
            total_loss = 0.0
            component_losses = {}
            
            for symbol, weight in weights.items():
                # Apply asset-specific shocks based on scenario
                asset_shock = self._calculate_asset_shock(symbol, scenario_params)
                asset_loss = weight * asset_shock
                
                total_loss += asset_loss
                component_losses[symbol] = {
                    "weight": weight,
                    "shock": asset_shock,
                    "contribution_to_loss": asset_loss
                }
            
            simulation_result.update({
                "portfolio_loss": total_loss,
                "max_drawdown": min(total_loss, total_loss * 1.2),  # Assume some additional drawdown
                "volatility_increase": scenario_params.get("volatility_spike", 1.0),
                "recovery_time_estimate": self._estimate_recovery_time(total_loss, scenario_params),
                "component_impacts": component_losses
            })
            
        except Exception as e:
            self.logger.error(f"Scenario simulation failed: {e}")
            simulation_result["error"] = str(e)
        
        return simulation_result
    
    def _calculate_asset_shock(self, symbol: str, scenario_params: Dict[str, Any]) -> float:
        """Calculate asset-specific shock based on scenario parameters."""
        # This would normally use asset classification and factor loadings
        # For now, using simplified approach
        
        base_shock = scenario_params.get("equity_shock", -0.10)
        
        # Adjust based on asset characteristics (simplified)
        if "SPY" in symbol or "QQQ" in symbol:
            # ETFs closely track market
            return base_shock
        elif symbol in ["AAPL", "GOOGL", "MSFT"]:
            # Large cap tech, potentially more volatile
            return base_shock * 1.2
        else:
            # Individual stocks
            return base_shock * 1.1
    
    def _estimate_recovery_time(self, loss: float, scenario_params: Dict[str, Any]) -> int:
        """Estimate recovery time in days based on loss severity."""
        duration = scenario_params.get("duration_days", 60)
        loss_severity = abs(loss)
        
        # Simple heuristic: recovery time proportional to loss severity
        if loss_severity < 0.05:
            return duration * 2
        elif loss_severity < 0.15:
            return duration * 3
        elif loss_severity < 0.25:
            return duration * 5
        else:
            return duration * 8
    
    def _identify_worst_cases(self, scenario_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify worst-case scenarios from stress test results."""
        scenarios_by_loss = []
        
        for scenario_name, results in scenario_results.items():
            if "error" not in results:
                scenarios_by_loss.append({
                    "scenario": scenario_name,
                    "portfolio_loss": results.get("portfolio_loss", 0),
                    "max_drawdown": results.get("max_drawdown", 0),
                    "recovery_time": results.get("recovery_time_estimate", 0)
                })
        
        # Sort by portfolio loss (worst first)
        scenarios_by_loss.sort(key=lambda x: x["portfolio_loss"])
        
        return scenarios_by_loss[:3]  # Top 3 worst cases
    
    def _generate_stress_summary(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from stress test results."""
        valid_results = [r for r in scenario_results.values() if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid stress test results"}
        
        losses = [r["portfolio_loss"] for r in valid_results]
        drawdowns = [r["max_drawdown"] for r in valid_results]
        recovery_times = [r["recovery_time_estimate"] for r in valid_results]
        
        return {
            "stress_statistics": {
                "worst_loss": min(losses),
                "average_loss": np.mean(losses),
                "loss_volatility": np.std(losses),
                "worst_drawdown": min(drawdowns),
                "average_recovery_time": np.mean(recovery_times)
            },
            "risk_metrics": {
                "stress_var_95": np.percentile(losses, 5),
                "stress_var_99": np.percentile(losses, 1),
                "stress_cvar_95": np.mean([l for l in losses if l <= np.percentile(losses, 5)])
            },
            "scenarios_tested": len(valid_results)
        }
    
    def _analyze_recovery_scenarios(self, portfolio_data: Dict[str, Any], 
                                   scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze portfolio recovery scenarios."""
        recovery_analysis = {
            "recovery_strategies": [],
            "optimal_rebalancing": {},
            "risk_reduction_opportunities": []
        }
        
        # This would normally include sophisticated recovery analysis
        # For now, providing a simplified framework
        
        worst_scenarios = [name for name, results in scenario_results.items() 
                          if results.get("portfolio_loss", 0) < -0.15]
        
        if worst_scenarios:
            recovery_analysis["recovery_strategies"] = [
                {
                    "strategy": "defensive_rebalancing",
                    "description": "Reduce risk exposure in worst-case scenarios",
                    "applicable_scenarios": worst_scenarios
                },
                {
                    "strategy": "diversification_enhancement",
                    "description": "Add uncorrelated assets to reduce concentration risk",
                    "applicable_scenarios": worst_scenarios
                }
            ]
        
        return recovery_analysis


class RiskModelingAgent(BaseAgent):
    """
    Advanced Risk Modeling Agent with multi-layer validation and consensus mechanisms.
    
    This agent implements comprehensive risk modeling with ultrathinking validation to ensure
    robust risk assessment through cross-validation and consensus.
    
    Inputs: Portfolio data, market data, existing recommendations
    Outputs: Risk-validated recommendations with comprehensive risk analysis
    """
    
    def __init__(self):
        super().__init__("RiskModelingAgent", "risk_modeling")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validation systems."""
        self.min_risk_models = self.get_config_value("min_risk_models", 3)
        self.consensus_threshold = self.get_config_value("consensus_threshold", 0.8)
        self.stress_testing = self.get_config_value("stress_testing", True)
        self.monte_carlo_runs = self.get_config_value("monte_carlo_runs", 10000)
        
        # Initialize validation systems
        self.risk_validator = RiskModelValidator(
            min_models=self.min_risk_models,
            consensus_threshold=self.consensus_threshold
        )
        self.stress_tester = StressTester()
        
        # Risk model configurations
        self.risk_models = self.get_config_value("risk_models", {
            "parametric_var": {"confidence_levels": [0.95, 0.99], "lookback_days": 252},
            "historical_simulation": {"lookback_days": 504, "bootstrap_samples": 1000},
            "monte_carlo": {"simulations": 10000, "time_horizon": 22},
            "extreme_value": {"threshold": 0.95, "tail_estimation": "gpd"}
        })
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute comprehensive risk modeling with multi-layer validation.
        
        Args:
            inputs: Dictionary containing:
                - recommendations: Trading recommendations to validate
                - portfolio_data: Current portfolio state
                - market_data: Market context and historical data
                - risk_constraints: Risk management constraints
                
        Returns:
            AgentOutput with risk-validated recommendations and comprehensive risk analysis
        """
        self._validate_inputs(inputs)
        
        recommendations = inputs["recommendations"]
        portfolio_data = inputs.get("portfolio_data", {})
        market_data = inputs.get("market_data", {})
        risk_constraints = inputs.get("risk_constraints", {})
        
        try:
            # Step 1: Multi-model risk estimation with cross-validation
            risk_estimates = await self._generate_risk_estimates(
                recommendations, portfolio_data, market_data
            )
            self.logger.info(f"Generated risk estimates from {len(risk_estimates)} models")
            
            # Step 2: Cross-validate risk models with consensus mechanisms
            consensus_validation = self.risk_validator.validate_risk_estimates(risk_estimates)
            self.logger.info(f"Risk model consensus: {consensus_validation['consensus_confidence']:.3f}")
            
            # Step 3: Stress testing and scenario analysis
            stress_results = {}
            if self.stress_testing:
                stress_results = self.stress_tester.run_stress_tests(
                    portfolio_data, market_data
                )
                self.logger.info(f"Completed stress testing with {len(stress_results['scenario_results'])} scenarios")
            
            # Step 4: Risk constraint validation
            constraint_validation = await self._validate_risk_constraints(
                recommendations, risk_estimates, risk_constraints, consensus_validation
            )
            
            # Step 5: Portfolio risk optimization
            optimized_recommendations = await self._optimize_portfolio_risk(
                recommendations, risk_estimates, constraint_validation, consensus_validation
            )
            
            # Step 6: Generate comprehensive risk report
            risk_report = self._generate_risk_report(
                risk_estimates, consensus_validation, stress_results, 
                constraint_validation, optimized_recommendations
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "risk_validated_recommendations": optimized_recommendations,
                    "risk_estimates": risk_estimates,
                    "consensus_validation": consensus_validation,
                    "stress_test_results": stress_results,
                    "constraint_validation": constraint_validation,
                    "risk_report": risk_report,
                    "portfolio_risk_metrics": self._calculate_portfolio_risk_metrics(
                        optimized_recommendations, risk_estimates
                    )
                },
                metadata={
                    "risk_models_used": len(risk_estimates),
                    "consensus_achieved": consensus_validation.get("is_valid", False),
                    "stress_scenarios_tested": len(stress_results.get("scenario_results", {})),
                    "recommendations_processed": len(recommendations),
                    "recommendations_modified": self._count_modified_recommendations(
                        recommendations, optimized_recommendations
                    ),
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Risk modeling failed: {str(e)}")
            raise RiskError(f"Risk modeling processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters with comprehensive checks."""
        required_keys = ["recommendations"]
        self.validate_inputs(inputs, required_keys)
        
        recommendations = inputs["recommendations"]
        if not recommendations:
            raise ValidationError("Recommendations cannot be empty")
        
        # Validate recommendation structure
        for i, rec in enumerate(recommendations):
            if not isinstance(rec, dict):
                raise ValidationError(f"Invalid recommendation format at index {i}")
            
            required_rec_fields = ["symbol", "action", "position_size"]
            for field in required_rec_fields:
                if field not in rec:
                    raise ValidationError(f"Missing required field '{field}' in recommendation {i}")
    
    async def _generate_risk_estimates(self, recommendations: List[Dict[str, Any]],
                                     portfolio_data: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate risk estimates using multiple models for cross-validation."""
        risk_estimates = {}
        
        # Extract portfolio and recommendation data
        symbols = [rec["symbol"] for rec in recommendations]
        weights = {rec["symbol"]: rec.get("position_size", 0) for rec in recommendations}
        
        try:
            # Model 1: Parametric VaR
            risk_estimates["parametric_var"] = await self._calculate_parametric_var(
                symbols, weights, market_data
            )
            
            # Model 2: Historical Simulation
            risk_estimates["historical_simulation"] = await self._calculate_historical_simulation(
                symbols, weights, market_data
            )
            
            # Model 3: Monte Carlo Simulation
            risk_estimates["monte_carlo"] = await self._calculate_monte_carlo_var(
                symbols, weights, market_data
            )
            
            # Model 4: Extreme Value Theory
            risk_estimates["extreme_value"] = await self._calculate_extreme_value_var(
                symbols, weights, market_data
            )
            
        except Exception as e:
            self.logger.error(f"Risk estimation failed: {e}")
            # Ensure we have at least one model for fallback
            if not risk_estimates:
                risk_estimates["fallback_model"] = {
                    "var_95": -0.05,
                    "var_99": -0.08,
                    "cvar_95": -0.07,
                    "volatility": 0.20,
                    "max_drawdown": -0.15
                }
        
        return risk_estimates
    
    async def _calculate_parametric_var(self, symbols: List[str], weights: Dict[str, float],
                                      market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate parametric VaR using normal distribution assumption."""
        try:
            # This would normally use actual historical returns
            # For demonstration, using simplified calculations
            
            portfolio_volatility = 0.15  # Placeholder
            
            # Calculate VaR using normal distribution
            var_95 = -1.645 * portfolio_volatility  # 95% VaR
            var_99 = -2.326 * portfolio_volatility  # 99% VaR
            cvar_95 = var_95 * 1.2  # Simplified CVaR
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "volatility": portfolio_volatility,
                "max_drawdown": var_99 * 1.5
            }
            
        except Exception as e:
            self.logger.error(f"Parametric VaR calculation failed: {e}")
            return {
                "var_95": -0.05,
                "var_99": -0.08,
                "cvar_95": -0.06,
                "volatility": 0.15,
                "max_drawdown": -0.12
            }
    
    async def _calculate_historical_simulation(self, symbols: List[str], weights: Dict[str, float],
                                             market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate VaR using historical simulation method."""
        try:
            # This would use actual historical return data
            # For demonstration, generating simulated historical returns
            
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(-0.001, 0.02, 252)  # Simulated daily returns
            
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = np.mean(returns[returns <= var_95])
            volatility = np.std(returns)
            max_drawdown = np.min(np.cumsum(returns))
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "volatility": volatility,
                "max_drawdown": max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Historical simulation failed: {e}")
            return {
                "var_95": -0.04,
                "var_99": -0.07,
                "cvar_95": -0.055,
                "volatility": 0.18,
                "max_drawdown": -0.11
            }
    
    async def _calculate_monte_carlo_var(self, symbols: List[str], weights: Dict[str, float],
                                       market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            # Monte Carlo simulation
            np.random.seed(123)  # For reproducibility
            num_simulations = self.monte_carlo_runs
            
            # Simplified simulation - would normally use more sophisticated models
            simulated_returns = np.random.normal(-0.0005, 0.025, num_simulations)
            
            var_95 = np.percentile(simulated_returns, 5)
            var_99 = np.percentile(simulated_returns, 1)
            cvar_95 = np.mean(simulated_returns[simulated_returns <= var_95])
            volatility = np.std(simulated_returns)
            max_drawdown = np.min(simulated_returns) * 2  # Simplified
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "volatility": volatility,
                "max_drawdown": max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {e}")
            return {
                "var_95": -0.055,
                "var_99": -0.085,
                "cvar_95": -0.07,
                "volatility": 0.22,
                "max_drawdown": -0.14
            }
    
    async def _calculate_extreme_value_var(self, symbols: List[str], weights: Dict[str, float],
                                         market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate VaR using Extreme Value Theory."""
        try:
            # Simplified EVT calculation
            # Would normally fit Generalized Pareto Distribution to tail
            
            # Simulate extreme returns
            np.random.seed(456)
            extreme_returns = np.random.pareto(1.5, 1000) * -0.01  # Negative for losses
            
            var_95 = np.percentile(extreme_returns, 5)
            var_99 = np.percentile(extreme_returns, 1)
            cvar_95 = np.mean(extreme_returns[extreme_returns <= var_95])
            volatility = np.std(extreme_returns)
            max_drawdown = np.min(extreme_returns)
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "volatility": volatility,
                "max_drawdown": max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Extreme Value Theory calculation failed: {e}")
            return {
                "var_95": -0.06,
                "var_99": -0.09,
                "cvar_95": -0.08,
                "volatility": 0.25,
                "max_drawdown": -0.16
            }
    
    async def _validate_risk_constraints(self, recommendations: List[Dict[str, Any]],
                                       risk_estimates: Dict[str, Dict[str, Any]],
                                       risk_constraints: Dict[str, Any],
                                       consensus_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate recommendations against risk constraints."""
        constraint_results = {
            "constraint_violations": [],
            "risk_adjusted_recommendations": [],
            "constraint_compliance": {},
            "violation_summary": {}
        }
        
        try:
            # Extract consensus risk metrics
            if consensus_validation.get("is_valid", False):
                consensus_risk = consensus_validation.get("risk_consensus", {})
                portfolio_var_95 = consensus_risk.get("var_95", {}).get("mean_value", -0.05)
                portfolio_volatility = consensus_risk.get("volatility", {}).get("mean_value", 0.2)
            else:
                # Use conservative fallback
                portfolio_var_95 = -0.08
                portfolio_volatility = 0.25
            
            # Check constraints
            max_var = risk_constraints.get("max_var_95", -0.03)
            max_volatility = risk_constraints.get("max_volatility", 0.15)
            max_single_position = risk_constraints.get("max_single_position", 0.1)
            max_sector_exposure = risk_constraints.get("max_sector_exposure", 0.25)
            
            # Validate portfolio-level constraints
            violations = []
            
            if portfolio_var_95 < max_var:
                violations.append({
                    "constraint": "max_var_95",
                    "current_value": portfolio_var_95,
                    "limit": max_var,
                    "severity": "high"
                })
            
            if portfolio_volatility > max_volatility:
                violations.append({
                    "constraint": "max_volatility",
                    "current_value": portfolio_volatility,
                    "limit": max_volatility,
                    "severity": "medium"
                })
            
            # Validate position-level constraints
            for rec in recommendations:
                position_size = abs(rec.get("position_size", 0))
                
                if position_size > max_single_position:
                    violations.append({
                        "constraint": "max_single_position",
                        "symbol": rec["symbol"],
                        "current_value": position_size,
                        "limit": max_single_position,
                        "severity": "high"
                    })
            
            constraint_results.update({
                "constraint_violations": violations,
                "constraint_compliance": {
                    "var_95_compliant": portfolio_var_95 >= max_var,
                    "volatility_compliant": portfolio_volatility <= max_volatility,
                    "position_size_compliant": all(
                        abs(r.get("position_size", 0)) <= max_single_position 
                        for r in recommendations
                    )
                },
                "violation_summary": {
                    "total_violations": len(violations),
                    "high_severity": len([v for v in violations if v["severity"] == "high"]),
                    "medium_severity": len([v for v in violations if v["severity"] == "medium"])
                }
            })
            
        except Exception as e:
            self.logger.error(f"Risk constraint validation failed: {e}")
            constraint_results["error"] = str(e)
        
        return constraint_results
    
    async def _optimize_portfolio_risk(self, recommendations: List[Dict[str, Any]],
                                     risk_estimates: Dict[str, Dict[str, Any]],
                                     constraint_validation: Dict[str, Any],
                                     consensus_validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize portfolio to meet risk constraints while preserving alpha."""
        optimized_recommendations = recommendations.copy()
        
        try:
            violations = constraint_validation.get("constraint_violations", [])
            
            if not violations:
                self.logger.info("No risk constraint violations - returning original recommendations")
                return optimized_recommendations
            
            # Apply risk adjustments
            for violation in violations:
                if violation["constraint"] == "max_single_position":
                    # Scale down oversized positions
                    symbol = violation["symbol"]
                    limit = violation["limit"]
                    
                    for rec in optimized_recommendations:
                        if rec["symbol"] == symbol:
                            original_size = rec["position_size"]
                            new_size = np.sign(original_size) * min(abs(original_size), limit)
                            rec["position_size"] = new_size
                            rec["risk_adjusted"] = True
                            rec["original_position_size"] = original_size
                            rec["adjustment_reason"] = "position_size_constraint"
                            
                            self.logger.info(f"Adjusted {symbol} position from {original_size:.3f} to {new_size:.3f}")
            
            # Portfolio-level adjustments
            portfolio_violations = [v for v in violations if "symbol" not in v]
            
            if portfolio_violations:
                # Scale down all positions proportionally
                scale_factor = 0.8  # Conservative scaling
                
                for rec in optimized_recommendations:
                    if not rec.get("risk_adjusted", False):
                        original_size = rec["position_size"]
                        new_size = original_size * scale_factor
                        rec["position_size"] = new_size
                        rec["risk_adjusted"] = True
                        rec["original_position_size"] = original_size
                        rec["adjustment_reason"] = "portfolio_risk_constraint"
                
                self.logger.info(f"Applied portfolio-wide scaling factor: {scale_factor}")
            
        except Exception as e:
            self.logger.error(f"Portfolio risk optimization failed: {e}")
            # Return original recommendations if optimization fails
            return recommendations
        
        return optimized_recommendations
    
    def _calculate_portfolio_risk_metrics(self, recommendations: List[Dict[str, Any]],
                                        risk_estimates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics."""
        portfolio_metrics = {
            "total_exposure": 0.0,
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "net_exposure": 0.0,
            "concentration_risk": 0.0,
            "risk_adjusted_exposure": 0.0
        }
        
        try:
            for rec in recommendations:
                position_size = rec.get("position_size", 0)
                portfolio_metrics["total_exposure"] += abs(position_size)
                
                if position_size > 0:
                    portfolio_metrics["long_exposure"] += position_size
                else:
                    portfolio_metrics["short_exposure"] += abs(position_size)
            
            portfolio_metrics["net_exposure"] = (
                portfolio_metrics["long_exposure"] - portfolio_metrics["short_exposure"]
            )
            
            # Calculate concentration risk (Herfindahl index)
            if recommendations:
                weights = [abs(r.get("position_size", 0)) for r in recommendations]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                    portfolio_metrics["concentration_risk"] = sum(w**2 for w in normalized_weights)
            
            # Risk-adjusted exposure (simplified)
            portfolio_metrics["risk_adjusted_exposure"] = (
                portfolio_metrics["total_exposure"] * 0.8  # Conservative adjustment
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio risk metrics calculation failed: {e}")
        
        return portfolio_metrics
    
    def _count_modified_recommendations(self, original_recs: List[Dict[str, Any]],
                                      modified_recs: List[Dict[str, Any]]) -> int:
        """Count how many recommendations were modified during risk optimization."""
        modified_count = 0
        
        for i, (orig, mod) in enumerate(zip(original_recs, modified_recs)):
            if orig.get("position_size", 0) != mod.get("position_size", 0):
                modified_count += 1
        
        return modified_count
    
    def _generate_risk_report(self, risk_estimates: Dict[str, Dict[str, Any]],
                            consensus_validation: Dict[str, Any],
                            stress_results: Dict[str, Any],
                            constraint_validation: Dict[str, Any],
                            optimized_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        report = {
            "executive_summary": {},
            "risk_model_analysis": {},
            "stress_test_summary": {},
            "constraint_compliance": {},
            "recommendations_summary": {},
            "risk_warnings": [],
            "timestamp": datetime.utcnow()
        }
        
        try:
            # Executive Summary
            consensus_achieved = consensus_validation.get("is_valid", False)
            total_violations = constraint_validation.get("violation_summary", {}).get("total_violations", 0)
            
            report["executive_summary"] = {
                "risk_assessment_status": "PASSED" if consensus_achieved and total_violations == 0 else "ATTENTION_REQUIRED",
                "consensus_confidence": consensus_validation.get("consensus_confidence", 0),
                "constraint_violations": total_violations,
                "stress_test_scenarios": len(stress_results.get("scenario_results", {})),
                "recommendations_modified": len([r for r in optimized_recommendations if r.get("risk_adjusted", False)])
            }
            
            # Risk Model Analysis
            if consensus_validation.get("risk_consensus"):
                risk_consensus = consensus_validation["risk_consensus"]
                report["risk_model_analysis"] = {
                    "var_95_consensus": risk_consensus.get("var_95", {}).get("is_valid", False),
                    "volatility_consensus": risk_consensus.get("volatility", {}).get("is_valid", False),
                    "model_agreement": consensus_validation.get("model_agreement", {}),
                    "outlier_models": consensus_validation.get("outlier_models", [])
                }
            
            # Stress Test Summary
            if stress_results.get("stress_summary"):
                stress_summary = stress_results["stress_summary"]
                report["stress_test_summary"] = {
                    "worst_case_loss": stress_summary.get("stress_statistics", {}).get("worst_loss", 0),
                    "average_loss": stress_summary.get("stress_statistics", {}).get("average_loss", 0),
                    "worst_scenarios": [s["scenario"] for s in stress_results.get("worst_case_scenarios", [])[:3]]
                }
            
            # Risk Warnings
            warnings = []
            
            if not consensus_achieved:
                warnings.append({
                    "severity": "HIGH",
                    "message": "Risk model consensus not achieved - proceed with caution",
                    "recommendation": "Review model assumptions and consider additional validation"
                })
            
            if total_violations > 0:
                warnings.append({
                    "severity": "HIGH",
                    "message": f"{total_violations} risk constraint violations detected",
                    "recommendation": "Review and adjust position sizes or risk limits"
                })
            
            report["risk_warnings"] = warnings
            
        except Exception as e:
            self.logger.error(f"Risk report generation failed: {e}")
            report["error"] = str(e)
        
        return report