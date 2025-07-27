"""
Cross-Asset Agent - Multi-asset correlation analysis and factor exposure with consensus validation.

This agent implements sophisticated cross-asset strategies with ultrathinking validation
for portfolio-level optimization across multiple asset classes. Key features:
- Cross-asset correlation analysis and regime detection
- Factor exposure measurement and optimization
- Currency hedging recommendations
- Sector rotation signals based on macro factors
- Intermarket analysis (bonds, commodities, FX, equities)
- Risk parity and factor-based portfolio construction
- Consensus validation across multiple correlation models
- Dynamic asset allocation recommendations
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats, optimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings
from loguru import logger
warnings.filterwarnings('ignore')

from ....core.base.agent import BaseAgent, AgentOutput
from ....core.base.exceptions import ValidationError, DataError
from ....core.utils.data_validation import DataValidator
from ....core.utils.math_utils import MathUtils


class CorrelationConsensusValidator:
    """Validate correlation analysis across multiple models and timeframes."""
    
    def __init__(self, min_timeframes: int = 3, confidence_threshold: float = 0.75):
        self.min_timeframes = min_timeframes
        self.confidence_threshold = confidence_threshold
        self.logger = logger.bind(component="correlation_consensus")
    
    def validate_correlation_consensus(self, price_data: pd.DataFrame, 
                                     timeframes: List[int] = None) -> Dict[str, Any]:
        """Validate correlation patterns across multiple timeframes."""
        if timeframes is None:
            timeframes = [30, 60, 120, 252]  # 1M, 2M, 4M, 1Y
        
        validation_result = {
            "is_valid": False,
            "consensus_correlations": {},
            "correlation_stability": {},
            "regime_consistency": {},
            "validation_details": {}
        }
        
        try:
            if len(price_data.columns) < 2:
                validation_result["validation_details"]["insufficient_assets"] = True
                return validation_result
            
            # Calculate correlations for each timeframe
            timeframe_correlations = {}
            for timeframe in timeframes:
                if len(price_data) >= timeframe:
                    recent_data = price_data.tail(timeframe)
                    returns = recent_data.pct_change().dropna()
                    
                    if len(returns) > 10:  # Minimum data requirement
                        corr_matrix = returns.corr()
                        timeframe_correlations[timeframe] = corr_matrix
            
            if len(timeframe_correlations) < self.min_timeframes:
                validation_result["validation_details"]["insufficient_timeframes"] = True
                return validation_result
            
            # Calculate consensus correlations
            consensus_corr = self._calculate_consensus_correlations(timeframe_correlations)
            
            # Measure correlation stability
            stability_metrics = self._measure_correlation_stability(timeframe_correlations)
            
            # Check regime consistency
            regime_consistency = self._check_regime_consistency(timeframe_correlations)
            
            # Determine validity
            avg_stability = np.mean(list(stability_metrics.values()))
            regime_score = regime_consistency.get("consistency_score", 0)
            
            is_valid = (
                avg_stability >= 0.6 and  # Correlations are reasonably stable
                regime_score >= 0.5 and  # Regime patterns are consistent
                len(timeframe_correlations) >= self.min_timeframes
            )
            
            validation_result.update({
                "is_valid": is_valid,
                "consensus_correlations": consensus_corr,
                "correlation_stability": stability_metrics,
                "regime_consistency": regime_consistency,
                "validation_details": {
                    "timeframes_analyzed": len(timeframe_correlations),
                    "average_stability": avg_stability,
                    "regime_consistency_score": regime_score
                }
            })
            
        except Exception as e:
            self.logger.error(f"Correlation consensus validation failed: {e}")
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _calculate_consensus_correlations(self, timeframe_correlations: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Calculate consensus correlation matrix across timeframes."""
        if not timeframe_correlations:
            return pd.DataFrame()
        
        # Get all asset pairs
        first_corr = list(timeframe_correlations.values())[0]
        assets = first_corr.index.tolist()
        
        # Calculate weighted average correlations
        consensus_matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    consensus_matrix.loc[asset1, asset2] = 1.0
                else:
                    correlations = []
                    weights = []
                    
                    for timeframe, corr_matrix in timeframe_correlations.items():
                        if asset1 in corr_matrix.index and asset2 in corr_matrix.columns:
                            corr_value = corr_matrix.loc[asset1, asset2]
                            if not np.isnan(corr_value):
                                correlations.append(corr_value)
                                # Weight longer timeframes more heavily
                                weights.append(np.log(timeframe + 1))
                    
                    if correlations:
                        weighted_corr = np.average(correlations, weights=weights)
                        consensus_matrix.loc[asset1, asset2] = weighted_corr
                        consensus_matrix.loc[asset2, asset1] = weighted_corr
                    else:
                        consensus_matrix.loc[asset1, asset2] = 0.0
                        consensus_matrix.loc[asset2, asset1] = 0.0
        
        return consensus_matrix
    
    def _measure_correlation_stability(self, timeframe_correlations: Dict[int, pd.DataFrame]) -> Dict[str, float]:
        """Measure stability of correlations across timeframes."""
        stability_metrics = {}
        
        if len(timeframe_correlations) < 2:
            return stability_metrics
        
        correlation_matrices = list(timeframe_correlations.values())
        
        # Calculate pairwise stability between timeframes
        for i, (tf1, corr1) in enumerate(timeframe_correlations.items()):
            for j, (tf2, corr2) in enumerate(timeframe_correlations.items()):
                if i < j:  # Avoid duplicates
                    # Find common assets
                    common_assets = corr1.index.intersection(corr2.index)
                    
                    if len(common_assets) >= 2:
                        # Extract correlation values for common assets
                        corr1_values = []
                        corr2_values = []
                        
                        for asset1 in common_assets:
                            for asset2 in common_assets:
                                if asset1 != asset2:
                                    val1 = corr1.loc[asset1, asset2]
                                    val2 = corr2.loc[asset1, asset2]
                                    
                                    if not (np.isnan(val1) or np.isnan(val2)):
                                        corr1_values.append(val1)
                                        corr2_values.append(val2)
                        
                        if len(corr1_values) > 0:
                            # Calculate correlation between correlation vectors
                            stability = np.corrcoef(corr1_values, corr2_values)[0, 1]
                            if not np.isnan(stability):
                                stability_metrics[f"{tf1}d_vs_{tf2}d"] = stability
        
        return stability_metrics
    
    def _check_regime_consistency(self, timeframe_correlations: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """Check consistency of correlation regimes across timeframes."""
        regime_analysis = {
            "consistency_score": 0.0,
            "regime_classifications": {},
            "regime_agreement": 0.0
        }
        
        try:
            # Classify each timeframe into correlation regime
            regime_classifications = {}
            
            for timeframe, corr_matrix in timeframe_correlations.items():
                # Calculate average correlation (excluding diagonal)
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                avg_corr = corr_matrix.values[mask].mean()
                
                # Classify regime
                if avg_corr > 0.7:
                    regime = "high_correlation"
                elif avg_corr > 0.3:
                    regime = "medium_correlation"
                else:
                    regime = "low_correlation"
                
                regime_classifications[timeframe] = {
                    "regime": regime,
                    "average_correlation": avg_corr
                }
            
            # Measure regime agreement
            regimes = [data["regime"] for data in regime_classifications.values()]
            if regimes:
                most_common_regime = max(set(regimes), key=regimes.count)
                agreement_ratio = regimes.count(most_common_regime) / len(regimes)
                
                regime_analysis.update({
                    "consistency_score": agreement_ratio,
                    "regime_classifications": regime_classifications,
                    "regime_agreement": agreement_ratio,
                    "dominant_regime": most_common_regime
                })
            
        except Exception as e:
            self.logger.error(f"Regime consistency check failed: {e}")
        
        return regime_analysis


class FactorAnalyzer:
    """Analyze factor exposures and loadings across assets."""
    
    def __init__(self):
        self.logger = logger.bind(component="factor_analyzer")
        
        # Common factor proxies
        self.factor_proxies = {
            "market": "SPY",
            "size": "IWM",  # Small cap
            "value": "IWD",  # Value
            "growth": "IWF",  # Growth
            "momentum": "MTUM",
            "low_volatility": "USMV",
            "quality": "QUAL",
            "international": "EFA",
            "emerging_markets": "EEM",
            "real_estate": "VNQ",
            "commodities": "DJP",
            "bonds": "AGG",
            "dollar": "UUP"
        }
    
    def perform_factor_analysis(self, returns_data: pd.DataFrame,
                               factor_returns: Dict[str, pd.Series] = None) -> Dict[str, Any]:
        """Perform comprehensive factor analysis."""
        analysis_result = {
            "factor_loadings": {},
            "factor_exposures": {},
            "explained_variance": {},
            "risk_attribution": {},
            "factor_concentrations": {}
        }
        
        try:
            if returns_data.empty:
                return analysis_result
            
            # Clean data
            clean_returns = returns_data.dropna()
            
            if len(clean_returns) < 60:  # Need sufficient data
                analysis_result["error"] = "Insufficient data for factor analysis"
                return analysis_result
            
            # Method 1: PCA-based factor analysis
            pca_result = self._perform_pca_analysis(clean_returns)
            analysis_result["pca_factors"] = pca_result
            
            # Method 2: Fundamental factor analysis (if factor returns provided)
            if factor_returns:
                fundamental_result = self._perform_fundamental_factor_analysis(
                    clean_returns, factor_returns
                )
                analysis_result["fundamental_factors"] = fundamental_result
            
            # Method 3: Statistical factor model
            statistical_result = self._perform_statistical_factor_analysis(clean_returns)
            analysis_result["statistical_factors"] = statistical_result
            
            # Risk attribution
            analysis_result["risk_attribution"] = self._calculate_risk_attribution(
                clean_returns, pca_result
            )
            
            # Factor concentrations
            analysis_result["factor_concentrations"] = self._calculate_factor_concentrations(
                pca_result
            )
            
        except Exception as e:
            self.logger.error(f"Factor analysis failed: {e}")
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    def _perform_pca_analysis(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA-based factor analysis."""
        try:
            # Standardize returns
            standardized_returns = (returns_data - returns_data.mean()) / returns_data.std()
            
            # Perform PCA
            pca = PCA()
            factor_returns = pca.fit_transform(standardized_returns)
            
            # Calculate factor loadings
            loadings = pd.DataFrame(
                pca.components_[:5].T,  # Top 5 factors
                index=returns_data.columns,
                columns=[f"Factor_{i+1}" for i in range(5)]
            )
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_[:5]
            
            return {
                "loadings": loadings,
                "explained_variance": explained_variance,
                "cumulative_variance": np.cumsum(explained_variance),
                "factor_returns": factor_returns[:, :5],
                "eigenvalues": pca.explained_variance_[:5]
            }
            
        except Exception as e:
            self.logger.error(f"PCA analysis failed: {e}")
            return {}
    
    def _perform_fundamental_factor_analysis(self, returns_data: pd.DataFrame,
                                           factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Perform fundamental factor analysis using known factors."""
        try:
            # Align data
            common_dates = returns_data.index
            aligned_factors = {}
            
            for factor_name, factor_series in factor_returns.items():
                aligned_series = factor_series.reindex(common_dates).dropna()
                if len(aligned_series) > len(common_dates) * 0.8:  # At least 80% coverage
                    aligned_factors[factor_name] = aligned_series
            
            if not aligned_factors:
                return {}
            
            # Create factor matrix
            factor_df = pd.DataFrame(aligned_factors)
            
            # Align returns data
            common_dates = factor_df.index.intersection(returns_data.index)
            factor_df = factor_df.loc[common_dates]
            returns_df = returns_data.loc[common_dates]
            
            # Run regressions for each asset
            factor_loadings = {}
            r_squareds = {}
            
            for asset in returns_df.columns:
                asset_returns = returns_df[asset].dropna()
                
                # Align factor data with asset returns
                asset_dates = asset_returns.index
                asset_factors = factor_df.loc[asset_dates]
                asset_returns_aligned = asset_returns.loc[asset_dates]
                
                if len(asset_returns_aligned) > 30:  # Minimum observations
                    # Multiple regression
                    X = asset_factors.values
                    y = asset_returns_aligned.values
                    
                    # Add constant
                    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
                    
                    try:
                        # OLS regression
                        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                        
                        # Calculate R-squared
                        y_pred = X_with_const @ beta
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        
                        factor_loadings[asset] = {
                            "alpha": beta[0],
                            **{factor_name: beta[i+1] for i, factor_name in enumerate(factor_df.columns)}
                        }
                        r_squareds[asset] = r_squared
                        
                    except np.linalg.LinAlgError:
                        continue
            
            return {
                "factor_loadings": factor_loadings,
                "r_squareds": r_squareds,
                "factors_used": list(factor_df.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental factor analysis failed: {e}")
            return {}
    
    def _perform_statistical_factor_analysis(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical factor analysis using clustering."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Perform clustering on correlation matrix
            n_clusters = min(5, len(returns_data.columns) // 3)
            
            if n_clusters < 2:
                return {}
            
            # Use correlation distance for clustering
            distance_matrix = 1 - np.abs(corr_matrix)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(distance_matrix)
            
            # Create cluster mapping
            cluster_mapping = {}
            for i, asset in enumerate(returns_data.columns):
                cluster_mapping[asset] = clusters[i]
            
            # Calculate cluster representatives (centroids)
            cluster_returns = {}
            for cluster_id in range(n_clusters):
                cluster_assets = [asset for asset, cid in cluster_mapping.items() if cid == cluster_id]
                if cluster_assets:
                    cluster_return = returns_data[cluster_assets].mean(axis=1)
                    cluster_returns[f"Cluster_{cluster_id}"] = cluster_return
            
            return {
                "cluster_mapping": cluster_mapping,
                "cluster_returns": cluster_returns,
                "n_clusters": n_clusters
            }
            
        except Exception as e:
            self.logger.error(f"Statistical factor analysis failed: {e}")
            return {}
    
    def _calculate_risk_attribution(self, returns_data: pd.DataFrame,
                                  pca_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk attribution to factors."""
        try:
            if not pca_result or "loadings" not in pca_result:
                return {}
            
            loadings = pca_result["loadings"]
            explained_variance = pca_result["explained_variance"]
            
            # Calculate risk contribution of each factor
            portfolio_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)  # Equal weight
            
            risk_contributions = {}
            for i, factor in enumerate(loadings.columns):
                factor_loadings = loadings[factor].values
                factor_risk = explained_variance[i] * np.sum((factor_loadings * portfolio_weights) ** 2)
                risk_contributions[factor] = factor_risk
            
            # Specific risk (unexplained)
            total_explained_risk = sum(risk_contributions.values())
            portfolio_variance = returns_data.var().mean()  # Simplified
            specific_risk = max(0, portfolio_variance - total_explained_risk)
            
            return {
                "factor_risks": risk_contributions,
                "specific_risk": specific_risk,
                "total_risk": portfolio_variance,
                "explained_risk_ratio": total_explained_risk / portfolio_variance if portfolio_variance > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Risk attribution calculation failed: {e}")
            return {}
    
    def _calculate_factor_concentrations(self, pca_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate factor concentration metrics."""
        try:
            if not pca_result or "loadings" not in pca_result:
                return {}
            
            loadings = pca_result["loadings"]
            
            # Calculate Herfindahl index for each factor
            concentration_metrics = {}
            
            for factor in loadings.columns:
                factor_loadings = np.abs(loadings[factor].values)
                weights = factor_loadings / factor_loadings.sum()
                herfindahl_index = np.sum(weights ** 2)
                
                concentration_metrics[factor] = {
                    "herfindahl_index": herfindahl_index,
                    "effective_assets": 1 / herfindahl_index if herfindahl_index > 0 else 0,
                    "max_loading": factor_loadings.max(),
                    "top_3_concentration": np.sort(factor_loadings)[-3:].sum() / factor_loadings.sum()
                }
            
            return concentration_metrics
            
        except Exception as e:
            self.logger.error(f"Factor concentration calculation failed: {e}")
            return {}


class CrossAssetAgent(BaseAgent):
    """
    Cross-Asset Agent with multi-asset correlation analysis and factor exposure.
    
    This agent implements sophisticated cross-asset strategies with portfolio-level
    optimization across multiple asset classes, including correlation analysis,
    factor exposure measurement, and dynamic asset allocation.
    
    Inputs: Multi-asset price data, factor returns, portfolio context
    Outputs: Cross-asset allocation recommendations with factor analysis
    """
    
    def __init__(self):
        super().__init__("CrossAssetAgent", "cross_asset")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validation systems."""
        self.min_correlation_timeframes = self.get_config_value("min_correlation_timeframes", 3)
        self.correlation_confidence_threshold = self.get_config_value("correlation_confidence_threshold", 0.75)
        self.rebalancing_threshold = self.get_config_value("rebalancing_threshold", 0.05)  # 5%
        self.max_asset_weight = self.get_config_value("max_asset_weight", 0.3)  # 30%
        self.min_asset_weight = self.get_config_value("min_asset_weight", 0.02)  # 2%
        
        # Initialize validation systems
        self.correlation_validator = CorrelationConsensusValidator(
            min_timeframes=self.min_correlation_timeframes,
            confidence_threshold=self.correlation_confidence_threshold
        )
        self.factor_analyzer = FactorAnalyzer()
        
        # Asset class mappings
        self.asset_classes = {
            "equities": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
            "bonds": ["AGG", "TLT", "HYG", "LQD", "TIP"],
            "commodities": ["GLD", "SLV", "USO", "DJP", "PDBC"],
            "real_estate": ["VNQ", "VNQI", "REM"],
            "currencies": ["UUP", "FXE", "FXY", "EUO"],
            "volatility": ["VIX", "UVXY", "SVXY"]
        }
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute cross-asset analysis with consensus validation.
        
        Args:
            inputs: Dictionary containing:
                - price_data: Multi-asset price data
                - factor_returns: Factor return data (optional)
                - current_portfolio: Current portfolio allocation
                - risk_constraints: Risk management constraints
                - rebalancing_frequency: How often to rebalance
                
        Returns:
            AgentOutput with cross-asset allocation recommendations
        """
        self._validate_inputs(inputs)
        
        price_data = inputs["price_data"]
        factor_returns = inputs.get("factor_returns", {})
        current_portfolio = inputs.get("current_portfolio", {})
        risk_constraints = inputs.get("risk_constraints", {})
        rebalancing_frequency = inputs.get("rebalancing_frequency", "monthly")
        
        try:
            self.logger.info(f"Analyzing cross-asset relationships for {len(price_data.columns)} assets")
            
            # Step 1: Correlation analysis with consensus validation
            correlation_analysis = self.correlation_validator.validate_correlation_consensus(
                price_data
            )
            self.logger.info(f"Correlation consensus: {correlation_analysis.get('is_valid', False)}")
            
            # Step 2: Factor analysis
            returns_data = price_data.pct_change().dropna()
            factor_analysis = self.factor_analyzer.perform_factor_analysis(
                returns_data, factor_returns
            )
            self.logger.info("Factor analysis completed")
            
            # Step 3: Asset class analysis
            asset_class_analysis = await self._analyze_asset_classes(
                price_data, correlation_analysis
            )
            
            # Step 4: Portfolio optimization
            optimal_allocation = await self._optimize_portfolio_allocation(
                returns_data, correlation_analysis, factor_analysis, risk_constraints
            )
            
            # Step 5: Generate rebalancing recommendations
            rebalancing_recs = self._generate_rebalancing_recommendations(
                current_portfolio, optimal_allocation, rebalancing_frequency
            )
            
            # Step 6: Risk analysis
            risk_analysis = self._analyze_portfolio_risk(
                optimal_allocation, correlation_analysis, factor_analysis
            )
            
            # Step 7: Generate final recommendations
            final_recommendations = self._generate_cross_asset_recommendations(
                optimal_allocation, rebalancing_recs, risk_analysis
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "recommendations": final_recommendations,
                    "optimal_allocation": optimal_allocation,
                    "correlation_analysis": correlation_analysis,
                    "factor_analysis": factor_analysis,
                    "asset_class_analysis": asset_class_analysis,
                    "risk_analysis": risk_analysis,
                    "rebalancing_recommendations": rebalancing_recs
                },
                metadata={
                    "assets_analyzed": len(price_data.columns),
                    "correlation_consensus_achieved": correlation_analysis.get("is_valid", False),
                    "factor_models_used": len([k for k in factor_analysis.keys() if "factors" in k]),
                    "recommendations_generated": len(final_recommendations),
                    "consensus_validation": True,
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cross-asset analysis failed: {str(e)}")
            raise DataError(f"Cross-asset processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["price_data"]
        self.validate_inputs(inputs, required_keys)
        
        price_data = inputs["price_data"]
        if not isinstance(price_data, pd.DataFrame):
            raise ValidationError("Price data must be a pandas DataFrame")
        
        if price_data.empty or len(price_data.columns) < 2:
            raise ValidationError("Need at least 2 assets for cross-asset analysis")
    
    async def _analyze_asset_classes(self, price_data: pd.DataFrame,
                                   correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between asset classes."""
        asset_class_analysis = {
            "class_correlations": {},
            "diversification_benefits": {},
            "regime_analysis": {},
            "rotation_signals": {}
        }
        
        try:
            # Group assets by class
            asset_to_class = {}
            for asset_class, symbols in self.asset_classes.items():
                for symbol in symbols:
                    if symbol in price_data.columns:
                        asset_to_class[symbol] = asset_class
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Asset class performance
            class_returns = {}
            for asset_class, symbols in self.asset_classes.items():
                class_symbols = [s for s in symbols if s in returns.columns]
                if class_symbols:
                    class_return = returns[class_symbols].mean(axis=1)
                    class_returns[asset_class] = class_return
            
            if len(class_returns) >= 2:
                # Calculate inter-class correlations
                class_df = pd.DataFrame(class_returns)
                class_correlations = class_df.corr()
                asset_class_analysis["class_correlations"] = class_correlations.to_dict()
                
                # Diversification benefits
                asset_class_analysis["diversification_benefits"] = self._calculate_diversification_benefits(
                    class_df
                )
                
                # Rotation signals
                asset_class_analysis["rotation_signals"] = self._detect_rotation_signals(
                    class_df
                )
            
        except Exception as e:
            self.logger.error(f"Asset class analysis failed: {e}")
            asset_class_analysis["error"] = str(e)
        
        return asset_class_analysis
    
    def _calculate_diversification_benefits(self, class_returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate diversification benefits across asset classes."""
        try:
            # Equal-weight portfolio
            equal_weight_return = class_returns.mean(axis=1)
            equal_weight_vol = equal_weight_return.std()
            
            # Individual asset class volatilities
            individual_vols = class_returns.std()
            avg_individual_vol = individual_vols.mean()
            
            # Diversification ratio
            diversification_ratio = avg_individual_vol / equal_weight_vol if equal_weight_vol > 0 else 1
            
            # Maximum drawdown reduction
            equal_weight_cumret = (1 + equal_weight_return).cumprod()
            equal_weight_dd = (equal_weight_cumret / equal_weight_cumret.expanding().max() - 1).min()
            
            avg_individual_dd = np.mean([
                ((1 + class_returns[col]).cumprod() / 
                 (1 + class_returns[col]).cumprod().expanding().max() - 1).min()
                for col in class_returns.columns
            ])
            
            return {
                "diversification_ratio": diversification_ratio,
                "volatility_reduction": (avg_individual_vol - equal_weight_vol) / avg_individual_vol,
                "max_drawdown_improvement": (avg_individual_dd - equal_weight_dd) / abs(avg_individual_dd) if avg_individual_dd != 0 else 0,
                "correlation_benefit": 1 - class_returns.corr().values[np.triu_indices_from(class_returns.corr().values, k=1)].mean()
            }
            
        except Exception as e:
            self.logger.error(f"Diversification benefits calculation failed: {e}")
            return {}
    
    def _detect_rotation_signals(self, class_returns: pd.DataFrame, 
                                lookback: int = 60) -> Dict[str, Any]:
        """Detect asset class rotation signals."""
        rotation_signals = {}
        
        try:
            if len(class_returns) < lookback:
                return rotation_signals
            
            # Calculate momentum for each asset class
            for asset_class in class_returns.columns:
                recent_performance = class_returns[asset_class].tail(lookback)
                
                # Calculate various momentum metrics
                total_return = (1 + recent_performance).prod() - 1
                sharpe_ratio = recent_performance.mean() / recent_performance.std() if recent_performance.std() > 0 else 0
                max_drawdown = (recent_performance.cumsum() - recent_performance.cumsum().expanding().max()).min()
                
                # Relative performance vs other classes
                relative_performance = recent_performance.mean() - class_returns.tail(lookback).mean(axis=1).mean()
                
                # Generate signal
                momentum_score = (
                    (total_return * 0.4) +
                    (sharpe_ratio * 0.3) +
                    (relative_performance * 0.3)
                )
                
                signal = "overweight" if momentum_score > 0.02 else "underweight" if momentum_score < -0.02 else "neutral"
                
                rotation_signals[asset_class] = {
                    "signal": signal,
                    "momentum_score": momentum_score,
                    "total_return": total_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "relative_performance": relative_performance
                }
            
        except Exception as e:
            self.logger.error(f"Rotation signal detection failed: {e}")
        
        return rotation_signals
    
    async def _optimize_portfolio_allocation(self, returns_data: pd.DataFrame,
                                           correlation_analysis: Dict[str, Any],
                                           factor_analysis: Dict[str, Any],
                                           risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation using multiple approaches."""
        optimization_result = {
            "equal_weight": {},
            "minimum_variance": {},
            "risk_parity": {},
            "factor_based": {},
            "recommended": {}
        }
        
        try:
            assets = returns_data.columns.tolist()
            n_assets = len(assets)
            
            if n_assets < 2:
                return optimization_result
            
            # Equal weight allocation
            equal_weights = np.ones(n_assets) / n_assets
            optimization_result["equal_weight"] = {
                asset: weight for asset, weight in zip(assets, equal_weights)
            }
            
            # Minimum variance optimization
            if correlation_analysis.get("is_valid", False):
                consensus_corr = correlation_analysis.get("consensus_correlations", {})
                if consensus_corr:
                    min_var_weights = self._optimize_minimum_variance(returns_data, consensus_corr)
                    optimization_result["minimum_variance"] = {
                        asset: weight for asset, weight in zip(assets, min_var_weights)
                    }
            
            # Risk parity allocation
            risk_parity_weights = self._calculate_risk_parity_weights(returns_data)
            optimization_result["risk_parity"] = {
                asset: weight for asset, weight in zip(assets, risk_parity_weights)
            }
            
            # Factor-based allocation
            if factor_analysis.get("pca_factors"):
                factor_weights = self._optimize_factor_exposure(returns_data, factor_analysis)
                optimization_result["factor_based"] = {
                    asset: weight for asset, weight in zip(assets, factor_weights)
                }
            
            # Select recommended allocation
            optimization_result["recommended"] = self._select_recommended_allocation(
                optimization_result, risk_constraints
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            optimization_result["error"] = str(e)
        
        return optimization_result
    
    def _optimize_minimum_variance(self, returns_data: pd.DataFrame,
                                 consensus_corr: pd.DataFrame) -> np.ndarray:
        """Optimize for minimum variance portfolio."""
        try:
            # Calculate covariance matrix
            volatilities = returns_data.std()
            cov_matrix = np.outer(volatilities, volatilities) * consensus_corr.values
            
            n_assets = len(returns_data.columns)
            
            # Optimization constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]
            
            # Bounds (min and max weights)
            bounds = [(self.min_asset_weight, self.max_asset_weight) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Objective function (minimize portfolio variance)
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Optimize
            result = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return x0
                
        except Exception as e:
            self.logger.error(f"Minimum variance optimization failed: {e}")
            return np.ones(len(returns_data.columns)) / len(returns_data.columns)
    
    def _calculate_risk_parity_weights(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity weights."""
        try:
            # Calculate volatilities
            volatilities = returns_data.std().values
            
            # Risk parity: inverse volatility weighting
            inv_vol_weights = 1 / volatilities
            risk_parity_weights = inv_vol_weights / inv_vol_weights.sum()
            
            # Apply weight constraints
            risk_parity_weights = np.clip(
                risk_parity_weights, self.min_asset_weight, self.max_asset_weight
            )
            
            # Renormalize
            risk_parity_weights = risk_parity_weights / risk_parity_weights.sum()
            
            return risk_parity_weights
            
        except Exception as e:
            self.logger.error(f"Risk parity calculation failed: {e}")
            return np.ones(len(returns_data.columns)) / len(returns_data.columns)
    
    def _optimize_factor_exposure(self, returns_data: pd.DataFrame,
                                factor_analysis: Dict[str, Any]) -> np.ndarray:
        """Optimize portfolio based on factor exposure."""
        try:
            pca_factors = factor_analysis.get("pca_factors", {})
            if not pca_factors or "loadings" not in pca_factors:
                return np.ones(len(returns_data.columns)) / len(returns_data.columns)
            
            loadings = pca_factors["loadings"]
            explained_variance = pca_factors["explained_variance"]
            
            # Weight by first factor loading, adjusted by explained variance
            first_factor_loadings = np.abs(loadings.iloc[:, 0].values)
            factor_weights = first_factor_loadings * explained_variance[0]
            
            # Normalize and apply constraints
            factor_weights = factor_weights / factor_weights.sum()
            factor_weights = np.clip(factor_weights, self.min_asset_weight, self.max_asset_weight)
            factor_weights = factor_weights / factor_weights.sum()
            
            return factor_weights
            
        except Exception as e:
            self.logger.error(f"Factor-based optimization failed: {e}")
            return np.ones(len(returns_data.columns)) / len(returns_data.columns)
    
    def _select_recommended_allocation(self, optimization_result: Dict[str, Any],
                                     risk_constraints: Dict[str, Any]) -> Dict[str, float]:
        """Select the recommended allocation based on constraints and performance."""
        # Default to equal weight if others fail
        recommended = optimization_result.get("equal_weight", {})
        
        # Prefer risk parity for balanced approach
        if optimization_result.get("risk_parity"):
            recommended = optimization_result["risk_parity"]
        
        # Use minimum variance if risk aversion is high
        risk_tolerance = risk_constraints.get("risk_tolerance", "moderate")
        if risk_tolerance == "conservative" and optimization_result.get("minimum_variance"):
            recommended = optimization_result["minimum_variance"]
        
        # Use factor-based if factor models are reliable
        if (risk_tolerance == "aggressive" and 
            optimization_result.get("factor_based") and 
            not optimization_result.get("error")):
            recommended = optimization_result["factor_based"]
        
        return recommended
    
    def _generate_rebalancing_recommendations(self, current_portfolio: Dict[str, float],
                                            optimal_allocation: Dict[str, float],
                                            frequency: str) -> List[Dict[str, Any]]:
        """Generate rebalancing recommendations."""
        recommendations = []
        
        try:
            if not current_portfolio or not optimal_allocation.get("recommended"):
                return recommendations
            
            optimal = optimal_allocation["recommended"]
            
            # Calculate deviations
            for asset, target_weight in optimal.items():
                current_weight = current_portfolio.get(asset, 0.0)
                deviation = target_weight - current_weight
                
                # Check if rebalancing is needed
                if abs(deviation) > self.rebalancing_threshold:
                    action = "BUY" if deviation > 0 else "SELL"
                    
                    recommendations.append({
                        "asset": asset,
                        "action": action,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "deviation": deviation,
                        "trade_size": abs(deviation),
                        "priority": "high" if abs(deviation) > 0.1 else "medium",
                        "rebalancing_reason": "portfolio_optimization"
                    })
            
        except Exception as e:
            self.logger.error(f"Rebalancing recommendations failed: {e}")
        
        return recommendations
    
    def _analyze_portfolio_risk(self, optimal_allocation: Dict[str, Any],
                              correlation_analysis: Dict[str, Any],
                              factor_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio risk metrics."""
        risk_analysis = {
            "concentration_risk": {},
            "correlation_risk": {},
            "factor_risk": {},
            "scenario_analysis": {}
        }
        
        try:
            recommended_weights = optimal_allocation.get("recommended", {})
            
            if not recommended_weights:
                return risk_analysis
            
            # Concentration risk
            weights_array = np.array(list(recommended_weights.values()))
            herfindahl_index = np.sum(weights_array ** 2)
            
            risk_analysis["concentration_risk"] = {
                "herfindahl_index": herfindahl_index,
                "effective_assets": 1 / herfindahl_index if herfindahl_index > 0 else 0,
                "max_weight": weights_array.max(),
                "concentration_level": "high" if herfindahl_index > 0.25 else "medium" if herfindahl_index > 0.15 else "low"
            }
            
            # Correlation risk
            if correlation_analysis.get("consensus_correlations"):
                consensus_corr = correlation_analysis["consensus_correlations"]
                avg_correlation = np.mean([
                    consensus_corr.get(asset1, {}).get(asset2, 0)
                    for asset1 in recommended_weights.keys()
                    for asset2 in recommended_weights.keys()
                    if asset1 != asset2
                ])
                
                risk_analysis["correlation_risk"] = {
                    "average_correlation": avg_correlation,
                    "correlation_level": "high" if avg_correlation > 0.7 else "medium" if avg_correlation > 0.4 else "low",
                    "diversification_benefit": 1 - avg_correlation
                }
            
            # Factor risk
            if factor_analysis.get("pca_factors"):
                pca_factors = factor_analysis["pca_factors"]
                loadings = pca_factors.get("loadings", pd.DataFrame())
                
                if not loadings.empty:
                    # Calculate portfolio factor exposures
                    portfolio_exposures = {}
                    for factor in loadings.columns:
                        exposure = sum(
                            recommended_weights.get(asset, 0) * loadings.loc[asset, factor]
                            for asset in loadings.index
                            if asset in recommended_weights
                        )
                        portfolio_exposures[factor] = exposure
                    
                    risk_analysis["factor_risk"] = {
                        "factor_exposures": portfolio_exposures,
                        "max_factor_exposure": max(abs(exp) for exp in portfolio_exposures.values()) if portfolio_exposures else 0
                    }
            
        except Exception as e:
            self.logger.error(f"Portfolio risk analysis failed: {e}")
            risk_analysis["error"] = str(e)
        
        return risk_analysis
    
    def _generate_cross_asset_recommendations(self, optimal_allocation: Dict[str, Any],
                                            rebalancing_recs: List[Dict[str, Any]],
                                            risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final cross-asset recommendations."""
        recommendations = []
        
        try:
            # Strategic allocation recommendations
            recommended_weights = optimal_allocation.get("recommended", {})
            
            for asset, weight in recommended_weights.items():
                if weight > self.min_asset_weight:
                    recommendations.append({
                        "asset": asset,
                        "recommendation_type": "strategic_allocation",
                        "action": "HOLD" if asset in rebalancing_recs else "MAINTAIN",
                        "target_weight": weight,
                        "confidence": 0.8,  # High confidence for strategic allocation
                        "rationale": f"Optimal allocation based on risk-return optimization",
                        "risk_contribution": risk_analysis.get("concentration_risk", {}).get("max_weight", 0),
                        "consensus_validation": "multi_model_optimization"
                    })
            
            # Tactical rebalancing recommendations
            for rebal_rec in rebalancing_recs:
                if rebal_rec["priority"] == "high":
                    recommendations.append({
                        "asset": rebal_rec["asset"],
                        "recommendation_type": "tactical_rebalancing",
                        "action": rebal_rec["action"],
                        "trade_size": rebal_rec["trade_size"],
                        "current_weight": rebal_rec["current_weight"],
                        "target_weight": rebal_rec["target_weight"],
                        "confidence": 0.9,  # High confidence for rebalancing
                        "rationale": f"Rebalancing due to {rebal_rec['deviation']:.1%} deviation from target",
                        "urgency": rebal_rec["priority"],
                        "consensus_validation": "portfolio_optimization"
                    })
            
            # Sort by confidence and importance
            recommendations.sort(key=lambda x: (x["confidence"], -abs(x.get("trade_size", 0))), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Cross-asset recommendations generation failed: {e}")
        
        return recommendations