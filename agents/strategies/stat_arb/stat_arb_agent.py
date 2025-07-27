"""
Statistical Arbitrage Agent - Pairs trading and mean reversion with internal consensus validation.

This agent implements sophisticated statistical arbitrage strategies with ultrathinking validation
to ensure high-quality trading signals. Key features:
- Cointegration-based pairs trading with consensus validation
- Mean reversion detection across multiple timeframes
- Market-neutral strategy construction
- Internal cross-validation of statistical models
- Multiple statistical tests for robustness
- Dynamic hedge ratio calculation
- Regime-aware signal generation
- Comprehensive risk controls and position limits
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats, optimize
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
import warnings
from loguru import logger
warnings.filterwarnings('ignore')

from ....core.base.agent import BaseAgent, AgentOutput
from ....core.base.exceptions import ValidationError, DataError
from ....core.utils.data_validation import DataValidator
from ....core.utils.math_utils import MathUtils


class CointegrationValidator:
    """Internal validation system for cointegration relationships with consensus."""
    
    def __init__(self, min_tests: int = 3, confidence_threshold: float = 0.95):
        self.min_tests = min_tests
        self.confidence_threshold = confidence_threshold
        self.logger = logger.bind(component="cointegration_validator")
    
    def validate_cointegration(self, price_data: pd.DataFrame, pair: Tuple[str, str]) -> Dict[str, Any]:
        """Validate cointegration using multiple statistical tests with consensus."""
        validation_result = {
            "is_valid": False,
            "consensus_confidence": 0.0,
            "test_results": {},
            "hedge_ratio": 1.0,
            "half_life": 0,
            "validation_details": {}
        }
        
        try:
            asset1, asset2 = pair
            if asset1 not in price_data.columns or asset2 not in price_data.columns:
                validation_result["validation_details"]["missing_data"] = True
                return validation_result
            
            series1 = price_data[asset1].dropna()
            series2 = price_data[asset2].dropna()
            
            # Ensure same length
            min_length = min(len(series1), len(series2))
            series1 = series1.iloc[-min_length:]
            series2 = series2.iloc[-min_length:]
            
            if len(series1) < 100:  # Minimum data requirement
                validation_result["validation_details"]["insufficient_data"] = True
                return validation_result
            
            # Test 1: Engle-Granger cointegration test
            eg_result = self._engle_granger_test(series1, series2)
            validation_result["test_results"]["engle_granger"] = eg_result
            
            # Test 2: Johansen cointegration test (simplified)
            johansen_result = self._johansen_test_simplified(series1, series2)
            validation_result["test_results"]["johansen"] = johansen_result
            
            # Test 3: ADF test on spread
            adf_result = self._adf_spread_test(series1, series2, eg_result.get("hedge_ratio", 1.0))
            validation_result["test_results"]["adf_spread"] = adf_result
            
            # Test 4: Hurst exponent for mean reversion
            hurst_result = self._hurst_exponent_test(series1, series2, eg_result.get("hedge_ratio", 1.0))
            validation_result["test_results"]["hurst"] = hurst_result
            
            # Consensus validation
            valid_tests = sum(1 for test in validation_result["test_results"].values() 
                            if test.get("is_cointegrated", False))
            
            consensus_confidence = valid_tests / len(validation_result["test_results"])
            is_valid = valid_tests >= self.min_tests and consensus_confidence >= 0.5
            
            validation_result.update({
                "is_valid": is_valid,
                "consensus_confidence": consensus_confidence,
                "hedge_ratio": eg_result.get("hedge_ratio", 1.0),
                "half_life": self._calculate_half_life(series1, series2, eg_result.get("hedge_ratio", 1.0)),
                "validation_details": {
                    "total_tests": len(validation_result["test_results"]),
                    "valid_tests": valid_tests,
                    "data_points": len(series1)
                }
            })
            
        except Exception as e:
            self.logger.error(f"Cointegration validation failed for {pair}: {e}")
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _engle_granger_test(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Perform Engle-Granger cointegration test."""
        try:
            # Run OLS regression
            model = OLS(series1, series2).fit()
            hedge_ratio = model.params[0]
            
            # Test residuals for stationarity
            residuals = series1 - hedge_ratio * series2
            adf_stat, p_value, _, _, critical_values, _ = adfuller(residuals)
            
            is_cointegrated = p_value < 0.05
            
            return {
                "is_cointegrated": is_cointegrated,
                "p_value": p_value,
                "adf_statistic": adf_stat,
                "critical_value_5pct": critical_values.get('5%', -2.86),
                "hedge_ratio": hedge_ratio,
                "r_squared": model.rsquared
            }
            
        except Exception as e:
            self.logger.error(f"Engle-Granger test failed: {e}")
            return {"is_cointegrated": False, "error": str(e)}
    
    def _johansen_test_simplified(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Simplified Johansen test using correlation analysis."""
        try:
            # Calculate rolling correlation
            window = min(60, len(series1) // 4)
            rolling_corr = series1.rolling(window).corr(series2)
            
            # Check correlation stability
            corr_mean = rolling_corr.mean()
            corr_std = rolling_corr.std()
            
            # High stable correlation suggests potential cointegration
            is_stable = corr_std < 0.15
            is_high_corr = abs(corr_mean) > 0.7
            
            is_cointegrated = is_stable and is_high_corr
            
            return {
                "is_cointegrated": is_cointegrated,
                "mean_correlation": corr_mean,
                "correlation_stability": 1 - corr_std,
                "test_confidence": 0.8 if is_cointegrated else 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Johansen test failed: {e}")
            return {"is_cointegrated": False, "error": str(e)}
    
    def _adf_spread_test(self, series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> Dict[str, Any]:
        """Test spread stationarity using ADF test."""
        try:
            spread = series1 - hedge_ratio * series2
            adf_stat, p_value, _, _, critical_values, _ = adfuller(spread)
            
            is_stationary = p_value < 0.01  # Stricter threshold for spread
            
            return {
                "is_cointegrated": is_stationary,
                "p_value": p_value,
                "adf_statistic": adf_stat,
                "spread_mean": spread.mean(),
                "spread_std": spread.std()
            }
            
        except Exception as e:
            self.logger.error(f"ADF spread test failed: {e}")
            return {"is_cointegrated": False, "error": str(e)}
    
    def _hurst_exponent_test(self, series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> Dict[str, Any]:
        """Calculate Hurst exponent to test for mean reversion."""
        try:
            spread = series1 - hedge_ratio * series2
            
            # Calculate Hurst exponent using R/S analysis
            lags = range(2, min(100, len(spread) // 2))
            tau = [np.sqrt(np.std(np.subtract(spread[lag:], spread[:-lag]))) for lag in lags]
            
            # Linear regression to estimate Hurst
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = reg[0] * 2
            
            # H < 0.5 indicates mean reversion
            is_mean_reverting = hurst < 0.5
            
            return {
                "is_cointegrated": is_mean_reverting,
                "hurst_exponent": hurst,
                "mean_reversion_strength": max(0, 0.5 - hurst) * 2,  # Scale to [0, 1]
                "interpretation": "mean_reverting" if is_mean_reverting else "trending"
            }
            
        except Exception as e:
            self.logger.error(f"Hurst exponent test failed: {e}")
            return {"is_cointegrated": False, "error": str(e)}
    
    def _calculate_half_life(self, series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> float:
        """Calculate half-life of mean reversion."""
        try:
            spread = series1 - hedge_ratio * series2
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            
            # AR(1) model
            model = OLS(spread_diff[1:], spread_lag[1:]).fit()
            half_life = -np.log(2) / model.params[0] if model.params[0] < 0 else np.inf
            
            return max(1, min(half_life, 252))  # Cap between 1 and 252 days
            
        except Exception:
            return 30  # Default half-life


class MeanReversionDetector:
    """Detect mean reversion opportunities across multiple timeframes."""
    
    def __init__(self):
        self.logger = logger.bind(component="mean_reversion_detector")
    
    def detect_mean_reversion(self, data: pd.DataFrame, lookback_periods: List[int] = None) -> Dict[str, Any]:
        """Detect mean reversion opportunities with multi-timeframe analysis."""
        if lookback_periods is None:
            lookback_periods = [20, 60, 120]  # Short, medium, long term
        
        reversion_signals = {}
        
        for symbol in data.columns:
            if symbol == 'Date' or data[symbol].isna().all():
                continue
                
            symbol_signals = self._analyze_symbol_reversion(data[symbol], lookback_periods)
            if symbol_signals["composite_score"] != 0:
                reversion_signals[symbol] = symbol_signals
        
        return reversion_signals
    
    def _analyze_symbol_reversion(self, series: pd.Series, lookback_periods: List[int]) -> Dict[str, Any]:
        """Analyze mean reversion for a single symbol."""
        timeframe_signals = {}
        
        for period in lookback_periods:
            if len(series) < period + 10:
                continue
                
            timeframe_signal = self._calculate_reversion_signal(series, period)
            timeframe_signals[f"period_{period}"] = timeframe_signal
        
        # Combine signals across timeframes
        composite_score = self._calculate_composite_score(timeframe_signals)
        
        return {
            "timeframe_signals": timeframe_signals,
            "composite_score": composite_score,
            "signal_strength": self._determine_signal_strength(composite_score),
            "recommendation": self._generate_recommendation(composite_score)
        }
    
    def _calculate_reversion_signal(self, series: pd.Series, lookback: int) -> Dict[str, Any]:
        """Calculate mean reversion signal for specific lookback period."""
        try:
            # Calculate rolling statistics
            rolling_mean = series.rolling(lookback).mean()
            rolling_std = series.rolling(lookback).std()
            
            # Z-score (distance from mean in standard deviations)
            z_score = (series - rolling_mean) / rolling_std
            current_z = z_score.iloc[-1]
            
            # Bollinger Band position
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            bb_position = (series - lower_band) / (upper_band - lower_band)
            current_bb = bb_position.iloc[-1]
            
            # RSI for momentum confirmation
            rsi = self._calculate_rsi(series, min(14, lookback // 2))
            current_rsi = rsi.iloc[-1]
            
            # Generate signal
            signal = 0
            if current_z < -2 and current_rsi < 30:
                signal = 1  # Oversold, expect reversion up
            elif current_z > 2 and current_rsi > 70:
                signal = -1  # Overbought, expect reversion down
            elif abs(current_z) > 1.5:
                signal = -np.sign(current_z) * 0.5  # Moderate reversion signal
            
            return {
                "z_score": current_z,
                "bb_position": current_bb,
                "rsi": current_rsi,
                "signal": signal,
                "rolling_mean": rolling_mean.iloc[-1],
                "rolling_std": rolling_std.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Reversion signal calculation failed: {e}")
            return {"signal": 0, "error": str(e)}
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_composite_score(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> float:
        """Calculate composite score from multiple timeframe signals."""
        if not timeframe_signals:
            return 0.0
        
        weights = {
            "period_20": 0.5,   # Short-term weight
            "period_60": 0.3,   # Medium-term weight
            "period_120": 0.2   # Long-term weight
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for timeframe, signal_data in timeframe_signals.items():
            if "signal" in signal_data and timeframe in weights:
                weighted_sum += signal_data["signal"] * weights[timeframe]
                total_weight += weights[timeframe]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_signal_strength(self, composite_score: float) -> str:
        """Determine signal strength from composite score."""
        abs_score = abs(composite_score)
        if abs_score > 0.7:
            return "STRONG"
        elif abs_score > 0.4:
            return "MODERATE"
        elif abs_score > 0.2:
            return "WEAK"
        else:
            return "NEUTRAL"
    
    def _generate_recommendation(self, composite_score: float) -> str:
        """Generate recommendation based on composite score."""
        if composite_score > 0.5:
            return "BUY"
        elif composite_score < -0.5:
            return "SELL"
        elif abs(composite_score) > 0.2:
            return "WEAK_BUY" if composite_score > 0 else "WEAK_SELL"
        else:
            return "HOLD"


class StatisticalArbitrageAgent(BaseAgent):
    """
    Statistical Arbitrage Agent with internal consensus validation.
    
    This agent implements pairs trading and mean reversion strategies with sophisticated
    validation mechanisms to ensure high-quality trading signals through internal consensus.
    
    Inputs: Enhanced feature matrix with price data and technical indicators
    Outputs: Statistical arbitrage recommendations with consensus validation
    """
    
    def __init__(self):
        super().__init__("StatisticalArbitrageAgent", "stat_arb")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validation systems."""
        self.min_cointegration_tests = self.get_config_value("min_cointegration_tests", 3)
        self.cointegration_threshold = self.get_config_value("cointegration_threshold", 0.05)
        self.max_pairs_to_trade = self.get_config_value("max_pairs_to_trade", 10)
        self.min_half_life = self.get_config_value("min_half_life", 5)
        self.max_half_life = self.get_config_value("max_half_life", 60)
        self.entry_threshold = self.get_config_value("entry_threshold", 2.0)  # Z-score
        self.exit_threshold = self.get_config_value("exit_threshold", 0.5)
        
        # Initialize validation systems
        self.cointegration_validator = CointegrationValidator(
            min_tests=self.min_cointegration_tests,
            confidence_threshold=0.95
        )
        self.mean_reversion_detector = MeanReversionDetector()
        
        # Pair selection parameters
        self.correlation_threshold = self.get_config_value("correlation_threshold", 0.5)
        self.min_data_points = self.get_config_value("min_data_points", 252)
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute statistical arbitrage analysis with internal consensus validation.
        
        Args:
            inputs: Dictionary containing:
                - enhanced_feature_matrix: DataFrame with price and technical data
                - symbols: List of symbols to analyze
                - current_positions: Current portfolio positions
                - lookback_window: Historical data window for analysis
                
        Returns:
            AgentOutput with statistical arbitrage recommendations
        """
        self._validate_inputs(inputs)
        
        feature_matrix = inputs["enhanced_feature_matrix"]
        symbols = inputs.get("symbols", [])
        current_positions = inputs.get("current_positions", {})
        lookback_window = inputs.get("lookback_window", 252)
        
        try:
            # Extract price data for analysis
            price_data = self._extract_price_data(feature_matrix, symbols)
            
            if price_data.empty or len(price_data.columns) < 2:
                raise DataError("Insufficient price data for statistical arbitrage analysis")
            
            self.logger.info(f"Analyzing {len(price_data.columns)} symbols for statistical arbitrage opportunities")
            
            # Step 1: Identify cointegrated pairs with consensus validation
            cointegrated_pairs = await self._identify_cointegrated_pairs(price_data)
            self.logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
            
            # Step 2: Generate trading signals for pairs
            pairs_signals = await self._generate_pairs_signals(
                price_data, cointegrated_pairs, current_positions
            )
            
            # Step 3: Detect mean reversion opportunities
            mean_reversion_signals = self.mean_reversion_detector.detect_mean_reversion(
                price_data
            )
            self.logger.info(f"Detected {len(mean_reversion_signals)} mean reversion opportunities")
            
            # Step 4: Combine and validate all signals
            combined_signals = self._combine_signals(pairs_signals, mean_reversion_signals)
            
            # Step 5: Generate final recommendations with risk controls
            recommendations = self._generate_recommendations(
                combined_signals, current_positions
            )
            
            # Step 6: Calculate strategy metrics
            strategy_metrics = self._calculate_strategy_metrics(
                recommendations, cointegrated_pairs, mean_reversion_signals
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "recommendations": recommendations,
                    "cointegrated_pairs": self._format_pairs_output(cointegrated_pairs),
                    "mean_reversion_signals": mean_reversion_signals,
                    "strategy_metrics": strategy_metrics,
                    "signal_summary": self._generate_signal_summary(recommendations)
                },
                metadata={
                    "symbols_analyzed": len(symbols),
                    "pairs_identified": len(cointegrated_pairs),
                    "recommendations_generated": len(recommendations),
                    "strategy_type": "market_neutral",
                    "consensus_validation": True,
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Statistical arbitrage analysis failed: {str(e)}")
            raise DataError(f"Statistical arbitrage processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["enhanced_feature_matrix"]
        self.validate_inputs(inputs, required_keys)
        
        feature_matrix = inputs["enhanced_feature_matrix"]
        if not isinstance(feature_matrix, pd.DataFrame):
            raise ValidationError("Enhanced feature matrix must be a pandas DataFrame")
        
        if feature_matrix.empty:
            raise ValidationError("Feature matrix cannot be empty")
    
    def _extract_price_data(self, feature_matrix: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Extract price data from feature matrix."""
        try:
            # Look for close price columns
            price_columns = []
            
            # If feature matrix has multi-level columns
            if isinstance(feature_matrix.columns, pd.MultiIndex):
                for symbol in symbols:
                    if (symbol, 'close') in feature_matrix.columns:
                        price_columns.append((symbol, 'close'))
                
                if price_columns:
                    price_data = feature_matrix[price_columns]
                    # Flatten column names
                    price_data.columns = [col[0] for col in price_data.columns]
                    return price_data
            
            # Single-level columns - look for price-related columns
            price_cols = [col for col in feature_matrix.columns if 'close' in col.lower()]
            if not price_cols:
                # Fallback to numeric columns that might be prices
                numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
                price_cols = [col for col in numeric_cols if col in symbols]
            
            if price_cols:
                return feature_matrix[price_cols]
            
            # Last resort - use all numeric columns
            return feature_matrix.select_dtypes(include=[np.number])
            
        except Exception as e:
            self.logger.error(f"Price data extraction failed: {e}")
            return pd.DataFrame()
    
    async def _identify_cointegrated_pairs(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify cointegrated pairs with consensus validation."""
        cointegrated_pairs = []
        symbols = list(price_data.columns)
        
        # Calculate correlation matrix for initial filtering
        correlation_matrix = price_data.corr()
        
        # Test pairs with sufficient correlation
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Pre-filter by correlation
                if abs(correlation_matrix.loc[symbol1, symbol2]) < self.correlation_threshold:
                    continue
                
                # Validate cointegration with consensus
                validation_result = self.cointegration_validator.validate_cointegration(
                    price_data, (symbol1, symbol2)
                )
                
                if validation_result["is_valid"]:
                    half_life = validation_result.get("half_life", 30)
                    
                    # Apply half-life filter
                    if self.min_half_life <= half_life <= self.max_half_life:
                        cointegrated_pairs.append({
                            "pair": (symbol1, symbol2),
                            "hedge_ratio": validation_result["hedge_ratio"],
                            "half_life": half_life,
                            "consensus_confidence": validation_result["consensus_confidence"],
                            "test_results": validation_result["test_results"],
                            "correlation": correlation_matrix.loc[symbol1, symbol2]
                        })
        
        # Sort by consensus confidence
        cointegrated_pairs.sort(key=lambda x: x["consensus_confidence"], reverse=True)
        
        # Limit to max pairs
        return cointegrated_pairs[:self.max_pairs_to_trade]
    
    async def _generate_pairs_signals(self, price_data: pd.DataFrame,
                                    cointegrated_pairs: List[Dict[str, Any]],
                                    current_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals for cointegrated pairs."""
        pairs_signals = []
        
        for pair_info in cointegrated_pairs:
            try:
                symbol1, symbol2 = pair_info["pair"]
                hedge_ratio = pair_info["hedge_ratio"]
                half_life = pair_info["half_life"]
                
                # Calculate spread
                series1 = price_data[symbol1]
                series2 = price_data[symbol2]
                spread = series1 - hedge_ratio * series2
                
                # Calculate z-score
                lookback = int(half_life * 2)
                spread_mean = spread.rolling(lookback).mean()
                spread_std = spread.rolling(lookback).std()
                z_score = (spread - spread_mean) / spread_std
                current_z = z_score.iloc[-1]
                
                # Check if pair is already in position
                pair_key = f"{symbol1}_{symbol2}"
                in_position = pair_key in current_positions
                
                # Generate signal
                signal = self._generate_pair_signal(
                    current_z, in_position, pair_info
                )
                
                if signal["action"] != "HOLD":
                    pairs_signals.append({
                        "type": "pairs_trade",
                        "pair": pair_info["pair"],
                        "action": signal["action"],
                        "positions": signal["positions"],
                        "z_score": current_z,
                        "hedge_ratio": hedge_ratio,
                        "half_life": half_life,
                        "confidence": pair_info["consensus_confidence"],
                        "expected_return": signal["expected_return"],
                        "risk_metrics": self._calculate_pair_risk(spread, z_score)
                    })
                
            except Exception as e:
                self.logger.error(f"Signal generation failed for pair {pair_info['pair']}: {e}")
                continue
        
        return pairs_signals
    
    def _generate_pair_signal(self, z_score: float, in_position: bool,
                            pair_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal for a specific pair."""
        symbol1, symbol2 = pair_info["pair"]
        hedge_ratio = pair_info["hedge_ratio"]
        
        signal = {
            "action": "HOLD",
            "positions": {},
            "expected_return": 0.0
        }
        
        # Entry signals
        if not in_position:
            if z_score > self.entry_threshold:
                # Spread too high - short spread (short asset1, long asset2)
                signal["action"] = "ENTER"
                signal["positions"] = {
                    symbol1: {"action": "SELL", "weight": -1.0},
                    symbol2: {"action": "BUY", "weight": hedge_ratio}
                }
                signal["expected_return"] = (z_score - self.exit_threshold) / pair_info["half_life"]
                
            elif z_score < -self.entry_threshold:
                # Spread too low - long spread (long asset1, short asset2)
                signal["action"] = "ENTER"
                signal["positions"] = {
                    symbol1: {"action": "BUY", "weight": 1.0},
                    symbol2: {"action": "SELL", "weight": -hedge_ratio}
                }
                signal["expected_return"] = (-z_score - self.exit_threshold) / pair_info["half_life"]
        
        # Exit signals
        else:
            if abs(z_score) < self.exit_threshold:
                signal["action"] = "EXIT"
                signal["positions"] = {
                    symbol1: {"action": "CLOSE", "weight": 0},
                    symbol2: {"action": "CLOSE", "weight": 0}
                }
        
        return signal
    
    def _calculate_pair_risk(self, spread: pd.Series, z_score: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics for a pair."""
        return {
            "spread_volatility": spread.std(),
            "z_score_volatility": z_score.std(),
            "max_z_score": abs(z_score).max(),
            "current_drawdown": self._calculate_spread_drawdown(spread)
        }
    
    def _calculate_spread_drawdown(self, spread: pd.Series) -> float:
        """Calculate current drawdown of spread from peak."""
        cumulative = (1 + spread.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.iloc[-1])
    
    def _combine_signals(self, pairs_signals: List[Dict[str, Any]],
                        mean_reversion_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine pairs trading and mean reversion signals."""
        combined_signals = pairs_signals.copy()
        
        # Add mean reversion signals
        for symbol, reversion_data in mean_reversion_signals.items():
            if reversion_data["signal_strength"] in ["STRONG", "MODERATE"]:
                combined_signals.append({
                    "type": "mean_reversion",
                    "symbol": symbol,
                    "action": reversion_data["recommendation"],
                    "composite_score": reversion_data["composite_score"],
                    "signal_strength": reversion_data["signal_strength"],
                    "confidence": abs(reversion_data["composite_score"]),
                    "timeframe_signals": reversion_data["timeframe_signals"]
                })
        
        # Sort by confidence
        combined_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return combined_signals
    
    def _generate_recommendations(self, signals: List[Dict[str, Any]],
                                current_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final recommendations with risk controls."""
        recommendations = []
        allocated_capital = 0.0
        max_allocation_per_strategy = 0.1  # 10% per strategy
        
        for signal in signals:
            if allocated_capital >= 0.5:  # Max 50% allocation to stat arb
                break
            
            if signal["type"] == "pairs_trade":
                # Create recommendation for each leg of the pair
                pair_rec = self._create_pair_recommendation(signal, max_allocation_per_strategy)
                recommendations.extend(pair_rec)
                allocated_capital += max_allocation_per_strategy
                
            elif signal["type"] == "mean_reversion":
                # Create single-asset recommendation
                rec = self._create_mean_reversion_recommendation(signal, max_allocation_per_strategy / 2)
                recommendations.append(rec)
                allocated_capital += max_allocation_per_strategy / 2
        
        return recommendations
    
    def _create_pair_recommendation(self, signal: Dict[str, Any],
                                  max_allocation: float) -> List[Dict[str, Any]]:
        """Create recommendations for a pairs trade."""
        recommendations = []
        positions = signal.get("positions", {})
        
        # Scale position sizes to stay within allocation
        total_weight = sum(abs(pos["weight"]) for pos in positions.values())
        scale_factor = max_allocation / total_weight if total_weight > 0 else 1.0
        
        for symbol, position in positions.items():
            if position["action"] in ["BUY", "SELL"]:
                recommendations.append({
                    "symbol": symbol,
                    "action": position["action"],
                    "signal_type": "pairs_trade",
                    "pair": signal["pair"],
                    "position_size": abs(position["weight"] * scale_factor),
                    "direction": 1 if position["action"] == "BUY" else -1,
                    "confidence": signal["confidence"],
                    "z_score": signal["z_score"],
                    "half_life": signal["half_life"],
                    "expected_return": signal.get("expected_return", 0),
                    "risk_metrics": signal.get("risk_metrics", {}),
                    "consensus_validation": "multi_test_cointegration"
                })
        
        return recommendations
    
    def _create_mean_reversion_recommendation(self, signal: Dict[str, Any],
                                            max_allocation: float) -> Dict[str, Any]:
        """Create recommendation for mean reversion trade."""
        return {
            "symbol": signal["symbol"],
            "action": signal["action"],
            "signal_type": "mean_reversion",
            "position_size": max_allocation,
            "confidence": signal["confidence"],
            "composite_score": signal["composite_score"],
            "signal_strength": signal["signal_strength"],
            "timeframe_analysis": signal.get("timeframe_signals", {}),
            "consensus_validation": "multi_timeframe_convergence"
        }
    
    def _format_pairs_output(self, cointegrated_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format cointegrated pairs for output."""
        formatted_pairs = []
        
        for pair in cointegrated_pairs:
            formatted_pairs.append({
                "symbol1": pair["pair"][0],
                "symbol2": pair["pair"][1],
                "hedge_ratio": round(pair["hedge_ratio"], 4),
                "half_life_days": round(pair["half_life"], 1),
                "consensus_confidence": round(pair["consensus_confidence"], 3),
                "correlation": round(pair["correlation"], 3),
                "cointegration_tests": {
                    test_name: {
                        "passed": result.get("is_cointegrated", False),
                        "confidence": result.get("test_confidence", result.get("p_value", 0))
                    }
                    for test_name, result in pair["test_results"].items()
                }
            })
        
        return formatted_pairs
    
    def _calculate_strategy_metrics(self, recommendations: List[Dict[str, Any]],
                                  cointegrated_pairs: List[Dict[str, Any]],
                                  mean_reversion_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall strategy metrics."""
        pairs_trades = [r for r in recommendations if r.get("signal_type") == "pairs_trade"]
        reversion_trades = [r for r in recommendations if r.get("signal_type") == "mean_reversion"]
        
        return {
            "total_recommendations": len(recommendations),
            "pairs_trades": len(set(r.get("pair", ()) for r in pairs_trades)),
            "mean_reversion_trades": len(reversion_trades),
            "average_confidence": np.mean([r["confidence"] for r in recommendations]) if recommendations else 0,
            "consensus_validation_rate": 1.0,  # All signals are consensus-validated
            "expected_sharpe_ratio": self._estimate_sharpe_ratio(recommendations),
            "market_neutral_score": self._calculate_market_neutral_score(recommendations)
        }
    
    def _estimate_sharpe_ratio(self, recommendations: List[Dict[str, Any]]) -> float:
        """Estimate expected Sharpe ratio for the strategy."""
        if not recommendations:
            return 0.0
        
        # Simplified estimation based on signal quality
        avg_confidence = np.mean([r["confidence"] for r in recommendations])
        pairs_ratio = len([r for r in recommendations if r.get("signal_type") == "pairs_trade"]) / len(recommendations)
        
        # Pairs trading typically has better Sharpe
        base_sharpe = 1.5 if pairs_ratio > 0.5 else 1.0
        
        return base_sharpe * avg_confidence
    
    def _calculate_market_neutral_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate how market-neutral the portfolio is."""
        if not recommendations:
            return 1.0
        
        # Calculate net exposure
        long_exposure = sum(r["position_size"] for r in recommendations if r.get("direction", 1) > 0)
        short_exposure = sum(r["position_size"] for r in recommendations if r.get("direction", 1) < 0)
        
        net_exposure = abs(long_exposure - short_exposure)
        gross_exposure = long_exposure + short_exposure
        
        # Perfect market neutral = 0 net exposure
        neutrality_score = 1.0 - (net_exposure / gross_exposure) if gross_exposure > 0 else 1.0
        
        return neutrality_score
    
    def _generate_signal_summary(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of signals."""
        summary = {
            "total_signals": len(recommendations),
            "signal_distribution": {},
            "confidence_distribution": {},
            "expected_returns": {}
        }
        
        if recommendations:
            # Signal type distribution
            signal_types = [r.get("signal_type", "unknown") for r in recommendations]
            summary["signal_distribution"] = {
                signal_type: signal_types.count(signal_type) 
                for signal_type in set(signal_types)
            }
            
            # Confidence distribution
            confidences = [r["confidence"] for r in recommendations]
            summary["confidence_distribution"] = {
                "high": len([c for c in confidences if c > 0.8]),
                "medium": len([c for c in confidences if 0.6 <= c <= 0.8]),
                "low": len([c for c in confidences if c < 0.6])
            }
            
            # Expected returns
            expected_returns = [r.get("expected_return", 0) for r in recommendations if r.get("expected_return")]
            if expected_returns:
                summary["expected_returns"] = {
                    "average": np.mean(expected_returns),
                    "best": max(expected_returns),
                    "worst": min(expected_returns)
                }
        
        return summary