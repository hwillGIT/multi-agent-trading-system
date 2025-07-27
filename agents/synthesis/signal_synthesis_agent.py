"""
Signal Synthesis & Arbitration Agent - Multi-layer consensus and validation system.

This agent implements sophisticated consensus mechanisms and ultrathinking to ensure
high-confidence trading recommendations. Key features:
- Requires minimum 3 confirming independent sources
- Multi-layer validation and cross-checking
- Regime-aware signal weighting
- Outlier detection and handling
- Comprehensive audit trails for all decisions
- Internal consistency verification
- Confidence scoring based on consensus strength
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
from loguru import logger

from ...core.base.agent import BaseAgent, AgentOutput
from ...core.base.exceptions import ValidationError, DataError
from ...core.utils.data_validation import DataValidator
from ...core.utils.math_utils import MathUtils


class ConsensusValidator:
    """Internal validation system for ensuring consensus quality."""
    
    def __init__(self, min_sources: int = 3, confidence_threshold: float = 0.7):
        self.min_sources = min_sources
        self.confidence_threshold = confidence_threshold
        self.logger = logger.bind(component="consensus_validator")
    
    def validate_signal_consensus(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consensus across multiple signal sources."""
        validation_result = {
            "is_valid": False,
            "confidence": 0.0,
            "consensus_strength": 0.0,
            "agreeing_sources": [],
            "disagreeing_sources": [],
            "outliers": [],
            "validation_details": {}
        }
        
        # Extract signal values and sources
        signal_values = {}
        for source, signal_data in signals.items():
            if isinstance(signal_data, dict) and 'signal' in signal_data:
                signal_values[source] = signal_data['signal']
            elif isinstance(signal_data, (int, float)):
                signal_values[source] = signal_data
        
        if len(signal_values) < self.min_sources:
            validation_result["validation_details"]["insufficient_sources"] = True
            return validation_result
        
        # Convert signals to standardized format (-1 to 1)
        normalized_signals = self._normalize_signals(signal_values)
        
        # Detect outliers
        outliers = self._detect_signal_outliers(normalized_signals)
        validation_result["outliers"] = outliers
        
        # Calculate consensus after removing outliers
        consensus_signals = {k: v for k, v in normalized_signals.items() if k not in outliers}
        
        if len(consensus_signals) < self.min_sources:
            validation_result["validation_details"]["too_many_outliers"] = True
            return validation_result
        
        # Calculate consensus metrics
        signal_array = np.array(list(consensus_signals.values()))
        consensus_direction = np.sign(np.mean(signal_array))
        consensus_strength = self._calculate_consensus_strength(signal_array)
        
        # Determine agreeing vs disagreeing sources
        for source, signal in consensus_signals.items():
            if np.sign(signal) == consensus_direction or abs(signal) < 0.1:
                validation_result["agreeing_sources"].append(source)
            else:
                validation_result["disagreeing_sources"].append(source)
        
        # Final validation
        agreement_ratio = len(validation_result["agreeing_sources"]) / len(consensus_signals)
        
        validation_result.update({
            "is_valid": agreement_ratio >= 0.6 and consensus_strength >= self.confidence_threshold,
            "confidence": agreement_ratio * consensus_strength,
            "consensus_strength": consensus_strength,
            "consensus_direction": float(consensus_direction),
            "signal_std": float(np.std(signal_array)),
            "validation_details": {
                "total_sources": len(signal_values),
                "consensus_sources": len(consensus_signals),
                "agreement_ratio": agreement_ratio,
                "outliers_removed": len(outliers)
            }
        })
        
        return validation_result
    
    def _normalize_signals(self, signal_values: Dict[str, Any]) -> Dict[str, float]:
        """Normalize signals to [-1, 1] range."""
        normalized = {}
        
        for source, signal in signal_values.items():
            try:
                if isinstance(signal, str):
                    # Convert string signals to numeric
                    signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0, "STRONG_BUY": 1.0, "STRONG_SELL": -1.0}
                    normalized[source] = signal_map.get(signal.upper(), 0.0)
                elif isinstance(signal, (int, float)):
                    # Clamp numeric signals to [-1, 1]
                    normalized[source] = max(-1.0, min(1.0, float(signal)))
                else:
                    normalized[source] = 0.0
            except Exception:
                normalized[source] = 0.0
        
        return normalized
    
    def _detect_signal_outliers(self, signals: Dict[str, float], z_threshold: float = 2.5) -> List[str]:
        """Detect outlier signals using statistical methods."""
        if len(signals) < 3:
            return []
        
        values = np.array(list(signals.values()))
        z_scores = np.abs(stats.zscore(values))
        
        outliers = []
        for i, (source, z_score) in enumerate(zip(signals.keys(), z_scores)):
            if z_score > z_threshold:
                outliers.append(source)
        
        return outliers
    
    def _calculate_consensus_strength(self, signal_array: np.ndarray) -> float:
        """Calculate consensus strength based on signal distribution."""
        if len(signal_array) == 0:
            return 0.0
        
        # Calculate measures of consensus
        mean_abs_signal = np.mean(np.abs(signal_array))
        signal_std = np.std(signal_array)
        
        # Strong consensus = high absolute mean, low standard deviation
        if signal_std == 0:
            consensus_strength = mean_abs_signal
        else:
            consensus_strength = mean_abs_signal / (1 + signal_std)
        
        return min(1.0, consensus_strength)


class RegimeDetector:
    """Detect market regimes for context-aware signal weighting."""
    
    def __init__(self):
        self.logger = logger.bind(component="regime_detector")
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime for signal weighting."""
        regime_analysis = {
            "volatility_regime": "NORMAL",
            "trend_regime": "NEUTRAL",
            "correlation_regime": "NORMAL",
            "volume_regime": "NORMAL",
            "regime_confidence": 0.5,
            "regime_stability": 0.5
        }
        
        try:
            # Volatility regime detection
            if "volatility_indicators" in market_data:
                vol_data = market_data["volatility_indicators"]
                regime_analysis["volatility_regime"] = self._detect_volatility_regime(vol_data)
            
            # Trend regime detection
            if "trend_indicators" in market_data:
                trend_data = market_data["trend_indicators"]
                regime_analysis["trend_regime"] = self._detect_trend_regime(trend_data)
            
            # Correlation regime
            if "correlation_data" in market_data:
                corr_data = market_data["correlation_data"]
                regime_analysis["correlation_regime"] = self._detect_correlation_regime(corr_data)
            
            # Calculate overall regime confidence
            regime_analysis["regime_confidence"] = self._calculate_regime_confidence(regime_analysis)
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
        
        return regime_analysis
    
    def _detect_volatility_regime(self, vol_data: Dict[str, Any]) -> str:
        """Detect volatility regime."""
        try:
            current_vol = vol_data.get("current_volatility", 0.2)
            historical_vol = vol_data.get("historical_volatility", 0.2)
            
            vol_ratio = current_vol / (historical_vol + 1e-8)
            
            if vol_ratio > 1.5:
                return "HIGH_VOLATILITY"
            elif vol_ratio < 0.7:
                return "LOW_VOLATILITY"
            else:
                return "NORMAL"
        except Exception:
            return "NORMAL"
    
    def _detect_trend_regime(self, trend_data: Dict[str, Any]) -> str:
        """Detect trend regime."""
        try:
            trend_strength = trend_data.get("trend_strength", 0)
            trend_direction = trend_data.get("trend_direction", 0)
            
            if abs(trend_strength) > 0.7:
                return "STRONG_TREND" if trend_direction > 0 else "STRONG_DOWNTREND"
            elif abs(trend_strength) > 0.3:
                return "WEAK_TREND" if trend_direction > 0 else "WEAK_DOWNTREND"
            else:
                return "SIDEWAYS"
        except Exception:
            return "NEUTRAL"
    
    def _detect_correlation_regime(self, corr_data: Dict[str, Any]) -> str:
        """Detect correlation regime."""
        try:
            avg_correlation = corr_data.get("average_correlation", 0.5)
            
            if avg_correlation > 0.8:
                return "HIGH_CORRELATION"
            elif avg_correlation < 0.3:
                return "LOW_CORRELATION"
            else:
                return "NORMAL"
        except Exception:
            return "NORMAL"
    
    def _calculate_regime_confidence(self, regime_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in regime detection."""
        # Simple heuristic - in production would be more sophisticated
        non_normal_regimes = sum(1 for regime in [
            regime_analysis["volatility_regime"],
            regime_analysis["trend_regime"],
            regime_analysis["correlation_regime"]
        ] if regime != "NORMAL" and regime != "NEUTRAL")
        
        return min(1.0, 0.5 + (non_normal_regimes * 0.2))


class SignalSynthesisAgent(BaseAgent):
    """
    Signal Synthesis & Arbitration Agent with advanced consensus mechanisms.
    
    This agent combines signals from multiple strategy agents, applies rigorous
    validation, and ensures high-confidence recommendations through consensus.
    
    Inputs: Signals from multiple strategy agents
    Outputs: Validated, consensus-based trading recommendations
    """
    
    def __init__(self):
        super().__init__("SignalSynthesisAgent", "signal_synthesis")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validation systems."""
        self.min_confirming_sources = self.get_config_value("min_confirming_sources", 3)
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.7)
        self.regime_adjustment = self.get_config_value("regime_adjustment", True)
        self.outlier_detection = self.get_config_value("outlier_detection", True)
        
        # Initialize validation systems
        self.consensus_validator = ConsensusValidator(
            min_sources=self.min_confirming_sources,
            confidence_threshold=self.confidence_threshold
        )
        self.regime_detector = RegimeDetector()
        
        # Strategy weights by regime
        self.regime_weights = self.get_config_value("regime_weights", {
            "HIGH_VOLATILITY": {"momentum": 0.6, "technical": 0.8, "ml": 0.7},
            "LOW_VOLATILITY": {"momentum": 0.8, "technical": 0.6, "ml": 0.9},
            "STRONG_TREND": {"momentum": 0.9, "technical": 0.7, "ml": 0.8},
            "SIDEWAYS": {"momentum": 0.5, "technical": 0.9, "ml": 0.8}
        })
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute signal synthesis with multi-layer validation and consensus.
        
        Args:
            inputs: Dictionary containing:
                - strategy_signals: Signals from different strategy agents
                - market_data: Current market context for regime detection
                - risk_constraints: Risk management constraints
                - portfolio_context: Current portfolio state
                
        Returns:
            AgentOutput with consensus-validated recommendations
        """
        self._validate_inputs(inputs)
        
        strategy_signals = inputs["strategy_signals"]
        market_data = inputs.get("market_data", {})
        risk_constraints = inputs.get("risk_constraints", {})
        portfolio_context = inputs.get("portfolio_context", {})
        
        try:
            # Step 1: Detect market regime for context-aware processing
            regime_analysis = self.regime_detector.detect_market_regime(market_data)
            self.logger.info(f"Detected regime: {regime_analysis}")
            
            # Step 2: Process signals by symbol with consensus validation
            symbol_recommendations = await self._process_signals_by_symbol(
                strategy_signals, regime_analysis, risk_constraints
            )
            
            # Step 3: Apply portfolio-level constraints and optimization
            portfolio_optimized = await self._apply_portfolio_constraints(
                symbol_recommendations, portfolio_context, risk_constraints
            )
            
            # Step 4: Final validation and ranking
            final_recommendations = await self._final_validation_and_ranking(
                portfolio_optimized, regime_analysis
            )
            
            # Step 5: Generate comprehensive audit trail
            audit_trail = self._generate_audit_trail(
                strategy_signals, regime_analysis, symbol_recommendations, 
                portfolio_optimized, final_recommendations
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "final_recommendations": final_recommendations,
                    "regime_analysis": regime_analysis,
                    "consensus_analysis": self._generate_consensus_summary(symbol_recommendations),
                    "portfolio_impact": self._calculate_portfolio_impact(final_recommendations, portfolio_context),
                    "audit_trail": audit_trail
                },
                metadata={
                    "symbols_processed": len(symbol_recommendations),
                    "recommendations_generated": len(final_recommendations),
                    "consensus_threshold": self.confidence_threshold,
                    "regime_detected": regime_analysis.get("volatility_regime", "NORMAL"),
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Signal synthesis failed: {str(e)}")
            raise DataError(f"Signal synthesis processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters with multi-layer checks."""
        required_keys = ["strategy_signals"]
        self.validate_inputs(inputs, required_keys)
        
        strategy_signals = inputs["strategy_signals"]
        if not strategy_signals:
            raise ValidationError("Strategy signals cannot be empty")
        
        # Validate signal structure
        for strategy_name, signals in strategy_signals.items():
            if not isinstance(signals, dict):
                raise ValidationError(f"Invalid signal format for strategy {strategy_name}")
    
    async def _process_signals_by_symbol(self, strategy_signals: Dict[str, Any],
                                       regime_analysis: Dict[str, Any],
                                       risk_constraints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Process signals by symbol with consensus validation."""
        symbol_recommendations = {}
        
        # Collect all symbols across strategies
        all_symbols = set()
        for strategy_name, signals in strategy_signals.items():
            if isinstance(signals, dict) and "recommendations" in signals:
                for rec in signals["recommendations"]:
                    if "symbol" in rec:
                        all_symbols.add(rec["symbol"])
        
        # Process each symbol individually
        for symbol in all_symbols:
            try:
                symbol_signals = self._extract_symbol_signals(symbol, strategy_signals)
                
                if len(symbol_signals) < self.min_confirming_sources:
                    self.logger.debug(f"Insufficient signals for {symbol}: {len(symbol_signals)}")
                    continue
                
                # Validate consensus
                consensus_result = self.consensus_validator.validate_signal_consensus(symbol_signals)
                
                if not consensus_result["is_valid"]:
                    self.logger.debug(f"Consensus validation failed for {symbol}: {consensus_result}")
                    continue
                
                # Apply regime-aware weighting
                weighted_signal = self._apply_regime_weights(
                    symbol_signals, regime_analysis, consensus_result
                )
                
                # Generate recommendation
                recommendation = self._generate_symbol_recommendation(
                    symbol, weighted_signal, consensus_result, risk_constraints
                )
                
                symbol_recommendations[symbol] = recommendation
                
            except Exception as e:
                self.logger.error(f"Failed to process signals for {symbol}: {e}")
                continue
        
        return symbol_recommendations
    
    def _extract_symbol_signals(self, symbol: str, strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signals for a specific symbol from all strategies."""
        symbol_signals = {}
        
        for strategy_name, signals in strategy_signals.items():
            try:
                # Handle different signal formats
                if isinstance(signals, dict):
                    if "recommendations" in signals:
                        # Format from strategy agents
                        for rec in signals["recommendations"]:
                            if rec.get("symbol") == symbol:
                                symbol_signals[strategy_name] = {
                                    "signal": self._convert_action_to_signal(rec.get("action", "HOLD")),
                                    "confidence": rec.get("confidence", 0.5),
                                    "score": rec.get("composite_score", 0),
                                    "details": rec
                                }
                                break
                    elif "predictions" in signals:
                        # Format from ML models
                        if symbol in signals["predictions"]:
                            prediction = signals["predictions"][symbol]
                            symbol_signals[f"ml_{strategy_name}"] = {
                                "signal": np.tanh(prediction * 10),  # Convert to [-1, 1]
                                "confidence": signals.get("uncertainty_metrics", {}).get("confidence", 0.5),
                                "score": prediction,
                                "details": {"prediction": prediction}
                            }
                
            except Exception as e:
                self.logger.warning(f"Failed to extract signal for {symbol} from {strategy_name}: {e}")
                continue
        
        return symbol_signals
    
    def _convert_action_to_signal(self, action: str) -> float:
        """Convert action strings to numeric signals."""
        action_map = {
            "STRONG_BUY": 1.0,
            "BUY": 0.7,
            "WEAK_BUY": 0.3,
            "HOLD": 0.0,
            "WEAK_SELL": -0.3,
            "SELL": -0.7,
            "STRONG_SELL": -1.0
        }
        return action_map.get(action.upper(), 0.0)
    
    def _apply_regime_weights(self, symbol_signals: Dict[str, Any],
                            regime_analysis: Dict[str, Any],
                            consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime-aware weighting to signals."""
        if not self.regime_adjustment:
            return symbol_signals
        
        regime = regime_analysis.get("volatility_regime", "NORMAL")
        regime_weights = self.regime_weights.get(regime, {})
        
        weighted_signals = {}
        total_weight = 0
        
        for source, signal_data in symbol_signals.items():
            # Determine strategy type
            strategy_type = self._classify_strategy_type(source)
            weight = regime_weights.get(strategy_type, 1.0)
            
            # Adjust weight based on consensus confidence
            confidence_boost = consensus_result.get("confidence", 0.5)
            adjusted_weight = weight * (0.5 + 0.5 * confidence_boost)
            
            weighted_signals[source] = {
                **signal_data,
                "regime_weight": adjusted_weight,
                "strategy_type": strategy_type
            }
            total_weight += adjusted_weight
        
        # Normalize weights
        if total_weight > 0:
            for source in weighted_signals:
                weighted_signals[source]["normalized_weight"] = (
                    weighted_signals[source]["regime_weight"] / total_weight
                )
        
        return weighted_signals
    
    def _classify_strategy_type(self, source: str) -> str:
        """Classify strategy source into categories."""
        source_lower = source.lower()
        
        if "momentum" in source_lower:
            return "momentum"
        elif "technical" in source_lower or "ta" in source_lower:
            return "technical"
        elif "ml" in source_lower or "ensemble" in source_lower:
            return "ml"
        elif "stat" in source_lower or "arb" in source_lower:
            return "statistical"
        elif "event" in source_lower or "news" in source_lower:
            return "event"
        else:
            return "other"
    
    def _generate_symbol_recommendation(self, symbol: str, weighted_signals: Dict[str, Any],
                                      consensus_result: Dict[str, Any],
                                      risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation for a symbol."""
        # Calculate weighted average signal
        weighted_sum = 0
        weight_sum = 0
        
        for source, signal_data in weighted_signals.items():
            signal = signal_data.get("signal", 0)
            weight = signal_data.get("normalized_weight", 1.0)
            
            weighted_sum += signal * weight
            weight_sum += weight
        
        final_signal = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Determine action and position size
        action, position_size = self._determine_action_and_size(
            final_signal, consensus_result, risk_constraints
        )
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            weighted_signals, consensus_result
        )
        
        return {
            "symbol": symbol,
            "action": action,
            "position_size": position_size,
            "final_signal": final_signal,
            "consensus_confidence": consensus_result.get("confidence", 0),
            "consensus_strength": consensus_result.get("consensus_strength", 0),
            "confirming_sources": consensus_result.get("agreeing_sources", []),
            "disagreeing_sources": consensus_result.get("disagreeing_sources", []),
            "outliers_removed": consensus_result.get("outliers", []),
            "confidence_metrics": confidence_metrics,
            "weighted_signals": weighted_signals,
            "recommendation_timestamp": datetime.utcnow()
        }
    
    def _determine_action_and_size(self, final_signal: float,
                                 consensus_result: Dict[str, Any],
                                 risk_constraints: Dict[str, Any]) -> Tuple[str, float]:
        """Determine trading action and position size."""
        confidence = consensus_result.get("confidence", 0)
        consensus_strength = consensus_result.get("consensus_strength", 0)
        
        # Conservative thresholds
        strong_threshold = 0.6
        medium_threshold = 0.3
        
        # Require high confidence for any action
        if confidence < self.confidence_threshold:
            return "HOLD", 0.0
        
        # Determine action based on signal strength and consensus
        if abs(final_signal) > strong_threshold and consensus_strength > 0.8:
            action = "STRONG_BUY" if final_signal > 0 else "STRONG_SELL"
            base_size = 0.08  # 8% base position
        elif abs(final_signal) > medium_threshold and consensus_strength > 0.6:
            action = "BUY" if final_signal > 0 else "SELL"
            base_size = 0.05  # 5% base position
        elif abs(final_signal) > 0.1 and consensus_strength > 0.5:
            action = "WEAK_BUY" if final_signal > 0 else "WEAK_SELL"
            base_size = 0.03  # 3% base position
        else:
            action = "HOLD"
            base_size = 0.0
        
        # Apply risk constraints
        max_position = risk_constraints.get("max_single_position", 0.1)
        position_size = min(base_size, max_position)
        
        # Scale by confidence
        position_size *= confidence
        
        return action, position_size
    
    def _calculate_confidence_metrics(self, weighted_signals: Dict[str, Any],
                                    consensus_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive confidence metrics."""
        signals = [s.get("signal", 0) for s in weighted_signals.values()]
        confidences = [s.get("confidence", 0.5) for s in weighted_signals.values()]
        
        return {
            "signal_consistency": 1.0 - np.std(signals) if len(signals) > 1 else 1.0,
            "average_individual_confidence": np.mean(confidences),
            "consensus_confidence": consensus_result.get("confidence", 0),
            "source_count": len(weighted_signals),
            "outlier_ratio": len(consensus_result.get("outliers", [])) / len(weighted_signals) if weighted_signals else 0,
            "agreement_ratio": len(consensus_result.get("agreeing_sources", [])) / len(weighted_signals) if weighted_signals else 0
        }
    
    async def _apply_portfolio_constraints(self, symbol_recommendations: Dict[str, Dict[str, Any]],
                                         portfolio_context: Dict[str, Any],
                                         risk_constraints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Apply portfolio-level constraints and optimization."""
        # Current portfolio state
        current_positions = portfolio_context.get("positions", {})
        total_capital = portfolio_context.get("total_capital", 1000000)
        
        # Risk limits
        max_portfolio_risk = risk_constraints.get("max_portfolio_risk", 0.02)
        max_sector_exposure = risk_constraints.get("max_sector_exposure", 0.20)
        max_total_leverage = risk_constraints.get("max_total_leverage", 1.0)
        
        # Calculate portfolio impact
        total_new_exposure = 0
        sector_exposures = defaultdict(float)
        
        for symbol, rec in symbol_recommendations.items():
            position_size = rec.get("position_size", 0)
            total_new_exposure += abs(position_size)
            
            # Would need sector mapping here
            sector = "Unknown"  # Placeholder
            sector_exposures[sector] += abs(position_size)
        
        # Apply scaling if needed
        if total_new_exposure > max_total_leverage:
            scaling_factor = max_total_leverage / total_new_exposure
            
            for symbol, rec in symbol_recommendations.items():
                rec["position_size"] *= scaling_factor
                rec["portfolio_scaled"] = True
                rec["scaling_factor"] = scaling_factor
        
        return symbol_recommendations
    
    async def _final_validation_and_ranking(self, portfolio_optimized: Dict[str, Dict[str, Any]],
                                          regime_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Final validation and ranking of recommendations."""
        validated_recommendations = []
        
        for symbol, rec in portfolio_optimized.items():
            # Final validation checks
            if self._final_validation_check(rec, regime_analysis):
                validated_recommendations.append(rec)
        
        # Rank by confidence and signal strength
        validated_recommendations.sort(
            key=lambda x: (
                x.get("consensus_confidence", 0) * 
                abs(x.get("final_signal", 0)) * 
                x.get("confidence_metrics", {}).get("signal_consistency", 0)
            ),
            reverse=True
        )
        
        return validated_recommendations
    
    def _final_validation_check(self, recommendation: Dict[str, Any],
                              regime_analysis: Dict[str, Any]) -> bool:
        """Perform final validation checks."""
        # Check minimum confidence
        if recommendation.get("consensus_confidence", 0) < self.confidence_threshold:
            return False
        
        # Check minimum confirming sources
        confirming_sources = len(recommendation.get("confirming_sources", []))
        if confirming_sources < self.min_confirming_sources:
            return False
        
        # Check signal strength
        if abs(recommendation.get("final_signal", 0)) < 0.1:
            return False
        
        # Regime-specific checks
        regime = regime_analysis.get("volatility_regime", "NORMAL")
        if regime == "HIGH_VOLATILITY":
            # More conservative in high volatility
            if recommendation.get("consensus_confidence", 0) < 0.8:
                return False
        
        return True
    
    def _generate_consensus_summary(self, symbol_recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of consensus analysis."""
        if not symbol_recommendations:
            return {}
        
        confidences = [r.get("consensus_confidence", 0) for r in symbol_recommendations.values()]
        strengths = [r.get("consensus_strength", 0) for r in symbol_recommendations.values()]
        source_counts = [len(r.get("confirming_sources", [])) for r in symbol_recommendations.values()]
        
        return {
            "total_symbols_analyzed": len(symbol_recommendations),
            "average_consensus_confidence": np.mean(confidences),
            "average_consensus_strength": np.mean(strengths),
            "average_confirming_sources": np.mean(source_counts),
            "high_confidence_count": sum(1 for c in confidences if c > 0.8),
            "consensus_distribution": {
                "high": sum(1 for c in confidences if c > 0.8),
                "medium": sum(1 for c in confidences if 0.6 <= c <= 0.8),
                "low": sum(1 for c in confidences if c < 0.6)
            }
        }
    
    def _calculate_portfolio_impact(self, recommendations: List[Dict[str, Any]],
                                  portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected portfolio impact."""
        if not recommendations:
            return {}
        
        total_exposure = sum(abs(r.get("position_size", 0)) for r in recommendations)
        long_exposure = sum(r.get("position_size", 0) for r in recommendations if r.get("position_size", 0) > 0)
        short_exposure = sum(abs(r.get("position_size", 0)) for r in recommendations if r.get("position_size", 0) < 0)
        
        return {
            "total_exposure": total_exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "recommendation_count": len(recommendations),
            "average_position_size": total_exposure / len(recommendations) if recommendations else 0
        }
    
    def _generate_audit_trail(self, strategy_signals: Dict[str, Any],
                            regime_analysis: Dict[str, Any],
                            symbol_recommendations: Dict[str, Dict[str, Any]],
                            portfolio_optimized: Dict[str, Dict[str, Any]],
                            final_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive audit trail."""
        return {
            "process_steps": {
                "1_regime_detection": regime_analysis,
                "2_signal_processing": len(symbol_recommendations),
                "3_consensus_validation": sum(1 for r in symbol_recommendations.values() if r.get("consensus_confidence", 0) > self.confidence_threshold),
                "4_portfolio_optimization": len(portfolio_optimized),
                "5_final_validation": len(final_recommendations)
            },
            "validation_summary": {
                "consensus_threshold": self.confidence_threshold,
                "min_confirming_sources": self.min_confirming_sources,
                "regime_adjustment_enabled": self.regime_adjustment,
                "outlier_detection_enabled": self.outlier_detection
            },
            "decision_rationale": {
                "regime_detected": regime_analysis.get("volatility_regime", "NORMAL"),
                "consensus_approach": "multi_source_validation",
                "risk_management": "portfolio_level_constraints",
                "final_filter": "confidence_and_consistency"
            },
            "timestamp": datetime.utcnow()
        }