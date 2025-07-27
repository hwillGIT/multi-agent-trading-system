"""
Momentum Strategy Agent - Multi-factor momentum, sector rotation, and breakout strategies.

This agent implements:
- Classic price momentum (time series momentum)
- Cross-sectional momentum (relative strength)
- Risk-adjusted momentum (Sharpe-based)
- Sector rotation strategies
- Technical breakout patterns
- Volume-confirmed momentum
- Multi-timeframe momentum alignment
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from ....core.base.agent import BaseAgent, AgentOutput
from ....core.base.exceptions import ValidationError, DataError
from ....core.utils.data_validation import DataValidator
from ....core.utils.math_utils import MathUtils


class MomentumAgent(BaseAgent):
    """
    Agent responsible for momentum-based trading strategies.
    
    This agent generates momentum signals based on price trends, relative strength,
    technical breakouts, and sector rotation patterns.
    
    Inputs: Enhanced feature matrix with technical indicators and ML predictions
    Outputs: Momentum signals, scores, and supporting analysis
    """
    
    def __init__(self):
        super().__init__("MomentumAgent", "strategies.momentum")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validate configuration."""
        self.lookback_period = self.get_config_value("lookback_period", 20)
        self.threshold = self.get_config_value("threshold", 0.02)
        self.min_volume = self.get_config_value("min_volume", 1000000)
        self.momentum_timeframes = self.get_config_value("timeframes", [5, 10, 20, 50])
        self.sector_rotation_enabled = self.get_config_value("sector_rotation", True)
        self.volume_confirmation = self.get_config_value("volume_confirmation", True)
        
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute momentum strategy analysis.
        
        Args:
            inputs: Dictionary containing:
                - enhanced_feature_matrix: DataFrame with technical features
                - ml_predictions: ML model predictions (optional)
                - symbols: List of symbols to analyze
                - current_positions: Current portfolio positions (optional)
                
        Returns:
            AgentOutput with momentum signals and analysis
        """
        self._validate_inputs(inputs)
        
        feature_matrix = inputs["enhanced_feature_matrix"]
        ml_predictions = inputs.get("ml_predictions", {})
        symbols = inputs.get("symbols", feature_matrix["symbol"].unique() if "symbol" in feature_matrix.columns else [])
        current_positions = inputs.get("current_positions", {})
        
        try:
            # Calculate momentum signals for each symbol
            momentum_signals = await self._calculate_momentum_signals(
                feature_matrix, symbols, ml_predictions
            )
            
            # Apply sector rotation analysis
            if self.sector_rotation_enabled:
                sector_analysis = await self._analyze_sector_rotation(
                    feature_matrix, momentum_signals
                )
            else:
                sector_analysis = {}
            
            # Filter and rank momentum candidates
            filtered_signals = await self._filter_momentum_candidates(
                momentum_signals, feature_matrix
            )
            
            # Generate final momentum recommendations
            recommendations = await self._generate_momentum_recommendations(
                filtered_signals, sector_analysis, current_positions
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "momentum_signals": momentum_signals,
                    "sector_analysis": sector_analysis,
                    "filtered_signals": filtered_signals,
                    "recommendations": recommendations,
                    "strategy_metadata": {
                        "lookback_period": self.lookback_period,
                        "threshold": self.threshold,
                        "timeframes_analyzed": self.momentum_timeframes
                    }
                },
                metadata={
                    "symbols_analyzed": len(symbols),
                    "signals_generated": len(momentum_signals),
                    "recommendations_count": len(recommendations),
                    "analysis_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Momentum strategy analysis failed: {str(e)}")
            raise DataError(f"Momentum strategy analysis failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["enhanced_feature_matrix"]
        self.validate_inputs(inputs, required_keys)
        
        feature_matrix = inputs["enhanced_feature_matrix"]
        if feature_matrix.empty:
            raise ValidationError("Enhanced feature matrix cannot be empty")
        
        # Check for required columns
        required_columns = ["close"]
        missing_columns = [col for col in required_columns if col not in feature_matrix.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
    
    async def _calculate_momentum_signals(self, feature_matrix: pd.DataFrame,
                                        symbols: List[str],
                                        ml_predictions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive momentum signals for each symbol."""
        momentum_signals = {}
        
        for symbol in symbols:
            try:
                # Get symbol data
                if "symbol" in feature_matrix.columns:
                    symbol_data = feature_matrix[feature_matrix["symbol"] == symbol].copy()
                else:
                    symbol_data = feature_matrix.copy()
                    symbol_data["symbol"] = symbol
                
                if symbol_data.empty or len(symbol_data) < self.lookback_period:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Sort by timestamp
                if "timestamp" in symbol_data.columns:
                    symbol_data = symbol_data.sort_values("timestamp")
                
                # Calculate different momentum types
                momentum_analysis = {
                    "symbol": symbol,
                    "price_momentum": self._calculate_price_momentum(symbol_data),
                    "technical_momentum": self._calculate_technical_momentum(symbol_data),
                    "risk_adjusted_momentum": self._calculate_risk_adjusted_momentum(symbol_data),
                    "volume_momentum": self._calculate_volume_momentum(symbol_data),
                    "multi_timeframe_momentum": self._calculate_multi_timeframe_momentum(symbol_data),
                    "breakout_signals": self._detect_momentum_breakouts(symbol_data),
                    "ml_confirmation": self._get_ml_confirmation(symbol, ml_predictions)
                }
                
                # Calculate composite momentum score
                momentum_analysis["composite_score"] = self._calculate_composite_momentum_score(momentum_analysis)
                
                # Determine signal strength and direction
                momentum_analysis["signal"] = self._determine_momentum_signal(momentum_analysis)
                
                momentum_signals[symbol] = momentum_analysis
                
            except Exception as e:
                self.logger.error(f"Failed to calculate momentum for {symbol}: {e}")
                continue
        
        return momentum_signals
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate traditional price momentum metrics."""
        prices = data["close"]
        
        momentum_metrics = {}
        
        # Calculate momentum for different periods
        for period in self.momentum_timeframes:
            if len(prices) > period:
                # Simple return momentum
                momentum_metrics[f"return_{period}d"] = (prices.iloc[-1] / prices.iloc[-period] - 1)
                
                # Log return momentum
                momentum_metrics[f"log_return_{period}d"] = np.log(prices.iloc[-1] / prices.iloc[-period])
                
                # Risk-adjusted momentum (return/volatility)
                returns = prices.pct_change().iloc[-period:]
                if returns.std() > 0:
                    momentum_metrics[f"risk_adj_{period}d"] = returns.mean() / returns.std()
                else:
                    momentum_metrics[f"risk_adj_{period}d"] = 0
        
        # Momentum acceleration (second derivative)
        if len(prices) >= 40:
            short_momentum = prices.iloc[-20] / prices.iloc[-40] - 1
            recent_momentum = prices.iloc[-1] / prices.iloc[-20] - 1
            momentum_metrics["acceleration"] = recent_momentum - short_momentum
        
        # Momentum persistence (correlation of returns)
        if len(prices) >= 50:
            returns = prices.pct_change().dropna()
            if len(returns) >= 20:
                momentum_metrics["persistence"] = returns.autocorr(lag=1)
        
        return momentum_metrics
    
    def _calculate_technical_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum based on technical indicators."""
        technical_momentum = {}
        
        # RSI momentum
        if "rsi_14" in data.columns:
            rsi = data["rsi_14"].iloc[-1]
            technical_momentum["rsi_momentum"] = (rsi - 50) / 50  # Normalized RSI
        
        # MACD momentum
        if "macd" in data.columns and "macd_signal" in data.columns:
            macd = data["macd"].iloc[-1]
            macd_signal = data["macd_signal"].iloc[-1]
            technical_momentum["macd_momentum"] = macd - macd_signal
        
        # Moving average momentum
        if "sma_20" in data.columns and "sma_50" in data.columns:
            sma_20 = data["sma_20"].iloc[-1]
            sma_50 = data["sma_50"].iloc[-1]
            price = data["close"].iloc[-1]
            
            technical_momentum["ma_momentum"] = (sma_20 / sma_50 - 1)
            technical_momentum["price_vs_ma"] = (price / sma_20 - 1)
        
        # Bollinger Band momentum
        if all(col in data.columns for col in ["bb_upper", "bb_lower", "bb_position"]):
            bb_position = data["bb_position"].iloc[-1]
            technical_momentum["bb_momentum"] = bb_position - 0.5  # Centered around 0
        
        # ADX trend strength
        if "adx_14" in data.columns:
            adx = data["adx_14"].iloc[-1]
            technical_momentum["trend_strength"] = adx / 100  # Normalized ADX
        
        return technical_momentum
    
    def _calculate_risk_adjusted_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted momentum metrics."""
        prices = data["close"]
        returns = prices.pct_change().dropna()
        
        risk_adjusted = {}
        
        if len(returns) >= 20:
            # Sharpe-based momentum
            for period in [10, 20, 50]:
                if len(returns) >= period:
                    period_returns = returns.iloc[-period:]
                    if period_returns.std() > 0:
                        sharpe = period_returns.mean() / period_returns.std() * np.sqrt(252)
                        risk_adjusted[f"sharpe_{period}d"] = sharpe
        
        # Sortino ratio momentum
        if len(returns) >= 20:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
                risk_adjusted["sortino"] = sortino
        
        # Maximum drawdown consideration
        cumulative = (1 + returns).cumprod()
        if len(cumulative) >= 20:
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            if max_dd < 0:
                # Return/max drawdown ratio
                total_return = cumulative.iloc[-1] - 1
                risk_adjusted["return_dd_ratio"] = total_return / abs(max_dd)
        
        return risk_adjusted
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based momentum indicators."""
        volume_momentum = {}
        
        if "volume" not in data.columns or data["volume"].sum() == 0:
            return volume_momentum
        
        volume = data["volume"]
        prices = data["close"]
        
        # Volume-weighted momentum
        if len(volume) >= 20:
            recent_volume = volume.iloc[-5:].mean()
            avg_volume = volume.iloc[-20:].mean()
            
            if avg_volume > 0:
                volume_momentum["volume_ratio"] = recent_volume / avg_volume
        
        # On-Balance Volume momentum
        if "obv" in data.columns:
            obv = data["obv"]
            if len(obv) >= 20:
                obv_sma = obv.rolling(window=20).mean()
                current_obv = obv.iloc[-1]
                avg_obv = obv_sma.iloc[-1]
                
                if avg_obv != 0:
                    volume_momentum["obv_momentum"] = (current_obv / avg_obv - 1)
        
        # Price-Volume correlation
        if len(data) >= 20:
            price_returns = prices.pct_change().iloc[-20:]
            volume_changes = volume.pct_change().iloc[-20:]
            
            # Remove NaN values
            valid_data = pd.DataFrame({
                "price_returns": price_returns,
                "volume_changes": volume_changes
            }).dropna()
            
            if len(valid_data) >= 10:
                correlation = valid_data["price_returns"].corr(valid_data["volume_changes"])
                volume_momentum["price_volume_corr"] = correlation if not np.isnan(correlation) else 0
        
        return volume_momentum
    
    def _calculate_multi_timeframe_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum alignment across multiple timeframes."""
        prices = data["close"]
        multi_tf = {}
        
        # Different timeframe periods
        timeframes = [5, 10, 20, 50, 100]
        momentum_scores = []
        
        for tf in timeframes:
            if len(prices) > tf:
                momentum = prices.iloc[-1] / prices.iloc[-tf] - 1
                momentum_scores.append(1 if momentum > 0 else -1)
        
        if momentum_scores:
            # Alignment score (-1 to 1)
            multi_tf["alignment_score"] = np.mean(momentum_scores)
            
            # Consistency (how many timeframes agree)
            multi_tf["consistency"] = len([x for x in momentum_scores if x == momentum_scores[0]]) / len(momentum_scores)
        
        # Trend acceleration across timeframes
        if len(prices) >= 100:
            short_trend = prices.iloc[-10:].pct_change().mean()
            medium_trend = prices.iloc[-30:].pct_change().mean()
            long_trend = prices.iloc[-100:].pct_change().mean()
            
            multi_tf["trend_acceleration"] = short_trend - long_trend
        
        return multi_tf
    
    def _detect_momentum_breakouts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect technical breakout patterns that support momentum."""
        breakouts = {}
        
        if len(data) < 20:
            return breakouts
        
        prices = data["close"]
        volume = data.get("volume", pd.Series([0] * len(data)))
        
        # Price breakouts from range
        lookback = min(20, len(prices) - 1)
        recent_high = prices.iloc[-lookback:].max()
        recent_low = prices.iloc[-lookback:].min()
        current_price = prices.iloc[-1]
        
        # Upward breakout
        if current_price > recent_high * 1.01:  # 1% above recent high
            breakouts["upward_breakout"] = True
            breakouts["breakout_strength"] = (current_price / recent_high - 1)
        else:
            breakouts["upward_breakout"] = False
        
        # Downward breakout
        if current_price < recent_low * 0.99:  # 1% below recent low
            breakouts["downward_breakout"] = True
            breakouts["breakdown_strength"] = (recent_low / current_price - 1)
        else:
            breakouts["downward_breakout"] = False
        
        # Volume confirmation
        if volume.sum() > 0:
            recent_volume = volume.iloc[-5:].mean()
            avg_volume = volume.iloc[-20:].mean()
            
            breakouts["volume_confirmation"] = recent_volume > avg_volume * 1.5
        
        # Support/resistance breakout
        if "support_resistance" in data.columns:
            sr_signal = data["support_resistance"].iloc[-1]
            breakouts["sr_breakout_signal"] = sr_signal
        
        # Moving average breakout
        if "sma_20" in data.columns:
            ma_20 = data["sma_20"].iloc[-1]
            breakouts["ma_breakout"] = current_price > ma_20 * 1.02
        
        return breakouts
    
    def _get_ml_confirmation(self, symbol: str, ml_predictions: Dict[str, Any]) -> Dict[str, float]:
        """Get ML model confirmation for momentum signals."""
        ml_confirmation = {}
        
        if not ml_predictions or "predictions" not in ml_predictions:
            return ml_confirmation
        
        try:
            predictions = ml_predictions.get("predictions", {})
            
            if hasattr(predictions, "get"):
                # If predictions is a dict
                symbol_prediction = predictions.get(symbol, 0)
            elif hasattr(predictions, "__getitem__"):
                # If predictions is array-like, use symbol index
                symbol_idx = list(ml_predictions.get("symbols", [symbol])).index(symbol) if symbol in ml_predictions.get("symbols", []) else 0
                symbol_prediction = predictions[symbol_idx] if symbol_idx < len(predictions) else 0
            else:
                symbol_prediction = 0
            
            ml_confirmation["ml_prediction"] = float(symbol_prediction)
            
            # Get prediction confidence if available
            uncertainty_metrics = ml_predictions.get("uncertainty_metrics", {})
            if uncertainty_metrics:
                ml_confirmation["prediction_confidence"] = 1 - uncertainty_metrics.get("mean_uncertainty", 0.5)
            
            # ML momentum alignment
            if abs(symbol_prediction) > 0.01:  # 1% threshold
                ml_confirmation["ml_momentum_signal"] = 1 if symbol_prediction > 0 else -1
            else:
                ml_confirmation["ml_momentum_signal"] = 0
                
        except Exception as e:
            self.logger.warning(f"Failed to get ML confirmation for {symbol}: {e}")
        
        return ml_confirmation
    
    def _calculate_composite_momentum_score(self, momentum_analysis: Dict[str, Any]) -> float:
        """Calculate composite momentum score from all indicators."""
        scores = []
        weights = []
        
        # Price momentum (weight: 30%)
        price_momentum = momentum_analysis.get("price_momentum", {})
        if price_momentum:
            # Use 20-day momentum as primary
            primary_momentum = price_momentum.get("return_20d", 0)
            scores.append(np.tanh(primary_momentum * 10))  # Sigmoid-like normalization
            weights.append(0.3)
        
        # Technical momentum (weight: 25%)
        technical_momentum = momentum_analysis.get("technical_momentum", {})
        if technical_momentum:
            # Average technical indicators
            tech_values = [v for v in technical_momentum.values() if isinstance(v, (int, float)) and not np.isnan(v)]
            if tech_values:
                avg_tech = np.mean(tech_values)
                scores.append(np.tanh(avg_tech))
                weights.append(0.25)
        
        # Risk-adjusted momentum (weight: 20%)
        risk_adjusted = momentum_analysis.get("risk_adjusted_momentum", {})
        if risk_adjusted:
            # Use Sharpe ratio as primary
            sharpe = risk_adjusted.get("sharpe_20d", 0)
            scores.append(np.tanh(sharpe / 2))  # Normalize Sharpe
            weights.append(0.2)
        
        # Multi-timeframe alignment (weight: 15%)
        multi_tf = momentum_analysis.get("multi_timeframe_momentum", {})
        if multi_tf:
            alignment = multi_tf.get("alignment_score", 0)
            scores.append(alignment)
            weights.append(0.15)
        
        # ML confirmation (weight: 10%)
        ml_confirmation = momentum_analysis.get("ml_confirmation", {})
        if ml_confirmation:
            ml_signal = ml_confirmation.get("ml_momentum_signal", 0)
            scores.append(ml_signal)
            weights.append(0.1)
        
        # Calculate weighted average
        if scores and weights:
            composite_score = np.average(scores, weights=weights)
        else:
            composite_score = 0
        
        return float(composite_score)
    
    def _determine_momentum_signal(self, momentum_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine final momentum signal and strength."""
        composite_score = momentum_analysis.get("composite_score", 0)
        
        # Determine signal direction
        if composite_score > self.threshold:
            direction = "BUY"
        elif composite_score < -self.threshold:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        # Determine signal strength
        abs_score = abs(composite_score)
        if abs_score > 0.7:
            strength = "STRONG"
        elif abs_score > 0.4:
            strength = "MEDIUM"
        elif abs_score > 0.1:
            strength = "WEAK"
        else:
            strength = "NEUTRAL"
        
        # Check for breakout confirmation
        breakouts = momentum_analysis.get("breakout_signals", {})
        breakout_confirmation = breakouts.get("upward_breakout", False) or breakouts.get("downward_breakout", False)
        
        # Volume confirmation
        volume_momentum = momentum_analysis.get("volume_momentum", {})
        volume_confirmation = volume_momentum.get("volume_ratio", 1) > 1.2  # 20% above average
        
        return {
            "direction": direction,
            "strength": strength,
            "score": composite_score,
            "breakout_confirmation": breakout_confirmation,
            "volume_confirmation": volume_confirmation,
            "confidence": min(abs_score, 1.0)
        }
    
    async def _analyze_sector_rotation(self, feature_matrix: pd.DataFrame,
                                     momentum_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sector rotation patterns."""
        if "metadata_sector" not in feature_matrix.columns:
            return {"sector_rotation_enabled": False}
        
        # Group signals by sector
        sector_momentum = {}
        
        for symbol, signal_data in momentum_signals.items():
            # Get sector for this symbol
            symbol_data = feature_matrix[feature_matrix["symbol"] == symbol]
            if symbol_data.empty:
                continue
            
            sector = symbol_data["metadata_sector"].iloc[0]
            if sector not in sector_momentum:
                sector_momentum[sector] = []
            
            sector_momentum[sector].append(signal_data["composite_score"])
        
        # Calculate sector averages
        sector_rankings = {}
        for sector, scores in sector_momentum.items():
            if scores:
                sector_rankings[sector] = {
                    "average_momentum": np.mean(scores),
                    "momentum_consistency": np.std(scores),
                    "stocks_count": len(scores),
                    "positive_momentum_ratio": len([s for s in scores if s > 0]) / len(scores)
                }
        
        # Rank sectors
        sorted_sectors = sorted(
            sector_rankings.items(),
            key=lambda x: x[1]["average_momentum"],
            reverse=True
        )
        
        return {
            "sector_rotation_enabled": True,
            "sector_rankings": sector_rankings,
            "top_sectors": [sector for sector, _ in sorted_sectors[:3]],
            "bottom_sectors": [sector for sector, _ in sorted_sectors[-3:]],
            "sector_rotation_signal": self._generate_sector_rotation_signal(sorted_sectors)
        }
    
    def _generate_sector_rotation_signal(self, sorted_sectors: List[Tuple[str, Dict[str, float]]]) -> str:
        """Generate sector rotation signal."""
        if len(sorted_sectors) < 2:
            return "NEUTRAL"
        
        top_sector_momentum = sorted_sectors[0][1]["average_momentum"]
        bottom_sector_momentum = sorted_sectors[-1][1]["average_momentum"]
        
        momentum_spread = top_sector_momentum - bottom_sector_momentum
        
        if momentum_spread > 0.3:
            return "STRONG_ROTATION"
        elif momentum_spread > 0.15:
            return "MODERATE_ROTATION"
        else:
            return "NEUTRAL"
    
    async def _filter_momentum_candidates(self, momentum_signals: Dict[str, Dict[str, Any]],
                                        feature_matrix: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Filter momentum candidates based on quality criteria."""
        filtered_signals = {}
        
        for symbol, signal_data in momentum_signals.items():
            # Get symbol data for additional filtering
            symbol_data = feature_matrix[feature_matrix["symbol"] == symbol] if "symbol" in feature_matrix.columns else feature_matrix
            
            if symbol_data.empty:
                continue
            
            # Volume filter
            if self.volume_confirmation and "volume" in symbol_data.columns:
                recent_volume = symbol_data["volume"].iloc[-5:].mean()
                if recent_volume < self.min_volume:
                    continue
            
            # Signal strength filter
            signal_info = signal_data.get("signal", {})
            if signal_info.get("strength") == "NEUTRAL":
                continue
            
            # Momentum consistency filter
            multi_tf = signal_data.get("multi_timeframe_momentum", {})
            consistency = multi_tf.get("consistency", 0)
            if consistency < 0.6:  # At least 60% timeframe agreement
                continue
            
            # Risk filter - avoid stocks with excessive volatility
            if "volatility_20" in symbol_data.columns:
                volatility = symbol_data["volatility_20"].iloc[-1]
                if volatility > 0.05:  # 5% daily volatility threshold
                    # Only allow if momentum is very strong
                    if abs(signal_data["composite_score"]) < 0.5:
                        continue
            
            filtered_signals[symbol] = signal_data
        
        return filtered_signals
    
    async def _generate_momentum_recommendations(self, filtered_signals: Dict[str, Dict[str, Any]],
                                               sector_analysis: Dict[str, Any],
                                               current_positions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate final momentum recommendations."""
        recommendations = []
        
        # Sort by composite score
        sorted_signals = sorted(
            filtered_signals.items(),
            key=lambda x: abs(x[1]["composite_score"]),
            reverse=True
        )
        
        for symbol, signal_data in sorted_signals[:20]:  # Top 20 momentum candidates
            signal_info = signal_data["signal"]
            
            # Position sizing based on signal strength and volatility
            base_position_size = 0.05  # 5% base position
            
            # Adjust for signal strength
            strength_multiplier = {
                "STRONG": 1.5,
                "MEDIUM": 1.0,
                "WEAK": 0.5
            }.get(signal_info["strength"], 1.0)
            
            # Adjust for sector rotation
            sector_boost = 1.0
            if sector_analysis.get("sector_rotation_enabled", False):
                # Would need sector info from feature matrix
                # For now, use default
                pass
            
            position_size = base_position_size * strength_multiplier * sector_boost
            position_size = min(position_size, 0.1)  # Max 10% position
            
            # Current position adjustment
            current_pos = current_positions.get(symbol, 0)
            if signal_info["direction"] == "SELL" and current_pos > 0:
                recommended_action = "REDUCE"
                target_position = max(0, current_pos - position_size)
            elif signal_info["direction"] == "BUY":
                recommended_action = "BUY" if current_pos == 0 else "INCREASE"
                target_position = current_pos + position_size
            else:
                recommended_action = "HOLD"
                target_position = current_pos
            
            recommendation = {
                "symbol": symbol,
                "action": recommended_action,
                "signal_direction": signal_info["direction"],
                "signal_strength": signal_info["strength"],
                "composite_score": signal_data["composite_score"],
                "confidence": signal_info["confidence"],
                "target_position_size": target_position,
                "current_position": current_pos,
                "position_change": target_position - current_pos,
                "momentum_factors": {
                    "price_momentum": signal_data.get("price_momentum", {}),
                    "technical_confirmation": signal_data.get("technical_momentum", {}),
                    "volume_confirmation": signal_info.get("volume_confirmation", False),
                    "breakout_confirmation": signal_info.get("breakout_confirmation", False),
                    "ml_confirmation": signal_data.get("ml_confirmation", {})
                },
                "risk_factors": {
                    "timeframe_consistency": signal_data.get("multi_timeframe_momentum", {}).get("consistency", 0),
                    "risk_adjusted_score": signal_data.get("risk_adjusted_momentum", {}),
                },
                "recommendation_timestamp": datetime.utcnow()
            }
            
            recommendations.append(recommendation)
        
        return recommendations