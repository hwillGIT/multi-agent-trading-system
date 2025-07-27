"""
Technical Analysis Agent - Extracts classic and exotic indicators, patterns, and statistical features.

This agent handles comprehensive technical analysis including:
- Classic indicators (RSI, MACD, Bollinger Bands, etc.)
- Exotic indicators (Chande, Ulcer Index, TSI, etc.)
- Pattern recognition (chart patterns, candlestick patterns)
- Statistical features (autocorrelation, regime detection)
- Market microstructure analysis
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from scipy import stats
from scipy.signal import find_peaks
from loguru import logger

from ...core.base.agent import BaseAgent, AgentOutput
from ...core.base.exceptions import DataError, ValidationError
from ...core.utils.data_validation import DataValidator
from ...core.utils.math_utils import MathUtils


class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent responsible for comprehensive technical analysis and feature engineering.
    
    This agent extracts technical indicators, recognizes patterns, and computes
    statistical features from price and volume data for use in trading strategies.
    
    Inputs: Price/volume data with OHLCV format
    Outputs: Feature-enhanced dataset with technical indicators and patterns
    """
    
    def __init__(self):
        super().__init__("TechnicalAnalysisAgent", "features")
        self._setup_dependencies()
    
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validate configuration."""
        self.lookback_periods = self.get_config_value("lookback_periods", [5, 10, 20, 50, 100, 200])
        self.technical_indicators = self.get_config_value("technical_indicators", [
            "sma", "ema", "rsi", "macd", "bollinger_bands", "atr", "obv", "adx"
        ])
        self.statistical_features = self.get_config_value("statistical_features", [
            "returns", "volatility", "skewness", "kurtosis", "autocorrelation"
        ])
        
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute technical analysis on price data.
        
        Args:
            inputs: Dictionary containing:
                - feature_matrix: DataFrame with OHLCV data from DataUniverseAgent
                - symbols: List of symbols to analyze
                - add_patterns: Whether to include pattern recognition
                - add_microstructure: Whether to include microstructure features
                
        Returns:
            AgentOutput with feature-enhanced dataset
        """
        self._validate_inputs(inputs)
        
        feature_matrix = inputs["feature_matrix"]
        symbols = inputs.get("symbols", feature_matrix["symbol"].unique() if "symbol" in feature_matrix.columns else [])
        add_patterns = inputs.get("add_patterns", True)
        add_microstructure = inputs.get("add_microstructure", False)
        
        try:
            # Process each symbol separately for technical analysis
            enhanced_data = await self._process_symbols(
                feature_matrix, symbols, add_patterns, add_microstructure
            )
            
            # Generate feature summary
            feature_summary = self._generate_feature_summary(enhanced_data)
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "enhanced_feature_matrix": enhanced_data,
                    "feature_summary": feature_summary,
                    "indicators_computed": self.technical_indicators,
                    "lookback_periods": self.lookback_periods
                },
                metadata={
                    "symbols_processed": len(symbols),
                    "features_added": len(feature_summary.get("new_features", [])),
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {str(e)}")
            raise DataError(f"Technical analysis processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["feature_matrix"]
        self.validate_inputs(inputs, required_keys)
        
        feature_matrix = inputs["feature_matrix"]
        if feature_matrix.empty:
            raise ValidationError("Feature matrix cannot be empty")
        
        # Check for required columns
        required_columns = ["close"]
        missing_columns = [col for col in required_columns if col not in feature_matrix.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
    
    async def _process_symbols(self, feature_matrix: pd.DataFrame, symbols: List[str],
                             add_patterns: bool, add_microstructure: bool) -> pd.DataFrame:
        """Process technical analysis for all symbols."""
        enhanced_parts = []
        
        for symbol in symbols:
            try:
                # Get symbol data
                if "symbol" in feature_matrix.columns:
                    symbol_data = feature_matrix[feature_matrix["symbol"] == symbol].copy()
                else:
                    symbol_data = feature_matrix.copy()
                    symbol_data["symbol"] = symbol
                
                if symbol_data.empty:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    continue
                
                # Sort by timestamp for proper technical analysis
                if "timestamp" in symbol_data.columns:
                    symbol_data = symbol_data.sort_values("timestamp")
                
                # Add technical indicators
                symbol_data = self._add_technical_indicators(symbol_data)
                
                # Add statistical features
                symbol_data = self._add_statistical_features(symbol_data)
                
                # Add pattern recognition if requested
                if add_patterns:
                    symbol_data = self._add_pattern_features(symbol_data)
                
                # Add microstructure features if requested
                if add_microstructure:
                    symbol_data = self._add_microstructure_features(symbol_data)
                
                enhanced_parts.append(symbol_data)
                
            except Exception as e:
                self.logger.error(f"Failed to process symbol {symbol}: {e}")
                continue
        
        if not enhanced_parts:
            raise DataError("No symbols were successfully processed")
        
        return pd.concat(enhanced_parts, ignore_index=True)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        df = data.copy()
        
        # Ensure we have the required price columns
        if not all(col in df.columns for col in ["open", "high", "low", "close"]):
            # If missing OHLC, use close price for all
            if "close" in df.columns:
                df["open"] = df["close"]
                df["high"] = df["close"]
                df["low"] = df["close"]
        
        # Volume - use 0 if not available
        if "volume" not in df.columns:
            df["volume"] = 0
        
        # Convert to numpy arrays for TA-Lib
        open_prices = df["open"].values.astype(float)
        high_prices = df["high"].values.astype(float)
        low_prices = df["low"].values.astype(float)
        close_prices = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)
        
        # Moving Averages
        for period in self.lookback_periods:
            if len(close_prices) > period:
                # Simple Moving Average
                df[f"sma_{period}"] = talib.SMA(close_prices, timeperiod=period)
                
                # Exponential Moving Average
                df[f"ema_{period}"] = talib.EMA(close_prices, timeperiod=period)
                
                # Weighted Moving Average
                df[f"wma_{period}"] = talib.WMA(close_prices, timeperiod=period)
        
        # Momentum Indicators
        if len(close_prices) > 14:
            # RSI
            df["rsi_14"] = talib.RSI(close_prices, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd
            
            # Williams %R
            df["williams_r"] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Rate of Change
            df["roc_10"] = talib.ROC(close_prices, timeperiod=10)
        
        # MACD
        if len(close_prices) > 26:
            macd, macdsignal, macdhist = talib.MACD(close_prices)
            df["macd"] = macd
            df["macd_signal"] = macdsignal
            df["macd_histogram"] = macdhist
        
        # Bollinger Bands
        if len(close_prices) > 20:
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
            df["bb_upper"] = upper
            df["bb_middle"] = middle
            df["bb_lower"] = lower
            df["bb_width"] = (upper - lower) / middle
            df["bb_position"] = (close_prices - lower) / (upper - lower)
        
        # Volatility Indicators
        if len(close_prices) > 14:
            # Average True Range
            df["atr_14"] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Normalized ATR
            df["natr_14"] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Volume Indicators
        if np.sum(volume) > 0:  # Only if we have volume data
            # On Balance Volume
            df["obv"] = talib.OBV(close_prices, volume)
            
            # Volume-weighted Average Price (approximation)
            if len(df) > 20:
                df["vwap_20"] = self._calculate_vwap(df, 20)
            
            # Accumulation/Distribution Line
            df["ad_line"] = talib.AD(high_prices, low_prices, close_prices, volume)
        
        # Trend Indicators
        if len(close_prices) > 14:
            # ADX (Average Directional Index)
            df["adx_14"] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Parabolic SAR
            df["sar"] = talib.SAR(high_prices, low_prices)
        
        # Exotic Indicators
        if len(close_prices) > 20:
            # Chande Momentum Oscillator
            df["cmo_14"] = talib.CMO(close_prices, timeperiod=14)
            
            # True Strength Index (approximation)
            df["tsi"] = self._calculate_tsi(close_prices)
            
            # Ulcer Index
            df["ulcer_index"] = self._calculate_ulcer_index(close_prices)
        
        return df
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features and regime detection."""
        df = data.copy()
        
        # Calculate returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Rolling statistical features
        for period in [10, 20, 50]:
            if len(df) > period:
                # Rolling volatility
                df[f"volatility_{period}"] = df["returns"].rolling(window=period).std()
                
                # Rolling skewness
                df[f"skewness_{period}"] = df["returns"].rolling(window=period).skew()
                
                # Rolling kurtosis
                df[f"kurtosis_{period}"] = df["returns"].rolling(window=period).kurt()
                
                # Rolling autocorrelation
                df[f"autocorr_{period}"] = df["returns"].rolling(window=period).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
                )
                
                # Rolling Sharpe ratio (annualized)
                df[f"sharpe_{period}"] = (
                    df["returns"].rolling(window=period).mean() * 252 /
                    df["returns"].rolling(window=period).std() / np.sqrt(252)
                )
        
        # Price momentum features
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1
                df[f"price_rank_{period}"] = df["close"].rolling(window=period).rank(pct=True)
        
        # Realized volatility clustering
        if len(df) > 50:
            df["vol_clustering"] = self._detect_volatility_clustering(df["returns"])
        
        # Regime detection
        if len(df) > 100:
            df["volatility_regime"] = self._detect_volatility_regime(df["returns"])
            df["trend_regime"] = self._detect_trend_regime(df["close"])
        
        return df
    
    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        df = data.copy()
        
        if not all(col in df.columns for col in ["open", "high", "low", "close"]):
            return df
        
        open_prices = df["open"].values.astype(float)
        high_prices = df["high"].values.astype(float)
        low_prices = df["low"].values.astype(float)
        close_prices = df["close"].values.astype(float)
        
        # Candlestick patterns
        candlestick_patterns = {
            "doji": talib.CDLDOJI,
            "hammer": talib.CDLHAMMER,
            "hanging_man": talib.CDLHANGINGMAN,
            "engulfing": talib.CDLENGULFING,
            "piercing": talib.CDLPIERCING,
            "dark_cloud": talib.CDLDARKCLOUDCOVER,
            "morning_star": talib.CDLMORNINGSTAR,
            "evening_star": talib.CDLEVENINGSTAR,
            "shooting_star": talib.CDLSHOOTINGSTAR,
            "inverted_hammer": talib.CDLINVERTEDHAMMER
        }
        
        for pattern_name, pattern_func in candlestick_patterns.items():
            try:
                df[f"pattern_{pattern_name}"] = pattern_func(
                    open_prices, high_prices, low_prices, close_prices
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate {pattern_name} pattern: {e}")
                df[f"pattern_{pattern_name}"] = 0
        
        # Chart patterns (simplified detection)
        if len(df) > 20:
            df["support_resistance"] = self._detect_support_resistance(df["close"])
            df["breakout_signal"] = self._detect_breakouts(df)
            df["consolidation"] = self._detect_consolidation(df["close"])
        
        # Fractal patterns
        if len(df) > 10:
            df["fractal_high"] = self._detect_fractals(df["high"], fractal_type="high")
            df["fractal_low"] = self._detect_fractals(df["low"], fractal_type="low")
        
        return df
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        df = data.copy()
        
        # Price impact and liquidity measures
        if "volume" in df.columns and df["volume"].sum() > 0:
            # Volume-weighted return impact
            df["volume_impact"] = df["returns"] / (df["volume"] + 1)
            
            # Liquidity measure (Amihud illiquidity)
            df["amihud_illiquidity"] = np.abs(df["returns"]) / (df["volume"] + 1)
            
            # Volume rate of change
            df["volume_roc"] = df["volume"].pct_change()
            
            # Volume momentum
            for period in [5, 10]:
                if len(df) > period:
                    df[f"volume_momentum_{period}"] = (
                        df["volume"] / df["volume"].rolling(window=period).mean() - 1
                    )
        
        # Price gaps and jumps
        if len(df) > 1:
            # Gap detection
            df["gap"] = df["open"] / df["close"].shift(1) - 1
            df["gap_up"] = (df["gap"] > 0.02).astype(int)  # 2% gap up
            df["gap_down"] = (df["gap"] < -0.02).astype(int)  # 2% gap down
            
            # Intraday range
            df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
            
            # Opening vs closing bias
            df["open_close_bias"] = (df["close"] - df["open"]) / df["open"]
        
        # Tick-by-tick proxies (using OHLC)
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # Approximated bid-ask spread
            df["approx_spread"] = (df["high"] - df["low"]) / df["close"]
            
            # Price reversal indicator
            df["price_reversal"] = self._detect_price_reversals(df)
        
        return df
    
    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap = (typical_price * data["volume"]).rolling(window=period).sum() / \
               data["volume"].rolling(window=period).sum()
        return vwap
    
    def _calculate_tsi(self, close_prices: np.ndarray, long_period: int = 25, short_period: int = 13) -> pd.Series:
        """Calculate True Strength Index."""
        price_changes = np.diff(close_prices)
        price_changes = np.concatenate([[0], price_changes])
        
        # Double smoothed momentum
        first_smooth = pd.Series(price_changes).ewm(span=long_period).mean()
        second_smooth = first_smooth.ewm(span=short_period).mean()
        
        # Double smoothed absolute momentum
        abs_first_smooth = pd.Series(np.abs(price_changes)).ewm(span=long_period).mean()
        abs_second_smooth = abs_first_smooth.ewm(span=short_period).mean()
        
        tsi = 100 * (second_smooth / abs_second_smooth)
        return tsi
    
    def _calculate_ulcer_index(self, close_prices: np.ndarray, period: int = 14) -> pd.Series:
        """Calculate Ulcer Index (downside volatility measure)."""
        close_series = pd.Series(close_prices)
        rolling_max = close_series.rolling(window=period).max()
        percentage_drawdown = 100 * (close_series - rolling_max) / rolling_max
        ulcer_index = np.sqrt(percentage_drawdown.rolling(window=period).apply(lambda x: (x**2).mean()))
        return ulcer_index
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> pd.Series:
        """Detect volatility clustering using GARCH-like approach."""
        squared_returns = returns**2
        
        # Simple volatility clustering indicator
        short_vol = squared_returns.rolling(window=5).mean()
        long_vol = squared_returns.rolling(window=20).mean()
        
        clustering = short_vol / (long_vol + 1e-8)  # Avoid division by zero
        return clustering
    
    def _detect_volatility_regime(self, returns: pd.Series, threshold: float = 1.5) -> pd.Series:
        """Detect high/low volatility regimes."""
        rolling_vol = returns.rolling(window=20).std()
        median_vol = rolling_vol.median()
        
        regime = pd.Series(0, index=returns.index)
        regime[rolling_vol > median_vol * threshold] = 1  # High volatility
        regime[rolling_vol < median_vol / threshold] = -1  # Low volatility
        
        return regime
    
    def _detect_trend_regime(self, prices: pd.Series) -> pd.Series:
        """Detect trend regimes using moving average crossovers."""
        short_ma = prices.rolling(window=20).mean()
        long_ma = prices.rolling(window=50).mean()
        
        regime = pd.Series(0, index=prices.index)
        regime[short_ma > long_ma] = 1  # Uptrend
        regime[short_ma < long_ma] = -1  # Downtrend
        
        return regime
    
    def _detect_support_resistance(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Detect support and resistance levels."""
        rolling_min = prices.rolling(window=window).min()
        rolling_max = prices.rolling(window=window).max()
        
        # Distance from support/resistance
        support_distance = (prices - rolling_min) / prices
        resistance_distance = (rolling_max - prices) / prices
        
        # Combine into single indicator
        sr_indicator = resistance_distance - support_distance
        return sr_indicator
    
    def _detect_breakouts(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Detect price breakouts from trading ranges."""
        if "close" not in data.columns:
            return pd.Series(0, index=data.index)
        
        prices = data["close"]
        rolling_max = prices.rolling(window=period).max()
        rolling_min = prices.rolling(window=period).min()
        
        breakout = pd.Series(0, index=data.index)
        breakout[prices > rolling_max.shift(1)] = 1  # Upward breakout
        breakout[prices < rolling_min.shift(1)] = -1  # Downward breakout
        
        return breakout
    
    def _detect_consolidation(self, prices: pd.Series, period: int = 20, threshold: float = 0.05) -> pd.Series:
        """Detect price consolidation periods."""
        rolling_std = prices.rolling(window=period).std()
        rolling_mean = prices.rolling(window=period).mean()
        
        # Normalized volatility
        normalized_vol = rolling_std / rolling_mean
        
        # Consolidation when volatility is below threshold
        consolidation = (normalized_vol < threshold).astype(int)
        return consolidation
    
    def _detect_fractals(self, prices: pd.Series, fractal_type: str = "high", window: int = 2) -> pd.Series:
        """Detect fractal highs and lows."""
        fractals = pd.Series(0, index=prices.index)
        
        if fractal_type == "high":
            # Find local maxima
            for i in range(window, len(prices) - window):
                if all(prices.iloc[i] >= prices.iloc[i-j] for j in range(1, window+1)) and \
                   all(prices.iloc[i] >= prices.iloc[i+j] for j in range(1, window+1)):
                    fractals.iloc[i] = 1
        else:  # fractal_type == "low"
            # Find local minima
            for i in range(window, len(prices) - window):
                if all(prices.iloc[i] <= prices.iloc[i-j] for j in range(1, window+1)) and \
                   all(prices.iloc[i] <= prices.iloc[i+j] for j in range(1, window+1)):
                    fractals.iloc[i] = 1
        
        return fractals
    
    def _detect_price_reversals(self, data: pd.DataFrame) -> pd.Series:
        """Detect price reversal patterns."""
        if not all(col in data.columns for col in ["high", "low", "close"]):
            return pd.Series(0, index=data.index)
        
        reversals = pd.Series(0, index=data.index)
        
        # Simple reversal detection
        for i in range(2, len(data)):
            prev_high = data["high"].iloc[i-1]
            prev_low = data["low"].iloc[i-1]
            curr_close = data["close"].iloc[i]
            
            # Reversal from high
            if data["close"].iloc[i-2] < prev_high and curr_close < prev_high * 0.98:
                reversals.iloc[i] = -1
            
            # Reversal from low
            elif data["close"].iloc[i-2] > prev_low and curr_close > prev_low * 1.02:
                reversals.iloc[i] = 1
        
        return reversals
    
    def _generate_feature_summary(self, enhanced_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of added features."""
        original_columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        new_features = [col for col in enhanced_data.columns if col not in original_columns]
        
        feature_categories = {
            "moving_averages": [f for f in new_features if any(ma in f for ma in ["sma", "ema", "wma"])],
            "momentum": [f for f in new_features if any(mom in f for mom in ["rsi", "stoch", "williams", "roc", "momentum"])],
            "volatility": [f for f in new_features if any(vol in f for vol in ["atr", "bb_", "volatility", "ulcer"])],
            "volume": [f for f in new_features if any(vol in f for vol in ["obv", "vwap", "ad_line", "volume_"])],
            "trend": [f for f in new_features if any(trend in f for trend in ["adx", "sar", "macd", "trend_"])],
            "patterns": [f for f in new_features if "pattern_" in f or "fractal" in f],
            "statistical": [f for f in new_features if any(stat in f for stat in ["returns", "skewness", "kurtosis", "autocorr", "sharpe"])],
            "microstructure": [f for f in new_features if any(micro in f for micro in ["gap", "spread", "reversal", "impact"])]
        }
        
        return {
            "total_features_added": len(new_features),
            "new_features": new_features,
            "feature_categories": feature_categories,
            "category_counts": {cat: len(features) for cat, features in feature_categories.items()},
            "data_shape": enhanced_data.shape,
            "missing_data_percentage": (enhanced_data.isnull().sum().sum() / enhanced_data.size) * 100
        }