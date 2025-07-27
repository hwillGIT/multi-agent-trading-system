"""
Data validation utilities for the trading system.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..base.exceptions import ValidationError


class DataValidator:
    """
    Comprehensive data validation for trading system data.
    """
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        check_nulls: bool = True,
        numeric_columns: Optional[List[str]] = None
    ) -> None:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            check_nulls: Whether to check for null values
            numeric_columns: Columns that must be numeric
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        
        if len(df) < min_rows:
            raise ValidationError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
        
        if check_nulls and df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            raise ValidationError(f"Found null values in columns: {null_cols}")
        
        if numeric_columns:
            for col in numeric_columns:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValidationError(f"Column '{col}' must be numeric")
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> None:
        """
        Validate price data format and consistency.
        
        Args:
            df: DataFrame with OHLCV data
            
        Raises:
            ValidationError: If validation fails
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        DataValidator.validate_dataframe(
            df,
            required_columns=required_columns,
            numeric_columns=required_columns
        )
        
        # Check OHLC consistency
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            raise ValidationError("Invalid OHLC data: high/low consistency check failed")
        
        # Check for negative values
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            raise ValidationError("Price and volume data cannot be negative")
        
        # Check for zero volume (might be suspicious)
        if (df['volume'] == 0).sum() > len(df) * 0.1:  # More than 10% zero volume
            raise ValidationError("Too many zero volume periods detected")
    
    @staticmethod
    def validate_returns(returns: Union[pd.Series, np.ndarray]) -> None:
        """
        Validate return data for extreme outliers.
        
        Args:
            returns: Return data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = returns
        
        # Remove NaN values for validation
        clean_returns = returns_array[~np.isnan(returns_array)]
        
        if len(clean_returns) == 0:
            raise ValidationError("No valid return data found")
        
        # Check for extreme outliers (> 100% daily return)
        extreme_returns = np.abs(clean_returns) > 1.0
        if extreme_returns.sum() > len(clean_returns) * 0.01:  # More than 1% extreme returns
            raise ValidationError("Too many extreme returns (>100%) detected")
        
        # Check for infinite values
        if np.isinf(clean_returns).any():
            raise ValidationError("Infinite values found in returns")
    
    @staticmethod
    def validate_signal(signal: Union[pd.Series, np.ndarray]) -> None:
        """
        Validate trading signal format.
        
        Args:
            signal: Trading signal to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(signal, pd.Series):
            signal_array = signal.values
        else:
            signal_array = signal
        
        # Check signal range (-1 to 1 for continuous signals, or specific discrete values)
        unique_values = np.unique(signal_array[~np.isnan(signal_array)])
        
        # Allow for continuous signals between -1 and 1, or discrete {-1, 0, 1}
        valid_continuous = np.all((unique_values >= -1) & (unique_values <= 1))
        valid_discrete = np.all(np.isin(unique_values, [-1, 0, 1]))
        
        if not (valid_continuous or valid_discrete):
            raise ValidationError("Signal values must be between -1 and 1, or discrete {-1, 0, 1}")
    
    @staticmethod
    def validate_risk_metrics(metrics: Dict[str, float]) -> None:
        """
        Validate risk metrics for reasonable ranges.
        
        Args:
            metrics: Dictionary of risk metrics
            
        Raises:
            ValidationError: If validation fails
        """
        # Expected ranges for common risk metrics
        metric_ranges = {
            'sharpe_ratio': (-5, 10),
            'sortino_ratio': (-5, 15),
            'max_drawdown': (0, 1),
            'volatility': (0, 5),
            'var_95': (-1, 0),
            'var_99': (-1, 0),
            'beta': (-3, 3),
            'alpha': (-1, 1)
        }
        
        for metric, value in metrics.items():
            if metric in metric_ranges:
                min_val, max_val = metric_ranges[metric]
                if not (min_val <= value <= max_val):
                    raise ValidationError(
                        f"Risk metric '{metric}' value {value} outside expected range [{min_val}, {max_val}]"
                    )
            
            # Check for NaN or infinite values
            if np.isnan(value) or np.isinf(value):
                raise ValidationError(f"Risk metric '{metric}' has invalid value: {value}")
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> None:
        """
        Validate portfolio weights sum to 1 and are within reasonable bounds.
        
        Args:
            weights: Dictionary of asset weights
            
        Raises:
            ValidationError: If validation fails
        """
        weight_values = list(weights.values())
        
        # Check weights sum to approximately 1
        total_weight = sum(weight_values)
        if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
            raise ValidationError(f"Portfolio weights sum to {total_weight}, should be 1.0")
        
        # Check individual weight bounds
        for asset, weight in weights.items():
            if weight < -1.0 or weight > 1.0:
                raise ValidationError(f"Weight for {asset} ({weight}) outside bounds [-1, 1]")
            
            if np.isnan(weight) or np.isinf(weight):
                raise ValidationError(f"Invalid weight for {asset}: {weight}")
    
    @staticmethod
    def validate_timestamp_series(timestamps: pd.Series) -> None:
        """
        Validate timestamp series for proper ordering and frequency.
        
        Args:
            timestamps: Series of timestamps
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(timestamps.iloc[0], (datetime, pd.Timestamp)):
            raise ValidationError("Timestamps must be datetime objects")
        
        # Check if sorted
        if not timestamps.is_monotonic_increasing:
            raise ValidationError("Timestamps must be in ascending order")
        
        # Check for duplicates
        if timestamps.duplicated().any():
            raise ValidationError("Duplicate timestamps found")
        
        # Check for reasonable gaps (no more than 7 days between consecutive points)
        time_diffs = timestamps.diff().dropna()
        max_gap = time_diffs.max()
        if max_gap > timedelta(days=7):
            raise ValidationError(f"Large time gap detected: {max_gap}")
    
    @staticmethod
    def validate_correlation_matrix(corr_matrix: pd.DataFrame) -> None:
        """
        Validate correlation matrix properties.
        
        Args:
            corr_matrix: Correlation matrix to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(corr_matrix, pd.DataFrame):
            raise ValidationError("Correlation matrix must be a DataFrame")
        
        # Check square matrix
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValidationError("Correlation matrix must be square")
        
        # Check symmetric
        if not np.allclose(corr_matrix.values, corr_matrix.T.values, atol=1e-6):
            raise ValidationError("Correlation matrix must be symmetric")
        
        # Check diagonal is 1
        diagonal = np.diag(corr_matrix.values)
        if not np.allclose(diagonal, 1.0, atol=1e-6):
            raise ValidationError("Correlation matrix diagonal must be 1")
        
        # Check values in [-1, 1]
        if (corr_matrix.abs() > 1.0).any().any():
            raise ValidationError("Correlation values must be between -1 and 1")
        
        # Check for NaN values
        if corr_matrix.isnull().any().any():
            raise ValidationError("Correlation matrix contains NaN values")