"""
Mathematical utilities for the trading system.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import warnings


class MathUtils:
    """
    Mathematical utility functions for trading and finance calculations.
    """
    
    @staticmethod
    def calculate_returns(prices: Union[pd.Series, np.ndarray], 
                         method: str = "simple") -> Union[pd.Series, np.ndarray]:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: 'simple' or 'log' returns
            
        Returns:
            Return series
        """
        if isinstance(prices, pd.Series):
            if method == "simple":
                return prices.pct_change()
            elif method == "log":
                return np.log(prices / prices.shift(1))
        else:
            if method == "simple":
                return np.diff(prices) / prices[:-1]
            elif method == "log":
                return np.diff(np.log(prices))
        
        raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def calculate_volatility(returns: Union[pd.Series, np.ndarray], 
                           annualize: bool = True, 
                           periods_per_year: int = 252) -> float:
        """
        Calculate volatility from returns.
        
        Args:
            returns: Return series
            annualize: Whether to annualize the volatility
            periods_per_year: Number of periods per year for annualization
            
        Returns:
            Volatility
        """
        if isinstance(returns, pd.Series):
            vol = returns.std()
        else:
            vol = np.std(returns, ddof=1)
        
        if annualize:
            vol *= np.sqrt(periods_per_year)
        
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], 
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        if isinstance(returns, pd.Series):
            mean_return = returns.mean()
            std_return = returns.std()
        else:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
        
        # Annualize
        annual_return = mean_return * periods_per_year
        annual_vol = std_return * np.sqrt(periods_per_year)
        
        if annual_vol == 0:
            return 0.0
        
        return (annual_return - risk_free_rate) / annual_vol
    
    @staticmethod
    def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], 
                               risk_free_rate: float = 0.0,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility).
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        if isinstance(returns, pd.Series):
            mean_return = returns.mean()
            downside_returns = returns[returns < 0]
        else:
            mean_return = np.mean(returns)
            downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns, ddof=1)
        
        # Annualize
        annual_return = mean_return * periods_per_year
        annual_downside_vol = downside_std * np.sqrt(periods_per_year)
        
        if annual_downside_vol == 0:
            return np.inf
        
        return (annual_return - risk_free_rate) / annual_downside_vol
    
    @staticmethod
    def calculate_max_drawdown(prices: Union[pd.Series, np.ndarray]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if isinstance(prices, pd.Series):
            cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
            prices_array = cumulative.values
        else:
            prices_array = prices
        
        peak = np.maximum.accumulate(prices_array)
        drawdown = (prices_array - peak) / peak
        
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        
        # Find the peak before the max drawdown
        start_idx = np.argmax(prices_array[:end_idx+1])
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calculate_var(returns: Union[pd.Series, np.ndarray], 
                     confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value (negative number)
        """
        if isinstance(returns, pd.Series):
            returns_clean = returns.dropna()
        else:
            returns_clean = returns[~np.isnan(returns)]
        
        return np.percentile(returns_clean, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: Union[pd.Series, np.ndarray], 
                      confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        var = MathUtils.calculate_var(returns, confidence_level)
        
        if isinstance(returns, pd.Series):
            tail_returns = returns[returns <= var]
        else:
            tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    @staticmethod
    def calculate_beta(asset_returns: Union[pd.Series, np.ndarray], 
                      market_returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate beta coefficient.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
        """
        if isinstance(asset_returns, pd.Series) and isinstance(market_returns, pd.Series):
            # Align the series
            combined = pd.concat([asset_returns, market_returns], axis=1).dropna()
            asset_clean = combined.iloc[:, 0]
            market_clean = combined.iloc[:, 1]
        else:
            # For numpy arrays, assume they're already aligned
            asset_clean = asset_returns
            market_clean = market_returns
        
        covariance = np.cov(asset_clean, market_clean)[0, 1]
        market_variance = np.var(market_clean, ddof=1)
        
        return covariance / market_variance if market_variance != 0 else 0
    
    @staticmethod
    def calculate_correlation(x: Union[pd.Series, np.ndarray], 
                            y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate correlation coefficient between two series.
        
        Args:
            x: First series
            y: Second series
            
        Returns:
            Correlation coefficient
        """
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            combined = pd.concat([x, y], axis=1).dropna()
            return combined.corr().iloc[0, 1]
        else:
            return np.corrcoef(x, y)[0, 1]
    
    @staticmethod
    def calculate_rolling_correlation(x: pd.Series, y: pd.Series, 
                                    window: int) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            x: First series
            y: Second series
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        return x.rolling(window).corr(y)
    
    @staticmethod
    def z_score(data: Union[pd.Series, np.ndarray], 
               window: Optional[int] = None) -> Union[pd.Series, np.ndarray]:
        """
        Calculate z-score of data.
        
        Args:
            data: Input data
            window: Rolling window size (if None, use entire series)
            
        Returns:
            Z-score values
        """
        if isinstance(data, pd.Series):
            if window is not None:
                mean = data.rolling(window).mean()
                std = data.rolling(window).std()
                return (data - mean) / std
            else:
                return (data - data.mean()) / data.std()
        else:
            if window is not None:
                # For numpy arrays with rolling window, need to implement manually
                z_scores = np.full_like(data, np.nan)
                for i in range(window-1, len(data)):
                    window_data = data[i-window+1:i+1]
                    z_scores[i] = (data[i] - np.mean(window_data)) / np.std(window_data, ddof=1)
                return z_scores
            else:
                return (data - np.mean(data)) / np.std(data, ddof=1)
    
    @staticmethod
    def calculate_information_ratio(active_returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate information ratio.
        
        Args:
            active_returns: Active returns (portfolio - benchmark)
            
        Returns:
            Information ratio
        """
        if isinstance(active_returns, pd.Series):
            mean_active = active_returns.mean()
            std_active = active_returns.std()
        else:
            mean_active = np.mean(active_returns)
            std_active = np.std(active_returns, ddof=1)
        
        return mean_active / std_active if std_active != 0 else 0
    
    @staticmethod
    def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray], 
                              periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (Annual Return / Max Drawdown).
        
        Args:
            returns: Return series
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        if isinstance(returns, pd.Series):
            annual_return = returns.mean() * periods_per_year
            cumulative_returns = (1 + returns).cumprod()
        else:
            annual_return = np.mean(returns) * periods_per_year
            cumulative_returns = np.cumprod(1 + returns)
        
        max_dd, _, _ = MathUtils.calculate_max_drawdown(cumulative_returns)
        
        return annual_return / abs(max_dd) if max_dd != 0 else np.inf
    
    @staticmethod
    def normalize_weights(weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to sum to 1.
        
        Args:
            weights: Weight array
            
        Returns:
            Normalized weights
        """
        weight_sum = np.sum(weights)
        return weights / weight_sum if weight_sum != 0 else weights
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray, 
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio expected return and volatility.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Tuple of (expected_return, volatility)
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    @staticmethod
    def detect_regime_change(data: Union[pd.Series, np.ndarray], 
                           method: str = "variance") -> np.ndarray:
        """
        Simple regime change detection.
        
        Args:
            data: Input data series
            method: Detection method ('variance' or 'mean')
            
        Returns:
            Array of regime indicators
        """
        if isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = data
        
        if method == "variance":
            # Use rolling variance to detect regime changes
            window = min(50, len(data_array) // 4)
            rolling_var = pd.Series(data_array).rolling(window).var()
            threshold = rolling_var.median() * 2
            regimes = (rolling_var > threshold).astype(int)
        elif method == "mean":
            # Use rolling mean to detect regime changes
            window = min(50, len(data_array) // 4)
            rolling_mean = pd.Series(data_array).rolling(window).mean()
            threshold = rolling_mean.std()
            regimes = (rolling_mean > rolling_mean.median() + threshold).astype(int)
        else:
            raise ValueError("Method must be 'variance' or 'mean'")
        
        return regimes.values if isinstance(regimes, pd.Series) else regimes