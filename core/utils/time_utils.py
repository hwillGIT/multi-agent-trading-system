"""
Time utilities for the trading system.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
import pandas as pd
import numpy as np


class TimeUtils:
    """
    Utility functions for time-related operations in trading systems.
    """
    
    @staticmethod
    def get_market_hours(exchange: str = "NYSE") -> tuple:
        """
        Get market open/close hours for different exchanges.
        
        Args:
            exchange: Exchange name (NYSE, NASDAQ, LSE, etc.)
            
        Returns:
            Tuple of (open_hour, close_hour) in UTC
        """
        market_hours = {
            "NYSE": (14, 21),      # 9:30 AM - 4:00 PM EST
            "NASDAQ": (14, 21),    # 9:30 AM - 4:00 PM EST
            "LSE": (8, 16),        # 8:00 AM - 4:30 PM GMT
            "TSE": (0, 6),         # 9:00 AM - 3:00 PM JST
            "HKE": (1, 8),         # 9:30 AM - 4:00 PM HKT
            "ASX": (23, 6),        # 10:00 AM - 4:00 PM AEST
        }
        
        return market_hours.get(exchange.upper(), (14, 21))
    
    @staticmethod
    def is_market_open(dt: datetime, exchange: str = "NYSE") -> bool:
        """
        Check if market is open at given datetime.
        
        Args:
            dt: Datetime to check
            exchange: Exchange name
            
        Returns:
            True if market is open
        """
        # Convert to UTC if timezone-aware
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        
        # Check if weekday (Monday=0, Sunday=6)
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        
        open_hour, close_hour = TimeUtils.get_market_hours(exchange)
        return open_hour <= dt.hour < close_hour
    
    @staticmethod
    def get_trading_days(start_date: datetime, end_date: datetime, 
                        exchange: str = "NYSE") -> List[datetime]:
        """
        Get list of trading days between start and end dates.
        
        Args:
            start_date: Start date
            end_date: End date
            exchange: Exchange name
            
        Returns:
            List of trading day datetimes
        """
        # US market holidays (simplified - in production use a proper holiday calendar)
        us_holidays = [
            "2023-01-02",  # New Year's Day (observed)
            "2023-01-16",  # Martin Luther King Jr. Day
            "2023-02-20",  # Presidents' Day
            "2023-04-07",  # Good Friday
            "2023-05-29",  # Memorial Day
            "2023-06-19",  # Juneteenth
            "2023-07-04",  # Independence Day
            "2023-09-04",  # Labor Day
            "2023-11-23",  # Thanksgiving
            "2023-12-25",  # Christmas Day
        ]
        
        holidays = pd.to_datetime(us_holidays).date if exchange.upper() in ["NYSE", "NASDAQ"] else []
        
        trading_days = []
        current_date = start_date.date() if isinstance(start_date, datetime) else start_date
        end_date = end_date.date() if isinstance(end_date, datetime) else end_date
        
        while current_date <= end_date:
            # Check if weekday and not holiday
            if current_date.weekday() < 5 and current_date not in holidays:
                trading_days.append(datetime.combine(current_date, datetime.min.time()))
            current_date += timedelta(days=1)
        
        return trading_days
    
    @staticmethod
    def align_timestamps(df1: pd.DataFrame, df2: pd.DataFrame, 
                        method: str = "inner") -> tuple:
        """
        Align timestamps between two DataFrames.
        
        Args:
            df1: First DataFrame with datetime index
            df2: Second DataFrame with datetime index
            method: Alignment method ('inner', 'outer', 'left', 'right')
            
        Returns:
            Tuple of aligned DataFrames
        """
        if method == "inner":
            common_index = df1.index.intersection(df2.index)
        elif method == "outer":
            common_index = df1.index.union(df2.index)
        elif method == "left":
            common_index = df1.index
        elif method == "right":
            common_index = df2.index
        else:
            raise ValueError("Method must be 'inner', 'outer', 'left', or 'right'")
        
        aligned_df1 = df1.reindex(common_index)
        aligned_df2 = df2.reindex(common_index)
        
        return aligned_df1, aligned_df2
    
    @staticmethod
    def resample_to_frequency(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample time series data to specified frequency.
        
        Args:
            df: DataFrame with datetime index
            frequency: Target frequency ('1D', '1H', '5T', etc.)
            
        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Define aggregation rules for different column types
        agg_rules = {}
        for col in df.columns:
            if col.lower() in ['open']:
                agg_rules[col] = 'first'
            elif col.lower() in ['high']:
                agg_rules[col] = 'max'
            elif col.lower() in ['low']:
                agg_rules[col] = 'min'
            elif col.lower() in ['close', 'price']:
                agg_rules[col] = 'last'
            elif col.lower() in ['volume']:
                agg_rules[col] = 'sum'
            else:
                agg_rules[col] = 'last'  # Default for other columns
        
        return df.resample(frequency).agg(agg_rules).dropna()
    
    @staticmethod
    def get_time_to_expiry(expiry_date: datetime, current_date: Optional[datetime] = None) -> float:
        """
        Calculate time to expiry in years (useful for options pricing).
        
        Args:
            expiry_date: Expiry datetime
            current_date: Current datetime (default: now)
            
        Returns:
            Time to expiry in years
        """
        if current_date is None:
            current_date = datetime.utcnow()
        
        time_diff = expiry_date - current_date
        return time_diff.total_seconds() / (365.25 * 24 * 3600)
    
    @staticmethod
    def create_time_buckets(df: pd.DataFrame, bucket_size: str = "1H") -> pd.DataFrame:
        """
        Create time buckets for intraday analysis.
        
        Args:
            df: DataFrame with datetime index
            bucket_size: Size of time buckets
            
        Returns:
            DataFrame with time bucket column
        """
        df_copy = df.copy()
        df_copy['time_bucket'] = df_copy.index.floor(bucket_size)
        return df_copy
    
    @staticmethod
    def get_business_day_count(start_date: datetime, end_date: datetime) -> int:
        """
        Count business days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of business days
        """
        return len(pd.bdate_range(start_date, end_date))
    
    @staticmethod
    def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """
        Convert datetime from one timezone to another.
        
        Args:
            dt: Datetime to convert
            from_tz: Source timezone
            to_tz: Target timezone
            
        Returns:
            Converted datetime
        """
        if dt.tzinfo is None:
            # Assume the datetime is in from_tz
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to target timezone
        return dt.astimezone(timezone.utc)
    
    @staticmethod
    def get_market_calendar(year: int, exchange: str = "NYSE") -> List[datetime]:
        """
        Get trading calendar for a specific year and exchange.
        
        Args:
            year: Year to get calendar for
            exchange: Exchange name
            
        Returns:
            List of trading dates
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        return TimeUtils.get_trading_days(start_date, end_date, exchange)