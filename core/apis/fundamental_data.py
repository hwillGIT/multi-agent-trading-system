"""
Fundamental data API interface.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from loguru import logger

from ..base.config import config
from ..base.exceptions import APIError, DataError


class FundamentalDataAPI:
    """
    API for fetching fundamental data (earnings, financials, ratios, etc.).
    """
    
    def __init__(self):
        self.alpha_vantage_key = config.alpha_vantage_api_key
        self.base_url = config.get("market_data.alpha_vantage.base_url")
        self.logger = logger.bind(service="fundamental_data_api")
        
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview and key metrics.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company overview data
        """
        if not self.alpha_vantage_key:
            raise APIError("Alpha Vantage API key not configured")
            
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'Symbol' not in data:
                raise DataError(f"No overview data found for {symbol}")
            
            return data
            
        except Exception as e:
            raise APIError(f"Fundamental data API error for {symbol}: {str(e)}")
    
    async def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """
        Get annual income statement data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with income statement data
        """
        try:
            params = {
                'function': 'INCOME_STATEMENT',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'annualReports' not in data:
                raise DataError(f"No income statement data found for {symbol}")
            
            df = pd.DataFrame(data['annualReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
            return df
            
        except Exception as e:
            raise APIError(f"Income statement API error for {symbol}: {str(e)}")
    
    async def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """
        Get annual balance sheet data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with balance sheet data
        """
        try:
            params = {
                'function': 'BALANCE_SHEET',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'annualReports' not in data:
                raise DataError(f"No balance sheet data found for {symbol}")
            
            df = pd.DataFrame(data['annualReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
            return df
            
        except Exception as e:
            raise APIError(f"Balance sheet API error for {symbol}: {str(e)}")
    
    async def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """
        Get annual cash flow statement data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with cash flow data
        """
        try:
            params = {
                'function': 'CASH_FLOW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'annualReports' not in data:
                raise DataError(f"No cash flow data found for {symbol}")
            
            df = pd.DataFrame(data['annualReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
            return df
            
        except Exception as e:
            raise APIError(f"Cash flow API error for {symbol}: {str(e)}")
    
    async def get_earnings_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get earnings data (annual and quarterly).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with annual and quarterly earnings DataFrames
        """
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            result = {}
            
            if 'annualEarnings' in data:
                annual_df = pd.DataFrame(data['annualEarnings'])
                annual_df['fiscalDateEnding'] = pd.to_datetime(annual_df['fiscalDateEnding'])
                result['annual'] = annual_df
            
            if 'quarterlyEarnings' in data:
                quarterly_df = pd.DataFrame(data['quarterlyEarnings'])
                quarterly_df['fiscalDateEnding'] = pd.to_datetime(quarterly_df['fiscalDateEnding'])
                quarterly_df['reportedDate'] = pd.to_datetime(quarterly_df['reportedDate'])
                result['quarterly'] = quarterly_df
            
            if not result:
                raise DataError(f"No earnings data found for {symbol}")
            
            return result
            
        except Exception as e:
            raise APIError(f"Earnings API error for {symbol}: {str(e)}")
    
    def calculate_financial_ratios(self, overview: Dict[str, Any], 
                                 income_statement: pd.DataFrame,
                                 balance_sheet: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key financial ratios from fundamental data.
        
        Args:
            overview: Company overview data
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            
        Returns:
            Dictionary of calculated financial ratios
        """
        try:
            ratios = {}
            
            # Get latest data
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[0]
            
            # Parse numeric values from strings (Alpha Vantage returns strings)
            def safe_float(value):
                try:
                    return float(value) if value and value != 'None' else 0.0
                except (ValueError, TypeError):
                    return 0.0
            
            # Profitability Ratios
            net_income = safe_float(latest_income.get('netIncome', 0))
            total_revenue = safe_float(latest_income.get('totalRevenue', 0))
            total_assets = safe_float(latest_balance.get('totalAssets', 0))
            shareholders_equity = safe_float(latest_balance.get('totalShareholderEquity', 0))
            
            if total_revenue > 0:
                ratios['net_profit_margin'] = net_income / total_revenue
                
            if total_assets > 0:
                ratios['roa'] = net_income / total_assets  # Return on Assets
                
            if shareholders_equity > 0:
                ratios['roe'] = net_income / shareholders_equity  # Return on Equity
            
            # Liquidity Ratios
            current_assets = safe_float(latest_balance.get('totalCurrentAssets', 0))
            current_liabilities = safe_float(latest_balance.get('totalCurrentLiabilities', 0))
            
            if current_liabilities > 0:
                ratios['current_ratio'] = current_assets / current_liabilities
            
            # Leverage Ratios
            total_debt = safe_float(latest_balance.get('longTermDebt', 0)) + safe_float(latest_balance.get('shortTermDebt', 0))
            
            if shareholders_equity > 0:
                ratios['debt_to_equity'] = total_debt / shareholders_equity
                
            if total_assets > 0:
                ratios['debt_to_assets'] = total_debt / total_assets
            
            # Valuation Ratios from overview
            market_cap = safe_float(overview.get('MarketCapitalization', 0))
            pe_ratio = safe_float(overview.get('PERatio', 0))
            pb_ratio = safe_float(overview.get('PriceToBookRatio', 0))
            
            ratios['market_cap'] = market_cap
            ratios['pe_ratio'] = pe_ratio
            ratios['pb_ratio'] = pb_ratio
            
            # Efficiency Ratios
            if total_assets > 0 and total_revenue > 0:
                ratios['asset_turnover'] = total_revenue / total_assets
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating financial ratios: {e}")
            return {}
    
    async def get_comprehensive_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with all fundamental data and calculated ratios
        """
        try:
            # Fetch all fundamental data concurrently
            tasks = [
                self.get_company_overview(symbol),
                self.get_income_statement(symbol),
                self.get_balance_sheet(symbol),
                self.get_cash_flow(symbol),
                self.get_earnings_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            overview = results[0] if not isinstance(results[0], Exception) else {}
            income_statement = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
            balance_sheet = results[2] if not isinstance(results[2], Exception) else pd.DataFrame()
            cash_flow = results[3] if not isinstance(results[3], Exception) else pd.DataFrame()
            earnings = results[4] if not isinstance(results[4], Exception) else {}
            
            # Calculate financial ratios
            financial_ratios = self.calculate_financial_ratios(overview, income_statement, balance_sheet)
            
            return {
                'symbol': symbol,
                'overview': overview,
                'income_statement': income_statement,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'earnings': earnings,
                'financial_ratios': financial_ratios,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            raise APIError(f"Comprehensive fundamental data error for {symbol}: {str(e)}")