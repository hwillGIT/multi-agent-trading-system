"""
Options data API interface.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
import yfinance as yf

from ..base.config import config
from ..base.exceptions import APIError, DataError
from ..utils.math_utils import MathUtils


class OptionsDataAPI:
    """
    API for fetching options chain data and calculating options metrics.
    """
    
    def __init__(self):
        self.logger = logger.bind(service="options_data_api")
        
    async def get_options_chain(self, symbol: str, 
                              expiration_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Get options chain data for a symbol.
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date (if None, gets all)
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration_date:
                exp_str = expiration_date.strftime('%Y-%m-%d')
                options_data = ticker.option_chain(exp_str)
            else:
                # Get the nearest expiration
                expirations = ticker.options
                if not expirations:
                    raise DataError(f"No options available for {symbol}")
                options_data = ticker.option_chain(expirations[0])
            
            return {
                'calls': options_data.calls,
                'puts': options_data.puts
            }
            
        except Exception as e:
            raise APIError(f"Options chain API error for {symbol}: {str(e)}")
    
    async def get_all_expirations(self, symbol: str) -> List[str]:
        """
        Get all available expiration dates for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of expiration date strings
        """
        try:
            ticker = yf.Ticker(symbol)
            return list(ticker.options)
            
        except Exception as e:
            raise APIError(f"Options expirations API error for {symbol}: {str(e)}")
    
    async def get_multi_expiration_chain(self, symbol: str, 
                                       max_expirations: int = 6) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get options chains for multiple expirations.
        
        Args:
            symbol: Stock symbol
            max_expirations: Maximum number of expirations to fetch
            
        Returns:
            Nested dictionary with expiration -> call/put type -> DataFrame
        """
        try:
            expirations = await self.get_all_expirations(symbol)
            
            if not expirations:
                raise DataError(f"No options expirations available for {symbol}")
            
            # Limit to max_expirations
            expirations_to_fetch = expirations[:max_expirations]
            
            all_chains = {}
            
            for exp_str in expirations_to_fetch:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                    chain = await self.get_options_chain(symbol, exp_date)
                    all_chains[exp_str] = chain
                except Exception as e:
                    self.logger.warning(f"Failed to fetch options for {symbol} expiration {exp_str}: {e}")
                    continue
            
            return all_chains
            
        except Exception as e:
            raise APIError(f"Multi-expiration options API error for {symbol}: {str(e)}")
    
    def calculate_options_greeks(self, options_df: pd.DataFrame, 
                               spot_price: float,
                               risk_free_rate: float = 0.05,
                               dividend_yield: float = 0.0) -> pd.DataFrame:
        """
        Calculate options Greeks using Black-Scholes model.
        
        Args:
            options_df: Options DataFrame
            spot_price: Current stock price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            DataFrame with Greeks added
        """
        try:
            df = options_df.copy()
            
            # Calculate time to expiration in years
            if 'expiration' in df.columns:
                df['time_to_expiry'] = df['expiration'].apply(
                    lambda x: MathUtils.get_time_to_expiry(pd.to_datetime(x))
                )
            else:
                # Estimate from lastTradeDate if available
                df['time_to_expiry'] = 0.25  # Default to 3 months
            
            # Calculate implied volatility (simplified - would use more sophisticated method in production)
            df['implied_volatility'] = df.get('impliedVolatility', 0.3)  # Default 30%
            
            # Black-Scholes calculations
            for idx, row in df.iterrows():
                try:
                    S = spot_price
                    K = row['strike']
                    T = row['time_to_expiry']
                    r = risk_free_rate
                    q = dividend_yield
                    sigma = row['implied_volatility']
                    
                    if T <= 0 or sigma <= 0:
                        continue
                    
                    # Calculate d1 and d2
                    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    
                    # Standard normal CDF and PDF
                    N_d1 = stats.norm.cdf(d1)
                    N_d2 = stats.norm.cdf(d2)
                    n_d1 = stats.norm.pdf(d1)
                    
                    # Determine if call or put
                    is_call = 'C' in str(row.get('contractSymbol', '')) or 'call' in str(row.get('type', '')).lower()
                    
                    if is_call:
                        # Call Greeks
                        delta = np.exp(-q*T) * N_d1
                        theta = (-S*n_d1*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                                - r*K*np.exp(-r*T)*N_d2 
                                + q*S*np.exp(-q*T)*N_d1) / 365
                    else:
                        # Put Greeks  
                        delta = np.exp(-q*T) * (N_d1 - 1)
                        theta = (-S*n_d1*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                                + r*K*np.exp(-r*T)*(1-N_d2) 
                                - q*S*np.exp(-q*T)*(1-N_d1)) / 365
                    
                    # Common Greeks
                    gamma = n_d1*np.exp(-q*T) / (S*sigma*np.sqrt(T))
                    vega = S*n_d1*np.sqrt(T)*np.exp(-q*T) / 100  # Per 1% change in volatility
                    rho = K*T*np.exp(-r*T)*N_d2 / 100 if is_call else -K*T*np.exp(-r*T)*(1-N_d2) / 100
                    
                    # Add to DataFrame
                    df.at[idx, 'delta'] = delta
                    df.at[idx, 'gamma'] = gamma
                    df.at[idx, 'theta'] = theta
                    df.at[idx, 'vega'] = vega
                    df.at[idx, 'rho'] = rho
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating Greeks for row {idx}: {e}")
                    continue
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating options Greeks: {e}")
            return options_df
    
    def analyze_options_flow(self, options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze options flow and identify unusual activity.
        
        Args:
            options_data: Dictionary with calls and puts DataFrames
            
        Returns:
            Dictionary with flow analysis
        """
        try:
            calls_df = options_data.get('calls', pd.DataFrame())
            puts_df = options_data.get('puts', pd.DataFrame())
            
            analysis = {
                'total_call_volume': 0,
                'total_put_volume': 0,
                'total_call_oi': 0,
                'total_put_oi': 0,
                'put_call_ratio_volume': 0,
                'put_call_ratio_oi': 0,
                'unusual_activity': [],
                'max_pain': None
            }
            
            if not calls_df.empty:
                analysis['total_call_volume'] = calls_df['volume'].fillna(0).sum()
                analysis['total_call_oi'] = calls_df['openInterest'].fillna(0).sum()
                
                # Find unusual call activity
                if 'volume' in calls_df.columns and 'openInterest' in calls_df.columns:
                    calls_df['volume_oi_ratio'] = calls_df['volume'] / (calls_df['openInterest'] + 1)
                    unusual_calls = calls_df[calls_df['volume_oi_ratio'] > 0.5]  # Volume > 50% of OI
                    
                    for _, row in unusual_calls.iterrows():
                        analysis['unusual_activity'].append({
                            'type': 'call',
                            'strike': row['strike'],
                            'volume': row['volume'],
                            'open_interest': row['openInterest'],
                            'ratio': row['volume_oi_ratio']
                        })
            
            if not puts_df.empty:
                analysis['total_put_volume'] = puts_df['volume'].fillna(0).sum()
                analysis['total_put_oi'] = puts_df['openInterest'].fillna(0).sum()
                
                # Find unusual put activity
                if 'volume' in puts_df.columns and 'openInterest' in puts_df.columns:
                    puts_df['volume_oi_ratio'] = puts_df['volume'] / (puts_df['openInterest'] + 1)
                    unusual_puts = puts_df[puts_df['volume_oi_ratio'] > 0.5]
                    
                    for _, row in unusual_puts.iterrows():
                        analysis['unusual_activity'].append({
                            'type': 'put',
                            'strike': row['strike'],
                            'volume': row['volume'],
                            'open_interest': row['openInterest'],
                            'ratio': row['volume_oi_ratio']
                        })
            
            # Calculate put/call ratios
            if analysis['total_call_volume'] > 0:
                analysis['put_call_ratio_volume'] = analysis['total_put_volume'] / analysis['total_call_volume']
            
            if analysis['total_call_oi'] > 0:
                analysis['put_call_ratio_oi'] = analysis['total_put_oi'] / analysis['total_call_oi']
            
            # Calculate max pain (strike with highest total open interest)
            analysis['max_pain'] = self._calculate_max_pain(calls_df, puts_df)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing options flow: {e}")
            return {}
    
    def _calculate_max_pain(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate max pain strike price.
        
        Args:
            calls_df: Calls DataFrame
            puts_df: Puts DataFrame
            
        Returns:
            Max pain strike price
        """
        try:
            all_strikes = set()
            
            if not calls_df.empty and 'strike' in calls_df.columns:
                all_strikes.update(calls_df['strike'].values)
            
            if not puts_df.empty and 'strike' in puts_df.columns:
                all_strikes.update(puts_df['strike'].values)
            
            if not all_strikes:
                return None
            
            max_pain_values = {}
            
            for strike in all_strikes:
                total_pain = 0
                
                # Calculate pain from calls (ITM calls cause pain to sellers)
                if not calls_df.empty:
                    itm_calls = calls_df[calls_df['strike'] < strike]
                    call_pain = ((strike - itm_calls['strike']) * itm_calls['openInterest']).sum()
                    total_pain += call_pain
                
                # Calculate pain from puts (ITM puts cause pain to sellers)
                if not puts_df.empty:
                    itm_puts = puts_df[puts_df['strike'] > strike]
                    put_pain = ((itm_puts['strike'] - strike) * itm_puts['openInterest']).sum()
                    total_pain += put_pain
                
                max_pain_values[strike] = total_pain
            
            # Return strike with minimum total pain
            return min(max_pain_values, key=max_pain_values.get) if max_pain_values else None
            
        except Exception as e:
            self.logger.error(f"Error calculating max pain: {e}")
            return None
    
    def identify_options_strategies(self, calls_df: pd.DataFrame, 
                                  puts_df: pd.DataFrame,
                                  spot_price: float) -> List[Dict[str, Any]]:
        """
        Identify potential options strategies based on current prices and Greeks.
        
        Args:
            calls_df: Calls DataFrame with Greeks
            puts_df: Puts DataFrame with Greeks
            spot_price: Current stock price
            
        Returns:
            List of recommended options strategies
        """
        strategies = []
        
        try:
            # Filter for liquid options (volume > 10, open interest > 50)
            liquid_calls = calls_df[
                (calls_df['volume'].fillna(0) > 10) & 
                (calls_df['openInterest'].fillna(0) > 50)
            ].copy()
            
            liquid_puts = puts_df[
                (puts_df['volume'].fillna(0) > 10) & 
                (puts_df['openInterest'].fillna(0) > 50)
            ].copy()
            
            # Iron Condor opportunities (low volatility expectation)
            if len(liquid_calls) >= 2 and len(liquid_puts) >= 2:
                # Find OTM calls and puts
                otm_calls = liquid_calls[liquid_calls['strike'] > spot_price * 1.02]
                otm_puts = liquid_puts[liquid_puts['strike'] < spot_price * 0.98]
                
                if len(otm_calls) >= 2 and len(otm_puts) >= 2:
                    strategies.append({
                        'name': 'Iron Condor',
                        'type': 'neutral',
                        'description': 'Profit from low volatility and sideways movement',
                        'max_profit_range': f"{otm_puts.iloc[-1]['strike']:.2f} - {otm_calls.iloc[0]['strike']:.2f}",
                        'risk_level': 'Medium'
                    })
            
            # Straddle opportunities (high volatility expectation)
            atm_calls = liquid_calls[abs(liquid_calls['strike'] - spot_price) < spot_price * 0.02]
            atm_puts = liquid_puts[abs(liquid_puts['strike'] - spot_price) < spot_price * 0.02]
            
            if not atm_calls.empty and not atm_puts.empty:
                strategies.append({
                    'name': 'Long Straddle',
                    'type': 'volatility',
                    'description': 'Profit from large price movements in either direction',
                    'breakeven_range': 'High volatility needed',
                    'risk_level': 'High'
                })
            
            # Covered Call opportunities
            if not liquid_calls.empty:
                otm_calls = liquid_calls[liquid_calls['strike'] > spot_price * 1.05]
                
                if not otm_calls.empty:
                    strategies.append({
                        'name': 'Covered Call',
                        'type': 'income',
                        'description': 'Generate income on existing stock position',
                        'suggested_strike': otm_calls.iloc[0]['strike'],
                        'risk_level': 'Low'
                    })
            
            # Protective Put opportunities
            if not liquid_puts.empty:
                otm_puts = liquid_puts[liquid_puts['strike'] < spot_price * 0.95]
                
                if not otm_puts.empty:
                    strategies.append({
                        'name': 'Protective Put',
                        'type': 'hedge',
                        'description': 'Protect existing stock position from downside',
                        'suggested_strike': otm_puts.iloc[-1]['strike'],
                        'risk_level': 'Low'
                    })
            
        except Exception as e:
            self.logger.error(f"Error identifying options strategies: {e}")
        
        return strategies
    
    async def get_options_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive options summary for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with comprehensive options analysis
        """
        try:
            # Get current stock price
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('regularMarketPrice', 0)
            
            if current_price == 0:
                raise DataError(f"Could not get current price for {symbol}")
            
            # Get options chains for multiple expirations
            multi_chain = await self.get_multi_expiration_chain(symbol, max_expirations=3)
            
            if not multi_chain:
                raise DataError(f"No options data available for {symbol}")
            
            # Analyze the nearest expiration
            nearest_exp = sorted(multi_chain.keys())[0]
            nearest_chain = multi_chain[nearest_exp]
            
            # Calculate Greeks
            calls_with_greeks = self.calculate_options_greeks(
                nearest_chain['calls'], current_price
            )
            puts_with_greeks = self.calculate_options_greeks(
                nearest_chain['puts'], current_price
            )
            
            # Analyze options flow
            flow_analysis = self.analyze_options_flow({
                'calls': calls_with_greeks,
                'puts': puts_with_greeks
            })
            
            # Identify strategies
            strategies = self.identify_options_strategies(
                calls_with_greeks, puts_with_greeks, current_price
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'nearest_expiration': nearest_exp,
                'available_expirations': list(multi_chain.keys()),
                'flow_analysis': flow_analysis,
                'strategy_recommendations': strategies,
                'calls_summary': {
                    'total_volume': calls_with_greeks['volume'].fillna(0).sum(),
                    'total_oi': calls_with_greeks['openInterest'].fillna(0).sum(),
                    'avg_iv': calls_with_greeks['impliedVolatility'].mean()
                },
                'puts_summary': {
                    'total_volume': puts_with_greeks['volume'].fillna(0).sum(),
                    'total_oi': puts_with_greeks['openInterest'].fillna(0).sum(),
                    'avg_iv': puts_with_greeks['impliedVolatility'].mean()
                },
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            raise APIError(f"Options summary error for {symbol}: {str(e)}")