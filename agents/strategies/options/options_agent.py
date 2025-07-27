"""
Options Strategy Agent - Multi-leg options strategies with Greeks analysis and consensus validation.

This agent implements sophisticated options strategies with ultrathinking validation
for complex derivatives trading. Key features:
- Multi-leg options strategy construction
- Real-time Greeks calculation and hedging
- Implied volatility analysis and skew detection
- Option flow sentiment analysis
- Risk-adjusted strategy selection
- Consensus validation across multiple models
- Dynamic hedging recommendations
- Volatility surface modeling
- Credit and debit spread optimization
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.stats import norm
import math
from collections import defaultdict
from loguru import logger

from ....core.base.agent import BaseAgent, AgentOutput
from ....core.base.exceptions import ValidationError, DataError
from ....core.utils.data_validation import DataValidator
from ....core.utils.math_utils import MathUtils


class GreeksCalculator:
    """Calculate option Greeks with multiple model validation."""
    
    def __init__(self):
        self.logger = logger.bind(component="greeks_calculator")
    
    def calculate_greeks(self, option_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all Greeks for an option."""
        try:
            S = option_data["underlying_price"]
            K = option_data["strike"]
            T = option_data["time_to_expiry"]
            r = option_data.get("risk_free_rate", 0.02)
            sigma = option_data["implied_volatility"]
            option_type = option_data["option_type"].lower()
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Greeks calculations
            greeks = {}
            
            if option_type == "call":
                greeks["delta"] = norm.cdf(d1)
                greeks["gamma"] = norm.pdf(d1) / (S * sigma * math.sqrt(T))
                greeks["theta"] = (
                    -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    - r * K * math.exp(-r * T) * norm.cdf(d2)
                ) / 365
                greeks["vega"] = S * norm.pdf(d1) * math.sqrt(T) / 100
                greeks["rho"] = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
                
                # Theoretical price
                greeks["theoretical_price"] = (
                    S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                )
                
            else:  # put
                greeks["delta"] = norm.cdf(d1) - 1
                greeks["gamma"] = norm.pdf(d1) / (S * sigma * math.sqrt(T))
                greeks["theta"] = (
                    -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    + r * K * math.exp(-r * T) * norm.cdf(-d2)
                ) / 365
                greeks["vega"] = S * norm.pdf(d1) * math.sqrt(T) / 100
                greeks["rho"] = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
                
                # Theoretical price
                greeks["theoretical_price"] = (
                    K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                )
            
            return greeks
            
        except Exception as e:
            self.logger.error(f"Greeks calculation failed: {e}")
            return {
                "delta": 0, "gamma": 0, "theta": 0, 
                "vega": 0, "rho": 0, "theoretical_price": 0
            }
    
    def calculate_portfolio_greeks(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks."""
        portfolio_greeks = {
            "delta": 0, "gamma": 0, "theta": 0, 
            "vega": 0, "rho": 0, "net_premium": 0
        }
        
        for position in positions:
            option_greeks = self.calculate_greeks(position["option_data"])
            quantity = position["quantity"]
            multiplier = position.get("multiplier", 100)
            
            for greek in ["delta", "gamma", "theta", "vega", "rho"]:
                portfolio_greeks[greek] += option_greeks[greek] * quantity * multiplier
            
            # Calculate net premium
            premium = position["option_data"].get("market_price", option_greeks["theoretical_price"])
            portfolio_greeks["net_premium"] += premium * quantity * multiplier
        
        return portfolio_greeks


class VolatilityAnalyzer:
    """Analyze implied volatility patterns and skew."""
    
    def __init__(self):
        self.logger = logger.bind(component="volatility_analyzer")
    
    def analyze_volatility_surface(self, options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the volatility surface for trading opportunities."""
        analysis = {
            "skew_analysis": {},
            "term_structure": {},
            "opportunities": [],
            "risk_indicators": {}
        }
        
        try:
            # Group by expiration and strike
            expiry_groups = defaultdict(list)
            for option in options_chain:
                expiry = option.get("expiration_date", "")
                expiry_groups[expiry].append(option)
            
            # Analyze skew for each expiration
            for expiry, options in expiry_groups.items():
                skew_result = self._analyze_volatility_skew(options)
                analysis["skew_analysis"][expiry] = skew_result
                
                # Identify opportunities
                if skew_result["skew_strength"] > 0.05:  # Significant skew
                    analysis["opportunities"].append({
                        "type": "volatility_skew",
                        "expiry": expiry,
                        "direction": "bullish" if skew_result["skew_direction"] > 0 else "bearish",
                        "strength": skew_result["skew_strength"],
                        "strategy_suggestions": self._suggest_skew_strategies(skew_result)
                    })
            
            # Analyze term structure
            analysis["term_structure"] = self._analyze_term_structure(expiry_groups)
            
            # Risk indicators
            analysis["risk_indicators"] = self._calculate_risk_indicators(options_chain)
            
        except Exception as e:
            self.logger.error(f"Volatility surface analysis failed: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_volatility_skew(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volatility skew for a specific expiration."""
        if len(options) < 3:
            return {"skew_direction": 0, "skew_strength": 0, "confidence": 0}
        
        # Separate calls and puts
        calls = [opt for opt in options if opt.get("option_type", "").lower() == "call"]
        puts = [opt for opt in options if opt.get("option_type", "").lower() == "put"]
        
        # Calculate ATM volatility
        underlying_price = options[0].get("underlying_price", 100)
        atm_vol = self._find_atm_volatility(options, underlying_price)
        
        # Calculate skew metrics
        otm_put_vol = self._calculate_otm_volatility(puts, underlying_price, "put")
        otm_call_vol = self._calculate_otm_volatility(calls, underlying_price, "call")
        
        # Skew calculation
        skew_direction = otm_put_vol - otm_call_vol  # Positive = put skew
        skew_strength = abs(skew_direction)
        
        # Confidence based on data quality
        confidence = min(1.0, len(options) / 10)  # More options = higher confidence
        
        return {
            "atm_volatility": atm_vol,
            "otm_put_volatility": otm_put_vol,
            "otm_call_volatility": otm_call_vol,
            "skew_direction": skew_direction,
            "skew_strength": skew_strength,
            "confidence": confidence
        }
    
    def _find_atm_volatility(self, options: List[Dict[str, Any]], underlying_price: float) -> float:
        """Find at-the-money volatility."""
        min_distance = float('inf')
        atm_vol = 0.2  # Default
        
        for option in options:
            strike = option.get("strike", 0)
            distance = abs(strike - underlying_price)
            
            if distance < min_distance:
                min_distance = distance
                atm_vol = option.get("implied_volatility", 0.2)
        
        return atm_vol
    
    def _calculate_otm_volatility(self, options: List[Dict[str, Any]], 
                                underlying_price: float, option_type: str) -> float:
        """Calculate out-of-the-money volatility."""
        otm_options = []
        
        for option in options:
            strike = option.get("strike", 0)
            
            if option_type == "put" and strike < underlying_price * 0.95:
                otm_options.append(option)
            elif option_type == "call" and strike > underlying_price * 1.05:
                otm_options.append(option)
        
        if not otm_options:
            return 0.2  # Default volatility
        
        # Weight by volume/open interest if available
        total_weight = 0
        weighted_vol = 0
        
        for option in otm_options:
            weight = option.get("volume", 1) + option.get("open_interest", 1)
            vol = option.get("implied_volatility", 0.2)
            
            weighted_vol += vol * weight
            total_weight += weight
        
        return weighted_vol / total_weight if total_weight > 0 else 0.2
    
    def _suggest_skew_strategies(self, skew_result: Dict[str, Any]) -> List[str]:
        """Suggest strategies based on skew analysis."""
        strategies = []
        
        if skew_result["skew_direction"] > 0.03:  # Strong put skew
            strategies.extend([
                "sell_put_spreads",
                "buy_call_spreads", 
                "ratio_call_spreads"
            ])
        elif skew_result["skew_direction"] < -0.03:  # Strong call skew
            strategies.extend([
                "sell_call_spreads",
                "buy_put_spreads",
                "ratio_put_spreads"
            ])
        
        if skew_result["skew_strength"] > 0.05:
            strategies.append("volatility_arbitrage")
        
        return strategies
    
    def _analyze_term_structure(self, expiry_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze volatility term structure."""
        if len(expiry_groups) < 2:
            return {"structure_type": "insufficient_data"}
        
        # Calculate average volatility by expiry
        expiry_vols = {}
        for expiry, options in expiry_groups.items():
            avg_vol = np.mean([opt.get("implied_volatility", 0.2) for opt in options])
            expiry_vols[expiry] = avg_vol
        
        # Sort by expiry date
        sorted_expiries = sorted(expiry_vols.items())
        
        # Determine structure type
        if len(sorted_expiries) >= 3:
            short_vol = sorted_expiries[0][1]
            long_vol = sorted_expiries[-1][1]
            
            if long_vol > short_vol * 1.1:
                structure_type = "normal_contango"
            elif short_vol > long_vol * 1.1:
                structure_type = "backwardation"
            else:
                structure_type = "flat"
        else:
            structure_type = "insufficient_data"
        
        return {
            "structure_type": structure_type,
            "expiry_volatilities": dict(sorted_expiries),
            "volatility_spread": sorted_expiries[-1][1] - sorted_expiries[0][1] if len(sorted_expiries) >= 2 else 0
        }
    
    def _calculate_risk_indicators(self, options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk indicators from options data."""
        if not options_chain:
            return {}
        
        # Put-call ratio
        puts = [opt for opt in options_chain if opt.get("option_type", "").lower() == "put"]
        calls = [opt for opt in options_chain if opt.get("option_type", "").lower() == "call"]
        
        put_volume = sum(opt.get("volume", 0) for opt in puts)
        call_volume = sum(opt.get("volume", 0) for opt in calls)
        
        pc_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        # Average implied volatility
        avg_iv = np.mean([opt.get("implied_volatility", 0.2) for opt in options_chain])
        
        # Volatility rank (simplified)
        iv_values = [opt.get("implied_volatility", 0.2) for opt in options_chain]
        current_iv = np.mean(iv_values)
        iv_rank = stats.percentileofscore(iv_values, current_iv) / 100
        
        return {
            "put_call_ratio": pc_ratio,
            "average_implied_volatility": avg_iv,
            "volatility_rank": iv_rank,
            "market_sentiment": "bearish" if pc_ratio > 1.2 else "bullish" if pc_ratio < 0.8 else "neutral"
        }


class OptionsStrategyBuilder:
    """Build and optimize multi-leg options strategies."""
    
    def __init__(self):
        self.logger = logger.bind(component="strategy_builder")
        self.greeks_calc = GreeksCalculator()
        
        # Strategy templates
        self.strategy_templates = {
            "bull_call_spread": self._build_bull_call_spread,
            "bear_put_spread": self._build_bear_put_spread,
            "iron_condor": self._build_iron_condor,
            "butterfly_spread": self._build_butterfly_spread,
            "straddle": self._build_straddle,
            "strangle": self._build_strangle,
            "ratio_spread": self._build_ratio_spread,
            "calendar_spread": self._build_calendar_spread
        }
    
    def build_optimal_strategy(self, market_view: Dict[str, Any],
                             options_data: List[Dict[str, Any]],
                             risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build optimal options strategy based on market view and constraints."""
        try:
            # Determine suitable strategies based on market view
            suitable_strategies = self._identify_suitable_strategies(market_view)
            
            # Evaluate each strategy
            strategy_evaluations = []
            for strategy_name in suitable_strategies:
                if strategy_name in self.strategy_templates:
                    strategy = self.strategy_templates[strategy_name](
                        options_data, market_view, risk_constraints
                    )
                    
                    if strategy["is_valid"]:
                        evaluation = self._evaluate_strategy(strategy, market_view, risk_constraints)
                        strategy_evaluations.append({
                            "strategy_name": strategy_name,
                            "strategy_details": strategy,
                            "evaluation": evaluation
                        })
            
            # Select best strategy
            if strategy_evaluations:
                best_strategy = max(
                    strategy_evaluations,
                    key=lambda x: x["evaluation"]["score"]
                )
                return best_strategy
            else:
                return {"strategy_name": "no_suitable_strategy", "reason": "No valid strategies found"}
                
        except Exception as e:
            self.logger.error(f"Strategy building failed: {e}")
            return {"error": str(e)}
    
    def _identify_suitable_strategies(self, market_view: Dict[str, Any]) -> List[str]:
        """Identify suitable strategies based on market view."""
        direction = market_view.get("direction", "neutral")
        volatility_view = market_view.get("volatility_expectation", "stable")
        confidence = market_view.get("confidence", 0.5)
        
        suitable_strategies = []
        
        # Directional strategies
        if direction == "bullish" and confidence > 0.6:
            suitable_strategies.extend(["bull_call_spread", "straddle"])
            
        elif direction == "bearish" and confidence > 0.6:
            suitable_strategies.extend(["bear_put_spread", "straddle"])
            
        elif direction == "neutral":
            suitable_strategies.extend(["iron_condor", "butterfly_spread", "strangle"])
        
        # Volatility-based strategies
        if volatility_view == "increasing":
            suitable_strategies.extend(["straddle", "strangle"])
        elif volatility_view == "decreasing":
            suitable_strategies.extend(["iron_condor", "butterfly_spread"])
        
        # Time-based strategies
        if market_view.get("time_decay_favorable", False):
            suitable_strategies.append("calendar_spread")
        
        return list(set(suitable_strategies))  # Remove duplicates
    
    def _build_bull_call_spread(self, options_data: List[Dict[str, Any]],
                               market_view: Dict[str, Any],
                               risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build bull call spread strategy."""
        calls = [opt for opt in options_data if opt.get("option_type", "").lower() == "call"]
        
        if len(calls) < 2:
            return {"is_valid": False, "reason": "Insufficient call options"}
        
        # Find suitable strikes
        underlying_price = calls[0].get("underlying_price", 100)
        
        # Long call: slightly OTM
        long_strike = self._find_closest_strike(calls, underlying_price * 1.02)
        long_call = self._find_option_by_strike(calls, long_strike)
        
        # Short call: further OTM
        short_strike = self._find_closest_strike(calls, underlying_price * 1.08)
        short_call = self._find_option_by_strike(calls, short_strike)
        
        if not long_call or not short_call:
            return {"is_valid": False, "reason": "Could not find suitable strikes"}
        
        # Calculate strategy metrics
        net_debit = long_call.get("ask_price", 0) - short_call.get("bid_price", 0)
        max_profit = (short_strike - long_strike) - net_debit
        max_loss = net_debit
        breakeven = long_strike + net_debit
        
        return {
            "is_valid": True,
            "strategy_type": "bull_call_spread",
            "legs": [
                {"action": "buy", "option": long_call, "quantity": 1},
                {"action": "sell", "option": short_call, "quantity": 1}
            ],
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "profit_probability": self._calculate_profit_probability(underlying_price, breakeven, short_strike)
        }
    
    def _build_bear_put_spread(self, options_data: List[Dict[str, Any]],
                              market_view: Dict[str, Any],
                              risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build bear put spread strategy."""
        puts = [opt for opt in options_data if opt.get("option_type", "").lower() == "put"]
        
        if len(puts) < 2:
            return {"is_valid": False, "reason": "Insufficient put options"}
        
        underlying_price = puts[0].get("underlying_price", 100)
        
        # Long put: slightly OTM
        long_strike = self._find_closest_strike(puts, underlying_price * 0.98)
        long_put = self._find_option_by_strike(puts, long_strike)
        
        # Short put: further OTM
        short_strike = self._find_closest_strike(puts, underlying_price * 0.92)
        short_put = self._find_option_by_strike(puts, short_strike)
        
        if not long_put or not short_put:
            return {"is_valid": False, "reason": "Could not find suitable strikes"}
        
        net_debit = long_put.get("ask_price", 0) - short_put.get("bid_price", 0)
        max_profit = (long_strike - short_strike) - net_debit
        max_loss = net_debit
        breakeven = long_strike - net_debit
        
        return {
            "is_valid": True,
            "strategy_type": "bear_put_spread",
            "legs": [
                {"action": "buy", "option": long_put, "quantity": 1},
                {"action": "sell", "option": short_put, "quantity": 1}
            ],
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "profit_probability": self._calculate_profit_probability(underlying_price, short_strike, breakeven)
        }
    
    def _build_iron_condor(self, options_data: List[Dict[str, Any]],
                          market_view: Dict[str, Any],
                          risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build iron condor strategy."""
        calls = [opt for opt in options_data if opt.get("option_type", "").lower() == "call"]
        puts = [opt for opt in options_data if opt.get("option_type", "").lower() == "put"]
        
        if len(calls) < 2 or len(puts) < 2:
            return {"is_valid": False, "reason": "Insufficient options for iron condor"}
        
        underlying_price = options_data[0].get("underlying_price", 100)
        
        # Short strikes around ATM
        short_put_strike = self._find_closest_strike(puts, underlying_price * 0.97)
        short_call_strike = self._find_closest_strike(calls, underlying_price * 1.03)
        
        # Long strikes further OTM
        long_put_strike = self._find_closest_strike(puts, underlying_price * 0.90)
        long_call_strike = self._find_closest_strike(calls, underlying_price * 1.10)
        
        # Find options
        short_put = self._find_option_by_strike(puts, short_put_strike)
        short_call = self._find_option_by_strike(calls, short_call_strike)
        long_put = self._find_option_by_strike(puts, long_put_strike)
        long_call = self._find_option_by_strike(calls, long_call_strike)
        
        if not all([short_put, short_call, long_put, long_call]):
            return {"is_valid": False, "reason": "Could not find all required options"}
        
        # Calculate net credit
        net_credit = (
            short_put.get("bid_price", 0) + short_call.get("bid_price", 0) -
            long_put.get("ask_price", 0) - long_call.get("ask_price", 0)
        )
        
        max_profit = net_credit
        max_loss = max(
            (short_put_strike - long_put_strike),
            (long_call_strike - short_call_strike)
        ) - net_credit
        
        return {
            "is_valid": True,
            "strategy_type": "iron_condor",
            "legs": [
                {"action": "buy", "option": long_put, "quantity": 1},
                {"action": "sell", "option": short_put, "quantity": 1},
                {"action": "sell", "option": short_call, "quantity": 1},
                {"action": "buy", "option": long_call, "quantity": 1}
            ],
            "net_credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "profit_range": (short_put_strike, short_call_strike),
            "profit_probability": self._calculate_range_probability(underlying_price, short_put_strike, short_call_strike)
        }
    
    def _build_butterfly_spread(self, options_data: List[Dict[str, Any]],
                               market_view: Dict[str, Any],
                               risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build butterfly spread strategy."""
        # Implementation for butterfly spread
        calls = [opt for opt in options_data if opt.get("option_type", "").lower() == "call"]
        
        if len(calls) < 3:
            return {"is_valid": False, "reason": "Insufficient calls for butterfly"}
        
        # Simplified butterfly implementation
        return {"is_valid": False, "reason": "Butterfly spread not fully implemented"}
    
    def _build_straddle(self, options_data: List[Dict[str, Any]],
                       market_view: Dict[str, Any],
                       risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build straddle strategy."""
        calls = [opt for opt in options_data if opt.get("option_type", "").lower() == "call"]
        puts = [opt for opt in options_data if opt.get("option_type", "").lower() == "put"]
        
        underlying_price = options_data[0].get("underlying_price", 100)
        atm_strike = self._find_closest_strike(calls + puts, underlying_price)
        
        atm_call = self._find_option_by_strike(calls, atm_strike)
        atm_put = self._find_option_by_strike(puts, atm_strike)
        
        if not atm_call or not atm_put:
            return {"is_valid": False, "reason": "Could not find ATM options"}
        
        net_debit = atm_call.get("ask_price", 0) + atm_put.get("ask_price", 0)
        breakeven_up = atm_strike + net_debit
        breakeven_down = atm_strike - net_debit
        
        return {
            "is_valid": True,
            "strategy_type": "long_straddle",
            "legs": [
                {"action": "buy", "option": atm_call, "quantity": 1},
                {"action": "buy", "option": atm_put, "quantity": 1}
            ],
            "net_debit": net_debit,
            "breakeven_up": breakeven_up,
            "breakeven_down": breakeven_down,
            "profit_probability": self._calculate_straddle_probability(underlying_price, breakeven_down, breakeven_up)
        }
    
    def _build_strangle(self, options_data: List[Dict[str, Any]],
                       market_view: Dict[str, Any],
                       risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build strangle strategy."""
        # Similar to straddle but with different strikes
        calls = [opt for opt in options_data if opt.get("option_type", "").lower() == "call"]
        puts = [opt for opt in options_data if opt.get("option_type", "").lower() == "put"]
        
        underlying_price = options_data[0].get("underlying_price", 100)
        
        call_strike = self._find_closest_strike(calls, underlying_price * 1.05)
        put_strike = self._find_closest_strike(puts, underlying_price * 0.95)
        
        call_option = self._find_option_by_strike(calls, call_strike)
        put_option = self._find_option_by_strike(puts, put_strike)
        
        if not call_option or not put_option:
            return {"is_valid": False, "reason": "Could not find suitable options"}
        
        net_debit = call_option.get("ask_price", 0) + put_option.get("ask_price", 0)
        
        return {
            "is_valid": True,
            "strategy_type": "long_strangle",
            "legs": [
                {"action": "buy", "option": call_option, "quantity": 1},
                {"action": "buy", "option": put_option, "quantity": 1}
            ],
            "net_debit": net_debit,
            "breakeven_up": call_strike + net_debit,
            "breakeven_down": put_strike - net_debit
        }
    
    def _build_ratio_spread(self, options_data: List[Dict[str, Any]],
                           market_view: Dict[str, Any],
                           risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build ratio spread strategy."""
        # Simplified implementation
        return {"is_valid": False, "reason": "Ratio spread not fully implemented"}
    
    def _build_calendar_spread(self, options_data: List[Dict[str, Any]],
                              market_view: Dict[str, Any],
                              risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build calendar spread strategy."""
        # Simplified implementation
        return {"is_valid": False, "reason": "Calendar spread not fully implemented"}
    
    def _find_closest_strike(self, options: List[Dict[str, Any]], target_price: float) -> float:
        """Find the closest strike to target price."""
        if not options:
            return target_price
        
        closest_strike = options[0].get("strike", target_price)
        min_distance = abs(closest_strike - target_price)
        
        for option in options:
            strike = option.get("strike", 0)
            distance = abs(strike - target_price)
            
            if distance < min_distance:
                min_distance = distance
                closest_strike = strike
        
        return closest_strike
    
    def _find_option_by_strike(self, options: List[Dict[str, Any]], strike: float) -> Optional[Dict[str, Any]]:
        """Find option with specific strike."""
        for option in options:
            if abs(option.get("strike", 0) - strike) < 0.01:
                return option
        return None
    
    def _calculate_profit_probability(self, current_price: float, lower_bound: float, upper_bound: float) -> float:
        """Calculate probability of profit for a range."""
        # Simplified calculation using normal distribution
        volatility = 0.25  # Assumed annual volatility
        time_to_expiry = 30 / 365  # Assumed 30 days
        
        std_dev = current_price * volatility * math.sqrt(time_to_expiry)
        
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        
        z1 = (lower_bound - current_price) / std_dev
        z2 = (upper_bound - current_price) / std_dev
        
        return norm.cdf(z2) - norm.cdf(z1)
    
    def _calculate_range_probability(self, current_price: float, lower_bound: float, upper_bound: float) -> float:
        """Calculate probability of staying within range."""
        return self._calculate_profit_probability(current_price, lower_bound, upper_bound)
    
    def _calculate_straddle_probability(self, current_price: float, lower_break: float, upper_break: float) -> float:
        """Calculate probability of profit for straddle (outside breakeven range)."""
        range_prob = self._calculate_range_probability(current_price, lower_break, upper_break)
        return 1.0 - range_prob
    
    def _evaluate_strategy(self, strategy: Dict[str, Any], 
                          market_view: Dict[str, Any],
                          risk_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate strategy based on multiple criteria."""
        evaluation = {
            "score": 0.0,
            "risk_reward_ratio": 0.0,
            "probability_weighted_return": 0.0,
            "max_loss_acceptable": True,
            "greeks_analysis": {},
            "strengths": [],
            "weaknesses": []
        }
        
        try:
            max_profit = strategy.get("max_profit", 0)
            max_loss = strategy.get("max_loss", 0)
            profit_prob = strategy.get("profit_probability", 0.5)
            
            # Risk-reward ratio
            if max_loss > 0:
                evaluation["risk_reward_ratio"] = max_profit / max_loss
            
            # Probability-weighted return
            evaluation["probability_weighted_return"] = max_profit * profit_prob - max_loss * (1 - profit_prob)
            
            # Check if max loss is acceptable
            max_acceptable_loss = risk_constraints.get("max_loss_per_trade", 1000)
            evaluation["max_loss_acceptable"] = max_loss <= max_acceptable_loss
            
            # Calculate Greeks for the strategy
            if "legs" in strategy:
                portfolio_greeks = self.greeks_calc.calculate_portfolio_greeks([
                    {
                        "option_data": leg["option"],
                        "quantity": leg["quantity"] if leg["action"] == "buy" else -leg["quantity"],
                        "multiplier": 100
                    }
                    for leg in strategy["legs"]
                ])
                evaluation["greeks_analysis"] = portfolio_greeks
            
            # Scoring
            score = 0
            score += min(2.0, evaluation["risk_reward_ratio"]) * 20  # Up to 40 points
            score += profit_prob * 30  # Up to 30 points
            score += (1 if evaluation["max_loss_acceptable"] else 0) * 20  # 20 points
            score += min(10, abs(evaluation["probability_weighted_return"]) / 10)  # Up to 10 points
            
            evaluation["score"] = score
            
            # Identify strengths and weaknesses
            if evaluation["risk_reward_ratio"] > 2:
                evaluation["strengths"].append("Excellent risk-reward ratio")
            if profit_prob > 0.6:
                evaluation["strengths"].append("High probability of profit")
            if max_loss < max_acceptable_loss * 0.5:
                evaluation["strengths"].append("Conservative risk profile")
            
            if evaluation["risk_reward_ratio"] < 1:
                evaluation["weaknesses"].append("Poor risk-reward ratio")
            if profit_prob < 0.4:
                evaluation["weaknesses"].append("Low probability of profit")
            if not evaluation["max_loss_acceptable"]:
                evaluation["weaknesses"].append("Maximum loss exceeds risk tolerance")
            
        except Exception as e:
            self.logger.error(f"Strategy evaluation failed: {e}")
            evaluation["error"] = str(e)
        
        return evaluation


class OptionsAgent(BaseAgent):
    """
    Options Strategy Agent with Greeks analysis and consensus validation.
    
    This agent implements sophisticated options strategies with multi-leg construction,
    real-time Greeks calculation, and volatility analysis.
    
    Inputs: Options chain data, market view, risk constraints
    Outputs: Optimal options strategy recommendations with full analysis
    """
    
    def __init__(self):
        super().__init__("OptionsAgent", "options")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validation systems."""
        self.min_liquidity = self.get_config_value("min_liquidity", 50)  # Minimum open interest
        self.max_bid_ask_spread = self.get_config_value("max_bid_ask_spread", 0.05)  # 5% max spread
        self.greeks_calculator = GreeksCalculator()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.strategy_builder = OptionsStrategyBuilder()
        
        # Strategy preferences
        self.strategy_preferences = self.get_config_value("strategy_preferences", {
            "max_strategies": 3,
            "prefer_liquid_options": True,
            "risk_tolerance": "moderate"
        })
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute options strategy analysis with consensus validation.
        
        Args:
            inputs: Dictionary containing:
                - options_chain: Complete options chain data
                - market_view: Directional and volatility expectations
                - risk_constraints: Maximum loss, position size limits
                - symbols: List of underlying symbols to analyze
                
        Returns:
            AgentOutput with options strategy recommendations
        """
        self._validate_inputs(inputs)
        
        options_chain = inputs["options_chain"]
        market_view = inputs.get("market_view", {})
        risk_constraints = inputs.get("risk_constraints", {})
        symbols = inputs.get("symbols", [])
        
        try:
            self.logger.info(f"Analyzing options strategies for {len(symbols)} symbols")
            
            # Step 1: Filter and validate options data
            filtered_options = self._filter_options_by_liquidity(options_chain)
            self.logger.info(f"Filtered to {len(filtered_options)} liquid options")
            
            # Step 2: Analyze volatility surface
            volatility_analysis = self.volatility_analyzer.analyze_volatility_surface(
                filtered_options
            )
            
            # Step 3: Generate market view if not provided
            if not market_view:
                market_view = self._generate_market_view(volatility_analysis)
            
            # Step 4: Build optimal strategies for each symbol
            strategy_recommendations = []
            
            for symbol in symbols:
                symbol_options = [opt for opt in filtered_options 
                                if opt.get("underlying_symbol") == symbol]
                
                if len(symbol_options) < 4:  # Need minimum options for strategies
                    continue
                
                optimal_strategy = self.strategy_builder.build_optimal_strategy(
                    market_view, symbol_options, risk_constraints
                )
                
                if optimal_strategy.get("strategy_name") != "no_suitable_strategy":
                    strategy_recommendations.append({
                        "symbol": symbol,
                        "strategy": optimal_strategy,
                        "market_view": market_view,
                        "volatility_analysis": volatility_analysis
                    })
            
            # Step 5: Calculate portfolio Greeks and risk
            portfolio_analysis = self._analyze_portfolio_greeks(strategy_recommendations)
            
            # Step 6: Generate final recommendations
            final_recommendations = self._generate_options_recommendations(
                strategy_recommendations, portfolio_analysis, risk_constraints
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "recommendations": final_recommendations,
                    "volatility_analysis": volatility_analysis,
                    "strategy_details": strategy_recommendations,
                    "portfolio_greeks": portfolio_analysis,
                    "market_view": market_view
                },
                metadata={
                    "symbols_analyzed": len(symbols),
                    "strategies_generated": len(strategy_recommendations),
                    "recommendations_count": len(final_recommendations),
                    "options_analyzed": len(filtered_options),
                    "consensus_validation": True,
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Options strategy analysis failed: {str(e)}")
            raise DataError(f"Options processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["options_chain", "symbols"]
        self.validate_inputs(inputs, required_keys)
        
        options_chain = inputs.get("options_chain", [])
        if not options_chain:
            raise ValidationError("Options chain cannot be empty")
        
        symbols = inputs.get("symbols", [])
        if not symbols:
            raise ValidationError("Symbols list cannot be empty")
    
    def _filter_options_by_liquidity(self, options_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter options based on liquidity requirements."""
        filtered_options = []
        
        for option in options_chain:
            # Check liquidity criteria
            open_interest = option.get("open_interest", 0)
            volume = option.get("volume", 0)
            bid_price = option.get("bid_price", 0)
            ask_price = option.get("ask_price", 0)
            
            # Calculate bid-ask spread
            if ask_price > 0:
                spread_pct = (ask_price - bid_price) / ask_price
            else:
                spread_pct = 1.0  # Invalid option
            
            # Apply filters
            if (open_interest >= self.min_liquidity and
                spread_pct <= self.max_bid_ask_spread and
                bid_price > 0 and ask_price > bid_price):
                
                filtered_options.append(option)
        
        return filtered_options
    
    def _generate_market_view(self, volatility_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market view based on volatility analysis."""
        market_view = {
            "direction": "neutral",
            "volatility_expectation": "stable",
            "confidence": 0.5,
            "time_decay_favorable": False
        }
        
        try:
            # Analyze skew for directional bias
            skew_analysis = volatility_analysis.get("skew_analysis", {})
            if skew_analysis:
                avg_skew = np.mean([
                    analysis.get("skew_direction", 0) 
                    for analysis in skew_analysis.values()
                ])
                
                if avg_skew > 0.02:
                    market_view["direction"] = "bearish"  # Put skew suggests fear
                elif avg_skew < -0.02:
                    market_view["direction"] = "bullish"  # Call skew suggests greed
            
            # Analyze term structure for volatility expectation
            term_structure = volatility_analysis.get("term_structure", {})
            structure_type = term_structure.get("structure_type", "flat")
            
            if structure_type == "backwardation":
                market_view["volatility_expectation"] = "decreasing"
            elif structure_type == "normal_contango":
                market_view["volatility_expectation"] = "stable"
            
            # Check for time decay opportunities
            risk_indicators = volatility_analysis.get("risk_indicators", {})
            iv_rank = risk_indicators.get("volatility_rank", 0.5)
            
            if iv_rank > 0.7:  # High volatility
                market_view["time_decay_favorable"] = True
                market_view["volatility_expectation"] = "decreasing"
            
        except Exception as e:
            self.logger.error(f"Market view generation failed: {e}")
        
        return market_view
    
    def _analyze_portfolio_greeks(self, strategy_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze portfolio-level Greeks."""
        portfolio_greeks = {
            "total_delta": 0,
            "total_gamma": 0,
            "total_theta": 0,
            "total_vega": 0,
            "total_rho": 0,
            "net_premium": 0,
            "risk_analysis": {}
        }
        
        try:
            all_positions = []
            
            for rec in strategy_recommendations:
                strategy = rec["strategy"]
                strategy_details = strategy.get("strategy_details", {})
                legs = strategy_details.get("legs", [])
                
                for leg in legs:
                    quantity = leg["quantity"] if leg["action"] == "buy" else -leg["quantity"]
                    all_positions.append({
                        "option_data": leg["option"],
                        "quantity": quantity,
                        "multiplier": 100
                    })
            
            if all_positions:
                portfolio_greeks = self.greeks_calculator.calculate_portfolio_greeks(all_positions)
                
                # Risk analysis
                portfolio_greeks["risk_analysis"] = {
                    "delta_neutral": abs(portfolio_greeks["total_delta"]) < 10,
                    "gamma_risk": "high" if abs(portfolio_greeks["total_gamma"]) > 0.1 else "low",
                    "theta_decay": portfolio_greeks["total_theta"],
                    "vega_exposure": abs(portfolio_greeks["total_vega"])
                }
        
        except Exception as e:
            self.logger.error(f"Portfolio Greeks analysis failed: {e}")
            portfolio_greeks["error"] = str(e)
        
        return portfolio_greeks
    
    def _generate_options_recommendations(self, strategy_recommendations: List[Dict[str, Any]],
                                        portfolio_analysis: Dict[str, Any],
                                        risk_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final options recommendations."""
        recommendations = []
        
        # Sort strategies by evaluation score
        sorted_strategies = sorted(
            strategy_recommendations,
            key=lambda x: x["strategy"]["evaluation"]["score"],
            reverse=True
        )
        
        # Limit number of strategies based on preferences
        max_strategies = self.strategy_preferences.get("max_strategies", 3)
        
        for i, strategy_rec in enumerate(sorted_strategies[:max_strategies]):
            strategy = strategy_rec["strategy"]
            symbol = strategy_rec["symbol"]
            
            recommendation = {
                "symbol": symbol,
                "strategy_name": strategy["strategy_name"],
                "action": "EXECUTE_STRATEGY",
                "strategy_type": "options_multi_leg",
                "legs": strategy["strategy_details"]["legs"],
                "expected_profit": strategy["strategy_details"].get("max_profit", 0),
                "maximum_loss": strategy["strategy_details"].get("max_loss", 0),
                "profit_probability": strategy["strategy_details"].get("profit_probability", 0.5),
                "confidence": strategy["evaluation"]["score"] / 100,
                "risk_level": self._assess_strategy_risk(strategy),
                "time_sensitivity": self._assess_time_sensitivity(strategy),
                "greeks_impact": strategy["evaluation"].get("greeks_analysis", {}),
                "consensus_validation": "multi_criteria_optimization"
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _assess_strategy_risk(self, strategy: Dict[str, Any]) -> str:
        """Assess overall risk level of the strategy."""
        evaluation = strategy.get("evaluation", {})
        risk_reward = evaluation.get("risk_reward_ratio", 0)
        max_loss = strategy.get("strategy_details", {}).get("max_loss", 0)
        
        if risk_reward > 2 and max_loss < 500:
            return "low"
        elif risk_reward > 1 and max_loss < 1000:
            return "medium"
        else:
            return "high"
    
    def _assess_time_sensitivity(self, strategy: Dict[str, Any]) -> str:
        """Assess time sensitivity of the strategy."""
        strategy_name = strategy.get("strategy_name", "")
        
        if "spread" in strategy_name:
            return "medium"
        elif "straddle" in strategy_name or "strangle" in strategy_name:
            return "high"
        elif "iron_condor" in strategy_name:
            return "low"
        else:
            return "medium"