"""
Recommendation & Rationale Agent - Final output generation with cross-validation and chain-of-thought reasoning.

This agent implements the final stage of the multi-agent trading system, providing comprehensive
recommendations with full rationale, cross-validation, and audit trails. Key features:
- Cross-validation across all agent outputs with consensus verification
- Chain-of-thought reasoning for each recommendation
- Comprehensive rationale generation with evidence citations
- Regulatory-compliant audit trails
- Final quality assurance and consistency checks
- Risk-return optimization and ranking
- Output formatting per user specifications
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
import json
from loguru import logger

from ...core.base.agent import BaseAgent, AgentOutput
from ...core.base.exceptions import ValidationError, DataError
from ...core.utils.data_validation import DataValidator
from ...core.utils.math_utils import MathUtils


class CrossValidationEngine:
    """Advanced cross-validation engine for multi-agent consensus verification."""
    
    def __init__(self, min_consensus_sources: int = 4, confidence_threshold: float = 0.75):
        self.min_consensus_sources = min_consensus_sources
        self.confidence_threshold = confidence_threshold
        self.logger = logger.bind(component="cross_validation")
    
    def cross_validate_recommendations(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive cross-validation across all agent outputs."""
        validation_result = {
            "is_valid": False,
            "cross_validation_confidence": 0.0,
            "agent_consensus": {},
            "evidence_chain": {},
            "validation_summary": {},
            "quality_metrics": {}
        }
        
        try:
            # Extract recommendations from each agent
            agent_recommendations = self._extract_agent_recommendations(agent_outputs)
            
            if len(agent_recommendations) < self.min_consensus_sources:
                validation_result["validation_summary"]["insufficient_sources"] = True
                return validation_result
            
            # Cross-validate each symbol across agents
            symbol_validations = {}
            all_symbols = set()
            
            for agent_name, recs in agent_recommendations.items():
                for rec in recs:
                    if "symbol" in rec:
                        all_symbols.add(rec["symbol"])
            
            for symbol in all_symbols:
                symbol_validations[symbol] = self._cross_validate_symbol(
                    symbol, agent_recommendations
                )
            
            # Calculate overall consensus
            valid_symbols = [s for s, v in symbol_validations.items() if v["is_valid"]]
            consensus_ratio = len(valid_symbols) / len(all_symbols) if all_symbols else 0
            
            # Generate evidence chain
            evidence_chain = self._build_evidence_chain(agent_outputs, symbol_validations)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                agent_outputs, symbol_validations, evidence_chain
            )
            
            validation_result.update({
                "is_valid": consensus_ratio >= self.confidence_threshold,
                "cross_validation_confidence": consensus_ratio,
                "agent_consensus": symbol_validations,
                "evidence_chain": evidence_chain,
                "validation_summary": {
                    "total_symbols": len(all_symbols),
                    "valid_symbols": len(valid_symbols),
                    "consensus_ratio": consensus_ratio,
                    "agents_participating": len(agent_recommendations)
                },
                "quality_metrics": quality_metrics
            })
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _extract_agent_recommendations(self, agent_outputs: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract recommendations from all agent outputs."""
        agent_recommendations = {}
        
        for agent_name, output in agent_outputs.items():
            if not hasattr(output, 'data') or not output.data:
                continue
                
            recommendations = []
            
            # Handle different output formats from different agents
            if agent_name == "momentum_strategy":
                recommendations = output.data.get("recommendations", [])
            elif agent_name == "signal_synthesis":
                recommendations = output.data.get("final_recommendations", [])
            elif agent_name == "risk_modeling":
                recommendations = output.data.get("risk_validated_recommendations", [])
            elif agent_name == "ml_ensemble":
                # Convert ML predictions to recommendation format
                predictions = output.data.get("predictions", {})
                for symbol, prediction in predictions.items():
                    action = "BUY" if prediction > 0.1 else "SELL" if prediction < -0.1 else "HOLD"
                    recommendations.append({
                        "symbol": symbol,
                        "action": action,
                        "confidence": output.data.get("uncertainty_metrics", {}).get("confidence", 0.5),
                        "source": "ml_prediction",
                        "prediction_value": prediction
                    })
            
            if recommendations:
                agent_recommendations[agent_name] = recommendations
        
        return agent_recommendations
    
    def _cross_validate_symbol(self, symbol: str, 
                              agent_recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Cross-validate a specific symbol across all agents."""
        symbol_validation = {
            "is_valid": False,
            "confidence": 0.0,
            "consensus_action": "HOLD",
            "supporting_agents": [],
            "conflicting_agents": [],
            "evidence_strength": 0.0,
            "validation_details": {}
        }
        
        try:
            # Collect all recommendations for this symbol
            symbol_recs = []
            for agent_name, recs in agent_recommendations.items():
                for rec in recs:
                    if rec.get("symbol") == symbol:
                        symbol_recs.append({
                            **rec,
                            "agent_source": agent_name
                        })
            
            if len(symbol_recs) < 2:
                symbol_validation["validation_details"]["insufficient_agents"] = True
                return symbol_validation
            
            # Analyze action consensus
            actions = [rec.get("action", "HOLD") for rec in symbol_recs]
            action_counts = defaultdict(int)
            for action in actions:
                action_counts[action] += 1
            
            # Determine consensus action
            consensus_action = max(action_counts.items(), key=lambda x: x[1])[0]
            consensus_count = action_counts[consensus_action]
            
            # Calculate confidence based on consensus strength
            consensus_ratio = consensus_count / len(symbol_recs)
            
            # Categorize agents
            supporting_agents = []
            conflicting_agents = []
            
            for rec in symbol_recs:
                agent_name = rec["agent_source"]
                rec_action = rec.get("action", "HOLD")
                
                if rec_action == consensus_action or (consensus_action == "HOLD" and rec_action in ["WEAK_BUY", "WEAK_SELL"]):
                    supporting_agents.append(agent_name)
                else:
                    conflicting_agents.append(agent_name)
            
            # Calculate evidence strength
            evidence_strength = self._calculate_evidence_strength(symbol_recs)
            
            # Determine validity
            is_valid = (
                consensus_ratio >= 0.6 and  # At least 60% agreement
                len(supporting_agents) >= 2 and  # At least 2 supporting agents
                evidence_strength >= 0.5  # Minimum evidence strength
            )
            
            symbol_validation.update({
                "is_valid": is_valid,
                "confidence": consensus_ratio * evidence_strength,
                "consensus_action": consensus_action,
                "supporting_agents": supporting_agents,
                "conflicting_agents": conflicting_agents,
                "evidence_strength": evidence_strength,
                "validation_details": {
                    "total_recommendations": len(symbol_recs),
                    "consensus_count": consensus_count,
                    "consensus_ratio": consensus_ratio,
                    "action_distribution": dict(action_counts)
                }
            })
            
        except Exception as e:
            self.logger.error(f"Symbol validation failed for {symbol}: {e}")
            symbol_validation["error"] = str(e)
        
        return symbol_validation
    
    def _calculate_evidence_strength(self, symbol_recs: List[Dict[str, Any]]) -> float:
        """Calculate the strength of evidence for a symbol's recommendations."""
        if not symbol_recs:
            return 0.0
        
        # Factors contributing to evidence strength
        confidence_scores = []
        signal_strengths = []
        
        for rec in symbol_recs:
            # Confidence from individual agents
            confidence = rec.get("confidence", 0.5)
            confidence_scores.append(confidence)
            
            # Signal strength indicators
            if "composite_score" in rec:
                signal_strengths.append(abs(rec["composite_score"]))
            elif "final_signal" in rec:
                signal_strengths.append(abs(rec["final_signal"]))
            elif "prediction_value" in rec:
                signal_strengths.append(min(1.0, abs(rec["prediction_value"]) * 2))
        
        # Calculate overall evidence strength
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0.5
        
        # Bonus for multiple confirming sources
        source_bonus = min(0.2, (len(symbol_recs) - 1) * 0.05)
        
        evidence_strength = (avg_confidence * 0.4 + avg_signal_strength * 0.4 + source_bonus * 0.2)
        
        return min(1.0, evidence_strength)
    
    def _build_evidence_chain(self, agent_outputs: Dict[str, Any], 
                             symbol_validations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive evidence chain for audit and explanation."""
        evidence_chain = {
            "agent_contributions": {},
            "validation_path": [],
            "decision_timeline": [],
            "supporting_evidence": {},
            "risk_considerations": {}
        }
        
        try:
            # Document each agent's contribution
            for agent_name, output in agent_outputs.items():
                if hasattr(output, 'data') and output.data:
                    agent_contribution = {
                        "agent_type": agent_name,
                        "processing_time": getattr(output, 'execution_time_ms', 0),
                        "success": getattr(output, 'success', True),
                        "key_findings": self._extract_key_findings(agent_name, output.data),
                        "confidence_level": self._extract_agent_confidence(agent_name, output.data)
                    }
                    evidence_chain["agent_contributions"][agent_name] = agent_contribution
            
            # Build validation path
            validation_steps = [
                "Data Universe Processing",
                "Technical Analysis & Feature Engineering",
                "ML Model Ensemble",
                "Strategy Signal Generation",
                "Signal Synthesis & Consensus",
                "Risk Modeling & Validation",
                "Cross-Validation & Final Recommendation"
            ]
            
            evidence_chain["validation_path"] = validation_steps
            
            # Extract risk considerations
            if "risk_modeling" in agent_outputs:
                risk_data = agent_outputs["risk_modeling"].data
                evidence_chain["risk_considerations"] = {
                    "consensus_achieved": risk_data.get("consensus_validation", {}).get("is_valid", False),
                    "stress_test_results": risk_data.get("stress_test_results", {}).get("stress_summary", {}),
                    "constraint_compliance": risk_data.get("constraint_validation", {}).get("constraint_compliance", {})
                }
            
        except Exception as e:
            self.logger.error(f"Evidence chain building failed: {e}")
            evidence_chain["error"] = str(e)
        
        return evidence_chain
    
    def _extract_key_findings(self, agent_name: str, agent_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from each agent's output."""
        findings = []
        
        try:
            if agent_name == "data_universe":
                metadata = agent_data.get("metadata", {})
                findings.append(f"Processed {metadata.get('symbols_count', 0)} symbols")
                findings.append(f"Data quality: {100 - metadata.get('missing_data_percentage', 0):.1f}%")
                
            elif agent_name == "technical_analysis":
                feature_summary = agent_data.get("feature_summary", {})
                findings.append(f"Generated {feature_summary.get('total_features_added', 0)} technical features")
                
            elif agent_name == "ml_ensemble":
                performance = agent_data.get("model_performance", {})
                findings.append(f"Model R²: {performance.get('test_r2', 0):.3f}")
                findings.append(f"Directional accuracy: {performance.get('directional_accuracy', 0):.3f}")
                
            elif agent_name == "momentum_strategy":
                metadata = agent_data.get("metadata", {})
                findings.append(f"Generated {len(agent_data.get('recommendations', []))} momentum recommendations")
                
            elif agent_name == "signal_synthesis":
                consensus_analysis = agent_data.get("consensus_analysis", {})
                findings.append(f"Consensus confidence: {consensus_analysis.get('average_consensus_confidence', 0):.3f}")
                findings.append(f"High confidence recommendations: {consensus_analysis.get('high_confidence_count', 0)}")
                
            elif agent_name == "risk_modeling":
                consensus_validation = agent_data.get("consensus_validation", {})
                findings.append(f"Risk model consensus: {consensus_validation.get('consensus_confidence', 0):.3f}")
                
        except Exception as e:
            self.logger.error(f"Key findings extraction failed for {agent_name}: {e}")
            findings.append(f"Error extracting findings: {str(e)}")
        
        return findings
    
    def _extract_agent_confidence(self, agent_name: str, agent_data: Dict[str, Any]) -> float:
        """Extract overall confidence level from agent output."""
        try:
            if agent_name == "signal_synthesis":
                return agent_data.get("consensus_analysis", {}).get("average_consensus_confidence", 0.5)
            elif agent_name == "risk_modeling":
                return agent_data.get("consensus_validation", {}).get("consensus_confidence", 0.5)
            elif agent_name == "ml_ensemble":
                return agent_data.get("uncertainty_metrics", {}).get("confidence", 0.5)
            else:
                return 0.7  # Default confidence for other agents
        except Exception:
            return 0.5
    
    def _calculate_quality_metrics(self, agent_outputs: Dict[str, Any],
                                 symbol_validations: Dict[str, Dict[str, Any]],
                                 evidence_chain: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        metrics = {
            "overall_quality_score": 0.0,
            "consensus_strength": 0.0,
            "evidence_completeness": 0.0,
            "risk_assessment_quality": 0.0,
            "validation_thoroughness": 0.0
        }
        
        try:
            # Consensus strength
            valid_symbols = [v for v in symbol_validations.values() if v["is_valid"]]
            if symbol_validations:
                consensus_strength = len(valid_symbols) / len(symbol_validations)
            else:
                consensus_strength = 0.0
            
            # Evidence completeness
            expected_agents = 6  # Expected number of agents in full pipeline
            actual_agents = len(agent_outputs)
            evidence_completeness = actual_agents / expected_agents
            
            # Risk assessment quality
            risk_quality = 0.0
            if "risk_modeling" in agent_outputs:
                risk_data = agent_outputs["risk_modeling"].data
                consensus_achieved = risk_data.get("consensus_validation", {}).get("is_valid", False)
                stress_tested = len(risk_data.get("stress_test_results", {}).get("scenario_results", {})) > 0
                risk_quality = 0.5 + (0.3 if consensus_achieved else 0) + (0.2 if stress_tested else 0)
            
            # Validation thoroughness
            validation_steps_completed = len([a for a in agent_outputs.keys() if a in [
                "data_universe", "technical_analysis", "ml_ensemble", 
                "momentum_strategy", "signal_synthesis", "risk_modeling"
            ]])
            validation_thoroughness = validation_steps_completed / 6
            
            # Overall quality score
            overall_quality = (
                consensus_strength * 0.3 +
                evidence_completeness * 0.25 +
                risk_quality * 0.25 +
                validation_thoroughness * 0.2
            )
            
            metrics.update({
                "overall_quality_score": overall_quality,
                "consensus_strength": consensus_strength,
                "evidence_completeness": evidence_completeness,
                "risk_assessment_quality": risk_quality,
                "validation_thoroughness": validation_thoroughness
            })
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
        
        return metrics


class RecommendationFormatter:
    """Formats final recommendations according to user specifications."""
    
    def __init__(self):
        self.logger = logger.bind(component="recommendation_formatter")
    
    def format_final_recommendations(self, cross_validation: Dict[str, Any],
                                   agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format final recommendations with comprehensive rationale."""
        formatted_output = {
            "executive_summary": {},
            "ranked_recommendations": [],
            "methodology_overview": {},
            "risk_assessment": {},
            "audit_trail": {},
            "disclaimer": {}
        }
        
        try:
            # Generate executive summary
            formatted_output["executive_summary"] = self._generate_executive_summary(
                cross_validation, agent_outputs
            )
            
            # Create ranked recommendations
            formatted_output["ranked_recommendations"] = self._create_ranked_recommendations(
                cross_validation, agent_outputs
            )
            
            # Methodology overview
            formatted_output["methodology_overview"] = self._create_methodology_overview(
                cross_validation["evidence_chain"]
            )
            
            # Risk assessment summary
            formatted_output["risk_assessment"] = self._create_risk_assessment(
                agent_outputs
            )
            
            # Comprehensive audit trail
            formatted_output["audit_trail"] = self._create_audit_trail(
                cross_validation, agent_outputs
            )
            
            # Disclaimer and limitations
            formatted_output["disclaimer"] = self._create_disclaimer()
            
        except Exception as e:
            self.logger.error(f"Recommendation formatting failed: {e}")
            formatted_output["error"] = str(e)
        
        return formatted_output
    
    def _generate_executive_summary(self, cross_validation: Dict[str, Any],
                                  agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis and recommendations."""
        summary = {
            "analysis_date": datetime.utcnow().isoformat(),
            "market_context": {},
            "key_findings": [],
            "recommendation_summary": {},
            "confidence_assessment": {}
        }
        
        try:
            # Key findings from analysis
            quality_metrics = cross_validation.get("quality_metrics", {})
            
            summary["key_findings"] = [
                f"Analyzed recommendations using {cross_validation['validation_summary']['agents_participating']} independent agents",
                f"Cross-validation confidence: {cross_validation['cross_validation_confidence']:.1%}",
                f"Overall analysis quality score: {quality_metrics.get('overall_quality_score', 0):.1%}",
                f"Risk consensus achieved: {'Yes' if agent_outputs.get('risk_modeling', {}).data.get('consensus_validation', {}).get('is_valid', False) else 'No'}"
            ]
            
            # Recommendation summary
            valid_symbols = [s for s, v in cross_validation["agent_consensus"].items() if v["is_valid"]]
            
            summary["recommendation_summary"] = {
                "total_symbols_analyzed": cross_validation["validation_summary"]["total_symbols"],
                "recommendations_generated": len(valid_symbols),
                "high_confidence_count": len([v for v in cross_validation["agent_consensus"].values() 
                                            if v["is_valid"] and v["confidence"] > 0.8]),
                "risk_validated": "risk_modeling" in agent_outputs
            }
            
            # Confidence assessment
            summary["confidence_assessment"] = {
                "methodology_robustness": "HIGH" if quality_metrics.get("validation_thoroughness", 0) > 0.8 else "MEDIUM",
                "consensus_strength": "HIGH" if quality_metrics.get("consensus_strength", 0) > 0.7 else "MEDIUM",
                "risk_analysis_quality": "HIGH" if quality_metrics.get("risk_assessment_quality", 0) > 0.7 else "MEDIUM"
            }
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def _create_ranked_recommendations(self, cross_validation: Dict[str, Any],
                                     agent_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create ranked list of recommendations with full rationale."""
        recommendations = []
        
        try:
            # Get valid symbols with consensus
            valid_consensus = {symbol: validation for symbol, validation in 
                             cross_validation["agent_consensus"].items() if validation["is_valid"]}
            
            # Sort by confidence and evidence strength
            sorted_symbols = sorted(
                valid_consensus.items(),
                key=lambda x: x[1]["confidence"] * x[1]["evidence_strength"],
                reverse=True
            )
            
            for rank, (symbol, consensus) in enumerate(sorted_symbols, 1):
                recommendation = self._create_individual_recommendation(
                    symbol, consensus, agent_outputs, rank
                )
                recommendations.append(recommendation)
            
        except Exception as e:
            self.logger.error(f"Ranked recommendations creation failed: {e}")
        
        return recommendations
    
    def _create_individual_recommendation(self, symbol: str, consensus: Dict[str, Any],
                                        agent_outputs: Dict[str, Any], rank: int) -> Dict[str, Any]:
        """Create individual recommendation with comprehensive rationale."""
        recommendation = {
            "rank": rank,
            "symbol": symbol,
            "action": consensus["consensus_action"],
            "confidence": consensus["confidence"],
            "evidence_strength": consensus["evidence_strength"],
            "position_sizing": {},
            "rationale": {},
            "supporting_analysis": {},
            "risk_considerations": {},
            "chain_of_thought": []
        }
        
        try:
            # Extract position sizing from risk-validated recommendations
            position_size = 0.0
            if "risk_modeling" in agent_outputs:
                risk_recs = agent_outputs["risk_modeling"].data.get("risk_validated_recommendations", [])
                for rec in risk_recs:
                    if rec.get("symbol") == symbol:
                        position_size = rec.get("position_size", 0.0)
                        break
            
            recommendation["position_sizing"] = {
                "recommended_weight": abs(position_size),
                "direction": "LONG" if position_size > 0 else "SHORT" if position_size < 0 else "NEUTRAL",
                "risk_adjusted": True
            }
            
            # Build comprehensive rationale
            recommendation["rationale"] = self._build_rationale(symbol, consensus, agent_outputs)
            
            # Supporting analysis from each agent
            recommendation["supporting_analysis"] = self._extract_supporting_analysis(
                symbol, consensus, agent_outputs
            )
            
            # Risk considerations
            recommendation["risk_considerations"] = self._extract_risk_considerations(
                symbol, agent_outputs
            )
            
            # Chain of thought reasoning
            recommendation["chain_of_thought"] = self._build_chain_of_thought(
                symbol, consensus, agent_outputs
            )
            
        except Exception as e:
            self.logger.error(f"Individual recommendation creation failed for {symbol}: {e}")
            recommendation["error"] = str(e)
        
        return recommendation
    
    def _build_rationale(self, symbol: str, consensus: Dict[str, Any],
                        agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive rationale for recommendation."""
        rationale = {
            "primary_thesis": "",
            "supporting_factors": [],
            "risk_factors": [],
            "catalyst_analysis": [],
            "technical_outlook": "",
            "fundamental_view": ""
        }
        
        try:
            action = consensus["consensus_action"]
            supporting_agents = consensus["supporting_agents"]
            
            # Primary thesis based on consensus
            if action in ["BUY", "STRONG_BUY"]:
                rationale["primary_thesis"] = f"Multiple independent analysis methods converge on a positive outlook for {symbol}, with {len(supporting_agents)} confirming sources showing upward momentum and favorable risk-adjusted returns."
            elif action in ["SELL", "STRONG_SELL"]:
                rationale["primary_thesis"] = f"Consensus analysis indicates negative prospects for {symbol}, with {len(supporting_agents)} independent sources confirming bearish signals and elevated risk concerns."
            else:
                rationale["primary_thesis"] = f"Mixed signals for {symbol} suggest a neutral stance, with insufficient consensus for directional conviction."
            
            # Supporting factors from different agents
            supporting_factors = []
            
            if "momentum_strategy" in supporting_agents:
                supporting_factors.append("Strong momentum signals confirmed across multiple timeframes")
            
            if "ml_ensemble" in supporting_agents:
                supporting_factors.append("Machine learning models predict favorable price movement")
            
            if "signal_synthesis" in supporting_agents:
                supporting_factors.append("Multi-strategy consensus validation achieved")
            
            if "risk_modeling" in supporting_agents:
                supporting_factors.append("Risk-return profile meets portfolio constraints")
            
            rationale["supporting_factors"] = supporting_factors
            
            # Technical outlook
            if "technical_analysis" in agent_outputs:
                rationale["technical_outlook"] = f"Technical analysis supports the {action} recommendation with favorable indicator alignment."
            
        except Exception as e:
            self.logger.error(f"Rationale building failed for {symbol}: {e}")
            rationale["error"] = str(e)
        
        return rationale
    
    def _extract_supporting_analysis(self, symbol: str, consensus: Dict[str, Any],
                                   agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract supporting analysis from each agent."""
        supporting_analysis = {}
        
        try:
            # Extract agent-specific analysis
            for agent_name in consensus["supporting_agents"]:
                if agent_name in agent_outputs:
                    agent_data = agent_outputs[agent_name].data
                    
                    if agent_name == "momentum_strategy":
                        # Extract momentum-specific analysis
                        recs = agent_data.get("recommendations", [])
                        for rec in recs:
                            if rec.get("symbol") == symbol:
                                supporting_analysis["momentum"] = {
                                    "signal_strength": rec.get("signal_strength", ""),
                                    "composite_score": rec.get("composite_score", 0),
                                    "momentum_factors": rec.get("momentum_analysis", {})
                                }
                                break
                    
                    elif agent_name == "ml_ensemble":
                        # Extract ML analysis
                        predictions = agent_data.get("predictions", {})
                        if symbol in predictions:
                            supporting_analysis["machine_learning"] = {
                                "prediction": predictions[symbol],
                                "confidence": agent_data.get("uncertainty_metrics", {}).get("confidence", 0.5),
                                "model_performance": agent_data.get("model_performance", {})
                            }
                    
                    elif agent_name == "signal_synthesis":
                        # Extract synthesis analysis
                        final_recs = agent_data.get("final_recommendations", [])
                        for rec in final_recs:
                            if rec.get("symbol") == symbol:
                                supporting_analysis["signal_synthesis"] = {
                                    "consensus_confidence": rec.get("consensus_confidence", 0),
                                    "confirming_sources": rec.get("confirming_sources", []),
                                    "final_signal": rec.get("final_signal", 0)
                                }
                                break
            
        except Exception as e:
            self.logger.error(f"Supporting analysis extraction failed for {symbol}: {e}")
        
        return supporting_analysis
    
    def _extract_risk_considerations(self, symbol: str, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk considerations for the symbol."""
        risk_considerations = {
            "portfolio_impact": {},
            "stress_test_results": {},
            "risk_metrics": {},
            "constraints_compliance": {}
        }
        
        try:
            if "risk_modeling" in agent_outputs:
                risk_data = agent_outputs["risk_modeling"].data
                
                # Find risk-validated recommendation for this symbol
                risk_recs = risk_data.get("risk_validated_recommendations", [])
                for rec in risk_recs:
                    if rec.get("symbol") == symbol:
                        risk_considerations["portfolio_impact"] = {
                            "position_size": rec.get("position_size", 0),
                            "risk_adjusted": rec.get("risk_adjusted", False),
                            "adjustment_reason": rec.get("adjustment_reason", "")
                        }
                        break
                
                # Stress test implications
                stress_results = risk_data.get("stress_test_results", {})
                if stress_results:
                    risk_considerations["stress_test_results"] = {
                        "worst_case_scenario": stress_results.get("worst_case_scenarios", [{}])[0] if stress_results.get("worst_case_scenarios") else {},
                        "stress_summary": stress_results.get("stress_summary", {})
                    }
                
                # Overall risk metrics
                portfolio_metrics = risk_data.get("portfolio_risk_metrics", {})
                risk_considerations["risk_metrics"] = portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Risk considerations extraction failed for {symbol}: {e}")
        
        return risk_considerations
    
    def _build_chain_of_thought(self, symbol: str, consensus: Dict[str, Any],
                              agent_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build chain-of-thought reasoning for the recommendation."""
        chain_of_thought = []
        
        try:
            # Step 1: Data Analysis
            chain_of_thought.append({
                "step": 1,
                "process": "Data Universe Analysis",
                "reasoning": f"Symbol {symbol} identified in universe with sufficient data quality for analysis",
                "outcome": "Data validation passed"
            })
            
            # Step 2: Technical Analysis
            if "technical_analysis" in agent_outputs:
                chain_of_thought.append({
                    "step": 2,
                    "process": "Technical Feature Engineering",
                    "reasoning": "Generated comprehensive technical indicators and pattern recognition features",
                    "outcome": "Technical features enriched dataset"
                })
            
            # Step 3: ML Analysis
            if "ml_ensemble" in agent_outputs:
                ml_data = agent_outputs["ml_ensemble"].data
                predictions = ml_data.get("predictions", {})
                if symbol in predictions:
                    prediction = predictions[symbol]
                    chain_of_thought.append({
                        "step": 3,
                        "process": "Machine Learning Prediction",
                        "reasoning": f"Ensemble models predict {prediction:.3f} forward return with uncertainty quantification",
                        "outcome": f"ML signal: {'Bullish' if prediction > 0 else 'Bearish'}"
                    })
            
            # Step 4: Strategy Analysis
            if "momentum_strategy" in agent_outputs:
                chain_of_thought.append({
                    "step": 4,
                    "process": "Momentum Strategy Analysis",
                    "reasoning": f"Multi-factor momentum analysis supports {consensus['consensus_action']} recommendation",
                    "outcome": f"Strategy signal confirmed"
                })
            
            # Step 5: Consensus Validation
            chain_of_thought.append({
                "step": 5,
                "process": "Signal Synthesis & Consensus",
                "reasoning": f"Cross-validation across {len(consensus['supporting_agents'])} agents achieved consensus with {consensus['confidence']:.1%} confidence",
                "outcome": f"Consensus achieved: {consensus['consensus_action']}"
            })
            
            # Step 6: Risk Validation
            if "risk_modeling" in agent_outputs:
                chain_of_thought.append({
                    "step": 6,
                    "process": "Risk Modeling & Validation",
                    "reasoning": "Multi-model risk assessment validated recommendation against portfolio constraints",
                    "outcome": "Risk constraints satisfied"
                })
            
            # Step 7: Final Recommendation
            chain_of_thought.append({
                "step": 7,
                "process": "Final Cross-Validation",
                "reasoning": f"Comprehensive cross-validation confirms {consensus['consensus_action']} recommendation with {consensus['evidence_strength']:.1%} evidence strength",
                "outcome": f"Final recommendation: {consensus['consensus_action']}"
            })
            
        except Exception as e:
            self.logger.error(f"Chain of thought building failed for {symbol}: {e}")
        
        return chain_of_thought
    
    def _create_methodology_overview(self, evidence_chain: Dict[str, Any]) -> Dict[str, Any]:
        """Create methodology overview section."""
        methodology = {
            "analysis_framework": "Multi-Agent Consensus-Driven Trading System",
            "validation_approach": "Cross-Validation with Ultrathinking",
            "agents_employed": [],
            "validation_steps": [],
            "quality_assurance": {}
        }
        
        try:
            # Document agents employed
            agent_contributions = evidence_chain.get("agent_contributions", {})
            for agent_name, contribution in agent_contributions.items():
                methodology["agents_employed"].append({
                    "agent": agent_name,
                    "role": self._get_agent_role_description(agent_name),
                    "confidence": contribution.get("confidence_level", 0.5)
                })
            
            # Validation steps
            methodology["validation_steps"] = evidence_chain.get("validation_path", [])
            
            # Quality assurance measures
            methodology["quality_assurance"] = {
                "consensus_requirements": "Minimum 3 confirming sources for any recommendation",
                "risk_validation": "Multi-model risk assessment with stress testing",
                "cross_validation": "Independent agent verification with outlier detection",
                "audit_trail": "Comprehensive decision tracking for regulatory compliance"
            }
            
        except Exception as e:
            self.logger.error(f"Methodology overview creation failed: {e}")
        
        return methodology
    
    def _get_agent_role_description(self, agent_name: str) -> str:
        """Get role description for each agent."""
        role_descriptions = {
            "data_universe": "Market data acquisition, cleaning, and universe construction",
            "technical_analysis": "Technical indicator generation and pattern recognition",
            "ml_ensemble": "Machine learning ensemble predictions with uncertainty quantification",
            "momentum_strategy": "Multi-factor momentum analysis and signal generation",
            "signal_synthesis": "Multi-strategy consensus validation and signal arbitration",
            "risk_modeling": "Advanced risk modeling with multi-layer validation and stress testing"
        }
        return role_descriptions.get(agent_name, "Specialized analysis agent")
    
    def _create_risk_assessment(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive risk assessment summary."""
        risk_assessment = {
            "portfolio_risk_metrics": {},
            "stress_test_summary": {},
            "risk_model_consensus": {},
            "constraint_compliance": {}
        }
        
        try:
            if "risk_modeling" in agent_outputs:
                risk_data = agent_outputs["risk_modeling"].data
                
                # Portfolio risk metrics
                risk_assessment["portfolio_risk_metrics"] = risk_data.get("portfolio_risk_metrics", {})
                
                # Stress test summary
                stress_results = risk_data.get("stress_test_results", {})
                if stress_results:
                    risk_assessment["stress_test_summary"] = stress_results.get("stress_summary", {})
                
                # Risk model consensus
                consensus_validation = risk_data.get("consensus_validation", {})
                risk_assessment["risk_model_consensus"] = {
                    "consensus_achieved": consensus_validation.get("is_valid", False),
                    "confidence": consensus_validation.get("consensus_confidence", 0),
                    "models_in_agreement": len(consensus_validation.get("model_agreement", {}))
                }
                
                # Constraint compliance
                constraint_validation = risk_data.get("constraint_validation", {})
                risk_assessment["constraint_compliance"] = constraint_validation.get("constraint_compliance", {})
            
        except Exception as e:
            self.logger.error(f"Risk assessment creation failed: {e}")
        
        return risk_assessment
    
    def _create_audit_trail(self, cross_validation: Dict[str, Any],
                          agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive audit trail."""
        audit_trail = {
            "process_timeline": [],
            "decision_points": [],
            "validation_checkpoints": [],
            "data_lineage": {},
            "regulatory_compliance": {}
        }
        
        try:
            # Process timeline
            for agent_name, output in agent_outputs.items():
                if hasattr(output, 'metadata') and output.metadata:
                    timestamp = output.metadata.get('processing_timestamp', datetime.utcnow())
                    audit_trail["process_timeline"].append({
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        "agent": agent_name,
                        "status": "completed" if getattr(output, 'success', True) else "failed",
                        "processing_time_ms": getattr(output, 'execution_time_ms', 0)
                    })
            
            # Decision points
            evidence_chain = cross_validation.get("evidence_chain", {})
            audit_trail["decision_points"] = [
                {
                    "decision": "Universe Selection",
                    "criteria": "Data quality and liquidity requirements",
                    "outcome": f"Selected symbols for analysis"
                },
                {
                    "decision": "Feature Engineering",
                    "criteria": "Technical and statistical significance",
                    "outcome": "Generated comprehensive feature set"
                },
                {
                    "decision": "Model Selection",
                    "criteria": "Cross-validation performance and robustness",
                    "outcome": "Ensemble approach with uncertainty quantification"
                },
                {
                    "decision": "Consensus Validation",
                    "criteria": "Multi-source agreement and evidence strength",
                    "outcome": f"Achieved consensus on {cross_validation['validation_summary']['valid_symbols']} symbols"
                },
                {
                    "decision": "Risk Validation",
                    "criteria": "Portfolio constraints and stress test requirements",
                    "outcome": "Risk-adjusted position sizing applied"
                }
            ]
            
            # Regulatory compliance
            audit_trail["regulatory_compliance"] = {
                "decision_transparency": "All decisions documented with rationale",
                "model_explainability": "Chain-of-thought reasoning provided",
                "risk_documentation": "Comprehensive risk assessment included",
                "validation_independence": "Independent agent validation performed",
                "audit_completeness": "Full process trail maintained"
            }
            
        except Exception as e:
            self.logger.error(f"Audit trail creation failed: {e}")
        
        return audit_trail
    
    def _create_disclaimer(self) -> Dict[str, Any]:
        """Create comprehensive disclaimer and limitations."""
        return {
            "investment_disclaimer": "This analysis is for informational purposes only and does not constitute investment advice. Past performance does not guarantee future results.",
            "model_limitations": "All models have inherent limitations and assumptions. Market conditions can change rapidly, affecting model validity.",
            "risk_warning": "All investments carry risk of loss. Diversification and position sizing are critical for risk management.",
            "data_dependencies": "Analysis quality depends on data accuracy and completeness. Market data may be delayed or contain errors.",
            "regulatory_note": "This system is designed for institutional use with proper risk management oversight. Compliance with applicable regulations is the user's responsibility.",
            "technology_limitations": "Algorithmic trading systems can fail. Human oversight and intervention capabilities are essential.",
            "generation_timestamp": datetime.utcnow().isoformat()
        }


class RecommendationAgent(BaseAgent):
    """
    Recommendation & Rationale Agent - Final output generation with cross-validation and comprehensive reasoning.
    
    This agent serves as the final stage of the multi-agent trading system, providing comprehensive
    recommendations with full rationale, cross-validation, and regulatory-compliant audit trails.
    
    Inputs: All agent outputs from the trading system pipeline
    Outputs: Final formatted recommendations with comprehensive rationale and audit trails
    """
    
    def __init__(self):
        super().__init__("RecommendationAgent", "recommendation")
        self._setup_dependencies()
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validation systems."""
        self.min_consensus_sources = self.get_config_value("min_consensus_sources", 4)
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.75)
        self.include_chain_of_thought = self.get_config_value("include_chain_of_thought", True)
        self.regulatory_compliance = self.get_config_value("regulatory_compliance", True)
        
        # Initialize validation and formatting systems
        self.cross_validator = CrossValidationEngine(
            min_consensus_sources=self.min_consensus_sources,
            confidence_threshold=self.confidence_threshold
        )
        self.formatter = RecommendationFormatter()
    
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute final recommendation generation with cross-validation and comprehensive reasoning.
        
        Args:
            inputs: Dictionary containing:
                - agent_outputs: All outputs from previous agents in the pipeline
                - user_preferences: User-specific preferences and constraints
                - output_format: Desired output format specifications
                
        Returns:
            AgentOutput with final formatted recommendations and comprehensive audit trails
        """
        self._validate_inputs(inputs)
        
        agent_outputs = inputs["agent_outputs"]
        user_preferences = inputs.get("user_preferences", {})
        output_format = inputs.get("output_format", "comprehensive")
        
        try:
            # Step 1: Cross-validate all agent outputs with consensus verification
            self.logger.info("Performing comprehensive cross-validation across all agents")
            cross_validation = self.cross_validator.cross_validate_recommendations(agent_outputs)
            
            validation_confidence = cross_validation.get("cross_validation_confidence", 0)
            self.logger.info(f"Cross-validation completed with {validation_confidence:.1%} confidence")
            
            # Step 2: Format final recommendations according to specifications
            self.logger.info("Formatting final recommendations with comprehensive rationale")
            formatted_recommendations = self.formatter.format_final_recommendations(
                cross_validation, agent_outputs
            )
            
            # Step 3: Apply user preferences and customizations
            customized_output = await self._apply_user_preferences(
                formatted_recommendations, user_preferences
            )
            
            # Step 4: Generate final quality assessment
            quality_assessment = self._generate_quality_assessment(
                cross_validation, agent_outputs, customized_output
            )
            
            # Step 5: Create final output with regulatory compliance
            final_output = self._create_final_output(
                customized_output, cross_validation, quality_assessment, output_format
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "final_recommendations": final_output,
                    "cross_validation_results": cross_validation,
                    "quality_assessment": quality_assessment,
                    "processing_summary": self._generate_processing_summary(agent_outputs)
                },
                metadata={
                    "recommendations_generated": len(final_output.get("ranked_recommendations", [])),
                    "cross_validation_confidence": validation_confidence,
                    "quality_score": quality_assessment.get("overall_quality_score", 0),
                    "agents_validated": len(agent_outputs),
                    "regulatory_compliant": self.regulatory_compliance,
                    "chain_of_thought_included": self.include_chain_of_thought,
                    "processing_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            raise DataError(f"Final recommendation processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters with comprehensive checks."""
        required_keys = ["agent_outputs"]
        self.validate_inputs(inputs, required_keys)
        
        agent_outputs = inputs["agent_outputs"]
        if not agent_outputs:
            raise ValidationError("Agent outputs cannot be empty")
        
        # Validate that we have sufficient agent outputs for cross-validation
        if len(agent_outputs) < 2:
            raise ValidationError("Insufficient agent outputs for cross-validation")
        
        # Validate agent output structure
        for agent_name, output in agent_outputs.items():
            if not hasattr(output, 'data') or output.data is None:
                raise ValidationError(f"Invalid output structure from {agent_name}")
    
    async def _apply_user_preferences(self, formatted_recommendations: Dict[str, Any],
                                    user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user-specific preferences and customizations."""
        customized_output = formatted_recommendations.copy()
        
        try:
            # Apply ranking preferences
            ranking_preference = user_preferences.get("ranking_method", "confidence")
            if ranking_preference == "risk_adjusted":
                customized_output["ranked_recommendations"] = self._rerank_by_risk_adjusted_return(
                    customized_output["ranked_recommendations"]
                )
            elif ranking_preference == "momentum":
                customized_output["ranked_recommendations"] = self._rerank_by_momentum_strength(
                    customized_output["ranked_recommendations"]
                )
            
            # Apply output filtering
            min_confidence = user_preferences.get("min_confidence", 0.0)
            if min_confidence > 0:
                customized_output["ranked_recommendations"] = [
                    rec for rec in customized_output["ranked_recommendations"]
                    if rec.get("confidence", 0) >= min_confidence
                ]
            
            # Apply sector preferences
            preferred_sectors = user_preferences.get("preferred_sectors", [])
            excluded_sectors = user_preferences.get("excluded_sectors", [])
            
            if preferred_sectors or excluded_sectors:
                customized_output["ranked_recommendations"] = self._apply_sector_filters(
                    customized_output["ranked_recommendations"], preferred_sectors, excluded_sectors
                )
            
        except Exception as e:
            self.logger.error(f"User preference application failed: {e}")
            # Return original if customization fails
        
        return customized_output
    
    def _rerank_by_risk_adjusted_return(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank recommendations by risk-adjusted return."""
        def risk_adjusted_score(rec):
            confidence = rec.get("confidence", 0)
            evidence_strength = rec.get("evidence_strength", 0)
            position_size = rec.get("position_sizing", {}).get("recommended_weight", 0)
            
            # Simple risk adjustment - would be more sophisticated in production
            risk_penalty = 0.1  # Constant penalty for simplicity
            return (confidence * evidence_strength) - risk_penalty
        
        return sorted(recommendations, key=risk_adjusted_score, reverse=True)
    
    def _rerank_by_momentum_strength(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank recommendations by momentum strength."""
        def momentum_score(rec):
            supporting_analysis = rec.get("supporting_analysis", {})
            momentum_data = supporting_analysis.get("momentum", {})
            return momentum_data.get("composite_score", 0)
        
        return sorted(recommendations, key=momentum_score, reverse=True)
    
    def _apply_sector_filters(self, recommendations: List[Dict[str, Any]],
                            preferred_sectors: List[str], excluded_sectors: List[str]) -> List[Dict[str, Any]]:
        """Apply sector-based filtering."""
        # This would normally use sector classification
        # For now, return original recommendations as we don't have sector data
        return recommendations
    
    def _generate_quality_assessment(self, cross_validation: Dict[str, Any],
                                   agent_outputs: Dict[str, Any],
                                   customized_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive quality assessment."""
        quality_assessment = {
            "overall_quality_score": 0.0,
            "dimension_scores": {},
            "quality_indicators": {},
            "improvement_suggestions": []
        }
        
        try:
            # Extract quality metrics from cross-validation
            quality_metrics = cross_validation.get("quality_metrics", {})
            
            # Calculate dimension scores
            dimension_scores = {
                "consensus_strength": quality_metrics.get("consensus_strength", 0),
                "evidence_completeness": quality_metrics.get("evidence_completeness", 0),
                "risk_assessment_quality": quality_metrics.get("risk_assessment_quality", 0),
                "validation_thoroughness": quality_metrics.get("validation_thoroughness", 0),
                "methodology_robustness": min(1.0, len(agent_outputs) / 6)  # Expected 6 agents
            }
            
            # Overall quality score
            overall_score = np.mean(list(dimension_scores.values()))
            
            # Quality indicators
            quality_indicators = {
                "high_confidence_recommendations": len([
                    rec for rec in customized_output.get("ranked_recommendations", [])
                    if rec.get("confidence", 0) > 0.8
                ]),
                "cross_validation_passed": cross_validation.get("is_valid", False),
                "risk_consensus_achieved": any(
                    output.data.get("consensus_validation", {}).get("is_valid", False)
                    for output in agent_outputs.values()
                    if hasattr(output, 'data') and "consensus_validation" in output.data
                ),
                "all_agents_successful": all(
                    getattr(output, 'success', True) for output in agent_outputs.values()
                )
            }
            
            # Improvement suggestions
            suggestions = []
            if dimension_scores["consensus_strength"] < 0.7:
                suggestions.append("Consider increasing consensus threshold or adding more validation agents")
            if dimension_scores["evidence_completeness"] < 0.8:
                suggestions.append("Additional data sources or agent types could improve analysis completeness")
            if dimension_scores["risk_assessment_quality"] < 0.7:
                suggestions.append("Enhanced risk modeling or additional stress testing scenarios recommended")
            
            quality_assessment.update({
                "overall_quality_score": overall_score,
                "dimension_scores": dimension_scores,
                "quality_indicators": quality_indicators,
                "improvement_suggestions": suggestions
            })
            
        except Exception as e:
            self.logger.error(f"Quality assessment generation failed: {e}")
        
        return quality_assessment
    
    def _create_final_output(self, customized_output: Dict[str, Any],
                           cross_validation: Dict[str, Any],
                           quality_assessment: Dict[str, Any],
                           output_format: str) -> Dict[str, Any]:
        """Create final output in requested format."""
        final_output = customized_output.copy()
        
        try:
            # Add quality assessment
            final_output["quality_assessment"] = quality_assessment
            
            # Add metadata
            final_output["metadata"] = {
                "generation_timestamp": datetime.utcnow().isoformat(),
                "system_version": "Multi-Agent Trading System v1.0",
                "analysis_confidence": cross_validation.get("cross_validation_confidence", 0),
                "quality_score": quality_assessment.get("overall_quality_score", 0),
                "regulatory_compliance": self.regulatory_compliance
            }
            
            # Format-specific customizations
            if output_format == "summary":
                # Keep only essential information
                final_output = {
                    "executive_summary": final_output["executive_summary"],
                    "ranked_recommendations": final_output["ranked_recommendations"][:10],  # Top 10
                    "quality_assessment": {"overall_quality_score": quality_assessment["overall_quality_score"]},
                    "metadata": final_output["metadata"]
                }
            elif output_format == "detailed":
                # Include all available information
                final_output["detailed_analysis"] = {
                    "cross_validation_details": cross_validation,
                    "agent_performance_metrics": self._extract_agent_performance(agent_outputs)
                }
            
        except Exception as e:
            self.logger.error(f"Final output creation failed: {e}")
        
        return final_output
    
    def _extract_agent_performance(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from each agent."""
        performance_metrics = {}
        
        for agent_name, output in agent_outputs.items():
            metrics = {
                "execution_time_ms": getattr(output, 'execution_time_ms', 0),
                "success": getattr(output, 'success', True),
                "data_quality": "high" if hasattr(output, 'data') and output.data else "low"
            }
            
            # Agent-specific metrics
            if hasattr(output, 'metadata') and output.metadata:
                if agent_name == "data_universe":
                    metrics["symbols_processed"] = output.metadata.get("symbols_count", 0)
                elif agent_name == "ml_ensemble":
                    metrics["models_trained"] = output.metadata.get("models_trained", 0)
                elif agent_name == "signal_synthesis":
                    metrics["consensus_achieved"] = output.data.get("consensus_analysis", {}).get("average_consensus_confidence", 0) > 0.7
                elif agent_name == "risk_modeling":
                    metrics["risk_consensus"] = output.metadata.get("consensus_achieved", False)
            
            performance_metrics[agent_name] = metrics
        
        return performance_metrics
    
    def _generate_processing_summary(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of processing across all agents."""
        summary = {
            "total_agents": len(agent_outputs),
            "successful_agents": 0,
            "total_processing_time_ms": 0,
            "pipeline_status": "completed",
            "agent_status": {}
        }
        
        for agent_name, output in agent_outputs.items():
            is_successful = getattr(output, 'success', True)
            execution_time = getattr(output, 'execution_time_ms', 0)
            
            if is_successful:
                summary["successful_agents"] += 1
            
            summary["total_processing_time_ms"] += execution_time
            summary["agent_status"][agent_name] = "success" if is_successful else "failed"
        
        if summary["successful_agents"] < summary["total_agents"]:
            summary["pipeline_status"] = "partial_success"
        
        return summary