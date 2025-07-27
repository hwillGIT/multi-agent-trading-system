"""
ML & Model Ensemble Agent - Runs custom ML/AI models and ensemble methods.

This agent handles:
- Traditional ML models (XGBoost, CatBoost, Random Forest, etc.)
- Deep learning models (LSTM, Transformers, etc.)
- Ensemble methods (stacking, voting, blending)
- Model uncertainty quantification
- LLM integration for context and reasoning
- Model interpretation and explanation
"""

import asyncio
import pickle
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import catboost as cb
import optuna
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ...core.base.agent import BaseAgent, AgentOutput
from ...core.base.exceptions import ModelError, ValidationError
from ...core.utils.data_validation import DataValidator
from ...core.utils.math_utils import MathUtils


class MLEnsembleAgent(BaseAgent):
    """
    Agent responsible for machine learning model ensemble and predictions.
    
    This agent trains and runs multiple ML models, combines them using ensemble
    methods, and provides predictions with uncertainty quantification.
    
    Inputs: Feature-enhanced dataset from TechnicalAnalysisAgent
    Outputs: Model predictions, confidence intervals, and feature importance
    """
    
    def __init__(self):
        super().__init__("MLEnsembleAgent", "ml_models")
        self._setup_dependencies()
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        
    def _setup_dependencies(self) -> None:
        """Initialize dependencies and validate configuration."""
        self.ensemble_models = self.get_config_value("ensemble_models", [
            "xgboost", "catboost", "random_forest"
        ])
        self.cv_folds = self.get_config_value("model_selection.cv_folds", 5)
        self.test_size = self.get_config_value("model_selection.test_size", 0.2)
        self.validation_size = self.get_config_value("model_selection.validation_size", 0.2)
        self.hyperparameter_tuning = self.get_config_value("hyperparameter_tuning.method", "optuna")
        self.n_trials = self.get_config_value("hyperparameter_tuning.n_trials", 100)
        
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute ML ensemble training and prediction.
        
        Args:
            inputs: Dictionary containing:
                - enhanced_feature_matrix: DataFrame with technical features
                - target_variable: Name of target variable (default: forward_return_1d)
                - train_models: Whether to train new models (default: True)
                - prediction_mode: 'regression' or 'classification'
                - symbols: Symbols to process
                
        Returns:
            AgentOutput with model predictions and metadata
        """
        self._validate_inputs(inputs)
        
        feature_matrix = inputs["enhanced_feature_matrix"]
        target_variable = inputs.get("target_variable", "forward_return_1d")
        train_models = inputs.get("train_models", True)
        prediction_mode = inputs.get("prediction_mode", "regression")
        symbols = inputs.get("symbols", feature_matrix["symbol"].unique() if "symbol" in feature_matrix.columns else [])
        
        try:
            # Prepare data for ML
            prepared_data = await self._prepare_ml_data(feature_matrix, target_variable, symbols)
            
            if train_models:
                # Train individual models
                models_results = await self._train_individual_models(
                    prepared_data, prediction_mode
                )
                
                # Create ensemble model
                ensemble_results = await self._create_ensemble_model(
                    prepared_data, models_results, prediction_mode
                )
            else:
                # Use existing models for prediction
                ensemble_results = await self._predict_with_existing_models(prepared_data)
            
            # Generate predictions with uncertainty
            predictions_with_uncertainty = await self._generate_predictions_with_uncertainty(
                prepared_data, ensemble_results
            )
            
            # Calculate feature importance
            feature_importance = self._calculate_ensemble_feature_importance()
            
            # Generate model explanations
            model_explanations = await self._generate_model_explanations(
                prepared_data, predictions_with_uncertainty
            )
            
            return AgentOutput(
                agent_name=self.name,
                data={
                    "predictions": predictions_with_uncertainty,
                    "model_performance": ensemble_results.get("performance_metrics", {}),
                    "feature_importance": feature_importance,
                    "model_explanations": model_explanations,
                    "ensemble_weights": ensemble_results.get("ensemble_weights", {}),
                    "uncertainty_metrics": self._calculate_uncertainty_metrics(predictions_with_uncertainty)
                },
                metadata={
                    "models_trained": len(self.models),
                    "symbols_processed": len(symbols),
                    "target_variable": target_variable,
                    "prediction_mode": prediction_mode,
                    "training_timestamp": datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"ML ensemble processing failed: {str(e)}")
            raise ModelError(f"ML ensemble processing failed: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters."""
        required_keys = ["enhanced_feature_matrix"]
        self.validate_inputs(inputs, required_keys)
        
        feature_matrix = inputs["enhanced_feature_matrix"]
        if feature_matrix.empty:
            raise ValidationError("Feature matrix cannot be empty")
        
        target_variable = inputs.get("target_variable", "forward_return_1d")
        if target_variable not in feature_matrix.columns:
            raise ValidationError(f"Target variable '{target_variable}' not found in feature matrix")
    
    async def _prepare_ml_data(self, feature_matrix: pd.DataFrame, 
                             target_variable: str, symbols: List[str]) -> Dict[str, Any]:
        """Prepare data for machine learning."""
        # Remove non-feature columns
        exclude_columns = [
            "symbol", "timestamp", "frequency", 
            col for col in feature_matrix.columns if col.startswith("forward_") and col != target_variable
        ]
        
        feature_columns = [col for col in feature_matrix.columns if col not in exclude_columns]
        
        # Prepare features and target
        X = feature_matrix[feature_columns].copy()
        y = feature_matrix[target_variable].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Remove rows where target is NaN
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ModelError("No valid data remaining after preprocessing")
        
        # Feature engineering for ML
        X = self._engineer_ml_features(X)
        
        # Split data chronologically for time series
        split_data = self._split_time_series_data(X, y, feature_matrix[valid_mask])
        
        return {
            "X": X,
            "y": y,
            "feature_columns": list(X.columns),
            "target_column": target_variable,
            "split_data": split_data,
            "symbols": symbols
        }
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill then backward fill
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaNs, fill with median
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        cols_to_keep = X.columns[X.isnull().mean() < missing_threshold]
        X = X[cols_to_keep]
        
        return X
    
    def _engineer_ml_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for ML models."""
        X_engineered = X.copy()
        
        # Feature interactions (for most important features)
        numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns[:10]  # Top 10 to avoid explosion
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Ratio features
                if (X_engineered[col2] != 0).all():
                    X_engineered[f"{col1}_div_{col2}"] = X_engineered[col1] / X_engineered[col2]
                
                # Product features (for selected pairs)
                if len(X_engineered.columns) < 200:  # Limit feature explosion
                    X_engineered[f"{col1}_mult_{col2}"] = X_engineered[col1] * X_engineered[col2]
        
        # Polynomial features for key indicators
        key_features = [col for col in X_engineered.columns if any(indicator in col for indicator in ['rsi', 'macd', 'bb_position'])]
        for feature in key_features[:5]:  # Limit to top 5
            X_engineered[f"{feature}_squared"] = X_engineered[feature] ** 2
        
        # Remove infinite values
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(X_engineered.median())
        
        return X_engineered
    
    def _split_time_series_data(self, X: pd.DataFrame, y: pd.Series, 
                              full_data: pd.DataFrame) -> Dict[str, Any]:
        """Split data chronologically for time series."""
        # Sort by timestamp if available
        if "timestamp" in full_data.columns:
            sort_idx = full_data["timestamp"].sort_values().index
            X = X.reindex(sort_idx)
            y = y.reindex(sort_idx)
        
        # Calculate split indices
        n_samples = len(X)
        train_size = int(n_samples * (1 - self.test_size - self.validation_size))
        val_size = int(n_samples * self.validation_size)
        
        # Split data
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        
        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }
    
    async def _train_individual_models(self, prepared_data: Dict[str, Any], 
                                     prediction_mode: str) -> Dict[str, Any]:
        """Train individual ML models."""
        split_data = prepared_data["split_data"]
        models_results = {}
        
        # XGBoost
        if "xgboost" in self.ensemble_models:
            self.logger.info("Training XGBoost model...")
            xgb_result = await self._train_xgboost(split_data, prediction_mode)
            models_results["xgboost"] = xgb_result
            self.models["xgboost"] = xgb_result["model"]
        
        # CatBoost
        if "catboost" in self.ensemble_models:
            self.logger.info("Training CatBoost model...")
            cb_result = await self._train_catboost(split_data, prediction_mode)
            models_results["catboost"] = cb_result
            self.models["catboost"] = cb_result["model"]
        
        # Random Forest
        if "random_forest" in self.ensemble_models:
            self.logger.info("Training Random Forest model...")
            rf_result = await self._train_random_forest(split_data, prediction_mode)
            models_results["random_forest"] = rf_result
            self.models["random_forest"] = rf_result["model"]
        
        return models_results
    
    async def _train_xgboost(self, split_data: Dict[str, Any], 
                           prediction_mode: str) -> Dict[str, Any]:
        """Train XGBoost model with hyperparameter optimization."""
        X_train, y_train = split_data["X_train"], split_data["y_train"]
        X_val, y_val = split_data["X_val"], split_data["y_val"]
        X_test, y_test = split_data["X_test"], split_data["y_test"]
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42
            }
            
            if prediction_mode == "regression":
                model = xgb.XGBRegressor(**params)
            else:
                model = xgb.XGBClassifier(**params)
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_val)
            
            if prediction_mode == "regression":
                return mean_squared_error(y_val, pred)
            else:
                from sklearn.metrics import accuracy_score
                return -accuracy_score(y_val, pred)
        
        # Optimize hyperparameters
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(self.n_trials, 50))
        
        # Train final model
        best_params = study.best_params
        best_params["random_state"] = 42
        
        if prediction_mode == "regression":
            model = xgb.XGBRegressor(**best_params)
        else:
            model = xgb.XGBClassifier(**best_params)
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        performance = self._calculate_model_performance(
            y_train, train_pred, y_val, val_pred, y_test, test_pred, prediction_mode
        )
        
        return {
            "model": model,
            "performance": performance,
            "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
            "best_params": best_params
        }
    
    async def _train_catboost(self, split_data: Dict[str, Any], 
                            prediction_mode: str) -> Dict[str, Any]:
        """Train CatBoost model with hyperparameter optimization."""
        X_train, y_train = split_data["X_train"], split_data["y_train"]
        X_val, y_val = split_data["X_val"], split_data["y_val"]
        X_test, y_test = split_data["X_test"], split_data["y_test"]
        
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_seed": 42,
                "verbose": False
            }
            
            if prediction_mode == "regression":
                model = cb.CatBoostRegressor(**params)
            else:
                model = cb.CatBoostClassifier(**params)
            
            model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)
            pred = model.predict(X_val)
            
            if prediction_mode == "regression":
                return mean_squared_error(y_val, pred)
            else:
                from sklearn.metrics import accuracy_score
                return -accuracy_score(y_val, pred)
        
        # Optimize hyperparameters
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(self.n_trials, 50))
        
        # Train final model
        best_params = study.best_params
        best_params["random_seed"] = 42
        best_params["verbose"] = False
        
        if prediction_mode == "regression":
            model = cb.CatBoostRegressor(**best_params)
        else:
            model = cb.CatBoostClassifier(**best_params)
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        performance = self._calculate_model_performance(
            y_train, train_pred, y_val, val_pred, y_test, test_pred, prediction_mode
        )
        
        return {
            "model": model,
            "performance": performance,
            "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
            "best_params": best_params
        }
    
    async def _train_random_forest(self, split_data: Dict[str, Any], 
                                 prediction_mode: str) -> Dict[str, Any]:
        """Train Random Forest model."""
        X_train, y_train = split_data["X_train"], split_data["y_train"]
        X_val, y_val = split_data["X_val"], split_data["y_val"]
        X_test, y_test = split_data["X_test"], split_data["y_test"]
        
        if prediction_mode == "regression":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        performance = self._calculate_model_performance(
            y_train, train_pred, y_val, val_pred, y_test, test_pred, prediction_mode
        )
        
        return {
            "model": model,
            "performance": performance,
            "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
            "best_params": {}
        }
    
    async def _create_ensemble_model(self, prepared_data: Dict[str, Any], 
                                   models_results: Dict[str, Any],
                                   prediction_mode: str) -> Dict[str, Any]:
        """Create ensemble model from individual models."""
        split_data = prepared_data["split_data"]
        X_train, y_train = split_data["X_train"], split_data["y_train"]
        X_val, y_val = split_data["X_val"], split_data["y_val"]
        X_test, y_test = split_data["X_test"], split_data["y_test"]
        
        # Prepare base models for ensemble
        base_models = []
        model_names = []
        
        for name, result in models_results.items():
            base_models.append((name, result["model"]))
            model_names.append(name)
        
        # Create stacking ensemble
        if prediction_mode == "regression":
            from sklearn.linear_model import Ridge
            meta_model = Ridge(alpha=1.0)
            ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=3
            )
        else:
            from sklearn.linear_model import LogisticRegression
            meta_model = LogisticRegression(random_state=42)
            from sklearn.ensemble import StackingClassifier
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=3
            )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.ensemble_model = ensemble
        
        # Evaluate ensemble
        train_pred = ensemble.predict(X_train)
        val_pred = ensemble.predict(X_val)
        test_pred = ensemble.predict(X_test)
        
        ensemble_performance = self._calculate_model_performance(
            y_train, train_pred, y_val, val_pred, y_test, test_pred, prediction_mode
        )
        
        # Calculate ensemble weights (approximate)
        ensemble_weights = self._calculate_ensemble_weights(models_results, ensemble_performance)
        
        return {
            "ensemble_model": ensemble,
            "performance_metrics": ensemble_performance,
            "ensemble_weights": ensemble_weights,
            "individual_models": models_results
        }
    
    async def _predict_with_existing_models(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use existing models for prediction."""
        if not self.models or not self.ensemble_model:
            raise ModelError("No trained models available for prediction")
        
        X = prepared_data["X"]
        
        # Generate predictions from ensemble
        predictions = self.ensemble_model.predict(X)
        
        return {
            "ensemble_model": self.ensemble_model,
            "predictions": predictions,
            "performance_metrics": {},
            "ensemble_weights": {}
        }
    
    async def _generate_predictions_with_uncertainty(self, prepared_data: Dict[str, Any],
                                                   ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions with uncertainty quantification."""
        X = prepared_data["X"]
        ensemble_model = ensemble_results["ensemble_model"]
        
        # Main ensemble prediction
        main_prediction = ensemble_model.predict(X)
        
        # Uncertainty estimation using bootstrap ensemble
        uncertainty_estimates = self._estimate_prediction_uncertainty(X, ensemble_model)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            main_prediction, uncertainty_estimates
        )
        
        return {
            "predictions": main_prediction,
            "prediction_std": uncertainty_estimates["std"],
            "confidence_intervals": confidence_intervals,
            "uncertainty_score": uncertainty_estimates["uncertainty_score"],
            "feature_columns": prepared_data["feature_columns"]
        }
    
    def _calculate_model_performance(self, y_train: pd.Series, train_pred: np.ndarray,
                                   y_val: pd.Series, val_pred: np.ndarray,
                                   y_test: pd.Series, test_pred: np.ndarray,
                                   prediction_mode: str) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics."""
        performance = {}
        
        if prediction_mode == "regression":
            # Training metrics
            performance["train_mse"] = mean_squared_error(y_train, train_pred)
            performance["train_mae"] = mean_absolute_error(y_train, train_pred)
            performance["train_r2"] = r2_score(y_train, train_pred)
            
            # Validation metrics
            performance["val_mse"] = mean_squared_error(y_val, val_pred)
            performance["val_mae"] = mean_absolute_error(y_val, val_pred)
            performance["val_r2"] = r2_score(y_val, val_pred)
            
            # Test metrics
            if len(y_test) > 0:
                performance["test_mse"] = mean_squared_error(y_test, test_pred)
                performance["test_mae"] = mean_absolute_error(y_test, test_pred)
                performance["test_r2"] = r2_score(y_test, test_pred)
            
            # Financial metrics
            performance["directional_accuracy"] = self._calculate_directional_accuracy(y_val, val_pred)
            
        else:  # classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Training metrics
            performance["train_accuracy"] = accuracy_score(y_train, train_pred)
            performance["train_precision"] = precision_score(y_train, train_pred, average='weighted')
            performance["train_recall"] = recall_score(y_train, train_pred, average='weighted')
            performance["train_f1"] = f1_score(y_train, train_pred, average='weighted')
            
            # Validation metrics
            performance["val_accuracy"] = accuracy_score(y_val, val_pred)
            performance["val_precision"] = precision_score(y_val, val_pred, average='weighted')
            performance["val_recall"] = recall_score(y_val, val_pred, average='weighted')
            performance["val_f1"] = f1_score(y_val, val_pred, average='weighted')
            
            # Test metrics
            if len(y_test) > 0:
                performance["test_accuracy"] = accuracy_score(y_test, test_pred)
                performance["test_precision"] = precision_score(y_test, test_pred, average='weighted')
                performance["test_recall"] = recall_score(y_test, test_pred, average='weighted')
                performance["test_f1"] = f1_score(y_test, test_pred, average='weighted')
        
        return performance
    
    def _calculate_directional_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for financial predictions."""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        return np.mean(true_direction == pred_direction)
    
    def _calculate_ensemble_weights(self, models_results: Dict[str, Any], 
                                  ensemble_performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate approximate ensemble weights based on individual model performance."""
        weights = {}
        total_performance = 0
        
        # Use validation R2 or accuracy as weight basis
        weight_metric = "val_r2" if "val_r2" in list(models_results.values())[0]["performance"] else "val_accuracy"
        
        for name, result in models_results.items():
            performance = result["performance"].get(weight_metric, 0)
            # Convert to positive weight (handle negative R2)
            weight = max(performance, 0.01)
            weights[name] = weight
            total_performance += weight
        
        # Normalize weights
        if total_performance > 0:
            weights = {name: weight / total_performance for name, weight in weights.items()}
        
        return weights
    
    def _estimate_prediction_uncertainty(self, X: pd.DataFrame, 
                                       ensemble_model) -> Dict[str, np.ndarray]:
        """Estimate prediction uncertainty using model ensemble."""
        # Get predictions from individual models if available
        individual_predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                individual_predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Failed to get prediction from {name}: {e}")
                continue
        
        if individual_predictions:
            # Calculate standard deviation across models
            pred_array = np.array(individual_predictions)
            pred_std = np.std(pred_array, axis=0)
            pred_mean = np.mean(pred_array, axis=0)
            
            # Uncertainty score (higher = more uncertain)
            uncertainty_score = pred_std / (np.abs(pred_mean) + 1e-8)
        else:
            # Fallback: use dummy uncertainty
            pred_std = np.ones(len(X)) * 0.1
            uncertainty_score = np.ones(len(X)) * 0.5
        
        return {
            "std": pred_std,
            "uncertainty_score": uncertainty_score
        }
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, 
                                      uncertainty_estimates: Dict[str, np.ndarray],
                                      confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions."""
        z_score = 1.96  # 95% confidence interval
        std = uncertainty_estimates["std"]
        
        lower_bound = predictions - z_score * std
        upper_bound = predictions + z_score * std
        
        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": confidence_level
        }
    
    def _calculate_ensemble_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance across ensemble."""
        if not self.models:
            return {}
        
        # Aggregate feature importance from all models
        all_importances = {}
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importance))]
                    
                    for fname, imp in zip(feature_names, importance):
                        if fname not in all_importances:
                            all_importances[fname] = 0
                        all_importances[fname] += imp
                    
                    total_weight += 1
            except Exception as e:
                self.logger.warning(f"Failed to get feature importance from {name}: {e}")
        
        # Normalize by number of models
        if total_weight > 0:
            all_importances = {name: imp / total_weight for name, imp in all_importances.items()}
        
        # Sort by importance
        sorted_importance = dict(sorted(all_importances.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    async def _generate_model_explanations(self, prepared_data: Dict[str, Any],
                                         predictions_with_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations for model predictions."""
        explanations = {
            "top_features": [],
            "model_agreement": {},
            "prediction_rationale": [],
            "uncertainty_analysis": {}
        }
        
        # Top contributing features
        feature_importance = self.feature_importance or self._calculate_ensemble_feature_importance()
        explanations["top_features"] = list(feature_importance.keys())[:10]
        
        # Model agreement analysis
        if len(self.models) > 1:
            explanations["model_agreement"] = self._analyze_model_agreement(prepared_data["X"])
        
        # Prediction rationale for sample predictions
        explanations["prediction_rationale"] = self._generate_prediction_rationale(
            prepared_data, predictions_with_uncertainty, feature_importance
        )
        
        # Uncertainty analysis
        uncertainty_score = predictions_with_uncertainty.get("uncertainty_score", np.array([]))
        if len(uncertainty_score) > 0:
            explanations["uncertainty_analysis"] = {
                "high_uncertainty_threshold": np.percentile(uncertainty_score, 90),
                "mean_uncertainty": np.mean(uncertainty_score),
                "uncertainty_distribution": {
                    "low": np.sum(uncertainty_score < 0.3),
                    "medium": np.sum((uncertainty_score >= 0.3) & (uncertainty_score < 0.7)),
                    "high": np.sum(uncertainty_score >= 0.7)
                }
            }
        
        return explanations
    
    def _analyze_model_agreement(self, X: pd.DataFrame) -> Dict[str, float]:
        """Analyze agreement between different models."""
        if len(self.models) < 2:
            return {}
        
        # Get predictions from all models
        all_predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                all_predictions[name] = pred
            except Exception:
                continue
        
        if len(all_predictions) < 2:
            return {}
        
        # Calculate pairwise correlations
        model_names = list(all_predictions.keys())
        correlations = {}
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                corr = np.corrcoef(all_predictions[name1], all_predictions[name2])[0, 1]
                correlations[f"{name1}_{name2}"] = corr
        
        # Overall agreement score
        avg_correlation = np.mean(list(correlations.values()))
        
        return {
            "pairwise_correlations": correlations,
            "average_agreement": avg_correlation,
            "agreement_level": "high" if avg_correlation > 0.8 else "medium" if avg_correlation > 0.5 else "low"
        }
    
    def _generate_prediction_rationale(self, prepared_data: Dict[str, Any],
                                     predictions_with_uncertainty: Dict[str, Any],
                                     feature_importance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rationale for sample predictions."""
        X = prepared_data["X"]
        predictions = predictions_with_uncertainty["predictions"]
        
        rationales = []
        
        # Sample a few predictions for explanation
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        
        for idx in sample_indices:
            sample_features = X.iloc[idx]
            prediction = predictions[idx]
            
            # Find most contributing features for this prediction
            feature_contributions = {}
            for feature, importance in list(feature_importance.items())[:5]:
                if feature in sample_features.index:
                    feature_contributions[feature] = {
                        "value": sample_features[feature],
                        "importance": importance
                    }
            
            rationale = {
                "sample_index": int(idx),
                "prediction": float(prediction),
                "top_contributing_features": feature_contributions,
                "prediction_strength": "strong" if abs(prediction) > 0.02 else "weak"
            }
            
            rationales.append(rationale)
        
        return rationales
    
    def _calculate_uncertainty_metrics(self, predictions_with_uncertainty: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall uncertainty metrics."""
        uncertainty_score = predictions_with_uncertainty.get("uncertainty_score", np.array([]))
        
        if len(uncertainty_score) == 0:
            return {}
        
        return {
            "mean_uncertainty": float(np.mean(uncertainty_score)),
            "max_uncertainty": float(np.max(uncertainty_score)),
            "uncertainty_std": float(np.std(uncertainty_score)),
            "high_uncertainty_ratio": float(np.mean(uncertainty_score > 0.7))
        }