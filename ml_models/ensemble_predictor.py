"""
Ensemble Model for Advanced Cross-Asset Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    Advanced ensemble model combining multiple ML algorithms for cross-asset prediction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'models': ['lstm', 'transformer', 'random_forest', 'gradient_boosting', 'svr'],
            'weights': [0.3, 0.2, 0.2, 0.2, 0.1],  # Model weights
            'prediction_horizon': 24,
            'confidence_threshold': 0.6,
            'ensemble_method': 'weighted_average'  # 'weighted_average', 'voting', 'stacking'
        }
        
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.symbol = None
        self.asset_class = None
        
    async def add_model(self, model_name: str, model_instance: Any, 
                       weight: float = 1.0) -> bool:
        """
        Add a model to the ensemble
        """
        try:
            self.models[model_name] = {
                'instance': model_instance,
                'weight': weight,
                'is_trained': False
            }
            return True
        except Exception as e:
            print(f"Error adding model {model_name}: {e}")
            return False
    
    async def train_ensemble(self, data: pd.DataFrame, symbol: str, 
                           asset_class: str = 'equity') -> Dict[str, Any]:
        """
        Train all models in the ensemble
        """
        try:
            print(f"🔬 Training Ensemble model for {symbol} ({asset_class})")
            
            self.symbol = symbol
            self.asset_class = asset_class
            
            # Prepare data
            X, y = self._prepare_ensemble_data(data)
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train each model
            training_results = {}
            for model_name, model_info in self.models.items():
                try:
                    if hasattr(model_info['instance'], 'train_model'):
                        # Async training for LSTM/Transformer
                        result = await model_info['instance'].train_model(data, symbol, asset_class)
                    else:
                        # Sync training for sklearn models
                        result = self._train_sklearn_model(
                            model_info['instance'], X_train, y_train, X_test, y_test
                        )
                    
                    training_results[model_name] = result
                    model_info['is_trained'] = result.get('success', False)
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    training_results[model_name] = {'success': False, 'error': str(e)}
            
            # Calculate ensemble performance
            ensemble_metrics = self._calculate_ensemble_metrics(training_results)
            
            self.is_trained = any(model_info['is_trained'] for model_info in self.models.values())
            
            return {
                'success': self.is_trained,
                'symbol': symbol,
                'asset_class': asset_class,
                'training_results': training_results,
                'ensemble_metrics': ensemble_metrics,
                'models_trained': sum(1 for model_info in self.models.values() if model_info['is_trained'])
            }
            
        except Exception as e:
            print(f"Error training ensemble for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_ensemble_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ensemble training
        """
        # Select features
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            available_features = ['Close', 'Volume']
        
        # Prepare feature data
        feature_data = data[available_features].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Create target (next period's close price)
        target = data['Close'].values[1:]  # Shift by 1 period
        features_for_target = scaled_features[:-1]  # Remove last row to match target
        
        return features_for_target, target
    
    def _train_sklearn_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train sklearn model
        """
        try:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            training_time = time.time() - start_time
            
            return {
                'success': True,
                'training_time': training_time,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_ensemble_metrics(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ensemble performance metrics
        """
        successful_models = [name for name, result in training_results.items() 
                           if result.get('success', False)]
        
        if not successful_models:
            return {'success': False, 'error': 'No models trained successfully'}
        
        # Calculate average metrics
        avg_test_rmse = np.mean([training_results[name].get('test_rmse', 0) 
                                for name in successful_models])
        avg_test_mae = np.mean([training_results[name].get('test_mae', 0) 
                               for name in successful_models])
        avg_test_r2 = np.mean([training_results[name].get('test_r2', 0) 
                              for name in successful_models])
        
        return {
            'success': True,
            'successful_models': successful_models,
            'num_models': len(successful_models),
            'avg_test_rmse': avg_test_rmse,
            'avg_test_mae': avg_test_mae,
            'avg_test_r2': avg_test_r2
        }
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make ensemble predictions
        """
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Ensemble not trained'}
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model_info in self.models.items():
                if not model_info['is_trained']:
                    continue
                
                try:
                    model_instance = model_info['instance']
                    
                    if hasattr(model_instance, 'predict'):
                        # Async prediction for LSTM/Transformer
                        if hasattr(model_instance, 'predict') and asyncio.iscoroutinefunction(model_instance.predict):
                            result = await model_instance.predict(data)
                        else:
                            # Sync prediction for sklearn models
                            X, _ = self._prepare_ensemble_data(data)
                            if len(X) > 0:
                                pred = model_instance.predict(X[-1:])[0]
                                result = {
                                    'success': True,
                                    'predicted_price': pred,
                                    'confidence': 0.7  # Default confidence
                                }
                            else:
                                continue
                    else:
                        continue
                    
                    if result.get('success', False):
                        if 'predicted_prices' in result:
                            # LSTM/Transformer returns multiple predictions
                            predictions[model_name] = result['predicted_prices']
                            confidences[model_name] = result.get('confidence', 0.7)
                        else:
                            # Sklearn returns single prediction
                            predictions[model_name] = [result['predicted_price']]
                            confidences[model_name] = result.get('confidence', 0.7)
                
                except Exception as e:
                    print(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            if not predictions:
                return {'success': False, 'error': 'No valid predictions from any model'}
            
            # Combine predictions using ensemble method
            ensemble_prediction = self._combine_predictions(predictions, confidences)
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(predictions, confidences)
            
            return {
                'success': True,
                'symbol': self.symbol,
                'asset_class': self.asset_class,
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'num_models_used': len(predictions)
            }
            
        except Exception as e:
            print(f"Error making ensemble prediction: {e}")
            return {'success': False, 'error': str(e)}
    
    def _combine_predictions(self, predictions: Dict[str, List[float]], 
                           confidences: Dict[str, float]) -> List[float]:
        """
        Combine predictions from multiple models
        """
        if self.config['ensemble_method'] == 'weighted_average':
            return self._weighted_average_combination(predictions, confidences)
        elif self.config['ensemble_method'] == 'voting':
            return self._voting_combination(predictions, confidences)
        else:
            return self._weighted_average_combination(predictions, confidences)
    
    def _weighted_average_combination(self, predictions: Dict[str, List[float]], 
                                    confidences: Dict[str, float]) -> List[float]:
        """
        Combine predictions using weighted average
        """
        # Get the maximum prediction horizon
        max_horizon = max(len(preds) for preds in predictions.values())
        
        # Initialize ensemble prediction
        ensemble_pred = [0.0] * max_horizon
        total_weights = [0.0] * max_horizon
        
        # Calculate weighted average
        for model_name, preds in predictions.items():
            weight = confidences.get(model_name, 0.5)
            
            for i in range(len(preds)):
                if i < max_horizon:
                    ensemble_pred[i] += preds[i] * weight
                    total_weights[i] += weight
        
        # Normalize by total weights
        for i in range(max_horizon):
            if total_weights[i] > 0:
                ensemble_pred[i] /= total_weights[i]
        
        return ensemble_pred
    
    def _voting_combination(self, predictions: Dict[str, List[float]], 
                          confidences: Dict[str, float]) -> List[float]:
        """
        Combine predictions using voting (median)
        """
        # Get the maximum prediction horizon
        max_horizon = max(len(preds) for preds in predictions.values())
        
        ensemble_pred = []
        
        for i in range(max_horizon):
            # Collect predictions for this horizon
            horizon_preds = []
            for preds in predictions.values():
                if i < len(preds):
                    horizon_preds.append(preds[i])
            
            if horizon_preds:
                # Use median for voting
                ensemble_pred.append(np.median(horizon_preds))
            else:
                ensemble_pred.append(0.0)
        
        return ensemble_pred
    
    def _calculate_ensemble_confidence(self, predictions: Dict[str, List[float]], 
                                     confidences: Dict[str, float]) -> float:
        """
        Calculate ensemble confidence based on model agreement and individual confidences
        """
        try:
            # Calculate prediction agreement
            if len(predictions) < 2:
                return np.mean(list(confidences.values()))
            
            # Calculate variance of predictions (lower variance = higher agreement)
            max_horizon = max(len(preds) for preds in predictions.values())
            prediction_variances = []
            
            for i in range(max_horizon):
                horizon_preds = []
                for preds in predictions.values():
                    if i < len(preds):
                        horizon_preds.append(preds[i])
                
                if len(horizon_preds) > 1:
                    prediction_variances.append(np.var(horizon_preds))
            
            # Agreement score (inverse of average variance)
            if prediction_variances:
                avg_variance = np.mean(prediction_variances)
                agreement_score = 1 / (1 + avg_variance)
            else:
                agreement_score = 0.5
            
            # Average confidence
            avg_confidence = np.mean(list(confidences.values()))
            
            # Combined confidence
            ensemble_confidence = (agreement_score * 0.6 + avg_confidence * 0.4)
            
            return min(1.0, max(0.0, ensemble_confidence))
            
        except Exception as e:
            print(f"Error calculating ensemble confidence: {e}")
            return 0.5
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """
        Get ensemble summary and statistics
        """
        if not self.is_trained:
            return {'success': False, 'error': 'Ensemble not trained'}
        
        trained_models = [name for name, info in self.models.items() if info['is_trained']]
        
        return {
            'success': True,
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'is_trained': self.is_trained,
            'config': self.config,
            'trained_models': trained_models,
            'num_trained_models': len(trained_models),
            'total_models': len(self.models)
        }
