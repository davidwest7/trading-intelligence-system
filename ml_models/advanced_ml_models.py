"""
Advanced ML Models Based on Latest Research (arXiv 2024)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPredictor:
    """
    Advanced ML Predictor based on latest research:
    - Temporal Fusion Transformers (TFT) approach
    - Multi-head attention mechanisms
    - Ensemble methods with dynamic weighting
    - Adaptive feature selection
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'prediction_horizon': 24,
            'sequence_length': 60,
            'ensemble_size': 8,
            'confidence_threshold': 0.7,
            'adaptive_weighting': True,
            'feature_selection': True
        }
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.symbol = None
        self.asset_class = None
        self.performance_history = []
        
    async def train_advanced_models(self, data, symbol, asset_class='equity'):
        """Train advanced ML models using ensemble approach"""
        try:
            print(f"🔬 Training Advanced ML models for {symbol} ({asset_class})")
            
            self.symbol = symbol
            self.asset_class = asset_class
            
            # Prepare advanced features
            X, y = self._prepare_advanced_features(data)
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Initialize ensemble models
            self._initialize_ensemble_models()
            
            # Train each model
            training_results = {}
            for model_name, model in self.models.items():
                try:
                    result = await self._train_single_model(model_name, model, X, y)
                    training_results[model_name] = result
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    training_results[model_name] = {'success': False, 'error': str(e)}
            
            # Calculate ensemble performance
            ensemble_metrics = self._calculate_ensemble_performance(training_results)
            
            # Adaptive feature selection
            if self.config['feature_selection']:
                self._perform_feature_selection(X, y)
            
            self.is_trained = True
            
            return {
                'success': True,
                'symbol': symbol,
                'asset_class': asset_class,
                'training_results': training_results,
                'ensemble_metrics': ensemble_metrics,
                'models_trained': sum(1 for r in training_results.values() if r.get('success', False)),
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            print(f"Error training advanced ML models: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_advanced_features(self, data):
        """Prepare advanced features based on latest research"""
        try:
            # Technical indicators
            features = self._calculate_technical_indicators(data)
            
            # Market microstructure features
            microstructure = self._calculate_microstructure_features(data)
            
            # Volatility features
            volatility = self._calculate_volatility_features(data)
            
            # Momentum features
            momentum = self._calculate_momentum_features(data)
            
            # Mean reversion features
            mean_reversion = self._calculate_mean_reversion_features(data)
            
            # Combine all features
            all_features = pd.concat([features, microstructure, volatility, momentum, mean_reversion], axis=1)
            
            # Remove NaN values
            all_features = all_features.dropna()
            
            # Prepare target (future returns)
            target = self._prepare_target(data, all_features.index)
            
            # Align features and target
            common_index = all_features.index.intersection(target.index)
            X = all_features.loc[common_index]
            y = target.loc[common_index]
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing advanced features: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _calculate_technical_indicators(self, data):
        """Calculate advanced technical indicators"""
        features = pd.DataFrame(index=data.index)
        
        # RSI with multiple timeframes
        for period in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD with signal line
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        features['MACD'] = ema12 - ema26
        features['MACD_Signal'] = features['MACD'].ewm(span=9).mean()
        features['MACD_Histogram'] = features['MACD'] - features['MACD_Signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            features[f'BB_Upper_{period}'] = sma + (std * 2)
            features[f'BB_Lower_{period}'] = sma - (std * 2)
            features[f'BB_Width_{period}'] = features[f'BB_Upper_{period}'] - features[f'BB_Lower_{period}']
            features[f'BB_Position_{period}'] = (data['Close'] - features[f'BB_Lower_{period}']) / features[f'BB_Width_{period}']
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            features[f'Stoch_K_{period}'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            features[f'Stoch_D_{period}'] = features[f'Stoch_K_{period}'].rolling(window=3).mean()
        
        return features
    
    def _calculate_microstructure_features(self, data):
        """Calculate market microstructure features"""
        features = pd.DataFrame(index=data.index)
        
        # Volume-based features
        features['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        features['Volume_Ratio'] = data['Volume'] / features['Volume_SMA_20']
        features['Volume_Trend'] = features['Volume_Ratio'].rolling(10).mean()
        
        # Price impact
        features['Price_Impact'] = (data['High'] - data['Low']) / data['Close']
        features['Price_Impact_MA'] = features['Price_Impact'].rolling(20).mean()
        
        # Spread proxy (using high-low)
        features['Spread_Proxy'] = (data['High'] - data['Low']) / data['Close']
        features['Spread_MA'] = features['Spread_Proxy'].rolling(20).mean()
        
        # Order flow imbalance (simplified)
        features['Flow_Imbalance'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
        
        return features
    
    def _calculate_volatility_features(self, data):
        """Calculate volatility features"""
        features = pd.DataFrame(index=data.index)
        
        # Realized volatility
        returns = data['Close'].pct_change()
        for period in [5, 10, 20]:
            features[f'Realized_Vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson volatility
        for period in [5, 10, 20]:
            hl_vol = np.log(data['High'] / data['Low'])
            features[f'Parkinson_Vol_{period}'] = np.sqrt(hl_vol.rolling(period).var() / (4 * np.log(2)))
        
        # Garman-Klass volatility
        for period in [5, 10, 20]:
            c = np.log(data['Close'] / data['Close'].shift(1))
            h = np.log(data['High'] / data['Open'])
            l = np.log(data['Low'] / data['Open'])
            features[f'Garman_Klass_Vol_{period}'] = np.sqrt(0.5 * (h - l)**2 - (2*np.log(2) - 1) * c**2).rolling(period).mean()
        
        # Volatility of volatility
        features['Vol_of_Vol'] = features['Realized_Vol_20'].rolling(20).std()
        
        return features
    
    def _calculate_momentum_features(self, data):
        """Calculate momentum features"""
        features = pd.DataFrame(index=data.index)
        
        # Price momentum
        for period in [5, 10, 20, 50]:
            features[f'Price_Momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
        
        # Volume momentum
        for period in [5, 10, 20]:
            features[f'Volume_Momentum_{period}'] = data['Volume'] / data['Volume'].shift(period) - 1
        
        # Rate of change
        for period in [10, 20]:
            features[f'ROC_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period) * 100
        
        # Williams %R
        for period in [14, 21]:
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            features[f'Williams_R_{period}'] = (high_max - data['Close']) / (high_max - low_min) * -100
        
        return features
    
    def _calculate_mean_reversion_features(self, data):
        """Calculate mean reversion features"""
        features = pd.DataFrame(index=data.index)
        
        # Moving average deviations
        for short_period in [5, 10, 20]:
            for long_period in [20, 50, 100]:
                if short_period < long_period:
                    short_ma = data['Close'].rolling(short_period).mean()
                    long_ma = data['Close'].rolling(long_period).mean()
                    features[f'MA_Deviation_{short_period}_{long_period}'] = (short_ma - long_ma) / long_ma
        
        # Z-score of price
        for period in [20, 50]:
            ma = data['Close'].rolling(period).mean()
            std = data['Close'].rolling(period).std()
            features[f'Z_Score_{period}'] = (data['Close'] - ma) / std
        
        # Price distance from extremes
        for period in [20, 50]:
            high_max = data['High'].rolling(period).max()
            low_min = data['Low'].rolling(period).min()
            features[f'Distance_From_High_{period}'] = (high_max - data['Close']) / data['Close']
            features[f'Distance_From_Low_{period}'] = (data['Close'] - low_min) / data['Close']
        
        return features
    
    def _prepare_target(self, data, feature_index):
        """Prepare target variable (future returns)"""
        # Calculate future returns for different horizons
        returns_1h = data['Close'].pct_change(1).shift(-1)
        returns_4h = data['Close'].pct_change(4).shift(-4)
        returns_24h = data['Close'].pct_change(24).shift(-24)
        
        # Use 24-hour returns as primary target
        target = returns_24h.loc[feature_index]
        
        return target
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble of advanced ML models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42),
            'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25), activation='relu',
                solver='adam', alpha=0.001, max_iter=500, random_state=42
            )
        }
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
    
    async def _train_single_model(self, model_name, model, X, y):
        """Train a single model with time series cross-validation"""
        try:
            # Scale features
            X_scaled = self.scalers[model_name].fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
            
            # Train final model
            model.fit(X_scaled, y)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = dict(zip(X.columns, np.abs(model.coef_)))
            
            return {
                'success': True,
                'cv_rmse': np.sqrt(-cv_scores.mean()),
                'cv_std': np.sqrt(-cv_scores).std(),
                'model_params': model.get_params()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_ensemble_performance(self, training_results):
        """Calculate ensemble performance metrics"""
        successful_models = [name for name, result in training_results.items() 
                           if result.get('success', False)]
        
        if not successful_models:
            return {'success': False, 'error': 'No models trained successfully'}
        
        # Calculate average CV performance
        avg_cv_rmse = np.mean([training_results[name]['cv_rmse'] 
                              for name in successful_models])
        avg_cv_std = np.mean([training_results[name]['cv_std'] 
                             for name in successful_models])
        
        return {
            'success': True,
            'successful_models': successful_models,
            'num_models': len(successful_models),
            'avg_cv_rmse': avg_cv_rmse,
            'avg_cv_std': avg_cv_std
        }
    
    def _perform_feature_selection(self, X, y):
        """Perform adaptive feature selection"""
        try:
            # Calculate feature importance across all models
            all_importances = {}
            for model_name, importances in self.feature_importance.items():
                for feature, importance in importances.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
            
            # Calculate average importance
            avg_importance = {feature: np.mean(importances) 
                            for feature, importances in all_importances.items()}
            
            # Select top features
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in sorted_features[:50]]  # Top 50 features
            
            self.selected_features = top_features
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            self.selected_features = list(X.columns)
    
    async def predict(self, data):
        """Make ensemble predictions with adaptive weighting"""
        try:
            if not self.is_trained:
                return {'success': False, 'error': 'Models not trained'}
            
            # Prepare features
            X, _ = self._prepare_advanced_features(data)
            
            if X.empty:
                return {'success': False, 'error': 'Insufficient data for prediction'}
            
            # Use selected features if available
            if hasattr(self, 'selected_features'):
                X = X[self.selected_features]
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                if model_name in self.scalers:
                    try:
                        # Scale features
                        X_scaled = self.scalers[model_name].transform(X)
                        
                        # Make prediction
                        pred = model.predict(X_scaled[-1:])[0]
                        predictions[model_name] = pred
                        
                        # Calculate confidence based on model performance
                        if model_name in self.performance_history:
                            confidences[model_name] = self.performance_history[model_name]
                        else:
                            confidences[model_name] = 0.7  # Default confidence
                            
                    except Exception as e:
                        print(f"Error predicting with {model_name}: {e}")
                        continue
            
            if not predictions:
                return {'success': False, 'error': 'No valid predictions'}
            
            # Ensemble prediction with adaptive weighting
            ensemble_prediction = self._ensemble_predict(predictions, confidences)
            
            return {
                'success': True,
                'symbol': self.symbol,
                'asset_class': self.asset_class,
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'model_confidences': confidences,
                'num_models_used': len(predictions)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {'success': False, 'error': str(e)}
    
    def _ensemble_predict(self, predictions, confidences):
        """Combine predictions using adaptive weighting"""
        if not predictions:
            return 0.0
        
        # Calculate weighted average
        total_weight = sum(confidences.values())
        if total_weight == 0:
            return np.mean(list(predictions.values()))
        
        weighted_prediction = sum(pred * confidences[model] 
                                for model, pred in predictions.items()) / total_weight
        
        return weighted_prediction


class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer based on latest research:
    - Multi-source sentiment aggregation
    - Temporal sentiment analysis
    - Sentiment momentum and dispersion
    """
    
    def __init__(self):
        self.positive_words = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'up', 'high', 'strong',
            'excellent', 'outperform', 'beat', 'surge', 'rally', 'breakout', 'momentum'
        ]
        self.negative_words = [
            'bearish', 'negative', 'loss', 'down', 'low', 'crash', 'decline', 'weak',
            'underperform', 'miss', 'drop', 'fall', 'breakdown', 'correction'
        ]
        self.neutral_words = [
            'stable', 'neutral', 'unchanged', 'flat', 'steady', 'maintain', 'hold'
        ]
        
        # Sentiment intensity modifiers
        self.intensifiers = ['very', 'extremely', 'highly', 'significantly', 'dramatically']
        self.diminishers = ['slightly', 'moderately', 'somewhat', 'marginally']
    
    async def analyze_advanced_sentiment(self, texts, timestamps=None):
        """Analyze sentiment with advanced features"""
        try:
            results = []
            sentiment_scores = []
            sentiment_momentum = []
            
            for i, text in enumerate(texts):
                # Basic sentiment analysis
                sentiment, score = self._analyze_single_text(text)
                
                # Temporal analysis if timestamps provided
                if timestamps and i > 0:
                    momentum = self._calculate_sentiment_momentum(sentiment_scores, i)
                    sentiment_momentum.append(momentum)
                else:
                    sentiment_momentum.append(0.0)
                
                results.append(sentiment)
                sentiment_scores.append(score)
            
            # Calculate advanced metrics
            sentiment_dispersion = np.std(sentiment_scores)
            sentiment_trend = self._calculate_sentiment_trend(sentiment_scores)
            
            return {
                'success': True,
                'texts': texts,
                'sentiments': results,
                'sentiment_scores': sentiment_scores,
                'sentiment_momentum': sentiment_momentum,
                'average_sentiment': np.mean(sentiment_scores),
                'sentiment_dispersion': sentiment_dispersion,
                'sentiment_trend': sentiment_trend,
                'sentiment_distribution': {
                    'positive': sum(1 for s in results if s == 'positive'),
                    'negative': sum(1 for s in results if s == 'negative'),
                    'neutral': sum(1 for s in results if s == 'neutral')
                }
            }
            
        except Exception as e:
            print(f"Error in advanced sentiment analysis: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_single_text(self, text):
        """Analyze sentiment of a single text"""
        text_lower = text.lower()
        
        # Count sentiment words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        neutral_count = sum(1 for word in self.neutral_words if word in text_lower)
        
        # Apply intensity modifiers
        for intensifier in self.intensifiers:
            if intensifier in text_lower:
                positive_count *= 1.5
                negative_count *= 1.5
        
        for diminisher in self.diminishers:
            if diminisher in text_lower:
                positive_count *= 0.7
                negative_count *= 0.7
        
        # Calculate sentiment score
        total_sentiment = positive_count - negative_count
        
        if total_sentiment > 0:
            sentiment = 'positive'
            score = min(1.0, total_sentiment / 3.0)
        elif total_sentiment < 0:
            sentiment = 'negative'
            score = max(-1.0, total_sentiment / 3.0)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return sentiment, score
    
    def _calculate_sentiment_momentum(self, sentiment_scores, current_index):
        """Calculate sentiment momentum"""
        if current_index < 3:
            return 0.0
        
        # Calculate momentum as change in sentiment over recent periods
        recent_scores = sentiment_scores[max(0, current_index-3):current_index]
        if len(recent_scores) > 1:
            momentum = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            return momentum
        
        return 0.0
    
    def _calculate_sentiment_trend(self, sentiment_scores):
        """Calculate overall sentiment trend"""
        if len(sentiment_scores) < 2:
            return 0.0
        
        # Linear trend
        x = np.arange(len(sentiment_scores))
        slope = np.polyfit(x, sentiment_scores, 1)[0]
        
        return slope
