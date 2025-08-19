"""
LSTM Neural Network for Cross-Asset Time Series Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from datetime import datetime, timedelta

class LSTMPredictor:
    """
    Advanced LSTM Neural Network for cross-asset time series prediction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'sequence_length': 60,  # 60 time steps (hours/days)
            'prediction_horizon': 24,  # 24 hours ahead
            'lstm_units': [128, 64, 32],  # LSTM layer sizes
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2
        }
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        self.asset_class = None
        self.symbol = None
        
    async def train_model(self, data: pd.DataFrame, symbol: str, 
                         asset_class: str = 'equity') -> Dict[str, Any]:
        """
        Train LSTM model on historical data
        """
        try:
            print(f"🔬 Training LSTM model for {symbol} ({asset_class})")
            
            self.symbol = symbol
            self.asset_class = asset_class
            
            # Prepare data
            X, y = self._prepare_data(data)
            
            if len(X) < 100:  # Need sufficient data
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            self.model = self._build_model(X_train.shape[1], X_train.shape[2])
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train model
            start_time = time.time()
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            training_time = time.time() - start_time
            
            # Evaluate model
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            self.is_trained = True
            
            return {
                'success': True,
                'symbol': symbol,
                'asset_class': asset_class,
                'training_time': training_time,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            }
            
        except Exception as e:
            print(f"Error training LSTM model for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        """
        # Select features
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            # Fallback to basic features
            available_features = ['Close', 'Volume']
        
        # Prepare feature data
        feature_data = data[available_features].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        
        for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i:i+prediction_horizon, 0])  # Predict Close price
        
        return np.array(X), np.array(y)
    
    def _build_model(self, sequence_length: int, n_features: int) -> Sequential:
        """
        Build LSTM model architecture
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.config['lstm_units'][0],
            return_sequences=True,
            input_shape=(sequence_length, n_features)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout_rate']))
        
        # Second LSTM layer
        if len(self.config['lstm_units']) > 1:
            model.add(LSTM(
                units=self.config['lstm_units'][1],
                return_sequences=True
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))
        
        # Third LSTM layer
        if len(self.config['lstm_units']) > 2:
            model.add(LSTM(
                units=self.config['lstm_units'][2],
                return_sequences=False
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(Dense(self.config['prediction_horizon']))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using trained model
        """
        try:
            if not self.is_trained or self.model is None:
                return {'success': False, 'error': 'Model not trained'}
            
            # Prepare input data
            features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 2:
                available_features = ['Close', 'Volume']
            
            feature_data = data[available_features].values
            
            # Scale data
            scaled_data = self.scaler.transform(feature_data)
            
            # Get last sequence
            sequence_length = self.config['sequence_length']
            if len(scaled_data) < sequence_length:
                return {'success': False, 'error': 'Insufficient data for prediction'}
            
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, -1)
            
            # Make prediction
            prediction = self.model.predict(last_sequence)
            
            # Inverse transform to get actual prices
            # Create dummy array for inverse transform
            dummy_array = np.zeros((len(prediction[0]), len(available_features)))
            dummy_array[:, 0] = prediction[0]  # Set Close price predictions
            
            # Inverse transform
            predicted_prices = self.scaler.inverse_transform(dummy_array)[:, 0]
            
            # Calculate prediction metrics
            current_price = data['Close'].iloc[-1]
            price_changes = [(price - current_price) / current_price for price in predicted_prices]
            
            return {
                'success': True,
                'symbol': self.symbol,
                'asset_class': self.asset_class,
                'current_price': current_price,
                'predicted_prices': predicted_prices.tolist(),
                'price_changes': price_changes,
                'prediction_horizon': self.config['prediction_horizon'],
                'confidence': self._calculate_prediction_confidence(predicted_prices, current_price)
            }
            
        except Exception as e:
            print(f"Error making LSTM prediction for {self.symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_prediction_confidence(self, predicted_prices: np.ndarray, 
                                       current_price: float) -> float:
        """
        Calculate prediction confidence based on price volatility and trend consistency
        """
        try:
            # Calculate price volatility
            price_std = np.std(predicted_prices)
            price_mean = np.mean(predicted_prices)
            
            # Calculate trend consistency
            price_changes = np.diff(predicted_prices)
            trend_consistency = 1 - (np.std(price_changes) / (np.mean(np.abs(price_changes)) + 1e-8))
            
            # Calculate confidence based on volatility and trend
            volatility_confidence = max(0, 1 - (price_std / current_price))
            trend_confidence = max(0, trend_consistency)
            
            # Combined confidence
            confidence = (volatility_confidence * 0.6 + trend_confidence * 0.4)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            print(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary and statistics
        """
        if not self.is_trained:
            return {'success': False, 'error': 'Model not trained'}
        
        return {
            'success': True,
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'is_trained': self.is_trained,
            'config': self.config,
            'model_params': self.model.count_params() if self.model else 0
        }
