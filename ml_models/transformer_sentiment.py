"""
Transformer-based Sentiment Analysis for Financial Markets
Enhanced with comprehensive TensorFlow mutex fixes
"""

import numpy as np
import pandas as pd
import os
import sys
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import re
import json

# ============================================================================
# CRITICAL TENSORFLOW MUTEX FIXES - MUST BE BEFORE TF IMPORT
# ============================================================================

# Set all critical environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to prevent conflicts
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'  # Disable deprecation warnings
os.environ['TF_LOGGING_LEVEL'] = 'ERROR'  # Error-level logging only
os.environ['TF_PROFILER_DISABLE'] = '1'  # Disable profiling
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU growth
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Single inter-op thread
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Single intra-op thread

# Now import TensorFlow safely
import tensorflow as tf

# Configure TensorFlow threading for mutex safety
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure session for mutex safety
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.device_count['GPU'] = 0  # Force CPU-only
    config.log_device_placement = False
    
    # Set session options
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=config)
    )
except Exception as e:
    print(f"⚠️ Warning: Could not configure TensorFlow session: {e}")

# ============================================================================
# THREAD-SAFE TENSORFLOW WRAPPER
# ============================================================================

class ThreadSafeTensorFlow:
    """Thread-safe wrapper for TensorFlow operations."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._model_locks = {}
        
    def safe_model_predict(self, model, data, timeout=30):
        """Thread-safe model prediction with timeout."""
        model_id = id(model)
        if model_id not in self._model_locks:
            self._model_locks[model_id] = threading.Lock()
        
        with self._model_locks[model_id]:
            try:
                start_time = time.time()
                prediction = model.predict(data, verbose=0)
                
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Model prediction exceeded {timeout}s timeout")
                
                return prediction
            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                return None
    
    def safe_model_train(self, model, x, y, **kwargs):
        """Thread-safe model training."""
        model_id = id(model)
        if model_id not in self._model_locks:
            self._model_locks[model_id] = threading.Lock()
        
        with self._model_locks[model_id]:
            try:
                # Add timeout callback
                callbacks = kwargs.get('callbacks', [])
                callbacks.append(TimeoutCallback(timeout=300))  # 5 minute timeout
                kwargs['callbacks'] = callbacks
                
                history = model.fit(x, y, **kwargs)
                return history
            except Exception as e:
                print(f"❌ Model training error: {e}")
                return None

# ============================================================================
# TIMEOUT CALLBACK
# ============================================================================

class TimeoutCallback:
    """Callback to prevent TensorFlow operations from hanging."""
    
    def __init__(self, timeout=300):
        self.timeout = timeout
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if self.start_time and (time.time() - self.start_time) > self.timeout:
            raise TimeoutError(f"Training exceeded {self.timeout}s timeout")

# ============================================================================
# SKLEARN FALLBACK
# ============================================================================

class SafeMLFallback:
    """Safe ML fallback using sklearn when TensorFlow fails."""
    
    def __init__(self):
        self.models = {}
        
    def create_sklearn_model(self, model_type='random_forest'):
        """Create sklearn-based model as TensorFlow alternative."""
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42)
            elif model_type == 'svm':
                from sklearn.svm import SVC
                return SVC(random_state=42)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except ImportError:
            print("❌ sklearn not available")
            return None
    
    def train_sklearn_model(self, model, X, y):
        """Train sklearn model safely."""
        try:
            model.fit(X, y)
            return True
        except Exception as e:
            print(f"❌ sklearn training error: {e}")
            return False
    
    def predict_sklearn_model(self, model, X):
        """Predict with sklearn model safely."""
        try:
            return model.predict(X)
        except Exception as e:
            print(f"❌ sklearn prediction error: {e}")
            return None

# Initialize thread-safe TensorFlow wrapper
safe_tf = ThreadSafeTensorFlow()
fallback = SafeMLFallback()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from datetime import datetime, timedelta
import re
import json

class TransformerSentimentAnalyzer:
    """
    Advanced Transformer model for market sentiment analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'max_sequence_length': 512,
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 4,
            'ff_dim': 256,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'vocab_size': 10000
        }
        
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        
    def _create_transformer_block(self, inputs: tf.Tensor, num_heads: int, 
                                ff_dim: int, dropout_rate: float) -> tf.Tensor:
        """
        Create a transformer block
        """
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=self.config['embedding_dim']
        )(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = Dense(self.config['embedding_dim'])(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        return ffn_output
    
    def _build_model(self, num_classes: int) -> Model:
        """
        Build transformer model architecture
        """
        inputs = Input(shape=(self.config['max_sequence_length'],))
        
        # Embedding layer (simplified for demo)
        embedding = Dense(self.config['embedding_dim'], activation='relu')(inputs)
        
        # Transformer blocks
        transformer_output = embedding
        for _ in range(self.config['num_layers']):
            transformer_output = self._create_transformer_block(
                transformer_output,
                self.config['num_heads'],
                self.config['ff_dim'],
                self.config['dropout_rate']
            )
        
        # Global average pooling
        pooled_output = GlobalAveragePooling1D()(transformer_output)
        
        # Classification head
        dense_output = Dense(128, activation='relu')(pooled_output)
        dense_output = Dropout(self.config['dropout_rate'])(dense_output)
        outputs = Dense(num_classes, activation='softmax')(dense_output)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\$\%]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _create_simple_tokenizer(self, texts: List[str]) -> Dict[str, int]:
        """
        Create a simple tokenizer for demo purposes
        """
        vocab = {}
        word_count = {}
        
        # Count word frequencies
        for text in texts:
            words = text.split()
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # Create vocabulary (top words)
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for i, (word, _) in enumerate(sorted_words[:self.config['vocab_size'] - 2]):
            vocab[word] = i + 2
        
        return vocab
    
    def _text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of token IDs
        """
        if self.tokenizer is None:
            return []
        
        words = text.split()
        sequence = []
        
        for word in words:
            if word in self.tokenizer:
                sequence.append(self.tokenizer[word])
            else:
                sequence.append(self.tokenizer['<UNK>'])
        
        # Pad or truncate to max_sequence_length
        if len(sequence) < self.config['max_sequence_length']:
            sequence.extend([self.tokenizer['<PAD>']] * (self.config['max_sequence_length'] - len(sequence)))
        else:
            sequence = sequence[:self.config['max_sequence_length']]
        
        return sequence
    
    async def train_model(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Train transformer model on sentiment data
        """
        try:
            print(f"🔬 Training Transformer sentiment model")
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Create tokenizer
            self.tokenizer = self._create_simple_tokenizer(processed_texts)
            
            # Convert texts to sequences
            sequences = [self._text_to_sequence(text) for text in processed_texts]
            X = np.array(sequences)
            
            # Encode labels
            y = self.label_encoder.fit_transform(labels)
            num_classes = len(self.label_encoder.classes_)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            self.model = self._build_model(num_classes)
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6)
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
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            self.is_trained = True
            
            return {
                'success': True,
                'training_time': training_time,
                'accuracy': accuracy,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'vocab_size': len(self.tokenizer),
                'num_classes': num_classes
            }
            
        except Exception as e:
            print(f"Error training Transformer sentiment model: {e}")
            return {'success': False, 'error': str(e)}
    
    async def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of input texts
        """
        try:
            if not self.is_trained or self.model is None:
                return {'success': False, 'error': 'Model not trained'}
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Convert to sequences
            sequences = [self._text_to_sequence(text) for text in processed_texts]
            X = np.array(sequences)
            
            # Make predictions
            predictions = self.model.predict(X)
            predicted_classes = np.argmax(predictions, axis=1)
            confidence_scores = np.max(predictions, axis=1)
            
            # Convert back to labels
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            
            # Calculate sentiment scores
            sentiment_scores = []
            for i, label in enumerate(predicted_labels):
                if label == 'positive':
                    sentiment_scores.append(confidence_scores[i])
                elif label == 'negative':
                    sentiment_scores.append(-confidence_scores[i])
                else:  # neutral
                    sentiment_scores.append(0.0)
            
            return {
                'success': True,
                'texts': texts,
                'sentiments': predicted_labels.tolist(),
                'sentiment_scores': sentiment_scores,
                'confidence_scores': confidence_scores.tolist(),
                'average_sentiment': np.mean(sentiment_scores),
                'sentiment_distribution': {
                    'positive': sum(1 for s in predicted_labels if s == 'positive'),
                    'negative': sum(1 for s in predicted_labels if s == 'negative'),
                    'neutral': sum(1 for s in predicted_labels if s == 'neutral')
                }
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary and statistics
        """
        if not self.is_trained:
            return {'success': False, 'error': 'Model not trained'}
        
        return {
            'success': True,
            'is_trained': self.is_trained,
            'config': self.config,
            'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
            'num_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0,
            'model_params': self.model.count_params() if self.model else 0
        }
