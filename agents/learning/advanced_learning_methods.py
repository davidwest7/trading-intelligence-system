"""
Advanced Learning Methods for Enhanced Learning Agent

Implements:
- Reinforcement Learning (Q-learning)
- Meta-Learning (Learning to Learn)
- Transfer Learning (Cross-market)
- Online Learning (Real-time adaptation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

@dataclass
class QLearningState:
    """State representation for Q-learning"""
    market_regime: str  # 'bull', 'bear', 'sideways'
    volatility_level: str  # 'low', 'medium', 'high'
    trend_strength: float  # 0-1
    volume_profile: str  # 'low', 'normal', 'high'
    technical_signal: str  # 'buy', 'sell', 'hold'

@dataclass
class QLearningAction:
    """Action representation for Q-learning"""
    action_type: str  # 'buy', 'sell', 'hold'
    position_size: float  # 0-1
    stop_loss: float  # percentage
    take_profit: float  # percentage

class ReinforcementLearningAgent:
    """Q-learning agent for strategy optimization"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def get_state_key(self, state: QLearningState) -> str:
        """Convert state to string key"""
        return f"{state.market_regime}_{state.volatility_level}_{state.trend_strength:.2f}_{state.volume_profile}_{state.technical_signal}"
    
    def get_action_key(self, action: QLearningAction) -> str:
        """Convert action to string key"""
        return f"{action.action_type}_{action.position_size:.2f}_{action.stop_loss:.2f}_{action.take_profit:.2f}"
    
    def get_q_value(self, state: QLearningState, action: QLearningAction) -> float:
        """Get Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        return self.q_table.get(f"{state_key}_{action_key}", 0.0)
    
    def set_q_value(self, state: QLearningState, action: QLearningAction, value: float):
        """Set Q-value for state-action pair"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        self.q_table[f"{state_key}_{action_key}"] = value
    
    def choose_action(self, state: QLearningState, available_actions: List[QLearningAction]) -> QLearningAction:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(available_actions)
        else:
            # Exploitation: best action
            best_action = None
            best_value = float('-inf')
            
            for action in available_actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            return best_action if best_action else np.random.choice(available_actions)
    
    def update_q_value(self, state: QLearningState, action: QLearningAction, 
                      reward: float, next_state: QLearningState, 
                      next_actions: List[QLearningAction]):
        """Update Q-value using Q-learning formula"""
        current_q = self.get_q_value(state, action)
        
        # Find maximum Q-value for next state
        max_next_q = 0.0
        for next_action in next_actions:
            next_q = self.get_q_value(next_state, next_action)
            max_next_q = max(max_next_q, next_q)
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.set_q_value(state, action, new_q)
    
    def calculate_reward(self, action: QLearningAction, actual_return: float, 
                        max_drawdown: float, volatility: float) -> float:
        """Calculate reward based on trading performance"""
        # Base reward from return
        reward = actual_return * 100  # Scale up for better learning
        
        # Penalty for high drawdown
        if max_drawdown < -0.1:  # 10% drawdown
            reward -= abs(max_drawdown) * 50
        
        # Penalty for high volatility
        if volatility > 0.02:  # 2% daily volatility
            reward -= volatility * 100
        
        # Bonus for good risk-adjusted returns
        if actual_return > 0 and volatility > 0:
            sharpe_ratio = actual_return / volatility
            if sharpe_ratio > 1.0:
                reward += sharpe_ratio * 10
        
        return reward
    
    def learn_from_experience(self, state: QLearningState, action: QLearningAction, 
                            reward: float, next_state: QLearningState, 
                            next_actions: List[QLearningAction]):
        """Learn from trading experience"""
        self.update_q_value(state, action, reward, next_state, next_actions)
        
        # Store experience for meta-learning
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)

class MetaLearningAgent:
    """Meta-learning agent for learning optimal learning strategies"""
    
    def __init__(self):
        self.meta_models = {}
        self.learning_strategies = {}
        self.performance_history = {}
        
    def create_meta_model(self, strategy_name: str, base_model_type: str):
        """Create meta-model for a learning strategy"""
        if base_model_type == 'random_forest':
            self.meta_models[strategy_name] = RandomForestRegressor(n_estimators=50, random_state=42)
        elif base_model_type == 'linear':
            self.meta_models[strategy_name] = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {base_model_type}")
    
    def extract_meta_features(self, performance_history: List[Dict[str, Any]]) -> np.ndarray:
        """Extract meta-features from performance history"""
        if len(performance_history) < 10:
            return np.array([])
        
        features = []
        for i in range(len(performance_history) - 1):
            current = performance_history[i]
            next_perf = performance_history[i + 1]
            
            # Performance change features
            sharpe_change = next_perf.get('sharpe_ratio', 0) - current.get('sharpe_ratio', 0)
            drawdown_change = next_perf.get('max_drawdown', 0) - current.get('max_drawdown', 0)
            return_change = next_perf.get('total_return', 0) - current.get('total_return', 0)
            
            # Market condition features
            volatility = current.get('volatility', 0)
            trend_strength = current.get('trend_strength', 0)
            
            # Learning features
            learning_rate = current.get('learning_rate', 0.001)
            convergence_epochs = current.get('convergence_epochs', 100)
            
            features.append([
                sharpe_change, drawdown_change, return_change,
                volatility, trend_strength, learning_rate, convergence_epochs
            ])
        
        return np.array(features)
    
    def learn_optimal_strategy(self, strategy_name: str, performance_history: List[Dict[str, Any]]):
        """Learn optimal learning strategy from performance history"""
        if strategy_name not in self.meta_models:
            self.create_meta_model(strategy_name, 'random_forest')
        
        meta_features = self.extract_meta_features(performance_history)
        if len(meta_features) == 0:
            return
        
        # Create target: next period performance improvement
        targets = []
        for i in range(len(performance_history) - 1):
            current_sharpe = performance_history[i].get('sharpe_ratio', 0)
            next_sharpe = performance_history[i + 1].get('sharpe_ratio', 0)
            improvement = next_sharpe - current_sharpe
            targets.append(improvement)
        
        if len(targets) > 0:
            self.meta_models[strategy_name].fit(meta_features, targets)
    
    def predict_optimal_parameters(self, strategy_name: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal parameters for current market state"""
        if strategy_name not in self.meta_models:
            return {}
        
        # Extract current state features
        current_features = np.array([[
            current_state.get('volatility', 0),
            current_state.get('trend_strength', 0),
            current_state.get('sharpe_ratio', 0),
            current_state.get('max_drawdown', 0),
            current_state.get('learning_rate', 0.001),
            current_state.get('convergence_epochs', 100)
        ]])
        
        # Predict performance improvement
        predicted_improvement = self.meta_models[strategy_name].predict(current_features)[0]
        
        # Generate optimal parameters based on prediction
        optimal_params = {
            'learning_rate': current_state.get('learning_rate', 0.001),
            'convergence_epochs': current_state.get('convergence_epochs', 100),
            'predicted_improvement': predicted_improvement
        }
        
        # Adjust parameters based on predicted improvement
        if predicted_improvement > 0.1:  # Good improvement expected
            optimal_params['learning_rate'] *= 1.2
            optimal_params['convergence_epochs'] = min(200, optimal_params['convergence_epochs'] + 20)
        elif predicted_improvement < -0.1:  # Poor performance expected
            optimal_params['learning_rate'] *= 0.8
            optimal_params['convergence_epochs'] = max(50, optimal_params['convergence_epochs'] - 20)
        
        return optimal_params

class TransferLearningAgent:
    """Transfer learning agent for cross-market knowledge transfer"""
    
    def __init__(self):
        self.source_models = {}
        self.target_models = {}
        self.transfer_weights = {}
        self.domain_adaptation = {}
        
    def train_source_model(self, market_name: str, data: pd.DataFrame, target_col: str):
        """Train model on source market"""
        if not ML_AVAILABLE:
            return
        
        # Prepare features (excluding target)
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Train source model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        self.source_models[market_name] = {
            'model': model,
            'feature_importance': model.feature_importances_,
            'feature_names': feature_cols
        }
    
    def adapt_to_target_market(self, source_market: str, target_market: str, 
                             target_data: pd.DataFrame, target_col: str):
        """Adapt source model to target market"""
        if source_market not in self.source_models:
            return
        
        source_model_info = self.source_models[source_market]
        source_model = source_model_info['model']
        source_features = source_model_info['feature_names']
        
        # Prepare target data
        target_features = [col for col in target_data.columns if col != target_col]
        X_target = target_data[target_features].values
        y_target = target_data[target_col].values
        
        # Create target model with transferred knowledge
        target_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Transfer feature importance from source
        if len(source_features) == len(target_features):
            # Align features and transfer importance
            feature_importance = source_model_info['feature_importance']
            target_model.feature_importances_ = feature_importance
        
        # Fine-tune on target data
        target_model.fit(X_target, y_target)
        
        self.target_models[f"{source_market}_to_{target_market}"] = {
            'model': target_model,
            'source_market': source_market,
            'target_market': target_market,
            'transfer_score': self._calculate_transfer_score(source_model, target_model, X_target, y_target)
        }
    
    def _calculate_transfer_score(self, source_model, target_model, X_target, y_target) -> float:
        """Calculate transfer learning effectiveness score"""
        # Compare performance of source vs target model
        source_pred = source_model.predict(X_target)
        target_pred = target_model.predict(X_target)
        
        source_mse = mean_squared_error(y_target, source_pred)
        target_mse = mean_squared_error(y_target, target_pred)
        
        # Transfer score: improvement over source model
        if source_mse > 0:
            transfer_score = (source_mse - target_mse) / source_mse
        else:
            transfer_score = 0.0
        
        return transfer_score
    
    def get_transfer_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for transfer learning"""
        recommendations = []
        
        for transfer_key, transfer_info in self.target_models.items():
            if transfer_info['transfer_score'] > 0.1:  # Good transfer
                recommendations.append({
                    'transfer': transfer_key,
                    'source_market': transfer_info['source_market'],
                    'target_market': transfer_info['target_market'],
                    'transfer_score': transfer_info['transfer_score'],
                    'recommendation': 'Use transfer learning'
                })
            elif transfer_info['transfer_score'] < -0.1:  # Poor transfer
                recommendations.append({
                    'transfer': transfer_key,
                    'source_market': transfer_info['source_market'],
                    'target_market': transfer_info['target_market'],
                    'transfer_score': transfer_info['transfer_score'],
                    'recommendation': 'Train from scratch'
                })
        
        return recommendations

class OnlineLearningAgent:
    """Online learning agent for real-time model adaptation"""
    
    def __init__(self, base_model_type='linear', learning_rate=0.01):
        self.base_model_type = base_model_type
        self.learning_rate = learning_rate
        self.online_models = {}
        self.performance_window = 100  # Rolling window for performance tracking
        self.adaptation_threshold = 0.1  # Threshold for model adaptation
        
    def create_online_model(self, model_name: str):
        """Create online learning model"""
        if self.base_model_type == 'linear':
            self.online_models[model_name] = LinearRegression()
        elif self.base_model_type == 'random_forest':
            self.online_models[model_name] = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.base_model_type}")
    
    def update_online_model(self, model_name: str, new_data: pd.DataFrame, 
                          target_col: str, performance_metric: float):
        """Update online model with new data"""
        if model_name not in self.online_models:
            self.create_online_model(model_name)
        
        model = self.online_models[model_name]
        
        # Prepare new data
        feature_cols = [col for col in new_data.columns if col != target_col]
        X_new = new_data[feature_cols].values
        y_new = new_data[target_col].values
        
        # Check if adaptation is needed
        if self._should_adapt(model_name, performance_metric):
            # Retrain model with new data
            if self.base_model_type == 'linear':
                # For linear models, we can do incremental updates
                if hasattr(model, 'coef_') and model.coef_ is not None:
                    # Incremental update
                    self._incremental_update(model, X_new, y_new)
                else:
                    # Initial training
                    model.fit(X_new, y_new)
            else:
                # For non-linear models, retrain
                model.fit(X_new, y_new)
            
            print(f"🔄 Online model {model_name} adapted (performance: {performance_metric:.3f})")
    
    def _should_adapt(self, model_name: str, performance_metric: float) -> bool:
        """Determine if model should be adapted based on performance"""
        # Simple threshold-based adaptation
        return performance_metric < self.adaptation_threshold
    
    def _incremental_update(self, model, X_new: np.ndarray, y_new: np.ndarray):
        """Incremental update for linear models"""
        # Simple stochastic gradient descent update
        if len(X_new) > 0:
            # Update coefficients incrementally
            for i in range(len(X_new)):
                x = X_new[i].reshape(1, -1)
                y = y_new[i]
                
                # Predict current output
                y_pred = model.predict(x)[0]
                
                # Calculate error
                error = y - y_pred
                
                # Update coefficients (simplified)
                if hasattr(model, 'coef_') and model.coef_ is not None:
                    model.coef_ += self.learning_rate * error * x.flatten()
                    if hasattr(model, 'intercept_'):
                        model.intercept_ += self.learning_rate * error
    
    def predict_online(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using online model"""
        if model_name in self.online_models:
            return self.online_models[model_name].predict(X)
        else:
            return np.zeros(len(X))
    
    def get_online_performance(self, model_name: str) -> Dict[str, Any]:
        """Get online learning performance metrics"""
        if model_name not in self.online_models:
            return {}
        
        model = self.online_models[model_name]
        
        performance = {
            'model_name': model_name,
            'model_type': self.base_model_type,
            'learning_rate': self.learning_rate,
            'adaptation_threshold': self.adaptation_threshold
        }
        
        # Add model-specific metrics
        if hasattr(model, 'coef_') and model.coef_ is not None:
            performance['feature_importance'] = model.coef_.tolist()
        
        return performance

class AdvancedLearningOrchestrator:
    """Orchestrates all advanced learning methods"""
    
    def __init__(self):
        self.rl_agent = ReinforcementLearningAgent()
        self.meta_agent = MetaLearningAgent()
        self.transfer_agent = TransferLearningAgent()
        self.online_agent = OnlineLearningAgent()
        
    def optimize_strategy(self, market_data: pd.DataFrame, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize trading strategy using all advanced learning methods"""
        results = {
            'reinforcement_learning': {},
            'meta_learning': {},
            'transfer_learning': {},
            'online_learning': {},
            'recommendations': []
        }
        
        # 1. Reinforcement Learning Optimization
        if len(performance_history) > 10:
            rl_recommendations = self._apply_reinforcement_learning(market_data, performance_history)
            results['reinforcement_learning'] = rl_recommendations
        
        # 2. Meta-Learning Optimization
        if len(performance_history) > 20:
            meta_recommendations = self._apply_meta_learning(performance_history)
            results['meta_learning'] = meta_recommendations
        
        # 3. Transfer Learning Analysis
        transfer_recommendations = self._apply_transfer_learning(market_data)
        results['transfer_learning'] = transfer_recommendations
        
        # 4. Online Learning Adaptation
        online_recommendations = self._apply_online_learning(market_data)
        results['online_learning'] = online_recommendations
        
        # 5. Generate combined recommendations
        results['recommendations'] = self._generate_combined_recommendations(results)
        
        return results
    
    def _apply_reinforcement_learning(self, market_data: pd.DataFrame, 
                                    performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply reinforcement learning for strategy optimization"""
        # Create market state from recent data
        recent_data = market_data.tail(20)
        
        # Determine market regime
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std()
        trend = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        
        if trend > 0.05:
            market_regime = 'bull'
        elif trend < -0.05:
            market_regime = 'bear'
        else:
            market_regime = 'sideways'
        
        # Create state
        state = QLearningState(
            market_regime=market_regime,
            volatility_level='high' if volatility > 0.02 else 'low',
            trend_strength=abs(trend),
            volume_profile='normal',
            technical_signal='hold'
        )
        
        # Create available actions
        actions = [
            QLearningAction('buy', 0.5, 0.02, 0.05),
            QLearningAction('sell', 0.5, 0.02, 0.05),
            QLearningAction('hold', 0.0, 0.0, 0.0)
        ]
        
        # Choose action
        action = self.rl_agent.choose_action(state, actions)
        
        return {
            'recommended_action': action.action_type,
            'position_size': action.position_size,
            'stop_loss': action.stop_loss,
            'take_profit': action.take_profit,
            'market_regime': market_regime,
            'volatility': volatility,
            'trend_strength': trend
        }
    
    def _apply_meta_learning(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply meta-learning for optimal learning strategy"""
        strategy_name = 'trading_strategy_1'
        
        # Learn optimal strategy
        self.meta_agent.learn_optimal_strategy(strategy_name, performance_history)
        
        # Get current state
        if len(performance_history) > 0:
            current_state = performance_history[-1]
        else:
            current_state = {}
        
        # Predict optimal parameters
        optimal_params = self.meta_agent.predict_optimal_parameters(strategy_name, current_state)
        
        return {
            'strategy_name': strategy_name,
            'optimal_parameters': optimal_params,
            'predicted_improvement': optimal_params.get('predicted_improvement', 0)
        }
    
    def _apply_transfer_learning(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply transfer learning for cross-market knowledge"""
        # This would typically involve multiple markets
        # For now, we'll simulate with different time periods
        
        # Split data into source and target periods
        split_point = len(market_data) // 2
        source_data = market_data.iloc[:split_point]
        target_data = market_data.iloc[split_point:]
        
        if len(source_data) > 50 and len(target_data) > 50:
            # Train source model
            self.transfer_agent.train_source_model('source_period', source_data, 'target')
            
            # Adapt to target period
            self.transfer_agent.adapt_to_target_market('source_period', 'target_period', target_data, 'target')
            
            # Get transfer recommendations
            recommendations = self.transfer_agent.get_transfer_recommendations()
            
            return {
                'transfer_recommendations': recommendations,
                'source_period_length': len(source_data),
                'target_period_length': len(target_data)
            }
        
        return {'transfer_recommendations': [], 'error': 'Insufficient data'}
    
    def _apply_online_learning(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply online learning for real-time adaptation"""
        model_name = 'online_trading_model'
        
        # Create online model if it doesn't exist
        if model_name not in self.online_agent.online_models:
            self.online_agent.create_online_model(model_name)
        
        # Use recent data for online update
        recent_data = market_data.tail(20)
        
        if len(recent_data) > 10:
            # Calculate performance metric (simplified)
            returns = recent_data['close'].pct_change().dropna()
            performance_metric = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Update online model
            self.online_agent.update_online_model(model_name, recent_data, 'target', performance_metric)
            
            # Get performance metrics
            performance = self.online_agent.get_online_performance(model_name)
            
            return {
                'model_name': model_name,
                'performance_metric': performance_metric,
                'online_performance': performance,
                'data_points_used': len(recent_data)
            }
        
        return {'error': 'Insufficient data for online learning'}
    
    def _generate_combined_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate combined recommendations from all learning methods"""
        recommendations = []
        
        # RL recommendations
        rl_results = results.get('reinforcement_learning', {})
        if rl_results:
            action = rl_results.get('recommended_action', 'hold')
            recommendations.append(f"🤖 RL recommends: {action.upper()} action")
        
        # Meta-learning recommendations
        meta_results = results.get('meta_learning', {})
        if meta_results:
            improvement = meta_results.get('predicted_improvement', 0)
            if improvement > 0.05:
                recommendations.append(f"🧠 Meta-learning predicts {improvement:.3f} improvement")
        
        # Transfer learning recommendations
        transfer_results = results.get('transfer_learning', {})
        if transfer_results:
            transfer_recs = transfer_results.get('transfer_recommendations', [])
            if transfer_recs:
                recommendations.append(f"🔄 Transfer learning: {len(transfer_recs)} recommendations")
        
        # Online learning recommendations
        online_results = results.get('online_learning', {})
        if online_results and 'performance_metric' in online_results:
            perf = online_results['performance_metric']
            if perf > 0.5:
                recommendations.append(f"📈 Online learning: Strong performance ({perf:.3f})")
        
        return recommendations
