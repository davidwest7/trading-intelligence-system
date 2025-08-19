"""
Autonomous Code Generation System

Implements:
- Genetic Programming
- Neural Architecture Search
- Hyperparameter Optimization
- Feature Selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import random
import json
import ast
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

@dataclass
class CodeIndividual:
    """Individual in genetic programming population"""
    code: str
    fitness: float
    generation: int
    mutation_count: int
    crossover_count: int
    performance_metrics: Dict[str, float]

@dataclass
class NeuralArchitecture:
    """Neural network architecture specification"""
    architecture_id: str
    layers: List[Dict[str, Any]]
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    performance_score: float
    complexity_score: float

@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration"""
    config_id: str
    parameters: Dict[str, Any]
    performance_score: float
    validation_score: float
    training_time: float
    convergence_epochs: int

@dataclass
class FeatureSet:
    """Feature set for model training"""
    feature_set_id: str
    selected_features: List[str]
    feature_importance: Dict[str, float]
    performance_score: float
    feature_count: int
    selection_method: str

class GeneticProgramming:
    """Genetic programming for trading strategy evolution"""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_individual = None
        self.generation_history = []
        
        # Trading strategy components
        self.indicators = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'sma_20', 'sma_50']
        self.operators = ['+', '-', '*', '/', '>', '<', '>=', '<=', '==', 'and', 'or']
        self.functions = ['abs', 'max', 'min', 'log', 'exp', 'sqrt']
        
    def initialize_population(self):
        """Initialize random population of trading strategies"""
        self.population = []
        
        for i in range(self.population_size):
            code = self._generate_random_strategy()
            individual = CodeIndividual(
                code=code,
                fitness=0.0,
                generation=0,
                mutation_count=0,
                crossover_count=0,
                performance_metrics={}
            )
            self.population.append(individual)
    
    def _generate_random_strategy(self) -> str:
        """Generate random trading strategy code"""
        strategy_templates = [
            "if {condition}: return 'buy' else: return 'sell'",
            "signal = 'buy' if {condition} else 'sell'",
            "return 'buy' if {condition} and {condition2} else 'sell'",
            "if {condition}: signal = 'buy' elif {condition2}: signal = 'sell' else: signal = 'hold'"
        ]
        
        template = random.choice(strategy_templates)
        
        # Generate random conditions
        conditions = []
        for _ in range(template.count('{condition}')):
            condition = self._generate_random_condition()
            conditions.append(condition)
        
        # Fill template
        for i, condition in enumerate(conditions):
            template = template.replace(f'{{condition}}', condition, 1)
            if i < len(conditions) - 1:
                template = template.replace(f'{{condition2}}', conditions[i+1], 1)
        
        return template
    
    def _generate_random_condition(self) -> str:
        """Generate random trading condition"""
        indicator1 = random.choice(self.indicators)
        indicator2 = random.choice(self.indicators)
        operator = random.choice(self.operators[:6])  # Use comparison operators
        
        if random.random() < 0.5:
            # Simple comparison
            threshold = random.uniform(0, 100)
            return f"{indicator1} {operator} {threshold}"
        else:
            # Complex condition
            return f"{indicator1} {operator} {indicator2}"
    
    def evaluate_fitness(self, individual: CodeIndividual, historical_data: pd.DataFrame) -> float:
        """Evaluate fitness of trading strategy"""
        try:
            # Execute strategy on historical data
            returns = self._execute_strategy(individual.code, historical_data)
            
            if returns is None or len(returns) == 0:
                return 0.0
            
            # Calculate fitness metrics
            total_return = (1 + returns).prod() - 1
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = (returns > 0).mean()
            
            # Composite fitness score
            fitness = (
                total_return * 0.4 +
                sharpe_ratio * 0.3 +
                (1 + max_drawdown) * 0.2 +  # Penalize drawdown
                win_rate * 0.1
            )
            
            # Store performance metrics
            individual.performance_metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
            return max(0, fitness)  # Ensure non-negative fitness
            
        except Exception as e:
            print(f"❌ Error evaluating fitness: {e}")
            return 0.0
    
    def _execute_strategy(self, code: str, data: pd.DataFrame) -> pd.Series:
        """Execute trading strategy on historical data"""
        try:
            # Create a safe execution environment
            local_vars = {
                'rsi': data.get('rsi', pd.Series(50, index=data.index)),
                'macd': data.get('macd', pd.Series(0, index=data.index)),
                'bb_position': data.get('bb_position', pd.Series(0.5, index=data.index)),
                'volume_ratio': data.get('volume_ratio', pd.Series(1.0, index=data.index)),
                'sma_20': data.get('sma_20', data['close']),
                'sma_50': data.get('sma_50', data['close']),
                'close': data['close'],
                'volume': data['volume']
            }
            
            # Execute strategy for each data point
            signals = []
            for i in range(len(data)):
                # Update local variables for current point
                for key, value in local_vars.items():
                    if isinstance(value, pd.Series):
                        local_vars[key] = value.iloc[i]
                
                try:
                    # Execute strategy code
                    exec(code, {}, local_vars)
                    signal = local_vars.get('signal', 'hold')
                    signals.append(signal)
                except:
                    signals.append('hold')
            
            # Convert signals to returns
            returns = []
            for i in range(1, len(signals)):
                if signals[i-1] == 'buy':
                    returns.append((data['close'].iloc[i] / data['close'].iloc[i-1]) - 1)
                elif signals[i-1] == 'sell':
                    returns.append((data['close'].iloc[i-1] / data['close'].iloc[i]) - 1)
                else:
                    returns.append(0.0)
            
            return pd.Series(returns)
            
        except Exception as e:
            print(f"❌ Error executing strategy: {e}")
            return pd.Series()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def evolve_population(self, historical_data: pd.DataFrame):
        """Evolve population using genetic operators"""
        for generation in range(self.generations):
            # Evaluate fitness
            for individual in self.population:
                individual.fitness = self.evaluate_fitness(individual, historical_data)
                individual.generation = generation
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best individual
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0]
            
            # Store generation statistics
            self.generation_history.append({
                'generation': generation,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'best_code': self.population[0].code
            })
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = int(0.1 * self.population_size)
            new_population.extend(self.population[:elite_size])
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = self._select_parent()
                    parent2 = self._select_parent()
                    child = self._crossover(parent1, parent2)
                else:
                    # Mutation
                    parent = self._select_parent()
                    child = self._mutate(parent)
                
                new_population.append(child)
            
            self.population = new_population[:self.population_size]
    
    def _select_parent(self) -> CodeIndividual:
        """Select parent using tournament selection"""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: CodeIndividual, parent2: CodeIndividual) -> CodeIndividual:
        """Perform crossover between two parents"""
        # Simple single-point crossover
        code1_lines = parent1.code.split('\n')
        code2_lines = parent2.code.split('\n')
        
        if len(code1_lines) > 1 and len(code2_lines) > 1:
            crossover_point = random.randint(1, min(len(code1_lines), len(code2_lines)) - 1)
            child_code = '\n'.join(code1_lines[:crossover_point] + code2_lines[crossover_point:])
        else:
            child_code = parent1.code if random.random() < 0.5 else parent2.code
        
        return CodeIndividual(
            code=child_code,
            fitness=0.0,
            generation=parent1.generation + 1,
            mutation_count=0,
            crossover_count=parent1.crossover_count + parent2.crossover_count + 1,
            performance_metrics={}
        )
    
    def _mutate(self, parent: CodeIndividual) -> CodeIndividual:
        """Perform mutation on parent"""
        code = parent.code
        
        # Random mutations
        if random.random() < self.mutation_rate:
            # Change operator
            for operator in self.operators:
                if operator in code:
                    new_operator = random.choice(self.operators)
                    code = code.replace(operator, new_operator, 1)
                    break
        
        if random.random() < self.mutation_rate:
            # Change threshold
            import re
            numbers = re.findall(r'\d+\.?\d*', code)
            if numbers:
                old_number = random.choice(numbers)
                new_number = str(random.uniform(0, 100))
                code = code.replace(old_number, new_number, 1)
        
        return CodeIndividual(
            code=code,
            fitness=0.0,
            generation=parent.generation + 1,
            mutation_count=parent.mutation_count + 1,
            crossover_count=parent.crossover_count,
            performance_metrics={}
        )
    
    def get_best_strategy(self) -> Optional[CodeIndividual]:
        """Get the best evolved strategy"""
        return self.best_individual

class NeuralArchitectureSearch:
    """Neural architecture search for optimal network design"""
    
    def __init__(self, max_layers=5, max_neurons=256):
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.architectures = []
        self.best_architecture = None
        
    def search_architectures(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray, 
                           n_trials=20) -> List[NeuralArchitecture]:
        """Search for optimal neural network architectures"""
        if not DL_AVAILABLE:
            return []
        
        self.architectures = []
        
        for trial in range(n_trials):
            architecture = self._generate_random_architecture()
            performance = self._evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
            
            architecture.performance_score = performance
            self.architectures.append(architecture)
            
            # Update best architecture
            if self.best_architecture is None or performance > self.best_architecture.performance_score:
                self.best_architecture = architecture
        
        # Sort by performance
        self.architectures.sort(key=lambda x: x.performance_score, reverse=True)
        
        return self.architectures
    
    def _generate_random_architecture(self) -> NeuralArchitecture:
        """Generate random neural network architecture"""
        n_layers = random.randint(1, self.max_layers)
        layers = []
        
        for i in range(n_layers):
            layer_type = random.choice(['dense', 'lstm', 'dropout'])
            
            if layer_type == 'dense':
                neurons = random.choice([32, 64, 128, 256])
                activation = random.choice(['relu', 'tanh', 'sigmoid'])
                layers.append({
                    'type': 'dense',
                    'neurons': neurons,
                    'activation': activation
                })
            elif layer_type == 'lstm':
                neurons = random.choice([32, 64, 128])
                return_sequences = i < n_layers - 1  # Last LSTM layer doesn't return sequences
                layers.append({
                    'type': 'lstm',
                    'neurons': neurons,
                    'return_sequences': return_sequences
                })
            elif layer_type == 'dropout':
                rate = random.choice([0.1, 0.2, 0.3, 0.5])
                layers.append({
                    'type': 'dropout',
                    'rate': rate
                })
        
        # Add output layer
        layers.append({
            'type': 'dense',
            'neurons': 1,
            'activation': 'linear'
        })
        
        optimizer = random.choice(['adam', 'sgd', 'rmsprop'])
        learning_rate = random.choice([0.001, 0.01, 0.1])
        batch_size = random.choice([16, 32, 64, 128])
        epochs = random.choice([50, 100, 200])
        
        return NeuralArchitecture(
            architecture_id=f"arch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trial}",
            layers=layers,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            performance_score=0.0,
            complexity_score=self._calculate_complexity(layers)
        )
    
    def _evaluate_architecture(self, architecture: NeuralArchitecture, 
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate neural network architecture"""
        try:
            # Build model
            model = self._build_model(architecture, X_train.shape[1])
            
            # Compile model
            if architecture.optimizer == 'adam':
                optimizer = Adam(learning_rate=architecture.learning_rate)
            elif architecture.optimizer == 'sgd':
                optimizer = SGD(learning_rate=architecture.learning_rate)
            else:
                optimizer = RMSprop(learning_rate=architecture.learning_rate)
            
            model.compile(optimizer=optimizer, loss='mse')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=architecture.batch_size,
                epochs=architecture.epochs,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Evaluate performance
            y_pred = model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, y_pred)
            
            # Performance score (inverse of MSE)
            performance = 1 / (1 + mse)
            
            return performance
            
        except Exception as e:
            print(f"❌ Error evaluating architecture: {e}")
            return 0.0
    
    def _build_model(self, architecture: NeuralArchitecture, input_dim: int) -> Sequential:
        """Build neural network model from architecture"""
        model = Sequential()
        
        for i, layer in enumerate(architecture.layers):
            if layer['type'] == 'dense':
                if i == 0:
                    model.add(Dense(layer['neurons'], activation=layer['activation'], 
                                  input_shape=(input_dim,)))
                else:
                    model.add(Dense(layer['neurons'], activation=layer['activation']))
            
            elif layer['type'] == 'lstm':
                if i == 0:
                    # Reshape input for LSTM
                    model.add(Dense(layer['neurons'], input_shape=(input_dim,)))
                else:
                    model.add(LSTM(layer['neurons'], return_sequences=layer['return_sequences']))
            
            elif layer['type'] == 'dropout':
                model.add(Dropout(layer['rate']))
        
        return model
    
    def _calculate_complexity(self, layers: List[Dict[str, Any]]) -> float:
        """Calculate architecture complexity score"""
        complexity = 0.0
        
        for layer in layers:
            if layer['type'] == 'dense':
                complexity += layer['neurons']
            elif layer['type'] == 'lstm':
                complexity += layer['neurons'] * 2  # LSTM is more complex
            elif layer['type'] == 'dropout':
                complexity += 1
        
        return complexity
    
    def get_best_architecture(self) -> Optional[NeuralArchitecture]:
        """Get the best neural architecture"""
        return self.best_architecture

class HyperparameterOptimizer:
    """Hyperparameter optimization using Bayesian optimization"""
    
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.configs = []
        self.best_config = None
        
    def optimize_hyperparameters(self, model_type: str, X_train: np.ndarray, 
                               y_train: np.ndarray, X_val: np.ndarray, 
                               y_val: np.ndarray) -> List[HyperparameterConfig]:
        """Optimize hyperparameters for given model type"""
        self.configs = []
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            param_grid = {
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [16, 32, 64, 128],
                'epochs': [50, 100, 200]
            }
        
        # Use randomized search
        if ML_AVAILABLE:
            if model_type == 'random_forest':
                base_model = RandomForestRegressor(random_state=42)
            else:
                base_model = RandomForestRegressor(random_state=42)  # Default
            
            random_search = RandomizedSearchCV(
                base_model, param_grid, n_iter=self.n_trials,
                cv=3, scoring='neg_mean_squared_error', random_state=42
            )
            
            random_search.fit(X_train, y_train)
            
            # Store results
            for i, (params, score) in enumerate(zip(random_search.cv_results_['params'], 
                                                   random_search.cv_results_['mean_test_score'])):
                config = HyperparameterConfig(
                    config_id=f"config_{model_type}_{i}",
                    parameters=params,
                    performance_score=-score,  # Convert back to positive
                    validation_score=-score,
                    training_time=0.0,
                    convergence_epochs=0
                )
                self.configs.append(config)
        
        # Sort by performance
        self.configs.sort(key=lambda x: x.performance_score, reverse=True)
        
        if self.configs:
            self.best_config = self.configs[0]
        
        return self.configs
    
    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get the best hyperparameter configuration"""
        return self.best_config

class FeatureSelector:
    """Automated feature selection"""
    
    def __init__(self):
        self.feature_sets = []
        self.best_feature_set = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       methods=['correlation', 'mutual_info', 'lasso', 'random_forest']) -> List[FeatureSet]:
        """Select optimal feature sets using multiple methods"""
        if not ML_AVAILABLE:
            return []
        
        self.feature_sets = []
        
        for method in methods:
            feature_set = self._apply_selection_method(X, y, method)
            if feature_set:
                self.feature_sets.append(feature_set)
        
        # Sort by performance
        self.feature_sets.sort(key=lambda x: x.performance_score, reverse=True)
        
        if self.feature_sets:
            self.best_feature_set = self.feature_sets[0]
        
        return self.feature_sets
    
    def _apply_selection_method(self, X: pd.DataFrame, y: pd.Series, method: str) -> Optional[FeatureSet]:
        """Apply specific feature selection method"""
        try:
            if method == 'correlation':
                return self._correlation_selection(X, y)
            elif method == 'mutual_info':
                return self._mutual_info_selection(X, y)
            elif method == 'lasso':
                return self._lasso_selection(X, y)
            elif method == 'random_forest':
                return self._random_forest_selection(X, y)
            else:
                return None
        except Exception as e:
            print(f"❌ Error in {method} selection: {e}")
            return None
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series) -> FeatureSet:
        """Feature selection based on correlation"""
        correlations = X.corrwith(y).abs()
        selected_features = correlations[correlations > 0.1].index.tolist()
        
        if len(selected_features) == 0:
            selected_features = correlations.nlargest(5).index.tolist()
        
        # Evaluate performance
        performance = self._evaluate_feature_set(X[selected_features], y)
        
        return FeatureSet(
            feature_set_id=f"correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            selected_features=selected_features,
            feature_importance=correlations[selected_features].to_dict(),
            performance_score=performance,
            feature_count=len(selected_features),
            selection_method='correlation'
        )
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> FeatureSet:
        """Feature selection based on mutual information"""
        selector = SelectKBest(score_func=f_regression, k=min(10, X.shape[1]))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        feature_importance = dict(zip(X.columns, selector.scores_))
        
        performance = self._evaluate_feature_set(X[selected_features], y)
        
        return FeatureSet(
            feature_set_id=f"mutual_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            selected_features=selected_features,
            feature_importance=feature_importance,
            performance_score=performance,
            feature_count=len(selected_features),
            selection_method='mutual_info'
        )
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> FeatureSet:
        """Feature selection using Lasso regularization"""
        from sklearn.linear_model import LassoCV
        
        lasso = LassoCV(cv=3, random_state=42)
        lasso.fit(X, y)
        
        selected_features = X.columns[lasso.coef_ != 0].tolist()
        feature_importance = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        performance = self._evaluate_feature_set(X[selected_features], y)
        
        return FeatureSet(
            feature_set_id=f"lasso_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            selected_features=selected_features,
            feature_importance=feature_importance,
            performance_score=performance,
            feature_count=len(selected_features),
            selection_method='lasso'
        )
    
    def _random_forest_selection(self, X: pd.DataFrame, y: pd.Series) -> FeatureSet:
        """Feature selection using Random Forest importance"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Select features with importance > mean
        importance_threshold = rf.feature_importances_.mean()
        selected_features = X.columns[rf.feature_importances_ > importance_threshold].tolist()
        
        if len(selected_features) == 0:
            selected_features = X.columns[rf.feature_importances_ > 0.01].tolist()
        
        feature_importance = dict(zip(X.columns, rf.feature_importances_))
        
        performance = self._evaluate_feature_set(X[selected_features], y)
        
        return FeatureSet(
            feature_set_id=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            selected_features=selected_features,
            feature_importance=feature_importance,
            performance_score=performance,
            feature_count=len(selected_features),
            selection_method='random_forest'
        )
    
    def _evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate feature set performance"""
        try:
            # Use simple linear regression for evaluation
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            
            model = LinearRegression()
            scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            return scores.mean()
        except:
            return 0.0
    
    def get_best_feature_set(self) -> Optional[FeatureSet]:
        """Get the best feature set"""
        return self.best_feature_set

class AutonomousCodeGenerator:
    """Orchestrates all autonomous code generation methods"""
    
    def __init__(self):
        self.genetic_programming = GeneticProgramming()
        self.neural_architecture_search = NeuralArchitectureSearch()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.feature_selector = FeatureSelector()
        
    def generate_optimized_code(self, historical_data: pd.DataFrame, 
                              target_col: str = 'target') -> Dict[str, Any]:
        """Generate optimized code using all methods"""
        results = {
            'genetic_programming': {},
            'neural_architecture_search': {},
            'hyperparameter_optimization': {},
            'feature_selection': {},
            'recommendations': []
        }
        
        # Prepare data
        feature_cols = [col for col in historical_data.columns if col != target_col]
        X = historical_data[feature_cols]
        y = historical_data[target_col]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 1. Genetic Programming
        print("🧬 Running Genetic Programming...")
        self.genetic_programming.initialize_population()
        self.genetic_programming.evolve_population(historical_data)
        best_strategy = self.genetic_programming.get_best_strategy()
        
        if best_strategy:
            results['genetic_programming'] = {
                'best_code': best_strategy.code,
                'fitness': best_strategy.fitness,
                'performance_metrics': best_strategy.performance_metrics,
                'generation_history': self.genetic_programming.generation_history
            }
        
        # 2. Neural Architecture Search
        print("🧠 Running Neural Architecture Search...")
        if DL_AVAILABLE:
            architectures = self.neural_architecture_search.search_architectures(
                X_train.values, y_train.values, X_val.values, y_val.values
            )
            best_architecture = self.neural_architecture_search.get_best_architecture()
            
            if best_architecture:
                results['neural_architecture_search'] = {
                    'best_architecture': best_architecture,
                    'all_architectures': architectures
                }
        
        # 3. Hyperparameter Optimization
        print("⚙️ Running Hyperparameter Optimization...")
        configs = self.hyperparameter_optimizer.optimize_hyperparameters(
            'random_forest', X_train.values, y_train.values, X_val.values, y_val.values
        )
        best_config = self.hyperparameter_optimizer.get_best_config()
        
        if best_config:
            results['hyperparameter_optimization'] = {
                'best_config': best_config,
                'all_configs': configs
            }
        
        # 4. Feature Selection
        print("🔍 Running Feature Selection...")
        feature_sets = self.feature_selector.select_features(X, y)
        best_feature_set = self.feature_selector.get_best_feature_set()
        
        if best_feature_set:
            results['feature_selection'] = {
                'best_feature_set': best_feature_set,
                'all_feature_sets': feature_sets
            }
        
        # 5. Generate recommendations
        results['recommendations'] = self._generate_code_recommendations(results)
        
        return results
    
    def _generate_code_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from code generation results"""
        recommendations = []
        
        # Genetic Programming recommendations
        gp_results = results.get('genetic_programming', {})
        if gp_results:
            fitness = gp_results.get('fitness', 0)
            if fitness > 0.5:
                recommendations.append("🧬 High-fitness strategy evolved: Consider implementing")
            else:
                recommendations.append("🧬 Low-fitness strategy: Continue evolution or adjust parameters")
        
        # Neural Architecture recommendations
        nas_results = results.get('neural_architecture_search', {})
        if nas_results:
            best_arch = nas_results.get('best_architecture')
            if best_arch and best_arch.performance_score > 0.7:
                recommendations.append("🧠 High-performance architecture found: Implement for production")
        
        # Hyperparameter recommendations
        hp_results = results.get('hyperparameter_optimization', {})
        if hp_results:
            best_config = hp_results.get('best_config')
            if best_config:
                recommendations.append(f"⚙️ Optimal hyperparameters: {best_config.parameters}")
        
        # Feature selection recommendations
        fs_results = results.get('feature_selection', {})
        if fs_results:
            best_features = fs_results.get('best_feature_set')
            if best_features:
                recommendations.append(f"🔍 Best features: {best_features.selected_features}")
        
        return recommendations
