#!/usr/bin/env python3
"""
Structural Bayesian Networks for Causal Inference
Domain-constrained causal graphs and policy-shock simulators for risk & scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CausalNode:
    """Node in the causal graph"""
    name: str
    node_type: str  # 'market', 'macro', 'policy', 'endogenous'
    description: str
    data_source: str
    update_frequency: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    conditional_distribution: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """Edge in the causal graph"""
    source: str
    target: str
    edge_type: str  # 'direct', 'indirect', 'confounded'
    mechanism: str  # 'linear', 'nonlinear', 'threshold', 'regime_dependent'
    strength: float  # Causal strength [0, 1]
    lag: int = 0  # Time lag in periods
    confidence: float = 0.5  # Confidence in causal relationship
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyShock:
    """Policy shock for simulation"""
    shock_id: str
    target_variable: str
    shock_type: str  # 'level', 'growth', 'volatility'
    magnitude: float
    duration: int  # Number of periods
    start_date: datetime
    propagation_mechanism: str
    affected_sectors: List[str] = field(default_factory=list)
    second_order_effects: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of causal simulation"""
    scenario_id: str
    baseline_forecast: Dict[str, np.ndarray]
    shocked_forecast: Dict[str, np.ndarray]
    treatment_effects: Dict[str, np.ndarray]
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    propagation_path: List[str]
    simulation_metadata: Dict[str, Any]
    timestamp: datetime


class DomainKnowledge:
    """Domain knowledge constraints for causal structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define domain constraints for financial markets
        self.forbidden_edges = self._get_forbidden_edges()
        self.required_edges = self._get_required_edges()
        self.temporal_constraints = self._get_temporal_constraints()
        self.sector_hierarchies = self._get_sector_hierarchies()
        
    def _get_forbidden_edges(self) -> List[Tuple[str, str]]:
        """Get forbidden causal edges based on domain knowledge"""
        return [
            # Stock prices cannot cause macro indicators instantaneously
            ('stock_prices', 'gdp_growth'),
            ('stock_prices', 'inflation'),
            ('stock_prices', 'unemployment'),
            
            # Future events cannot cause past events
            ('future_earnings', 'current_price'),
            
            # Individual stocks cannot cause macro policy
            ('individual_stock', 'fed_policy'),
            ('individual_stock', 'fiscal_policy'),
            
            # Sector-specific constraints
            ('tech_stocks', 'oil_prices'),  # Unless through broader market
            ('bank_stocks', 'tech_earnings'),  # Direct causation unlikely
        ]
    
    def _get_required_edges(self) -> List[Tuple[str, str]]:
        """Get required causal edges based on economic theory"""
        return [
            # Central bank policy affects interest rates
            ('fed_policy', 'interest_rates'),
            
            # Interest rates affect bond yields
            ('interest_rates', 'bond_yields'),
            
            # Economic growth affects earnings
            ('gdp_growth', 'corporate_earnings'),
            
            # Volatility affects option prices
            ('volatility', 'option_prices'),
            
            # Credit spreads affect corporate bonds
            ('credit_risk', 'corporate_bond_spreads'),
            
            # Inflation affects real rates
            ('inflation', 'real_interest_rates'),
        ]
    
    def _get_temporal_constraints(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Get temporal constraints for causal relationships"""
        return {
            ('fed_policy', 'interest_rates'): {'min_lag': 0, 'max_lag': 1},
            ('interest_rates', 'stock_prices'): {'min_lag': 0, 'max_lag': 5},
            ('gdp_growth', 'corporate_earnings'): {'min_lag': 1, 'max_lag': 4},
            ('earnings_surprise', 'stock_prices'): {'min_lag': 0, 'max_lag': 1},
            ('geopolitical_risk', 'volatility'): {'min_lag': 0, 'max_lag': 2},
            ('monetary_policy', 'exchange_rates'): {'min_lag': 0, 'max_lag': 3},
        }
    
    def _get_sector_hierarchies(self) -> Dict[str, List[str]]:
        """Get sector hierarchies for causation"""
        return {
            'financials': ['banks', 'insurance', 'real_estate', 'fintech'],
            'technology': ['software', 'hardware', 'semiconductors', 'internet'],
            'healthcare': ['pharmaceuticals', 'biotech', 'medical_devices', 'healthcare_services'],
            'energy': ['oil_gas', 'renewables', 'utilities', 'coal'],
            'consumer': ['consumer_discretionary', 'consumer_staples', 'retail'],
            'industrials': ['aerospace', 'defense', 'manufacturing', 'transportation'],
            'materials': ['chemicals', 'metals', 'mining', 'forestry'],
            'communications': ['telecom', 'media', 'entertainment'],
            'utilities': ['electric', 'gas', 'water', 'renewable_energy'],
            'real_estate': ['reits', 'real_estate_services', 'homebuilders']
        }
    
    def validate_edge(self, source: str, target: str, lag: int = 0) -> Tuple[bool, str]:
        """Validate if an edge is allowed by domain knowledge"""
        # Check forbidden edges
        if (source, target) in self.forbidden_edges:
            return False, f"Forbidden edge: {source} -> {target}"
        
        # Check temporal constraints
        if (source, target) in self.temporal_constraints:
            constraints = self.temporal_constraints[(source, target)]
            if lag < constraints['min_lag'] or lag > constraints['max_lag']:
                return False, f"Temporal constraint violation: lag {lag} not in [{constraints['min_lag']}, {constraints['max_lag']}]"
        
        return True, "Valid edge"
    
    def get_prior_strength(self, source: str, target: str) -> float:
        """Get prior strength for causal relationship"""
        # Required edges have high prior strength
        if (source, target) in self.required_edges:
            return 0.8
        
        # Sector hierarchies
        for sector, subsectors in self.sector_hierarchies.items():
            if source == sector and target in subsectors:
                return 0.6
            if source in subsectors and target == sector:
                return 0.4
        
        # Default low prior for unknown relationships
        return 0.1


class CausalGraphLearner:
    """Learn causal graph structure with domain constraints"""
    
    def __init__(self, domain_knowledge: DomainKnowledge):
        self.domain_knowledge = domain_knowledge
        self.logger = logging.getLogger(__name__)
        self.learned_graph = None
        self.node_ordering = []
        
    def learn_structure(self, data: pd.DataFrame, 
                       variable_types: Dict[str, str],
                       max_lag: int = 5) -> nx.DiGraph:
        """Learn causal graph structure from data"""
        try:
            variables = list(data.columns)
            n_vars = len(variables)
            
            # Initialize graph
            graph = nx.DiGraph()
            
            # Add nodes
            for var in variables:
                var_type = variable_types.get(var, 'endogenous')
                graph.add_node(var, node_type=var_type)
            
            # Learn edges using PC algorithm with domain constraints
            edges_to_test = self._generate_candidate_edges(variables, max_lag)
            
            # Test independence relationships
            independence_results = {}
            for source, target, lag in edges_to_test:
                # Check domain constraints
                is_valid, reason = self.domain_knowledge.validate_edge(source, target, lag)
                if not is_valid:
                    self.logger.debug(f"Skipped edge {source} -> {target} (lag {lag}): {reason}")
                    continue
                
                # Test conditional independence
                p_value = self._test_conditional_independence(data, source, target, lag)
                independence_results[(source, target, lag)] = p_value
                
                # Add edge if significant and passes domain constraints
                if p_value < 0.05:  # Significance threshold
                    prior_strength = self.domain_knowledge.get_prior_strength(source, target)
                    
                    graph.add_edge(
                        source, target,
                        lag=lag,
                        strength=1 - p_value,
                        prior_strength=prior_strength,
                        p_value=p_value
                    )
            
            # Enforce required edges
            for source, target in self.domain_knowledge.required_edges:
                if source in variables and target in variables:
                    if not graph.has_edge(source, target):
                        graph.add_edge(
                            source, target,
                            lag=0,
                            strength=0.8,
                            prior_strength=0.8,
                            p_value=0.01,
                            required=True
                        )
            
            # Remove cycles using domain knowledge
            graph = self._remove_cycles(graph)
            
            self.learned_graph = graph
            self.logger.info(f"Learned causal graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error learning causal structure: {e}")
            # Return empty graph as fallback
            graph = nx.DiGraph()
            for var in data.columns:
                graph.add_node(var)
            return graph
    
    def _generate_candidate_edges(self, variables: List[str], max_lag: int) -> List[Tuple[str, str, int]]:
        """Generate candidate edges to test"""
        candidates = []
        
        for source in variables:
            for target in variables:
                if source != target:
                    for lag in range(max_lag + 1):
                        candidates.append((source, target, lag))
        
        return candidates
    
    def _test_conditional_independence(self, data: pd.DataFrame, 
                                     source: str, target: str, lag: int) -> float:
        """Test conditional independence between variables"""
        try:
            # Create lagged data
            if lag > 0:
                if len(data) <= lag:
                    return 1.0  # Cannot test with insufficient data
                
                source_data = data[source].iloc[:-lag].values
                target_data = data[target].iloc[lag:].values
                
                # Use remaining variables as conditioning set (simplified)
                conditioning_vars = [var for var in data.columns if var not in [source, target]]
                if conditioning_vars:
                    conditioning_data = data[conditioning_vars].iloc[lag:].values
                else:
                    conditioning_data = None
            else:
                source_data = data[source].values
                target_data = data[target].values
                conditioning_vars = [var for var in data.columns if var not in [source, target]]
                if conditioning_vars:
                    conditioning_data = data[conditioning_vars].values
                else:
                    conditioning_data = None
            
            # Perform partial correlation test
            if conditioning_data is not None and conditioning_data.shape[1] > 0:
                p_value = self._partial_correlation_test(source_data, target_data, conditioning_data)
            else:
                # Simple correlation test
                correlation, p_value = stats.pearsonr(source_data, target_data)
            
            return p_value
            
        except Exception as e:
            self.logger.warning(f"Error testing independence {source} -> {target} (lag {lag}): {e}")
            return 1.0  # Assume independence if test fails
    
    def _partial_correlation_test(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Test partial correlation between x and y given z"""
        try:
            # Linear regression approach
            # Regress x on z
            reg_x = LinearRegression().fit(z, x)
            residuals_x = x - reg_x.predict(z)
            
            # Regress y on z
            reg_y = LinearRegression().fit(z, y)
            residuals_y = y - reg_y.predict(z)
            
            # Test correlation between residuals
            if np.std(residuals_x) > 0 and np.std(residuals_y) > 0:
                correlation, p_value = stats.pearsonr(residuals_x, residuals_y)
                return p_value
            else:
                return 1.0
                
        except Exception as e:
            return 1.0
    
    def _remove_cycles(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles using domain knowledge priorities"""
        try:
            # Find cycles
            cycles = list(nx.simple_cycles(graph))
            
            for cycle in cycles:
                if len(cycle) <= 1:
                    continue
                
                # Find weakest edge in cycle to remove
                weakest_edge = None
                min_strength = float('inf')
                
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    
                    if graph.has_edge(source, target):
                        edge_data = graph[source][target]
                        
                        # Don't remove required edges
                        if edge_data.get('required', False):
                            continue
                        
                        strength = edge_data.get('strength', 0) + edge_data.get('prior_strength', 0)
                        
                        if strength < min_strength:
                            min_strength = strength
                            weakest_edge = (source, target)
                
                # Remove weakest edge
                if weakest_edge:
                    graph.remove_edge(weakest_edge[0], weakest_edge[1])
                    self.logger.debug(f"Removed edge {weakest_edge[0]} -> {weakest_edge[1]} to break cycle")
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error removing cycles: {e}")
            return graph


class StructuralEquationModel:
    """Structural equation model for causal inference"""
    
    def __init__(self, causal_graph: nx.DiGraph):
        self.graph = causal_graph
        self.parameters = {}
        self.noise_models = {}
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        
    def fit(self, data: pd.DataFrame) -> None:
        """Fit structural equation model"""
        try:
            # Topological sort to determine fitting order
            node_order = list(nx.topological_sort(self.graph))
            
            for node in node_order:
                parents = list(self.graph.predecessors(node))
                
                if len(parents) == 0:
                    # Exogenous variable - fit marginal distribution
                    self._fit_marginal_distribution(node, data[node])
                else:
                    # Endogenous variable - fit conditional distribution
                    self._fit_conditional_distribution(node, parents, data)
            
            self.is_fitted = True
            self.logger.info("Fitted structural equation model")
            
        except Exception as e:
            self.logger.error(f"Error fitting SEM: {e}")
            self.is_fitted = False
    
    def _fit_marginal_distribution(self, node: str, data: pd.Series) -> None:
        """Fit marginal distribution for exogenous variable"""
        try:
            # Fit normal distribution (can be extended to other distributions)
            mu = np.mean(data)
            sigma = np.std(data)
            
            self.parameters[node] = {
                'type': 'marginal',
                'distribution': 'normal',
                'parameters': {'mu': mu, 'sigma': sigma}
            }
            
            # Noise model (identity for exogenous variables)
            self.noise_models[node] = {
                'type': 'independent',
                'distribution': 'normal',
                'parameters': {'mu': 0, 'sigma': sigma}
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting marginal distribution for {node}: {e}")
    
    def _fit_conditional_distribution(self, node: str, parents: List[str], data: pd.DataFrame) -> None:
        """Fit conditional distribution for endogenous variable"""
        try:
            y = data[node].values
            X = data[parents].values
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # Fit linear regression
            reg = LinearRegression().fit(X, y)
            
            # Calculate residuals
            y_pred = reg.predict(X)
            residuals = y - y_pred
            residual_std = np.std(residuals)
            
            self.parameters[node] = {
                'type': 'conditional',
                'mechanism': 'linear',
                'parents': parents,
                'coefficients': reg.coef_,
                'intercept': reg.intercept_,
                'r_squared': reg.score(X, y)
            }
            
            # Noise model
            self.noise_models[node] = {
                'type': 'additive',
                'distribution': 'normal',
                'parameters': {'mu': 0, 'sigma': residual_std}
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting conditional distribution for {node}: {e}")
    
    def simulate(self, n_samples: int, 
                interventions: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Simulate data from structural equation model"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before simulation")
            
            # Initialize simulation data
            sim_data = {}
            node_order = list(nx.topological_sort(self.graph))
            
            for node in node_order:
                if interventions and node in interventions:
                    # Apply intervention
                    sim_data[node] = np.full(n_samples, interventions[node])
                else:
                    # Generate from structural equation
                    sim_data[node] = self._generate_node_values(node, sim_data, n_samples)
            
            return pd.DataFrame(sim_data)
            
        except Exception as e:
            self.logger.error(f"Error simulating from SEM: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def _generate_node_values(self, node: str, existing_data: Dict[str, np.ndarray], n_samples: int) -> np.ndarray:
        """Generate values for a single node"""
        try:
            params = self.parameters[node]
            noise_params = self.noise_models[node]
            
            if params['type'] == 'marginal':
                # Generate from marginal distribution
                dist_params = params['parameters']
                return np.random.normal(dist_params['mu'], dist_params['sigma'], n_samples)
            
            elif params['type'] == 'conditional':
                # Generate from conditional distribution
                parents = params['parents']
                
                # Get parent values
                parent_values = np.column_stack([existing_data[parent] for parent in parents])
                
                # Calculate structural component
                structural_component = (
                    params['intercept'] + 
                    np.sum(parent_values * params['coefficients'], axis=1)
                )
                
                # Add noise
                noise_dist_params = noise_params['parameters']
                noise = np.random.normal(noise_dist_params['mu'], noise_dist_params['sigma'], n_samples)
                
                return structural_component + noise
            
            else:
                # Fallback
                return np.random.normal(0, 1, n_samples)
                
        except Exception as e:
            self.logger.error(f"Error generating values for node {node}: {e}")
            return np.random.normal(0, 1, n_samples)


class PolicyShockSimulator:
    """Simulate policy shocks and their propagation through causal graph"""
    
    def __init__(self, structural_model: StructuralEquationModel, domain_knowledge: DomainKnowledge):
        self.sem = structural_model
        self.domain_knowledge = domain_knowledge
        self.logger = logging.getLogger(__name__)
        
        # Define shock propagation mechanisms
        self.propagation_mechanisms = {
            'monetary_policy': self._monetary_policy_propagation,
            'fiscal_policy': self._fiscal_policy_propagation,
            'regulatory_change': self._regulatory_change_propagation,
            'trade_policy': self._trade_policy_propagation,
            'geopolitical_shock': self._geopolitical_shock_propagation
        }
    
    async def simulate_policy_shock(self, shock: PolicyShock, 
                                  baseline_data: pd.DataFrame,
                                  simulation_horizon: int = 252) -> SimulationResult:
        """Simulate impact of policy shock"""
        try:
            # Generate baseline forecast
            baseline_forecast = self._generate_baseline_forecast(baseline_data, simulation_horizon)
            
            # Apply shock
            interventions = self._create_interventions(shock, simulation_horizon)
            
            # Generate shocked forecast
            shocked_forecast = self._generate_shocked_forecast(
                baseline_data, interventions, simulation_horizon
            )
            
            # Calculate treatment effects
            treatment_effects = {}
            confidence_intervals = {}
            
            for variable in baseline_forecast.keys():
                if variable in shocked_forecast:
                    treatment_effect = shocked_forecast[variable] - baseline_forecast[variable]
                    treatment_effects[variable] = treatment_effect
                    
                    # Calculate confidence intervals (simplified)
                    std_effect = np.std(treatment_effect)
                    ci_lower = treatment_effect - 1.96 * std_effect
                    ci_upper = treatment_effect + 1.96 * std_effect
                    confidence_intervals[variable] = (ci_lower, ci_upper)
            
            # Trace propagation path
            propagation_path = self._trace_propagation_path(shock)
            
            # Create simulation metadata
            simulation_metadata = {
                'shock_type': shock.shock_type,
                'shock_magnitude': shock.magnitude,
                'shock_duration': shock.duration,
                'simulation_horizon': simulation_horizon,
                'propagation_mechanism': shock.propagation_mechanism,
                'affected_sectors': shock.affected_sectors,
                'model_nodes': list(self.sem.graph.nodes()),
                'model_edges': list(self.sem.graph.edges())
            }
            
            return SimulationResult(
                scenario_id=f"{shock.shock_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                baseline_forecast=baseline_forecast,
                shocked_forecast=shocked_forecast,
                treatment_effects=treatment_effects,
                confidence_intervals=confidence_intervals,
                propagation_path=propagation_path,
                simulation_metadata=simulation_metadata,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error simulating policy shock: {e}")
            # Return empty result as fallback
            return SimulationResult(
                scenario_id="error",
                baseline_forecast={},
                shocked_forecast={},
                treatment_effects={},
                confidence_intervals={},
                propagation_path=[],
                simulation_metadata={},
                timestamp=datetime.now()
            )
    
    def _generate_baseline_forecast(self, baseline_data: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
        """Generate baseline forecast without intervention"""
        try:
            # Use last observation as initial condition
            initial_values = baseline_data.iloc[-1].to_dict()
            
            # Simulate forward
            simulated_data = self.sem.simulate(horizon, interventions=None)
            
            baseline_forecast = {}
            for column in simulated_data.columns:
                baseline_forecast[column] = simulated_data[column].values
            
            return baseline_forecast
            
        except Exception as e:
            self.logger.error(f"Error generating baseline forecast: {e}")
            return {}
    
    def _create_interventions(self, shock: PolicyShock, horizon: int) -> Dict[str, np.ndarray]:
        """Create intervention time series for shock"""
        interventions = {}
        
        # Create shock profile
        shock_profile = np.zeros(horizon)
        
        # Apply shock based on type
        if shock.shock_type == 'level':
            # Level shock
            shock_end = min(shock.duration, horizon)
            shock_profile[:shock_end] = shock.magnitude
        elif shock.shock_type == 'growth':
            # Growth shock (permanent change)
            shock_profile[:] = shock.magnitude
        elif shock.shock_type == 'volatility':
            # Volatility shock (affects variance)
            shock_profile[:] = shock.magnitude
        
        interventions[shock.target_variable] = shock_profile
        
        # Add second-order effects
        for variable, effect_magnitude in shock.second_order_effects.items():
            secondary_profile = shock_profile * effect_magnitude
            interventions[variable] = secondary_profile
        
        return interventions
    
    def _generate_shocked_forecast(self, baseline_data: pd.DataFrame,
                                 interventions: Dict[str, np.ndarray],
                                 horizon: int) -> Dict[str, np.ndarray]:
        """Generate forecast with interventions applied"""
        try:
            # Convert interventions to format expected by SEM
            intervention_dict = {}
            for variable, time_series in interventions.items():
                # Use average intervention value (simplified)
                intervention_dict[variable] = np.mean(time_series)
            
            # Simulate with interventions
            simulated_data = self.sem.simulate(horizon, interventions=intervention_dict)
            
            shocked_forecast = {}
            for column in simulated_data.columns:
                shocked_forecast[column] = simulated_data[column].values
            
            return shocked_forecast
            
        except Exception as e:
            self.logger.error(f"Error generating shocked forecast: {e}")
            return {}
    
    def _trace_propagation_path(self, shock: PolicyShock) -> List[str]:
        """Trace the propagation path of the shock through the causal graph"""
        try:
            # Find paths from shock target to other variables
            target_variable = shock.target_variable
            
            if target_variable not in self.sem.graph.nodes():
                return [target_variable]
            
            # BFS to find propagation paths
            visited = set()
            propagation_path = [target_variable]
            queue = [target_variable]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Get children (variables affected by current)
                children = list(self.sem.graph.successors(current))
                for child in children:
                    if child not in visited:
                        queue.append(child)
                        propagation_path.append(child)
            
            return propagation_path
            
        except Exception as e:
            self.logger.error(f"Error tracing propagation path: {e}")
            return [shock.target_variable]
    
    def _monetary_policy_propagation(self, shock: PolicyShock) -> Dict[str, float]:
        """Define monetary policy shock propagation"""
        return {
            'interest_rates': 1.0,  # Direct effect
            'bond_yields': 0.8,     # Strong indirect effect
            'exchange_rates': 0.6,  # Moderate effect
            'stock_prices': -0.4,   # Negative effect
            'credit_spreads': -0.3, # Risk premium effect
            'inflation_expectations': 0.2  # Long-term effect
        }
    
    def _fiscal_policy_propagation(self, shock: PolicyShock) -> Dict[str, float]:
        """Define fiscal policy shock propagation"""
        return {
            'government_spending': 1.0,  # Direct effect
            'gdp_growth': 0.6,          # Multiplier effect
            'inflation': 0.3,           # Demand pressure
            'bond_yields': 0.4,         # Financing pressure
            'corporate_earnings': 0.5,   # Economic stimulus
            'employment': 0.4           # Job creation
        }
    
    def _regulatory_change_propagation(self, shock: PolicyShock) -> Dict[str, float]:
        """Define regulatory change shock propagation"""
        return {
            'compliance_costs': 1.0,     # Direct effect
            'sector_valuations': -0.6,   # Negative valuation impact
            'competition': 0.3,          # Market structure change
            'innovation': -0.2,          # Potential drag on innovation
            'systemic_risk': -0.4        # Risk reduction
        }
    
    def _trade_policy_propagation(self, shock: PolicyShock) -> Dict[str, float]:
        """Define trade policy shock propagation"""
        return {
            'import_prices': 1.0,        # Direct tariff effect
            'export_competitiveness': -0.6,  # Retaliatory effects
            'domestic_production': 0.4,  # Import substitution
            'inflation': 0.3,            # Cost-push inflation
            'exchange_rates': 0.2,       # Trade balance effect
            'global_trade': -0.5         # Trade reduction
        }
    
    def _geopolitical_shock_propagation(self, shock: PolicyShock) -> Dict[str, float]:
        """Define geopolitical shock propagation"""
        return {
            'volatility': 1.0,           # Direct effect
            'safe_haven_demand': 0.8,    # Flight to quality
            'risk_premiums': 0.6,        # Increased uncertainty
            'commodity_prices': 0.4,     # Supply disruption
            'exchange_rates': 0.3,       # Risk-off flows
            'equity_valuations': -0.5    # Risk aversion
        }


class StructuralBayesianNetwork:
    """Main class for structural Bayesian network analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.domain_knowledge = DomainKnowledge()
        self.graph_learner = CausalGraphLearner(self.domain_knowledge)
        self.structural_model = None
        self.shock_simulator = None
        
        # State
        self.is_fitted = False
        self.learned_graph = None
        
        self.logger.info("Initialized Structural Bayesian Network")
    
    async def fit(self, data: pd.DataFrame, variable_types: Optional[Dict[str, str]] = None) -> None:
        """Fit the structural Bayesian network"""
        try:
            # Default variable types
            if variable_types is None:
                variable_types = {col: 'endogenous' for col in data.columns}
            
            # Learn causal graph structure
            self.logger.info("Learning causal graph structure...")
            self.learned_graph = self.graph_learner.learn_structure(data, variable_types)
            
            # Fit structural equation model
            self.logger.info("Fitting structural equation model...")
            self.structural_model = StructuralEquationModel(self.learned_graph)
            self.structural_model.fit(data)
            
            # Initialize shock simulator
            self.shock_simulator = PolicyShockSimulator(self.structural_model, self.domain_knowledge)
            
            self.is_fitted = True
            self.logger.info("Successfully fitted Structural Bayesian Network")
            
        except Exception as e:
            self.logger.error(f"Error fitting Structural BN: {e}")
            self.is_fitted = False
    
    async def simulate_scenario(self, shock: PolicyShock, 
                              baseline_data: pd.DataFrame,
                              simulation_horizon: int = 252) -> SimulationResult:
        """Simulate policy shock scenario"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        return await self.shock_simulator.simulate_policy_shock(
            shock, baseline_data, simulation_horizon
        )
    
    async def get_causal_summary(self) -> Dict[str, Any]:
        """Get summary of causal structure"""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        return {
            "graph_structure": {
                "nodes": list(self.learned_graph.nodes()),
                "edges": list(self.learned_graph.edges()),
                "n_nodes": self.learned_graph.number_of_nodes(),
                "n_edges": self.learned_graph.number_of_edges()
            },
            "model_parameters": {
                "fitted_equations": len(self.structural_model.parameters),
                "exogenous_variables": [
                    node for node in self.learned_graph.nodes()
                    if self.learned_graph.in_degree(node) == 0
                ],
                "endogenous_variables": [
                    node for node in self.learned_graph.nodes()
                    if self.learned_graph.in_degree(node) > 0
                ]
            },
            "domain_constraints": {
                "forbidden_edges": len(self.domain_knowledge.forbidden_edges),
                "required_edges": len(self.domain_knowledge.required_edges),
                "temporal_constraints": len(self.domain_knowledge.temporal_constraints)
            }
        }


# Factory function
async def create_structural_bn() -> StructuralBayesianNetwork:
    """Create and initialize structural Bayesian network"""
    return StructuralBayesianNetwork()


# Example usage
async def main():
    """Example usage of structural Bayesian network"""
    # Create sample data
    np.random.seed(42)
    n_periods = 500
    
    # Sample economic time series
    data = pd.DataFrame({
        'fed_policy': np.random.randn(n_periods) * 0.1,
        'interest_rates': np.random.randn(n_periods) * 0.5,
        'bond_yields': np.random.randn(n_periods) * 0.3,
        'stock_prices': np.random.randn(n_periods).cumsum() + 100,
        'volatility': np.abs(np.random.randn(n_periods)) * 10 + 15,
        'gdp_growth': np.random.randn(n_periods) * 0.02 + 0.03,
        'inflation': np.random.randn(n_periods) * 0.01 + 0.02
    })
    
    # Add some causal relationships
    for i in range(1, n_periods):
        data.loc[i, 'interest_rates'] += 0.7 * data.loc[i, 'fed_policy'] + 0.3 * data.loc[i-1, 'interest_rates']
        data.loc[i, 'bond_yields'] += 0.8 * data.loc[i, 'interest_rates'] + 0.2 * data.loc[i-1, 'bond_yields']
        data.loc[i, 'stock_prices'] += -0.5 * data.loc[i, 'interest_rates'] + 0.3 * data.loc[i, 'gdp_growth']
    
    # Define variable types
    variable_types = {
        'fed_policy': 'policy',
        'interest_rates': 'macro',
        'bond_yields': 'market',
        'stock_prices': 'market',
        'volatility': 'market',
        'gdp_growth': 'macro',
        'inflation': 'macro'
    }
    
    # Create and fit structural BN
    sbn = await create_structural_bn()
    await sbn.fit(data, variable_types)
    
    # Create policy shock
    shock = PolicyShock(
        shock_id="fed_rate_hike_2024",
        target_variable="fed_policy",
        shock_type="level",
        magnitude=0.5,  # 50 basis point rate hike
        duration=10,    # 10 periods
        start_date=datetime.now(),
        propagation_mechanism="monetary_policy",
        affected_sectors=["financials", "real_estate"],
        second_order_effects={
            "credit_spreads": -0.2,
            "mortgage_rates": 0.6
        }
    )
    
    # Simulate shock
    result = await sbn.simulate_scenario(shock, data.tail(50))
    
    print("Causal Analysis Results:")
    print(f"Scenario: {result.scenario_id}")
    print(f"Propagation path: {' -> '.join(result.propagation_path)}")
    print(f"Treatment effects calculated for {len(result.treatment_effects)} variables")
    
    # Get causal summary
    summary = await sbn.get_causal_summary()
    print(f"\nCausal Graph: {summary['graph_structure']['n_nodes']} nodes, {summary['graph_structure']['n_edges']} edges")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
