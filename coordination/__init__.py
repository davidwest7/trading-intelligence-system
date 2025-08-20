"""
Coordination Layer - Core Decision Making System

This module implements the coordination layer that orchestrates all 12 agents
and manages the decision-making pipeline for the trading intelligence system.

Components:
- Meta-Weighter: Calibrated blend of agent signals
- Top-K Selector: Diversified bandit for opportunity selection
- Opportunity Builder: Merge + costs + constraints
"""

from .meta_weighter import MetaWeighter
from .top_k_selector import TopKSelector
from .opportunity_builder import OpportunityBuilder

__all__ = ['MetaWeighter', 'TopKSelector', 'OpportunityBuilder']
