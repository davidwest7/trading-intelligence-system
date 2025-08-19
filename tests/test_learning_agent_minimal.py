"""
Minimal tests for the Learning Agent
"""

import pytest

from agents.learning.agent import LearningAgent


@pytest.mark.asyncio
async def test_learning_agent_basic_output_structure():
    agent = LearningAgent()

    result = await agent.process()

    # Top-level structure
    assert isinstance(result, dict)
    assert 'learning_analysis' in result

    analysis = result['learning_analysis']

    # Required keys from LearningAnalysis.to_dict()
    expected_keys = {
        'system_id',
        'timestamp',
        'active_models',
        'best_performing_model',
        'ensemble_performance',
        'recent_adaptations',
        'learning_trajectory',
        'feature_importance',
        'model_correlations',
        'regime_detection',
        'expected_performance_improvement',
        'recommended_adaptations',
        'next_learning_cycle',
    }

    assert expected_keys.issubset(set(analysis.keys()))

    # Active models
    assert isinstance(analysis['active_models'], list)
    assert len(analysis['active_models']) >= 3
    for m in analysis['active_models']:
        assert isinstance(m, dict)
        assert 'model_id' in m and 'model_type' in m and 'sharpe_ratio' in m

    # Ensemble
    ensemble = analysis['ensemble_performance']
    assert isinstance(ensemble, dict)
    assert ensemble.get('model_id') == 'ensemble'

    # Best model reference should be non-empty and reference one of the active models
    best_id = analysis['best_performing_model']
    assert isinstance(best_id, str) and len(best_id) > 0
    assert any(m['model_id'] == best_id for m in analysis['active_models'])


