import pytest
from agents.ppo_agent import PPOAgent
from experiments.trial import Trial
from aerofoil_env.env import AerofoilOptimisationEnv

def test_trial_initialization():
    config = {
        'env_params': {
            'naca_code': '0012',
            'num_points': 100,
            'reynolds': 1e6,
            'alpha': 0.0
        },
        'seeds': [42],
        'total_timesteps': 2,  # Small number for testing
        'name_prefix': 'test_trial'
    }
    trial = Trial(PPOAgent, AerofoilOptimisationEnv, config)
    assert trial.seeds == [42]

def test_trial_run(tmp_path):
    config = {
        'env_params': {
            'naca_code': '0012',
            'num_points': 100,
            'reynolds': 1e6,
            'alpha': 0.0
        },
        'seeds': [42],
        'total_timesteps': 2,  # Small number for testing
        'name_prefix': 'test_trial',
        'save_path': str(tmp_path)
    }
    trial = Trial(PPOAgent, AerofoilOptimisationEnv, config)
    results = trial.run()
    assert len(results) == 1  # One result per seed 