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
        'model_params': {
            'learning_rate': 3e-4,
            'n_steps': 4,      # Minimal steps
            'batch_size': 2,   # Minimal batch size
            'n_epochs': 1,     # Minimal epochs
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'normalize_advantage': True,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
        },
        'seeds': [42],
        'total_timesteps': 4,  # Small number for testing
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
        'model_params': {
            'learning_rate': 3e-4,
            'n_steps': 4,      # Minimal steps
            'batch_size': 2,   # Minimal batch size
            'n_epochs': 1,     # Minimal epochs
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'normalize_advantage': True,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
        },
        'seeds': [42],
        'total_timesteps': 4,  # Small number for testing
        'name_prefix': 'test_trial',
        'save_path': str(tmp_path)
    }
    trial = Trial(PPOAgent, AerofoilOptimisationEnv, config)
    results = trial.run()
    assert len(results) == 1  # One result per seed 