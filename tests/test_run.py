import pytest
from agents.ppo_agent import PPOAgent
from experiments.run import Run
from aerofoil_env.env import AerofoilOptimisationEnv

def test_run_initialization(tmp_path):
    env = AerofoilOptimisationEnv()
    agent = PPOAgent(env)
    config = {
        'total_timesteps': 4,  # Small number for testing
        'save_path': str(tmp_path),
        'name_prefix': 'test_run',
        'checkpoint_freq': 2
    }
    run = Run(agent, config)
    assert run.checkpoint_callback is not None

def test_run_execution(tmp_path):
    env = AerofoilOptimisationEnv()
    test_model_params = {
        'learning_rate': 3e-4,
        'n_steps': 4,  # Reduced for testing
        'batch_size': 2,  # Must be smaller than n_steps
        'n_epochs': 1,  # Reduced for testing
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'normalize_advantage': True,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
    }
    agent = PPOAgent(env, model_params=test_model_params)
    config = {
        'total_timesteps': 4,
        'save_path': str(tmp_path),
        'name_prefix': 'test_run',
        'checkpoint_freq': 2
    }
    run = Run(agent, config)
    result = run.execute()
    assert result is not None