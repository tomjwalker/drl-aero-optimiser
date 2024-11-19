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
    agent = PPOAgent(env)
    config = {
        'total_timesteps': 4,  # Minimal number for testing
        'save_path': str(tmp_path),
        'name_prefix': 'test_run',
        'checkpoint_freq': 2
    }
    run = Run(agent, config)
    result = run.execute()
    assert result is not None