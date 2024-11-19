import pytest
import numpy as np
from agents.ppo_agent import PPOAgent
from experiments.evaluate import evaluate_agent
from aerofoil_env.env import AerofoilOptimisationEnv

def test_evaluation():
    env = AerofoilOptimisationEnv()
    agent = PPOAgent(env)
    state, _ = env.reset()
    results = evaluate_agent(agent, env, num_episodes=2)
    
    assert 'mean_reward' in results
    assert 'std_reward' in results
    assert 'best_design' in results
    assert isinstance(results['mean_reward'], float)
    assert isinstance(results['std_reward'], float)