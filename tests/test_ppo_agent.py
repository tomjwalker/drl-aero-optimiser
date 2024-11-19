import pytest
import numpy as np
from agents.ppo_agent import PPOAgent
from aerofoil_env.env import AerofoilOptimisationEnv

def test_agent_initialization():
    env = AerofoilOptimisationEnv()
    agent = PPOAgent(env)
    assert agent.model is not None
    assert agent.env is not None

def test_agent_prediction():
    env = AerofoilOptimisationEnv()
    agent = PPOAgent(env)
    state = env.reset()[0]
    action, _ = agent.predict(state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)  # For your 3-parameter action space

def test_agent_save_load(tmp_path):
    env = AerofoilOptimisationEnv()
    agent = PPOAgent(env)
    
    # Save agent
    save_path = str(tmp_path)
    agent.save(save_path)
    
    # Load agent
    loaded_agent = PPOAgent.load(save_path, env)
    assert loaded_agent.model is not None 