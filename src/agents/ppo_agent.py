"""
PPO (Proximal Policy Optimization) agent implementation for reinforcement learning.

This module provides a wrapper around Stable-Baselines3's PPO implementation with
additional functionality for environment normalization and easy saving/loading.
Includes customizable network architecture and hyperparameters.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any, Union

class PPOAgent:
    """A PPO agent wrapper with environment normalization and convenient interfaces."""
    
    def __init__(self, env: Any, model_params: Optional[Dict] = None) -> None:
        """
        Initialize the PPO agent with an environment and optional parameters.

        Args:
            env: A Gym-compatible environment
            model_params: Dictionary of PPO hyperparameters. If None, uses default values
        """
        # Default model parameters if none provided
        self.model_params = model_params or {
            'learning_rate': 3e-4,        # Step size for policy/value network updates
            'n_steps': 128,              # Steps of experience (S, A, R, S') to collect before updating
            'batch_size': 32,             # Mini-batch size for SGD (how to split up the experience into batches, e.g. 128/32 = 4 batches)
            'n_epochs': 5,               # Number of passes over the data
            'gamma': 0.99,                # Discount factor (same as in Q-learning)
            'gae_lambda': 0.95,           # Trade-off parameter for GAE (Generalized Advantage Estimation). Lambda=0 is like TD(0), lambda=1 is like Monte Carlo.
            'clip_range': 0.2,            # How far the new policy can deviate from old. A PPO-specific parameter. 0.2 signifies 20% deviation.
            'normalize_advantage': True,   # Normalize advantages for stable training
            'ent_coef': 0.01,             # Entropy coefficient for exploration. A more sophisticated version of epsilon-greedy.
            'max_grad_norm': 0.5,         # Gradient clipping threshold
            
            # Network architecture
            'policy_kwargs': {
                'net_arch': [dict(pi=[32], vf=[32])],  # Later, try 2-hidden-layer NNs, e.g. [dict(pi=[64, 64], vf=[64, 64])]
                'activation_fn': torch.nn.ReLU
            }
        }
        
        # Wrap environment
        self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(
            self.env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=self.model_params['gamma'],
        )
        
        # Initialize PPO agent with modified network architecture
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            verbose=1,
            **self.model_params
        )
    
    def train(self, total_timesteps: int, callback: Optional[Any] = None) -> PPO:
        """
        Train the agent for a specified number of timesteps.

        Args:
            total_timesteps: Number of environment steps to train for
            callback: Optional callback for training events

        Returns:
            Trained PPO model
        """
        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, state: Union[np.ndarray, Tuple], deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for a given state.

        Args:
            state: Environment state or state-info tuple
            deterministic: If True, return best action; if False, sample from action distribution

        Returns:
            Tuple of (action, states)
        """
        if isinstance(state, tuple):
            state = state[0]  # Handle Gym API returning (state, info)
        return self.model.predict(state, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """
        Save the model and environment normalization parameters.

        Args:
            path: Directory path to save the model and normalizer
        """
        self.model.save(f"{path}/model")
        self.env.save(f"{path}/vec_normalize.pkl")
    
    @classmethod
    def load(cls, path: str, env: Any) -> 'PPOAgent':
        """
        Load a saved agent and environment normalizer.

        Args:
            path: Directory path containing the saved model and normalizer
            env: A Gym-compatible environment

        Returns:
            Loaded PPOAgent instance
        """
        agent = cls(env)
        agent.model = PPO.load(f"{path}/model")
        agent.env = VecNormalize.load(f"{path}/vec_normalize.pkl", agent.env)
        agent.env.training = False
        agent.env.norm_reward = False
        return agent 