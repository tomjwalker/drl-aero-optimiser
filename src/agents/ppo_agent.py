from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import torch

class PPOAgent:
    def __init__(self, env, model_params=None):
        # Default model parameters if none provided
        self.model_params = model_params or {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'normalize_advantage': True,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
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
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': torch.nn.ReLU,
        }
        
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            **self.model_params
        )
    
    def train(self, total_timesteps, callback=None):
        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, state, deterministic=True):
        # Ensure state is properly normalized
        if isinstance(state, tuple):
            state = state[0]  # Handle Gym API returning (state, info)
        return self.model.predict(state, deterministic=deterministic)
    
    def save(self, path):
        self.model.save(f"{path}/model")
        self.env.save(f"{path}/vec_normalize.pkl")
    
    @classmethod
    def load(cls, path, env):
        agent = cls(env)
        agent.model = PPO.load(f"{path}/model")
        agent.env = VecNormalize.load(f"{path}/vec_normalize.pkl", agent.env)
        agent.env.training = False
        agent.env.norm_reward = False
        return agent 