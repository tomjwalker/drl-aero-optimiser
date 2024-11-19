from stable_baselines3.common.callbacks import CheckpointCallback
import os

class Run:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        
        # Setup checkpointing
        self.checkpoint_callback = CheckpointCallback(
            save_freq=config.get('checkpoint_freq', 10000),
            save_path=config['save_path'],
            name_prefix=config['name_prefix']
        )
    
    def execute(self):
        self.agent.train(
            total_timesteps=self.config['total_timesteps'],
            callback=self.checkpoint_callback
        )
        self.agent.save(self.config['save_path']) 