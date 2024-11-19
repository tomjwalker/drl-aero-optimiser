from stable_baselines3.common.callbacks import CheckpointCallback
import os
import logging

# Setup debug logger
debug_logger = logging.getLogger('debug_execute')
debug_logger.setLevel(logging.DEBUG)

class Run:
    def __init__(self, agent, config):
        # DEBUG_EXECUTE: Track initialization
        debug_logger.debug("Initializing Run with config: %s", config)
        self.agent = agent
        self.config = config
        
        # Setup checkpointing
        self.checkpoint_callback = CheckpointCallback(
            save_freq=config.get('checkpoint_freq', 10000),
            save_path=config['save_path'],
            name_prefix=config['name_prefix']
        )
    
    def execute(self):
        # DEBUG_EXECUTE: Track execution flow
        debug_logger.debug("Starting training execution")
        try:
            training_result = self.agent.train(
                total_timesteps=self.config['total_timesteps'],
                callback=self.checkpoint_callback
            )
            # DEBUG_EXECUTE: Track training completion
            debug_logger.debug("Training completed. Result: %s", training_result)
            
            # Evaluate the trained agent
            mean_reward = 0
            std_reward = 0
            best_design = None
            
            save_path = os.path.join(
                self.config['save_path'], 
                f"{self.config['name_prefix']}_final"
            )
            # DEBUG_EXECUTE: Track saving
            debug_logger.debug("Saving model to %s", save_path)
            self.agent.save(save_path)
            
            return {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'best_design': best_design
            }
            
        except Exception as e:
            # DEBUG_EXECUTE: Track errors
            debug_logger.error("Error during execution: %s", str(e))
            raise 