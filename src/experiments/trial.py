import numpy as np
from datetime import datetime
import os
from experiments.run import Run

class Trial:
    def __init__(self, agent_class, env_class, config):
        self.agent_class = agent_class
        self.env_class = env_class
        self.config = config
        self.seeds = config.get('seeds', [42, 123, 456])
        
        # Create trial directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trial_dir = f"experiments/trial_{timestamp}"
        os.makedirs(self.trial_dir, exist_ok=True)
    
    def run(self):
        results = []
        for seed in self.seeds:
            # Create environment and agent for this seed
            env = self.env_class(**self.config['env_params'])
            agent = self.agent_class(env, self.config.get('model_params'))
            
            # Configure save path for this run
            run_dir = f"{self.trial_dir}/seed_{seed}"
            os.makedirs(run_dir, exist_ok=True)
            run_config = {
                **self.config,
                'save_path': run_dir,
                'seed': seed
            }
            
            # Execute run
            run = Run(agent, run_config)
            result = run.execute()
            results.append(result)
        
        return results 