from aerofoil_env.env import AerofoilOptimisationEnv
from agents.ppo_agent import PPOAgent
from experiments.trial import Trial

def main():
    # Define configuration
    config = {
        'env_params': {
            'naca_code': '0012',
            'num_points': 100,
            'reynolds': 1e6,
            'alpha': 0.0,
            'max_action_delta': 1
        },
        'model_params': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
        },
        'seeds': [42, 123, 456],
        'checkpoint_freq': 10000,
        'name_prefix': 'ppo_aerofoil',
        'total_timesteps': 1000000
    }
    
    # Create and run trial
    trial = Trial(PPOAgent, AerofoilOptimisationEnv, config)
    results = trial.run()
    
    # Print results
    print("\nTrial Results:")
    for seed, result in zip(config['seeds'], results):
        print(f"\nSeed {seed}:")
        print(f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"Best Design: NACA {result['best_design']['naca']}")
        print(f"Best CL/CD: {result['best_design']['coefficients']['CL']:.3f}/{result['best_design']['coefficients']['CD']:.3f}")

if __name__ == "__main__":
    main() 