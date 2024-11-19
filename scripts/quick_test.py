from aerofoil_env.env import AerofoilOptimisationEnv
from agents.ppo_agent import PPOAgent
from experiments.trial import Trial
from experiments.evaluate import evaluate_agent

def main():
    # Quick training configuration
    config = {
        'env_params': {
            'naca_code': '0012',
            'num_points': 100,
            'reynolds': 1e6,
            'alpha': 0.0
        },
        'model_params': {
            'learning_rate': 3e-4,
            'n_steps': 128,
            'batch_size': 64,
            'n_epochs': 1
        },
        'seeds': [42],
        'total_timesteps': 1000,
        'save_path': 'test_results',
        'name_prefix': 'quick_test'
    }
    
    print("Starting quick test...")
    trial = Trial(PPOAgent, AerofoilOptimisationEnv, config)
    results = trial.run()
    
    print("\nResults:")
    print(f"Mean Reward: {results[0]['mean_reward']:.2f}")
    print(f"Best Design: {results[0]['best_design']['naca']}")

if __name__ == "__main__":
    main() 