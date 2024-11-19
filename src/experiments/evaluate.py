import numpy as np

def evaluate_agent(agent, env, num_episodes=10):
    episode_rewards = []
    best_design = {'naca': None, 'reward': -np.inf}
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = agent.predict(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Track best design
            if reward > best_design['reward']:
                best_design['reward'] = reward
                best_design['naca'] = info['naca_code']
                best_design['coefficients'] = info['coefficients']
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'best_design': best_design
    } 