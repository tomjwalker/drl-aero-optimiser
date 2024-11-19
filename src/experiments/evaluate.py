import numpy as np

def evaluate_agent(agent, env, num_episodes=10):
    episode_rewards = []
    best_design = {'naca': None, 'reward': -np.inf}

    for episode in range(num_episodes):
        state = env.reset()[0]  # Get only the state from reset
        terminated = truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action, _ = agent.predict(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update best design if applicable
            if reward > best_design['reward']:
                best_design['naca'] = info['naca_code']
                best_design['reward'] = reward

        episode_rewards.append(total_reward)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'best_design': best_design
    } 