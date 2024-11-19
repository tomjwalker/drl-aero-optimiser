from gymnasium import Env
from gymnasium import spaces
import numpy as np
from aerofoil_env.naca_aerofoil import naca4
from aerofoil_env.aeropy_interface import compute_aero_coefficients


class AerofoilOptimisationEnv(Env):
    def __init__(self, naca_code='0012', num_points=100, reynolds=1e6, alpha=0.0, max_action_delta=1):
        super().__init__()
        
        # Store parameters
        self.naca_code = naca_code
        self.num_points = num_points
        self.reynolds = reynolds
        self.alpha = alpha
        self.max_action_delta = max_action_delta
        
        # Define action space (increment/decrement for C, P, T)
        # Each parameter can be changed by -max_delta to +max_delta
        n_actions = 2 * max_action_delta + 1  # e.g., for max_delta=3: [-3,-2,-1,0,1,2,3]
        self.action_space = spaces.MultiDiscrete([n_actions, n_actions, n_actions])
        
        # Define observation space (C, P, T values)
        # C: 0-9, P: 0-9, T: 1-99
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]),
            high=np.array([9, 9, 99]),
            dtype=np.int32
        )
        
        # Initialize state
        self.state = self._naca_to_state(self.naca_code)
        
    def _naca_to_state(self, naca_code):
        """Convert NACA code string to state array"""
        return np.array([
            int(naca_code[0]),    # C
            int(naca_code[1]),    # P
            int(naca_code[2:])    # T
        ])
    
    def _state_to_naca(self, state):
        """Convert state array to NACA code string"""
        return f"{state[0]}{state[1]}{state[2]:02d}"
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial NACA code
        self.state = self._naca_to_state(self.naca_code)
        
        return self.state, {}  # Return state and info dict (Gymnasium convention)
    
    def step(self, action):
        # Convert action to actual changes
        action = action - self.max_action_delta  # Convert [0,1,2,...,2*max_delta] to [-max_delta,...,0,...,+max_delta]
        
        # Update state based on action
        new_state = self.state + action
        
        # Ensure state stays within bounds
        new_state = np.clip(
            new_state,
            self.observation_space.low,
            self.observation_space.high
        )
        
        # Update state and NACA code
        self.state = new_state
        new_naca = self._state_to_naca(self.state)
        
        print(f"Debug - New NACA code: {new_naca}")  # Add debug print
        
        # Compute aerodynamic coefficients
        coeffs = compute_aero_coefficients(
            airfoil=new_naca,
            num_points=self.num_points,
            Reynolds=self.reynolds,
            alpha=self.alpha
        )
        
        # Calculate reward with clipping and scaling
        if coeffs['converged']:
            if coeffs['CD'] != 0:
                reward = coeffs['CL'] / coeffs['CD']
                # Clip reward to reasonable range
                reward = np.clip(reward, -10.0, 10.0)
            else:
                reward = -10.0  # Penalty for zero drag (likely numerical issue)
        else:
            reward = -10.0  # Penalty for non-convergent shapes
        
        # Add small penalty for extreme shapes to encourage exploration of reasonable designs
        c, p, t = self._naca_to_state(new_naca)
        shape_penalty = -0.01 * (c/9 + p/9 + t/99)  # Small penalty increasing with parameter values
        reward += shape_penalty
        
        # Check if done (we'll use a simple episode length limit)
        done = False
        
        # Additional info
        info = {
            'naca_code': new_naca,
            'coefficients': coeffs
        }
        
        return self.state, reward, done, False, info  # False is for truncated (Gymnasium convention)

if __name__ == "__main__":
    # Quick manual testing
    env = AerofoilOptimisationEnv(naca_code='0012')
    
    # Test reset
    state, _ = env.reset()
    print(f"Initial state: {state}")
    print(f"Initial NACA: {env._state_to_naca(state)}")
    
    # Test a few random actions
    for i in range(5):
        action = env.action_space.sample()  # Random action
        state, reward, done, truncated, info = env.step(action)
        print(f"\nStep {i+1}")
        print(f"Action taken: {action}")
        print(f"New state: {state}")
        print(f"New NACA: {info['naca_code']}")
        print(f"Reward: {reward:.3f}")
        print(f"CL/CD: {info['coefficients']['CL']:.3f}/{info['coefficients']['CD']:.3f}")
