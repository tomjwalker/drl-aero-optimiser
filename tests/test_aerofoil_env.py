import numpy as np
import pytest  # noqa
from aerofoil_env.env import AerofoilOptimisationEnv

def test_env_initialization():
    env = AerofoilOptimisationEnv(naca_code='0012')
    assert env.naca_code == '0012'
    assert isinstance(env.observation_space.sample(), np.ndarray)
    assert isinstance(env.action_space.sample(), np.ndarray)

def test_reset():
    env = AerofoilOptimisationEnv(naca_code='2412')
    state, _ = env.reset()
    assert np.array_equal(state, np.array([2, 4, 12]))

def test_step():
    env = AerofoilOptimisationEnv()
    state, _ = env.reset()
    action = np.array([1, 1, 1])  # Increase all parameters
    next_state, reward, done, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(info, dict)
    assert 'naca_code' in info
    assert 'coefficients' in info
    assert 'converged' in info['coefficients']

def test_bounds():
    env = AerofoilOptimisationEnv(naca_code='0012')
    state, _ = env.reset()
    
    # Try to decrease beyond minimum bounds
    action = np.array([0, 0, 0])  # All -1 after conversion
    next_state, _, _, _, _ = env.step(action)
    assert np.all(next_state >= env.observation_space.low)
    
    # Try to increase beyond maximum bounds
    action = np.array([2, 2, 2])  # All +1 after conversion
    for _ in range(10):  # Reduced from 100 to avoid too many XFOIL calls
        next_state, _, _, _, _ = env.step(action)
        assert np.all(next_state <= env.observation_space.high)

def test_naca_conversion():
    env = AerofoilOptimisationEnv()
    test_state = np.array([2, 4, 12])
    naca_code = env._state_to_naca(test_state)
    assert naca_code == '2412'
    converted_state = env._naca_to_state(naca_code)
    assert np.array_equal(test_state, converted_state)

def test_non_convergent_case():
    env = AerofoilOptimisationEnv(naca_code='9999')  # An extreme case likely to fail
    state, _ = env.reset()
    action = np.array([1, 1, 1])
    _, reward, _, _, info = env.step(action)
    assert 'converged' in info['coefficients']
    if not info['coefficients']['converged']:
        assert reward == -1.0  # Check penalty for non-convergent shapes