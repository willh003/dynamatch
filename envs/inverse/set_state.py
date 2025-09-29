import gymnasium as gym
import numpy as np


def set_state(env: gym.Env, state: np.ndarray):
    """
    Set the state of the environment to the given state

    Args:
        env: The environment to set the state of
        state: The state to set the environment to (shape: [cart_pos, pole_angle, cart_vel, pole_angular_vel])
    
    Returns:
        np.ndarray: The state of the environment after setting it
    """
    env_id = env.unwrapped.spec.id
    if "InvertedPendulum" in env_id:
        set_state_pendulum(env, state)
    else:
        raise ValueError(f"Environment {env_id} not supported for state setting - implement set_state for this environment") 

def set_state_pendulum(env: gym.Env, state: np.ndarray):
    """
    Set the state of the environment to the given state

    Args:
        env: The environment to set the state of
        state: The state to set the environment to (shape: [cart_pos, pole_angle, cart_vel, pole_angular_vel])
    
    Returns:
        np.ndarray: The state of the environment after setting it
    """
    # Extract position and velocity from the state vector
    # State format: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
    qpos = state[:2]  # [cart_pos, pole_angle]
    qvel = state[2:]  # [cart_vel, pole_angular_vel]
    
    # Use the environment's set_state method
    env.unwrapped.set_state(qpos, qvel)
    
    # Return the current state of the environment
    current_qpos = env.unwrapped.data.qpos[:2]
    current_qvel = env.unwrapped.data.qvel[:2]
    current_state = np.concatenate([current_qpos, current_qvel])

    assert np.allclose(current_state, state, atol=1e-10), "ERROR: state not set correctly"
    