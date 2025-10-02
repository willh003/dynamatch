import gymnasium as gym
import numpy as np
import mujoco


def set_state(env: gym.Env, state: np.ndarray):
    """
    Set the state of the environment to the given state

    Args:
        env: The environment to set the state of
        state: The state to set the environment to (shape: [cart_pos, pole_angle, cart_vel, pole_angular_vel])
        info: The info from stepping (often includes full state variables, in case they are masked in the env observation)
    
    Returns:
        np.ndarray: The state of the environment after setting it
    """
    env_id = env.unwrapped.spec.id
    if "InvertedPendulum" in env_id:
        set_state_pendulum(env, state)
    elif "Ant" in env_id:
        set_state_ant(env, state)
    else:
        raise ValueError(f"Environment {env_id} not supported for state setting - implement set_state for this environment") 

def set_state_pendulum(env: gym.Env, state: np.ndarray):
    """
    Set the state of the environment to the given state

    Args:
        env: The environment to set the state of
        state: The state to set the environment to (shape: [cart_pos, pole_angle, cart_vel, pole_angular_vel])
    
    Returns: None
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


def set_state_ant(env: gym.Env, state: np.ndarray):
    """
    Set the MuJoCo state from Gym observation and info.
    
    Args:
        env: Ant-v5 gym environment
        obs: observation array (105 elements: 13 qpos + 14 qvel + 78 cfrc_ext)
        info: info dict containing 'x_position' and 'y_position'
    """
    # Extract components from observation
    qpos= state[:15]  # x,y,z, quat(4), joints(8)
    qvel = state[15:29]   # all 14 velocities
    
    # Set the state
    env.unwrapped.set_state(qpos, qvel)

