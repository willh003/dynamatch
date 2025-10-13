import gymnasium as gym
import numpy as np
import mujoco
from typing import Dict, Any

def modify_env_integrator(env, integrator=None, frame_skip=None):
    if integrator is not None:
        assert integrator in ['euler', 'rk4'], "ERROR: integrator must be euler or rk4"
        if integrator == 'euler':
            env.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        elif integrator == 'rk4':
            env.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    else:
        print(f"WARNING: using default integrator for forward, {env.unwrapped.model.opt.integrator}")
    
    if frame_skip is not None:
        env.unwrapped.frame_skip = frame_skip
    else:
        print(f"WARNING: using default frame skip for forward, {env.unwrapped.frame_skip}")

    return env

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


def parse_raw_observations_pendulum(obs_array: np.ndarray, obs_shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Parse raw observation array into dictionary format expected by the dataset.
    
    For InvertedPendulum-v5, the observation array typically contains:
    [x,theta, x_dot, theta_dot] where:
    - x: cart position
    - x_dot: cart velocity  
    - theta: pole angle
    - theta_dot: pole velocity
    
    Args:
        obs_array: Raw observation array from environment
        obs_shape_meta: Shape metadata for observations
        
    Returns:
        Dictionary with parsed observations
    """
    obs_dict = {}
    
    # For InvertedPendulum-v5, map the 4-element array to the expected keys
    # Environment observation order: [cart_pos, pole_angle, cart_vel,  pole_vel]
    if len(obs_array.shape) == 1 and obs_array.shape[0] == 4:
        # Single observation - reorder to match set_state expectation
        obs_dict["cart_position"] = np.array([obs_array[0]])  # cart_pos
        obs_dict["pole_angle"] = np.array([obs_array[1]])     # pole_angle  
        obs_dict["cart_velocity"] = np.array([obs_array[2]])  # cart_vel
        obs_dict["pole_velocity"] = np.array([obs_array[3]])  # pole_vel
    elif len(obs_array.shape) == 2 and obs_array.shape[1] == 4:
        # Multiple observations (trajectory) - reorder to match set_state expectation
        obs_dict["cart_position"] = obs_array[:, 0:1]  # cart_pos
        obs_dict["pole_angle"] = obs_array[:, 1:2]     # pole_angle
        obs_dict["cart_velocity"] = obs_array[:, 2:3]  # cart_vel
        obs_dict["pole_velocity"] = obs_array[:, 3:4]  # pole_vel
    
    return obs_dict

def parse_raw_observations_ant(obs_array: np.ndarray, info:dict, obs_shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Parse raw observation array into dictionary format expected by the dataset.
    
    For Ant-v5, the observation array typically contains:
    """
    x_pos = np.array([d["x_position"] for d in info])[:,None]
    y_pos = np.array([d["y_position"] for d in info])[:,None]

    full_obs = np.concatenate([x_pos, y_pos,obs_array], axis=1)
    obs_dict = {"full_obs": full_obs}
    return obs_dict

def parse_raw_observations(obs_array: np.ndarray, info:dict, obs_shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Parse raw observation array into dictionary format expected by the dataset.
    """
    if "cart_position" in obs_shape_meta:
        # TODO: better check for env type
        return parse_raw_observations_pendulum(obs_array, obs_shape_meta)
    else:
        return parse_raw_observations_ant(obs_array, info, obs_shape_meta)

def get_state_from_obs_pendulum(obs_array: np.ndarray) -> np.ndarray:
    """
    Get state from observation array for pendulum environment.
    """
    return obs_array

def get_state_from_obs_ant(obs_array: np.ndarray, info:dict) -> np.ndarray:
    """
    Get state from observation array for ant environment.
    """
    x_pos = info['x_position']
    y_pos = info['y_position']

    full_obs = np.concatenate([[x_pos], [y_pos],obs_array], axis=-1)
    return full_obs


def get_state_from_obs_fetch(obs: dict, info:dict) -> np.ndarray:
    """
    Get state from observation array for fetch environment.
    """

    obs_array = obs['observation']
    goal_array = obs['desired_goal']
    full_obs = np.concatenate([obs_array, goal_array], axis=-1)
    return full_obs


def get_state_from_obs(obs: np.ndarray, info:dict, env_id: str) -> np.ndarray:
    """
    Get state from observation array for environment.
    """
    if "Pendulum" in env_id:
        return get_state_from_obs_pendulum(obs)
    elif "Ant" in env_id:
        return get_state_from_obs_ant(obs, info)
    elif "Fetch" in env_id:
        return get_state_from_obs_fetch(obs, info)
    else:
        raise ValueError(f"Environment {env_id} not supported for state getting - implement get_state_from_obs for this environment")


