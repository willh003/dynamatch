import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from cluster_utils import set_cluster_graphics_vars
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.register_envs import register_custom_envs
from envs.env_utils import modify_env_integrator
from tqdm import tqdm
import copy
import mujoco
import zarr


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

    full_obs = np.concatenate([[x_pos], [y_pos],obs_array], axis=0)
    return full_obs

def get_state_from_obs(obs_array: np.ndarray, info:dict, env_id: str) -> np.ndarray:
    """
    Get state from observation array for environment.
    """
    if "Pendulum" in env_id:
        return get_state_from_obs_pendulum(obs_array)
    elif "Ant" in env_id:
        return get_state_from_obs_ant(obs_array, info)
    else:
        raise ValueError(f"Environment {env_id} not supported for state getting - implement get_state_from_obs for this environment")


def make_env(env_id: str, env_kwargs: Dict[str, Any], seed: Optional[int] = None):
    """
    Create a single environment for vectorized environment.
    """
    def _init():
        # Register custom environments in each subprocess
        register_custom_envs()
        env = gym.make(env_id, **env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def collect_transitions_parallel(
    model_path: str,
    env_id: str,
    env_kwargs: Dict[str, Any] = None,
    num_transitions: int = 10000,
    max_steps_per_episode: int = 1000,
    deterministic: bool = False,
    seed: Optional[int] = None,
    n_envs: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect transitions (s, a, s') in parallel using vectorized environments.
    
    Args:
        model_path: Path to the trained model .zip file
        env_id: Gymnasium environment ID
        env_kwargs: Additional environment arguments
        num_transitions: Number of transitions to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        n_envs: Number of parallel environments
        
    Returns:
        Tuple of (states, actions, next_states) as numpy arrays
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, env_kwargs, seed) for _ in range(n_envs)])
    
    states = []
    actions = []
    next_states = []
    all_rewards = []
    
    # Initialize environments
    obs = vec_env.reset()
    episode_steps = [0] * n_envs
    episode_dones = [False] * n_envs
    transitions_collected = 0
    
    # Store previous observations for each environment
    prev_obs = [None] * n_envs
    prev_infos = [None] * n_envs
    prev_actions = [None] * n_envs
    
    # Collect transitions in parallel
    with tqdm(total=num_transitions, desc="Collecting transitions") as pbar:
        while transitions_collected < num_transitions:
            # Get actions from policy
            policy_actions, _ = model.predict(obs, deterministic=deterministic)
            
            # Step all environments
            next_obs, rewards, dones, infos = vec_env.step(policy_actions)
            
            # Store transitions for each environment
            for env_idx in range(n_envs):
                if not episode_dones[env_idx] and transitions_collected < num_transitions:
                    # Only store transition if we have a previous observation (not at episode start)
                    if prev_obs[env_idx] is not None:
                        # Get state from previous observation (before action was taken)
                        state = get_state_from_obs(prev_obs[env_idx], prev_infos[env_idx], env_id)
                        next_state = get_state_from_obs(next_obs[env_idx], infos[env_idx], env_id)
                        
                        states.append(state)
                        actions.append(prev_actions[env_idx])  # Action that was taken
                        next_states.append(next_state)
                        all_rewards.append(rewards[env_idx])
                        
                        transitions_collected += 1
                        pbar.update(1)
                    
                    episode_steps[env_idx] += 1
                    
                    if dones[env_idx] or episode_steps[env_idx] >= max_steps_per_episode:
                        # Episode finished for this environment
                        episode_dones[env_idx] = True
                        episode_steps[env_idx] = 0
                        # Reset environment
                        reset_result = vec_env.env_method('reset', indices=[env_idx])
                        reset_obs, reset_info = reset_result[0]  # Unpack the tuple from the list
                        obs[env_idx] = reset_obs
                        episode_dones[env_idx] = False
                        # Clear previous observation since we're starting a new episode
                        prev_obs[env_idx] = None
                        prev_infos[env_idx] = None
                        prev_actions[env_idx] = None
                    else:
                        # Store current observation as previous for next iteration
                        prev_obs[env_idx] = obs[env_idx]
                        prev_infos[env_idx] = infos[env_idx]
                        prev_actions[env_idx] = policy_actions[env_idx]
                else:
                    # Store current observation as previous for next iteration (even if episode done)
                    if not episode_dones[env_idx]:
                        prev_obs[env_idx] = obs[env_idx]
                        prev_infos[env_idx] = infos[env_idx]
                        prev_actions[env_idx] = policy_actions[env_idx]
            
            # Update observation for next step
            obs = next_obs
    
    vec_env.close()
    
    # Convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    
    return states, actions, next_states, all_rewards


def save_transitions_to_zarr(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    output_path: str,
    append: bool = False,
) -> str:
    """
    Save transition data to zarr format compatible with inverse dynamics training.
    
    Args:
        states: State array (s)
        actions: Action array (a)
        next_states: Next state array (s')
        output_path: Path to save the zarr file
        append: Whether to append to existing file
        
    Returns:
        Path to the saved zarr file
    """
    if zarr is None:
        raise ImportError("zarr is required for saving datasets. Please install with: pip install zarr")
    
    # Check if file exists and we want to append
    file_exists = os.path.exists(output_path)
    
    if append and file_exists:
        # Open existing zarr store in read/write mode
        store = zarr.open(output_path, mode='r+')
        
        # Get existing data
        data_group = store['data']
        meta_group = store['meta']
        
        # Get existing data and append
        existing_states = data_group['state'][:]
        existing_actions = data_group['action'][:]
        existing_next_states = data_group['next_state'][:]
        
        combined_states = np.concatenate([existing_states, states], axis=0)
        combined_actions = np.concatenate([existing_actions, actions], axis=0)
        combined_next_states = np.concatenate([existing_next_states, next_states], axis=0)
        
        # Resize and update arrays
        data_group['state'].resize(combined_states.shape)
        data_group['state'][:] = combined_states
        
        data_group['action'].resize(combined_actions.shape)
        data_group['action'][:] = combined_actions
        
        data_group['next_state'].resize(combined_next_states.shape)
        data_group['next_state'][:] = combined_next_states
        
        # Update metadata
        meta_group['num_samples'][:] = [len(combined_states)]
        
        print(f"Appended {len(states)} new transitions to existing file")
        print(f"Total transitions: {len(combined_states)}")
        
    else:
        # Create new zarr store (overwrites if exists)
        store = zarr.open(output_path, mode='w')
        
        # Create data group
        data_group = store.create_group('data')
        
        # Save states, actions, and next states
        data_group.create_array('state', data=states.astype(np.float32))
        data_group.create_array('action', data=actions.astype(np.float32))
        data_group.create_array('next_state', data=next_states.astype(np.float32))
        
        # Create meta group
        meta_group = store.create_group('meta')
        
        # Save dataset info
        meta_group.create_array('num_samples', data=np.array([len(states)]))
        
        print(f"Total transitions: {len(states)}")
        print(f"State shape: {states.shape}")
        print(f"Action shape: {actions.shape}")
        print(f"Next state shape: {next_states.shape}")
    
    return output_path


def create_output_path_from_config(config):
    """Create output path by replacing 'sequence' with 'transitions' in the buffer path."""
    buffer_path = config['buffer_path']
    output_path = buffer_path.replace('/sequence/', '/transitions/')
    return output_path


def collect_transition_dataset_parallel(config_path: str, n_envs: int = 4) -> str:
    """
    Collect transition dataset by rolling out a policy in parallel and saving to zarr format.
    
    Args:
        config_path: Path to the dataset config YAML file
        n_envs: Number of parallel environments
        
    Returns:
        Path to the saved zarr file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_id = config['env_id']
    model_path = config['model_path']
    env_kwargs = config.get('env_kwargs', None)
    max_steps_per_episode = config.get('max_steps_per_episode', 1000)
    num_transitions = config.get('num_episodes', 1000) * max_steps_per_episode
    deterministic = config.get('deterministic', False)
    seed = config.get('seed', None)
    append = config.get('append', False)
    
    # Create output path
    output_path = create_output_path_from_config(config)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Collect transitions in parallel
    print(f"Collecting {num_transitions} transitions using {n_envs} parallel environments...")

    states, actions, next_states, all_rewards = collect_transitions_parallel(
        model_path=model_path,
        env_id=env_id,
        env_kwargs=env_kwargs,
        num_transitions=num_transitions,
        max_steps_per_episode=max_steps_per_episode,
        deterministic=deterministic,
        seed=seed,
        n_envs=n_envs,
    )
    
    # Save to zarr
    print(f"Saving transitions to {output_path}...")
    saved_path = save_transitions_to_zarr(
        states=states,
        actions=actions,
        next_states=next_states,
        output_path=output_path,
        append=append,
    )

    print(f"Mean reward: {np.mean(all_rewards):.3f}, Std reward: {np.std(all_rewards):.3f}, Max reward: {np.max(all_rewards):.3f}, Min reward: {np.min(all_rewards):.3f}")
    print(f"Transition dataset collection complete! Saved to: {saved_path}")

    # Create plots
    plot_path = os.path.join(os.path.dirname(output_path),'dataset_plots')
    os.makedirs(plot_path, exist_ok=True)
    
    plt.hist(actions.flatten(), label="actions", color="blue", alpha=0.5, bins=50)
    plt.legend()
    plt.title(f"Actions distribution ({len(actions)} transitions)")
    plt.savefig(os.path.join(plot_path, "collected_actions_dist.png"))
    plt.clf()

    plt.hist(all_rewards, label="rewards", color="red", alpha=0.5, bins=50)
    plt.legend()
    plt.title(f"Rewards distribution: ({len(all_rewards)} transitions)")
    plt.savefig(os.path.join(plot_path, "collected_rewards_dist.png"))
    plt.clf()

    # Plot actions over time for first 1000 transitions
    if len(actions) > 0:
        n_plot = min(1000, len(actions))
        plt.figure(figsize=(10, 6))
        plt.plot(actions[:n_plot], 'b-', linewidth=1, alpha=0.7, label='Actions')
        plt.xlabel('Transition Index')
        plt.ylabel('Action Value')
        plt.title(f'Actions over First {n_plot} Transitions - {env_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "actions_over_time.png"), dpi=150, bbox_inches='tight')
        plt.clf()
        print(f"Saved actions over time plot to: {os.path.join(plot_path, 'actions_over_time.png')}")

    return saved_path


def parse_args():
    parser = argparse.ArgumentParser(description="Collect transition dataset by rolling out an RL policy in parallel.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML file")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    
    collect_transition_dataset_parallel(config_path=args.config, n_envs=args.n_envs)
