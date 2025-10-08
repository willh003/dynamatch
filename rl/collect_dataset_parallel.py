import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
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


def flatten_actions_list(actions_list: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a jagged list of action arrays into a single 1D array.
    
    Args:
        actions_list: List of action arrays with potentially different shapes
        
    Returns:
        Flattened 1D numpy array containing all actions
    """
    flattened = []
    for actions in actions_list:
        flattened.extend(actions.flatten())
    return np.array(flattened, dtype=np.float32)


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


def rollout_policy_parallel(
    model_path: str,
    env_id: str,
    env_kwargs: Dict[str, Any] = None,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    deterministic: bool = False,
    seed: Optional[int] = None,
    n_envs: int = 4,
) -> tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]:
    """
    Rollout a policy in parallel using vectorized environments and collect trajectories.
    
    Args:
        model_path: Path to the trained model .zip file
        env_id: Gymnasium environment ID
        env_kwargs: Additional environment arguments
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        n_envs: Number of parallel environments
        
    Returns:
        Tuple of (observations_list, actions_list) where each list contains
        numpy arrays for each episode
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, env_kwargs, seed) for _ in range(n_envs)])
    
    observations_list = []
    actions_list = []
    infos_list = []
    all_rewards = []
    
    # Initialize environments
    obs = vec_env.reset()
    episode_obs = [[] for _ in range(n_envs)]
    episode_actions = [[] for _ in range(n_envs)]
    episode_infos = [[] for _ in range(n_envs)]
    episode_rewards = [[] for _ in range(n_envs)]
    episode_dones = [False] * n_envs
    episode_count = 0
    
    # Collect episodes in parallel
    with tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        while episode_count < num_episodes:
            # Get actions from policy
            actions, _ = model.predict(obs, deterministic=deterministic)
            
            # Step all environments
            next_obs, rewards, dones, infos = vec_env.step(actions)
            
            # Store data for each environment
            for env_idx in range(n_envs):
                if not episode_dones[env_idx]:
                    episode_obs[env_idx].append(obs[env_idx])
                    episode_actions[env_idx].append(actions[env_idx])
                    episode_infos[env_idx].append(infos[env_idx])
                    episode_rewards[env_idx].append(rewards[env_idx])
                    
                    if dones[env_idx]:
                        # Episode finished for this environment
                        episode_dones[env_idx] = True
                        episode_count += 1
                        
                        # Convert to numpy arrays and store
                        if len(episode_obs[env_idx]) > 0:
                            episode_obs_array = np.array(episode_obs[env_idx], dtype=np.float32)
                            episode_actions_array = np.array(episode_actions[env_idx], dtype=np.float32)
                            
                            observations_list.append(episode_obs_array)
                            actions_list.append(episode_actions_array)
                            infos_list.append(episode_infos[env_idx])
                            all_rewards.extend(episode_rewards[env_idx])
                            
                            # Reset for next episode
                            episode_obs[env_idx] = []
                            episode_actions[env_idx] = []
                            episode_infos[env_idx] = []
                            episode_rewards[env_idx] = []
                            episode_dones[env_idx] = False
                        
                        pbar.update(1)
            
            # Update observation for next step
            obs = next_obs
    
    vec_env.close()
    return observations_list, infos_list, actions_list, all_rewards


def save_trajectories_to_zarr(
    observations_list: List[Dict[str, np.ndarray]],
    infos_list: List[Dict[str, np.ndarray]],
    actions_list: List[np.ndarray],
    output_path: str,
    shape_meta: Dict[str, Any],
    append: bool = False,
) -> str:
    """
    Save trajectory data to zarr format compatible with DroidDataset.
    
    Args:
        observations_list: List of observation dictionaries for each episode
        actions_list: List of action arrays for each episode
        output_path: Path to save the zarr file
        shape_meta: Shape metadata from config file
        append: Whether to append to existing file
        
    Returns:
        Path to the saved zarr file
    """
    if zarr is None:
        raise ImportError("zarr is required for saving datasets. Please install with: pip install zarr")
    # Flatten the trajectory data
    flat_observations = {key: [] for key in shape_meta["obs"].keys()}
    flat_actions = []
    episode_ends = []
    
    current_end = 0
    for traj_obs, traj_info, traj_actions in zip(observations_list, infos_list, actions_list):
        # Handle different observation types
        if isinstance(traj_obs, dict):
            # If observations are already in dictionary format
            for key in shape_meta["obs"].keys():
                if key in traj_obs:
                    flat_observations[key].extend(traj_obs[key])
                else:
                    # If observation key not found, use the raw observation
                    flat_observations[key].extend(traj_obs)
        else:
            # If observations are in array format (like InvertedPendulum-v5)
            # Parse the raw observation array into the expected dictionary format
            obs_dict = parse_raw_observations(traj_obs, traj_info, shape_meta["obs"])
            for key in shape_meta["obs"].keys():
                flat_observations[key].extend(obs_dict[key])
        
        flat_actions.extend(traj_actions)
        current_end += len(traj_actions)
        episode_ends.append(current_end)
    
    # Convert to numpy arrays
    for key in flat_observations:
        flat_observations[key] = np.array(flat_observations[key], dtype=np.float32)

        if np.isnan(flat_observations[key]).any():
            raise ValueError(f"NaN found in observations for key: {key}")
        
    flat_actions = np.array(flat_actions, dtype=np.float32)
    episode_ends = np.array(episode_ends, dtype=np.int64)
    
    # Check if file exists and we want to append
    file_exists = os.path.exists(output_path)
    
    if append and file_exists:
        # Open existing zarr store in read/write mode
        store = zarr.open(output_path, mode='r+')
        
        # Get existing data
        data_group = store['data']
        meta_group = store['meta']
        
        # Get existing episode ends and calculate offset
        existing_episode_ends = meta_group['episode_ends'][:]
        offset = existing_episode_ends[-1] if len(existing_episode_ends) > 0 else 0
        
        # Append new episode ends with offset
        new_episode_ends = episode_ends + offset
        combined_episode_ends = np.concatenate([existing_episode_ends, new_episode_ends])
        
        # Append new data to existing datasets
        for key, data in flat_observations.items():
            existing_data = data_group[f'obs.{key}'][:]
            combined_data = np.concatenate([existing_data, data], axis=0)
            data_group[f'obs.{key}'].resize(combined_data.shape)
            data_group[f'obs.{key}'][:] = combined_data
        
        existing_actions = data_group['action'][:]
        combined_actions = np.concatenate([existing_actions, flat_actions], axis=0)
        data_group['action'].resize(combined_actions.shape)
        data_group['action'][:] = combined_actions
        
        # Update episode ends
        meta_group['episode_ends'].resize(combined_episode_ends.shape)
        meta_group['episode_ends'][:] = combined_episode_ends
        
        print(f"Appended {len(flat_actions)} new timesteps to existing file")
        print(f"Total timesteps: {len(combined_actions)}")
        print(f"Episode ends: {combined_episode_ends}")
        
    else:
        # Create new zarr store (overwrites if exists)
        store = zarr.open(output_path, mode='w')
        
        # Create data group
        data_group = store.create_group('data')
        
        # Save observations
        for key, data in flat_observations.items():
            data_group.create_array(f'obs.{key}', data=data)
        
        # Save actions
        data_group.create_array('action', data=flat_actions)
        
        # Create meta group
        meta_group = store.create_group('meta')
        
        # Save episode ends
        meta_group.create_array('episode_ends', data=episode_ends)
        
        print(f"Total timesteps: {len(flat_actions)}")
        print(f"Episode ends: {episode_ends}")
        for key, data in flat_observations.items():
            print(f"Observation '{key}' shape: {data.shape}")
        print(f"Action shape: {flat_actions.shape}")
    
    return output_path


def collect_dataset_parallel(config_path: str, n_envs: int = 4) -> str:
    """
    Collect dataset by rolling out a policy in parallel and saving to zarr format.
    
    Args:
        config_path: Path to the dataset config YAML file
        n_envs: Number of parallel environments
        
    Returns:
        Path to the saved zarr file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    shape_meta = config['shape_meta']
    env_id = config['env_id']
    model_path = config['model_path']
    env_kwargs = config.get('env_kwargs', None)
    num_episodes = config.get('num_episodes', 100)
    max_steps_per_episode = config.get('max_steps_per_episode', 1000)
    deterministic = config.get('deterministic', False)
    seed = config.get('seed', None)
    append = config.get('append', False)
    
    # Rollout policy in parallel
    print(f"Rolling out policy for {num_episodes} episodes using {n_envs} parallel environments...")

    observations_list, infos_list, actions_list, all_rewards = rollout_policy_parallel(
        model_path=model_path,
        env_id=env_id,
        env_kwargs=env_kwargs,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        deterministic=deterministic,
        seed=seed,
        n_envs=n_envs,
    )
    
    # Save to zarr
    output_path = config['buffer_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving trajectories to {output_path}...")
    saved_path = save_trajectories_to_zarr(
        observations_list=observations_list,
        infos_list=infos_list,
        actions_list=actions_list,
        output_path=output_path,
        shape_meta=shape_meta,
        append=append,
    )

    print(f"Mean reward: {np.mean(all_rewards):.3f}, Std reward: {np.std(all_rewards):.3f}, Max reward: {np.max(all_rewards):.3f}, Min reward: {np.min(all_rewards):.3f}")
    print(f"Dataset collection complete! Saved to: {saved_path}")

    # Create plots
    plot_path = os.path.join(os.path.dirname(output_path),'dataset_plots')
    os.makedirs(plot_path, exist_ok=True)
    all_actions = flatten_actions_list(actions_list)
    
    plt.hist(all_actions, label="actions", color="blue", alpha=0.5)
    plt.legend()
    plt.title(f"Actions distribution ({len(all_actions)} samples)")
    plt.savefig(os.path.join(plot_path, "collected_actions_dist.png"))
    plt.clf()

    plt.hist(all_rewards, label="rewards", color="red", alpha=0.5)
    plt.legend()
    plt.title(f"Rewards distribution: ({len(all_rewards)} samples)")
    plt.savefig(os.path.join(plot_path, "collected_rewards_dist.png"))
    plt.clf()

    # Plot actions over the course of the first episode for pendulum environments
    if "pendulum" in env_id.lower():
        if len(actions_list) > 0:
            first_episode_actions = actions_list[0]
            plt.figure(figsize=(10, 6))
            plt.plot(first_episode_actions, 'b-', linewidth=2, label='Actions')
            plt.xlabel('Time Step')
            plt.ylabel('Action Value')
            plt.title(f'Actions over First Episode - {env_id}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, "first_episode_actions.png"), dpi=150, bbox_inches='tight')
            plt.clf()
            print(f"Saved first episode actions plot to: {os.path.join(plot_path, 'first_episode_actions.png')}")

    return saved_path


def parse_args():
    parser = argparse.ArgumentParser(description="Collect dataset by rolling out an RL policy in parallel.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML file")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    
    collect_dataset_parallel(config_path=args.config, n_envs=args.n_envs)
