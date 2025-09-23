import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from cluster_utils import set_cluster_graphics_vars
from utils import modify_env_gravity
from register_envs import register_custom_envs
from tqdm import tqdm
# Try to import zarr, fall back to alternative if not available
try:
    import zarr
except ImportError:
    print("Warning: zarr not available. Please install with: pip install zarr")
    zarr = None


def rollout_policy(
    model_path: str,
    env_id: str,
    env_kwargs: Dict[str, Any] = None,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]:
    """
    Rollout a policy in the given environment and collect trajectories.
    
    Args:
        model_path: Path to the trained model .zip file
        env_id: Gymnasium environment ID
        env_kwargs: Additional environment arguments
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        
    Returns:
        Tuple of (observations_list, actions_list) where each list contains
        numpy arrays for each episode
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make(env_id, **env_kwargs)
    
    observations_list = []
    actions_list = []
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=seed)
        episode_obs = []
        episode_actions = []
        
        for step in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=deterministic)
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
        
        # Convert to numpy arrays
        episode_obs = np.array(episode_obs, dtype=np.float32)
        episode_actions = np.array(episode_actions, dtype=np.float32)
        
        observations_list.append(episode_obs)
        actions_list.append(episode_actions)
        

    env.close()
    return observations_list, actions_list


def save_trajectories_to_zarr(
    observations_list: List[Dict[str, np.ndarray]],
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
    for traj_obs, traj_actions in zip(observations_list, actions_list):
        # Handle different observation types
        for key in shape_meta["obs"].keys():
            if key in traj_obs:
                flat_observations[key].extend(traj_obs[key])
            else:
                # If observation key not found, use the raw observation
                flat_observations[key].extend(traj_obs)
        
        flat_actions.extend(traj_actions)
        current_end += len(traj_actions)
        episode_ends.append(current_end)
    
    # Convert to numpy arrays
    for key in flat_observations:
        flat_observations[key] = np.array(flat_observations[key], dtype=np.float32)
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


def collect_dataset(
    model_path: str,
    env_id: str,
    config_path: str,
    output_path: str,
    env_kwargs: Dict[str, Any] = None,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    seed: Optional[int] = None,
    append: bool = False,
) -> str:
    """
    Collect dataset by rolling out a policy and saving to zarr format.
    
    Args:
        model_path: Path to the trained model .zip file
        env_id: Gymnasium environment ID
        config_path: Path to the dataset config YAML file
        output_path: Path to save the zarr file
        env_kwargs: Additional environment arguments
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        append: Whether to append to existing file
        
    Returns:
        Path to the saved zarr file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    shape_meta = config['shape_meta']
    
    # Rollout policy
    print(f"Rolling out policy for {num_episodes} episodes...")
    observations_list, actions_list = rollout_policy(
        model_path=model_path,
        env_id=env_id,
        env_kwargs=env_kwargs,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        deterministic=deterministic,
        seed=seed,
    )
    
    # Save to zarr
    print(f"Saving trajectories to {output_path}...")
    saved_path = save_trajectories_to_zarr(
        observations_list=observations_list,
        actions_list=actions_list,
        output_path=output_path,
        shape_meta=shape_meta,
        append=append,
    )
    
    print(f"Dataset collection complete! Saved to: {saved_path}")
    return saved_path


def parse_args():
    parser = argparse.ArgumentParser(description="Collect dataset by rolling out an RL policy.")
    parser.add_argument("--model", required=True, help="Path to the trained model .zip file")
    parser.add_argument("--env_id", required=True, help="Gymnasium environment ID")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML file")
    parser.add_argument("--output", required=True, help="Path to save the zarr file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for resets")
    parser.add_argument("--append", action="store_true", help="Append to existing zarr file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    register_custom_envs()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    collect_dataset(
        model_path=args.model,
        env_id=args.env_id,
        config_path=args.config,
        output_path=args.output,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
        append=args.append,
    )
