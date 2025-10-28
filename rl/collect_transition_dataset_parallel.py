import os
import argparse
import yaml
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym

from cluster_utils import set_cluster_graphics_vars
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.register_envs import register_custom_envs
from utils.data_utils import get_transition_path_from_dataset_config
from tqdm import tqdm
import zarr
from envs.env_utils import get_state_from_obs

import gymnasium as gym
from inverse.physics_inverse_dynamics import gym_inverse_dynamics
from envs.register_envs import register_custom_envs
from envs.env_utils import make_vec_env
from utils.model_utils import load_source_policy_from_config
register_custom_envs()


def resolve_hydra_config_and_get_source_policy_config(config_path, source_policy_checkpoint=None):
    """
    Resolve a Hydra config and return the complete source policy config.
    Reused from eval_policy.py to maintain consistency.
    
    Returns:
        dict: Complete source policy configuration including _target_, checkpoint_path, etc.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Get the config directory to resolve relative paths
        config_dir = os.path.dirname(config_path)
        
        # Load source policy config
        source_policy_name = config['defaults'][0]['source_policy']
        # Navigate up to the configs directory, then to source_policy
        base_policy_config_path = os.path.join(config_dir, '..', '..', 'source_policy', f'{source_policy_name}.yaml')
        base_policy_config_path = os.path.normpath(base_policy_config_path)
        
        with open(base_policy_config_path, 'r', encoding='utf-8') as f:
            base_policy_config = yaml.safe_load(f)
        
        # Override checkpoint path if provided
        if source_policy_checkpoint:
            base_policy_config['checkpoint_path'] = source_policy_checkpoint
            
        return base_policy_config
            
    except Exception as e:
        print(f"Warning: Could not resolve Hydra config: {e}")
        # Fallback to basic config if resolution fails
        if source_policy_checkpoint:
            return {
                'checkpoint_path': source_policy_checkpoint,
                '_target_': 'stable_baselines3.PPO',
            }
        else:
            raise ValueError(f"Could not resolve source policy config: {e}")


def get_state_from_vec_env(obs, infos, env_id: str) -> np.ndarray:
    """
    Get state from vectorized environment.
    """
    if isinstance(obs, dict):
        return get_state_from_obs(obs, infos, env_id)
    
    n_envs = len(obs)
    states = []
    for env_idx in range(n_envs):
        state = get_state_from_obs(obs[env_idx], infos[env_idx], env_id)
        states.append(state)
    return np.array(states)

def collect_transitions_parallel(
    source_policy_config: Dict[str, Any],
    env_id: str,
    env_kwargs: Dict[str, Any] = None,
    num_transitions: int = 10000,
    max_steps_per_episode: int = 1000,
    deterministic: bool = False,
    seed: Optional[int] = None,
    validate_physics_id: bool = False,
    addl_noise_std: float = 0.0,
    n_envs: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect transitions (s, a, s') in parallel using vectorized environments.
    
    Args:
        source_policy_config: Config for the source policy
        env_id: Gymnasium environment ID
        env_kwargs: Additional environment arguments
        num_transitions: Number of transitions to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        validate_physics_id: Whether to validate using physics ID
        addl_noise_std: Standard deviation of additional noise to add to the actions (if 0, then no noise is added)
        n_envs: Number of parallel environments
        
    Returns:
        Tuple of (states, actions, next_states) as numpy arrays
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Load the trained model
    model = load_source_policy_from_config(source_policy_config)

    if validate_physics_id:
        physics_env = gym.make(env_id)
    
    # Create vectorized environment
    vec_env = make_vec_env(env_id, n_envs=n_envs, **env_kwargs)
    
    all_states = []
    all_next_states = []
    all_actions = []
    all_rewards = []
    all_gt_id_errors = []
    
    # Initialize environments
    obs = vec_env.reset()
    infos = vec_env.reset_infos

    episode_steps = np.zeros(n_envs)
    
    # Collect transitions in parallel
    with tqdm(total=num_transitions, desc="Collecting transitions") as pbar:
        while len(all_states) < num_transitions:
            policy_actions, _ = model.predict(obs, deterministic=deterministic)

            if addl_noise_std > 0:
                policy_actions = policy_actions + np.random.normal(loc=0.0, scale=addl_noise_std, size=policy_actions.shape)

            next_obs, rewards, dones, next_infos = vec_env.step(policy_actions)
            
            states = get_state_from_vec_env(obs, infos, env_id)
            next_states = get_state_from_vec_env(next_obs, next_infos, env_id)
            episode_steps += 1

            for env_idx in range(n_envs):

                if not dones[env_idx]:
                    
                    state = states[env_idx]
                    next_state = next_states[env_idx]
                    action = policy_actions[env_idx]
                    
                    all_states.append(state)
                    all_next_states.append(next_state)
                    all_actions.append(action)
                    all_rewards.append(rewards[env_idx])
                    
                    if validate_physics_id:
                        error = get_physics_id_error(physics_env, state, next_state, action)
                        all_gt_id_errors.append(error)
                    else:
                        all_gt_id_errors.append(0.0)
                        
                    pbar.update(1)
                else:
                    episode_steps[env_idx] = 0

            if any(episode_steps >= max_steps_per_episode):
                # If any reach max steps, reset all environments
                next_obs = vec_env.reset()
                next_infos = vec_env.reset_infos
                episode_steps = np.zeros(n_envs)

            obs = next_obs
            infos = next_infos

    vec_env.close()

    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_gt_id_errors = np.array(all_gt_id_errors, dtype=np.float32)

    return all_states, all_actions, all_next_states, all_rewards, all_gt_id_errors


def get_physics_id_error(physics_env, state, next_state, action):
    """
    Get the physics ID error for a given set of states, next states, and actions.
    """
    id_action = gym_inverse_dynamics(physics_env, state, next_state)
    error = np.linalg.norm(action - id_action)
    return error

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
        store['meta']['num_samples'][:] = [len(combined_states)]
        
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
    buffer_dir = config['buffer_dir']
    output_path = buffer_dir.replace('/sequence/', '/transitions/')
    return output_path


def validate_id_on_dataset(dataset_path, env_id, max_samples=1000):
    """Load inverse dynamics dataset from zarr file."""
    print("=== Loading Inverse Dynamics Dataset ===")
    
    store = zarr.open(dataset_path, mode='r')
    data_group = store['data']
    meta_group = store['meta']
    
    states = data_group['state'][:]
    actions = data_group['action'][:]
    next_states = data_group['next_state'][:]

    physics_env = gym.make(env_id)

    id_errors = []

    for state, action, next_state in tqdm(zip(states, actions, next_states), total=min(max_samples, len(states))):
        error = get_physics_id_error(physics_env, state, next_state, action)
        id_errors.append(error)

        if len(id_errors) >= max_samples:
            break    

    
    print(f"Mean ID error on {len(id_errors)} samples from loaded dataset: {np.mean(id_errors):.3e}, Std ID error: {np.std(id_errors):.3e}, Max ID error: {np.max(id_errors):.3e}, Min ID error: {np.min(id_errors):.3e}")
    
    return states, actions, next_states


def validate_dataset_saved_data(dataset_path, true_states,true_next_states, true_actions):
    
    store = zarr.open(dataset_path, mode='r')
    data_group = store['data']

    states = data_group['state'][:]
    actions = data_group['action'][:]
    next_states = data_group['next_state'][:]
    

    state_errors = states - true_states
    next_state_errors = next_states - true_next_states
    action_errors = actions - true_actions

    print(f"State errors: {np.linalg.norm(state_errors)}")
    print(f"Next state errors: {np.linalg.norm(next_state_errors)}")
    print(f"Action errors: {np.linalg.norm(action_errors)}")

def collect_transition_dataset_parallel(config_path: str, n_envs: int = 4, validate_physics_id: bool = False) -> str:
    """
    Collect transition dataset by rolling out a policy in parallel and saving to zarr format.
    
    Args:
        config_path: Path to the dataset config YAML file
        n_envs: Number of parallel environments
        
    Returns:
        Path to the saved zarr file
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    env_id = config['env_id']
    
    env_kwargs = config.get('env_kwargs', None)
    max_steps_per_episode = config.get('max_steps_per_episode', 1000)
    num_transitions = config.get('num_episodes', 1000) * max_steps_per_episode
    deterministic = config.get('deterministic', False)
    addl_noise_std = config.get('addl_noise_std', 0.0)
    seed = config.get('seed', None)
    append = config.get('append', False)
        
    # Create output path
    output_path = get_transition_path_from_dataset_config(config_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Collect transitions in parallel
    print(f"Collecting {num_transitions} transitions using {n_envs} parallel environments...")
    
    # Resolve source policy config using the same logic as eval_policy.py
    if 'defaults' in config and config['defaults']:
        # Hydra-style config with defaults
        source_policy_config = resolve_hydra_config_and_get_source_policy_config(config_path)
    elif 'source_policy' in config:
        # Direct source policy config
        source_policy_config = config['source_policy']
    elif 'model_path' in config:
        # Legacy model_path format
        source_policy_config = {
            'checkpoint_path': config['model_path'],
            '_target_': 'stable_baselines3.PPO',
        }
    else:
        raise ValueError("No valid source policy configuration found in config. Expected 'defaults', 'source_policy', or 'model_path'")
    
    states, actions, next_states, all_rewards, all_gt_id_errors = collect_transitions_parallel(
        source_policy_config=source_policy_config,
        env_id=env_id,
        env_kwargs=env_kwargs,
        num_transitions=num_transitions,
        max_steps_per_episode=max_steps_per_episode,
        deterministic=deterministic,
        seed=seed,
        n_envs=n_envs,
        validate_physics_id=validate_physics_id,
        addl_noise_std=addl_noise_std,
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


    # validate ID on the zarr file
    


    print(f"Transition dataset collection complete! Saved to: {saved_path}")


    print(f"Mean reward: {np.mean(all_rewards):.3f}, Std reward: {np.std(all_rewards):.3f}, Max reward: {np.max(all_rewards):.3f}, Min reward: {np.min(all_rewards):.3f}")

    validate_dataset_saved_data(saved_path, states, next_states, actions)
    if validate_physics_id:
        validate_id_on_dataset(saved_path, env_id)
    
    all_gt_id_errors = np.array(all_gt_id_errors)
    print(f"Mean Physics ID error: {np.mean(all_gt_id_errors):.3e}, Std Physics ID error: {np.std(all_gt_id_errors):.3e}, Max Physics ID error: {np.max(all_gt_id_errors):.3e}, Min Physics ID error: {np.min(all_gt_id_errors):.3e}")

    # Create plots
    plot_path = os.path.join(os.path.dirname(output_path),'dataset_plots')
    os.makedirs(plot_path, exist_ok=True)
    
    plt.hist(all_gt_id_errors, label="Physics ID errors", color="green", alpha=0.5, bins=50)
    plt.legend()
    plt.title(f"Physics ID errors distribution: ({len(all_gt_id_errors)} transitions)")
    plt.savefig(os.path.join(plot_path, "collected_physics_id_errors_dist.png"))
    plt.clf()
    
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
    parser.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments (default: 4)")
    parser.add_argument("--validate_physics_id", type=bool, default=False, help="Whether to validate physics ID (default: False)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    
    collect_transition_dataset_parallel(config_path=args.config, n_envs=args.n_envs, validate_physics_id=args.validate_physics_id)
