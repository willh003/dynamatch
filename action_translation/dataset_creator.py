import sys
import os
import argparse
import yaml
import numpy as np
import zarr
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.trajectory_dataset import make_trajectory_dataset
from utils.config_utils import filter_config_with_debug, load_yaml_config
from inverse.physics_inverse_dynamics import gym_inverse_dynamics
from envs.register_envs import register_custom_envs



def plot_action_distributions(original_actions, shifted_actions, output_path='action_distributions.png'):
    """Plot action distributions and correlation."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(original_actions, alpha=0.7, label='Original Actions (ID)', bins=50)
    plt.hist(shifted_actions, alpha=0.7, label='Shifted Actions', bins=50)
    plt.title(f'Action Distributions (total {len(original_actions)} samples)')
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(original_actions, shifted_actions, alpha=0.6)
    plt.plot([original_actions.min(), original_actions.max()], 
             [original_actions.min(), original_actions.max()], 'r--', label='y=x')
    plt.title('Original vs Shifted Actions')
    plt.xlabel('Original Action')
    plt.ylabel('Shifted Action')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Action distribution plots saved to {output_path}")


def load_shifted_dynamics_dataset(dataset_config, state_keys):
    """Load dataset with shifted dynamics."""

    dataset_template_vars = {'num_frames': 18, 'obs_num_frames': 2}
    
    # Filter config to only include valid kwargs for make_trajectory_dataset
    filtered_config = filter_config_with_debug(dataset_config, make_trajectory_dataset, debug=True, template_vars=dataset_template_vars)

    print("\n=== Creating Dataset ===")
    train_set, val_set = make_trajectory_dataset(**filtered_config)
    
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    
    return train_set, val_set, state_keys


def get_original_actions(train_set, state_keys, inverse_dynamics_env, max_samples=None):
    """Get original actions using inverse dynamics for each (s, s') pair."""
    print("=== Computing Original Actions with Inverse Dynamics ===")
    
    if max_samples is None:
        max_samples = len(train_set)
    
    original_actions = []
    shifted_actions = []
    states = []
    next_states = []
    
    print("Processing dataset and relabeling using inverse dynamics")
    for i in tqdm(range(min(max_samples, len(train_set)))):
        seq = train_set[i]

        shifted_action = seq['action'][0]        
        state = []
        # reconstruct state from dataset obs
        for state_key in state_keys:
            state.append(seq['obs'][state_key][0])

        state = np.array(state).squeeze()

        next_state = []
        for state_key in state_keys:
            next_state.append(seq['obs'][state_key][1])
        next_state = np.array(next_state).squeeze()

        # Unnormalize the observations before passing to inverse dynamics
        if hasattr(train_set, 'lowdim_normalizer'):
            state_unnorm = []
            next_state_unnorm = []
            for j, state_key in enumerate(state_keys):
                state_unnorm.append(train_set.lowdim_normalizer[state_key].reconstruct(state[j:j+1]))
                next_state_unnorm.append(train_set.lowdim_normalizer[state_key].reconstruct(next_state[j:j+1]))
            state_unnorm = np.array(state_unnorm).squeeze()
            next_state_unnorm = np.array(next_state_unnorm).squeeze()
        else:
            state_unnorm = state
            next_state_unnorm = next_state

        # Get original action using inverse dynamics
        original_action = gym_inverse_dynamics(inverse_dynamics_env, state_unnorm, next_state_unnorm)
        
        # Unnormalize the shifted action from the dataset for fair comparison
        if hasattr(train_set, 'action_normalizer'):
            shifted_action_unnorm = train_set.action_normalizer.reconstruct(shifted_action.cpu().numpy())
        else:
            shifted_action_unnorm = shifted_action.cpu().numpy()
        
        
        original_actions.append(original_action)
        shifted_actions.append(shifted_action_unnorm)
        states.append(state_unnorm)
        next_states.append(next_state_unnorm)
        
        # Debug output for first few samples
        if i < 3:
            print(f"Sample {i}:")
            print(f"  State (unnormalized): {state_unnorm}")
            print(f"  Next state (unnormalized): {next_state_unnorm}")
            print(f"  Action in dataset: {shifted_action_unnorm}")
            print(f"  ID predicted action: {original_action}")
            print(f"  Error: {np.linalg.norm(shifted_action_unnorm - original_action):.4f}")
            print(f"  MAPE: {np.mean(np.abs((shifted_action_unnorm - original_action)/shifted_action_unnorm))*100:.4f}%")
            print()

    return np.array(states), np.array(original_actions), np.array(shifted_actions)


def create_action_translation_dataset(states, original_actions, shifted_actions, output_path):
    """Create and save dataset with (s, a_original, a_shifted) pairs."""
    print("=== Creating Action Translation Dataset ===")
    
    # Create zarr store
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    store = zarr.open(output_path, mode='w')
    
    # Create data group
    data_group = store.create_group('data')
    
    # Save states
    data_group.create_array('state', data=states.astype(np.float32))
    
    # Save original and shifted actions
    data_group.create_array('original_action', data=original_actions.astype(np.float32))
    data_group.create_array('shifted_action', data=shifted_actions.astype(np.float32))
    
    # Create meta group
    meta_group = store.create_group('meta')
    
    # Save dataset info
    meta_group.create_array('num_samples', data=np.array([len(states)]))
    
    print(f"Saved action translation dataset to {output_path}")
    print(f"Dataset size: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Original action shape: {original_actions.shape}")
    print(f"Shifted action shape: {shifted_actions.shape}")
    
    return output_path


def create_output_path_from_config(config):
    """Create output path by replacing 'sequence' with 'relabeled_actions' in the buffer path."""

    buffer_path = config['buffer_path']
    # Replace 'sequence' with 'relabeled_actions' in the path
    output_path = buffer_path.replace('/sequence/', '/relabeled_actions/')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create action translation dataset from sequence dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    args = parser.parse_args()

    register_custom_envs()

    # Load config
    config = load_yaml_config(args.config)
    
    # Create output path from config
    output_path = create_output_path_from_config(config)
    print(f"Output path: {output_path}")

    # Define state keys for pendulum environment
    state_keys = OrderedDict(config['shape_meta']['obs']).keys()
    
    # Load shifted dynamics dataset
    train_set, _, state_keys = load_shifted_dynamics_dataset(config, state_keys)

    # Create inverse dynamics environment
    inverse_dynamics_env_id = config['source_env_id']
    inverse_dynamics_env = gym.make(inverse_dynamics_env_id)
    
    # Get original actions using inverse dynamics
    states, original_actions, shifted_actions = get_original_actions(
        train_set, state_keys, inverse_dynamics_env, args.max_samples
    )
    
    # Create and save action translation dataset
    create_action_translation_dataset(states, original_actions, shifted_actions, output_path)
    

    plot_action_distributions(original_actions, shifted_actions, os.path.join(os.path.dirname(output_path), 'action_distributions.png'))
    print("Action translation dataset creation completed successfully!")


if __name__ == "__main__":
    main()
