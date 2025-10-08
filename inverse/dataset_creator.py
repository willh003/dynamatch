import sys
import os
import argparse
import numpy as np
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.trajectory_dataset import make_trajectory_dataset


from utils.config_utils import filter_config_with_debug, load_yaml_config
from envs.register_envs import register_custom_envs


def plot_action_distributions(actions, output_path='action_distributions.png'):
    """Plot action distributions."""
    plt.figure(figsize=(8, 5))
    
    plt.hist(actions, alpha=0.7, label='Actions', bins=50)
    plt.title(f'Action Distributions (total {len(actions)} samples)')
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Action distribution plots saved to {output_path}")


def load_sequence_dataset(dataset_config, state_keys):
    """Load sequence dataset for inverse dynamics training."""

    dataset_template_vars = {'num_frames': 18, 'obs_num_frames': 2}
    
    # Filter config to only include valid kwargs for make_trajectory_dataset
    filtered_config = filter_config_with_debug(dataset_config, make_trajectory_dataset, debug=True, template_vars=dataset_template_vars)

    print("\n=== Creating Dataset ===")
    train_set, val_set = make_trajectory_dataset(**filtered_config)
    
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    
    return train_set, val_set, state_keys


def get_inverse_dynamics_data(train_set, state_keys, max_samples=None):
    """Extract (s, a, s') triplets from sequence dataset."""
    print("=== Extracting Inverse Dynamics Data ===")
    
    if max_samples is None:
        max_samples = len(train_set)
    
    states = []
    actions = []
    next_states = []
    
    print("Processing dataset to extract (s, a, s') triplets")
    for i in tqdm(range(min(max_samples, len(train_set)))):
        seq = train_set[i]

        action = seq['action'][0]        
        state = []
        # reconstruct state from dataset obs
        for state_key in state_keys:
            state.append(seq['obs'][state_key][0])

        state = np.array(state).squeeze()

        next_state = []
        for state_key in state_keys:
            next_state.append(seq['obs'][state_key][1])
        next_state = np.array(next_state).squeeze()

        # Unnormalize the observations if normalizer exists
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

        # Unnormalize the action if normalizer exists
        if hasattr(train_set, 'action_normalizer'):
            action_unnorm = train_set.action_normalizer.reconstruct(action.cpu().numpy())
        else:
            action_unnorm = action.cpu().numpy()
        
        states.append(state_unnorm)
        actions.append(action_unnorm)
        next_states.append(next_state_unnorm)
        
        # Debug output for first few samples
        if i < 3:
            print(f"Sample {i}:")
            print(f"  State (unnormalized): {state_unnorm}")
            print(f"  Next state (unnormalized): {next_state_unnorm}")
            print(f"  Action: {action_unnorm}")
            print()

    return np.array(states), np.array(actions), np.array(next_states)


def create_inverse_dynamics_dataset(states, actions, next_states, output_path):
    """Create and save dataset with (s, a, s') triplets."""
    print("=== Creating Inverse Dynamics Dataset ===")
    
    # Create zarr store
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    
    print(f"Saved inverse dynamics dataset to {output_path}")
    print(f"Dataset size: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Next state shape: {next_states.shape}")
    
    return output_path


def create_output_path_from_config(config):
    """Create output path by replacing 'sequence' with 'inverse_dynamics' in the buffer path."""

    buffer_path = config['buffer_path']
    output_path = buffer_path.replace('/sequence/', '/transitions/')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create inverse dynamics dataset from sequence dataset')
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

    # Define state keys for environment
    state_keys = OrderedDict(config['shape_meta']['obs']).keys()
    
    # Load sequence dataset
    train_set, _, state_keys = load_sequence_dataset(config, state_keys)

    # Extract inverse dynamics data
    states, actions, next_states = get_inverse_dynamics_data(
        train_set, state_keys, args.max_samples
    )
    
    # Create and save inverse dynamics dataset
    create_inverse_dynamics_dataset(states, actions, next_states, output_path)
    
    # Plot action distributions
    plot_action_distributions(actions, os.path.join(os.path.dirname(output_path), 'action_distributions.png'))
    print("Inverse dynamics dataset creation completed successfully!")


if __name__ == "__main__":
    main()
