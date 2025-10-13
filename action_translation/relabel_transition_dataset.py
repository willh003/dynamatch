import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from collections import OrderedDict
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config_utils import load_yaml_config
from inverse.physics_inverse_dynamics import PhysicsInverseDynamicsModel
from generative_policies.inverse_dynamics.interface import InverseDynamicsInterface
from envs.register_envs import register_custom_envs
from utils.model_utils import load_inverse_dynamics_model_from_config
from utils.data_utils import load_transition_dataset, get_transition_path_from_dataset_config, get_relabeled_actions_path_from_config
from torch.utils.data import TensorDataset

def get_original_actions(train_set, 
                        inverse_dynamics_model: Optional[InverseDynamicsInterface] = None, 
                        physics_inverse_dynamics_model: Optional[InverseDynamicsInterface] = None, max_samples=None):
    """
    Get original actions using inverse dynamics for each (s, s') pair.
    If no model is provided, the physics model is used for relabeling. If both are provided, the physics is used only for validation.
    Raises an error if no model is provided.
    Args:
        train_set: Train set
        state_keys: State keys
        inverse_dynamics_model: Inverse dynamics model
        physics_inverse_dynamics_model: Physics inverse dynamics model
        max_samples: Maximum number of samples to process
    """
    print("=== Computing Original Actions with Inverse Dynamics ===")
    
    if max_samples is None:
        max_samples = len(train_set)
    
    original_actions = []
    physics_original_actions = []
    shifted_actions = []
    states = []
    next_states = []
    validate_physics = True

    if inverse_dynamics_model is None and physics_inverse_dynamics_model is None:
        raise ValueError("Either inverse_dynamics_model or physics_inverse_dynamics_model must be provided")
    elif inverse_dynamics_model is None and physics_inverse_dynamics_model is not None:
        inverse_dynamics_model = physics_inverse_dynamics_model
    elif inverse_dynamics_model is not None and physics_inverse_dynamics_model is None:
        validate_physics = False
    
    print("Processing dataset and relabeling using inverse dynamics")
    for i in tqdm(range(min(max_samples, len(train_set)))):
        # get state, next_state, action from train set
        state, shifted_action, next_state = train_set[i]
        
        # Convert tensors to numpy arrays
        state_np = state.numpy() if hasattr(state, 'numpy') else state
        next_state_np = next_state.numpy() if hasattr(next_state, 'numpy') else next_state
        shifted_action_np = shifted_action.numpy() if hasattr(shifted_action, 'numpy') else shifted_action
        
        # run inverse dynamics on (state, next_state) to obtain original action
        try:
            # Use the inverse dynamics model to predict the original action
            original_action = inverse_dynamics_model.predict(
                state_np.reshape(1, -1), 
                next_state_np.reshape(1, -1)
            )
            original_action = original_action[0]  # Remove batch dimension

            if validate_physics:
                physics_original_action = physics_inverse_dynamics_model.predict(
                    state_np.reshape(1, -1),
                    next_state_np.reshape(1, -1)
                )
                physics_original_actions.append(physics_original_action[0])  # Remove batch
                
                if i < 3:
                    print(f"Sample {i}:")
                    print(f"  State: {state_np}")
                    print(f"  Next state: {next_state_np}")
                    print(f"  Action in dataset: {shifted_action_np}")
                    print(f"  Original action prediction: {original_action}")
                    print(f"  Physics original action: {physics_original_action}")
                    
            # Store the results
            states.append(state_np)
            next_states.append(next_state_np)
            original_actions.append(original_action)
            shifted_actions.append(shifted_action_np)
            
        except Exception as e:
            print(f"Warning: Inverse dynamics failed for sample {i}: {e}")
            continue

    if validate_physics:
        error = np.array(original_actions) - np.array(physics_original_actions)
        print(f"Mean Diff from physics ID: {np.mean(error)}, Std Diff from physics ID: {np.std(error)}, Max Diff from physics ID: {np.max(error)}, Min Diff from physics ID: {np.min(error)}")

    return np.array(states), np.array(original_actions), np.array(shifted_actions), np.array(next_states)


def create_action_translation_dataset(states, original_actions, shifted_actions, next_states, output_path):
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
    data_group.create_array('next_state', data=next_states.astype(np.float32))
    
    # Create meta group
    meta_group = store.create_group('meta')
    
    # Save dataset info
    meta_group.create_array('num_samples', data=np.array([len(states)]))
    
    print(f"Saved action translation dataset to {output_path}")
    print(f"Dataset size: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Original action shape: {original_actions.shape}")
    print(f"Shifted action shape: {shifted_actions.shape}")
    print(f"Next state shape: {next_states.shape}")
    
    return output_path


def plot_action_distributions(original_actions, shifted_actions, output_path):
    """
    Plot distributions of original_actions and shifted_actions for each action dimension.
    
    Args:
        original_actions: Array of original actions (n_samples, action_dim)
        shifted_actions: Array of shifted actions (n_samples, action_dim)
        output_path: Path where the plot should be saved
    """
    print("=== Plotting Action Distributions ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    action_dim = original_actions.shape[1]
    
    # Create subplots - 2 columns for original and shifted, rows for each action dimension
    fig, axes = plt.subplots(action_dim, 2, figsize=(12, 3 * action_dim))
    
    # If only one action dimension, ensure axes is 2D
    if action_dim == 1:
        axes = axes.reshape(1, -1)
    
    for dim in range(action_dim):
        # Plot original actions
        axes[dim, 0].hist(original_actions[:, dim], bins=50, alpha=0.7, density=True, 
                         label='Original Actions', color='blue')
        axes[dim, 0].set_title(f'Original Actions - Dimension {dim}')
        axes[dim, 0].set_xlabel('Action Value')
        axes[dim, 0].set_ylabel('Density')
        axes[dim, 0].grid(True, alpha=0.3)
        axes[dim, 0].legend()
        
        # Plot shifted actions
        axes[dim, 1].hist(shifted_actions[:, dim], bins=50, alpha=0.7, density=True, 
                         label='Shifted Actions', color='red')
        axes[dim, 1].set_title(f'Shifted Actions - Dimension {dim}')
        axes[dim, 1].set_xlabel('Action Value')
        axes[dim, 1].set_ylabel('Density')
        axes[dim, 1].grid(True, alpha=0.3)
        axes[dim, 1].legend()
        
        # Add statistics as text
        orig_mean = np.mean(original_actions[:, dim])
        orig_std = np.std(original_actions[:, dim])
        shift_mean = np.mean(shifted_actions[:, dim])
        shift_std = np.std(shifted_actions[:, dim])
        
        axes[dim, 0].text(0.02, 0.98, f'Mean: {orig_mean:.3f}\nStd: {orig_std:.3f}', 
                         transform=axes[dim, 0].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[dim, 1].text(0.02, 0.98, f'Mean: {shift_mean:.3f}\nStd: {shift_std:.3f}', 
                         transform=axes[dim, 1].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    plot_dir = os.path.dirname(output_path)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(plot_dir, 'action_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Action distributions plot saved to: {plot_path}")
    
    # Also create a comparison plot showing both distributions on the same axes
    fig2, axes2 = plt.subplots(action_dim, 1, figsize=(8, 3 * action_dim))
    if action_dim == 1:
        axes2 = [axes2]
    
    for dim in range(action_dim):
        axes2[dim].hist(original_actions[:, dim], bins=50, alpha=0.6, density=True, 
                       label='Original Actions', color='blue')
        axes2[dim].hist(shifted_actions[:, dim], bins=50, alpha=0.6, density=True, 
                       label='Shifted Actions', color='red')
        axes2[dim].set_title(f'Action Distributions Comparison - Dimension {dim}')
        axes2[dim].set_xlabel('Action Value')
        axes2[dim].set_ylabel('Density')
        axes2[dim].grid(True, alpha=0.3)
        axes2[dim].legend()
        
        # Add statistics
        orig_mean = np.mean(original_actions[:, dim])
        orig_std = np.std(original_actions[:, dim])
        shift_mean = np.mean(shifted_actions[:, dim])
        shift_std = np.std(shifted_actions[:, dim])
        
        axes2[dim].text(0.02, 0.98, 
                       f'Original: μ={orig_mean:.3f}, σ={orig_std:.3f}\n'
                       f'Shifted: μ={shift_mean:.3f}, σ={shift_std:.3f}', 
                       transform=axes2[dim].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_plot_path = os.path.join(plot_dir, 'action_distributions_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Action distributions comparison plot saved to: {comparison_plot_path}")
    
    plt.close('all')  # Close all figures to free memory


def main():
    parser = argparse.ArgumentParser(description='Relabel actions in transition dataset with inverse dynamics model')
    parser.add_argument('--model_config', type=str, default=None,
                       help='Path to inverse dynamics model config YAML file (if None, uses physics-based inverse dynamics)')
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--validate_physics_id', type=bool, default=False, help="Whether to validate physics ID (default: False)")
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--inverse_dynamics_env_id', type=str, default=None,
                       help='Environment ID for inverse dynamics model, only matters if validate_physics_id is True')
    args = parser.parse_args()

    register_custom_envs()
    
    # Load dataset config

    dataset_config_path = args.dataset_config
    dataset_config = load_yaml_config(dataset_config_path)
    dataset_path = get_transition_path_from_dataset_config(dataset_config_path)
    
    
    # Create output path
    output_path = get_relabeled_actions_path_from_config(dataset_config_path)
    print(f"Output path: {output_path}")

    # Set up inverse dynamics model
    inverse_dynamics_model = None
    if args.model_config is not None:        
        inverse_dynamics_model = load_inverse_dynamics_model_from_config(args.model_config, load_checkpoint=True)

    if args.validate_physics_id:
        inverse_dynamics_env_id = args.inverse_dynamics_env_id
        inverse_dynamics_env = gym.make(inverse_dynamics_env_id)
        physics_inverse_dynamics_model = PhysicsInverseDynamicsModel(inverse_dynamics_env)
    else:
        physics_inverse_dynamics_model = None
    
    # Load transition dataset
    states, actions, next_states = load_transition_dataset(dataset_path)


    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    next_states_tensor = torch.FloatTensor(next_states)

    train_set = TensorDataset(states_tensor, actions_tensor, next_states_tensor)

    # Get original actions using inverse dynamics
    states, original_actions, shifted_actions, next_states = get_original_actions(
        train_set, inverse_dynamics_model, physics_inverse_dynamics_model, args.max_samples
    )
    
    # Plot action distributions
    plot_action_distributions(original_actions, shifted_actions, output_path)
    
    # Create and save action translation dataset
    create_action_translation_dataset(states, original_actions, shifted_actions, next_states, output_path)
    print("Action translation dataset creation completed successfully!")


if __name__ == "__main__":
    main()

