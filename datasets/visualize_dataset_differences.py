import zarr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Dict, Any


def load_zarr_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a zarr dataset and extract state and next_state data for (s, s') analysis.
    
    Args:
        dataset_path: Path to the zarr file
        
    Returns:
        states: Array of current states (N, state_dim)
        next_states: Array of next states (N, state_dim)
        metadata: Dictionary containing dataset metadata
    """
    print(f"Loading dataset from: {dataset_path}")
    
    store = zarr.open(dataset_path, mode='r')
    data_group = store['data']
    meta_group = store['meta']
    
    # Check if this is a transition dataset (with state/next_state keys)
    if 'state' in data_group and 'next_state' in data_group:
        # Transition dataset
        states = data_group['state'][:]
        next_states = data_group['next_state'][:]
        print(f"Loaded transition dataset with {len(states)} samples")
        print(f"State shape: {states.shape}")
        print(f"Next state shape: {next_states.shape}")
        metadata = {
            'type': 'transition',
            'num_samples': len(states)
        }
    else:
        # For trajectory datasets, we need to reconstruct (s, s') pairs from sequential observations
        obs_keys = [key for key in data_group.keys() if key.startswith('obs.')]
        if not obs_keys:
            raise ValueError(f"No observation keys found in dataset at {dataset_path}")
        
        # Use the first observation key
        obs_key = obs_keys[0]
        observations = data_group[obs_key][:]
        
        # Get episode ends to reconstruct state transitions
        if 'episode_ends' in meta_group:
            episode_ends = meta_group['episode_ends'][:]
            print(f"Loaded trajectory dataset with {len(observations)} observations across {len(episode_ends)} episodes")
            
            # Create (s, s') pairs by taking consecutive observations within episodes
            states = []
            next_states = []
            
            start_idx = 0
            for end_idx in episode_ends:
                episode_obs = observations[start_idx:end_idx]
                if len(episode_obs) > 1:
                    # Create pairs of consecutive observations
                    states.extend(episode_obs[:-1])
                    next_states.extend(episode_obs[1:])
                start_idx = end_idx
            
            states = np.array(states)
            next_states = np.array(next_states)
        else:
            # If no episode structure, assume all observations are sequential
            if len(observations) > 1:
                states = observations[:-1]
                next_states = observations[1:]
            else:
                raise ValueError("Not enough observations to create state transitions")
        
        print(f"Created {len(states)} state transition pairs")
        print(f"State shape: {states.shape}")
        print(f"Next state shape: {next_states.shape}")
        metadata = {
            'type': 'trajectory',
            'obs_key': obs_key,
            'num_samples': len(states),
            'all_obs_keys': obs_keys
        }
    
    return states, next_states, metadata


def plot_observation_distributions(obs1: np.ndarray, obs2: np.ndarray, 
                                 dataset1_name: str, dataset2_name: str,
                                 output_path: str = None):
    """
    Create subplots comparing the distribution of the first few observation dimensions.
    
    Args:
        obs1: First dataset observations (N1, obs_dim)
        obs2: Second dataset observations (N2, obs_dim)
        dataset1_name: Name for first dataset (for legend)
        dataset2_name: Name for second dataset (for legend)
        output_path: Optional path to save the plot
    """
    # Ensure we have at least 4 dimensions
    min_dims = min(obs1.shape[1], obs2.shape[1], 4)
    
    fig, axes = plt.subplots(1, min_dims, figsize=(5 * min_dims, 4))
    if min_dims == 1:
        axes = [axes]
    
    for i in range(min_dims):
        ax = axes[i]
        
        # Extract the i-th dimension from both datasets
        dim1 = obs1[:, i]
        dim2 = obs2[:, i]
        
        # Create histogram comparison
        ax.hist(dim1, bins=50, alpha=0.7, label=dataset1_name, density=True, color='blue')
        ax.hist(dim2, bins=50, alpha=0.7, label=dataset2_name, density=True, color='red')
        
        ax.set_xlabel(f'Observation Dimension {i+1}')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of Observation Dimension {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean1, std1 = np.mean(dim1), np.std(dim1)
        mean2, std2 = np.mean(dim2), np.std(dim2)
        
        ax.text(0.02, 0.98, f'{dataset1_name}:\nμ={mean1:.3f}, σ={std1:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.text(0.02, 0.75, f'{dataset2_name}:\nμ={mean2:.3f}, σ={std2:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Observation distribution plot saved to: {output_path}")
    else:
        plt.show()


def plot_state_transition_heatmaps(states1: np.ndarray, next_states1: np.ndarray,
                                  states2: np.ndarray, next_states2: np.ndarray,
                                  dataset1_name: str, dataset2_name: str,
                                  output_path: str = None):
    """
    Create 2D heatmaps showing (s, s') distributions for the first three state dimensions.
    
    Args:
        states1: First dataset current states (N1, state_dim)
        next_states1: First dataset next states (N1, state_dim)
        states2: Second dataset current states (N2, state_dim)
        next_states2: Second dataset next states (N2, state_dim)
        dataset1_name: Name for first dataset
        dataset2_name: Name for second dataset
        output_path: Optional path to save the plot
    """
    # Ensure we have at least 3 dimensions
    min_dims = min(states1.shape[1], states2.shape[1], 4)
    
    fig, axes = plt.subplots(2, min_dims, figsize=(5 * min_dims, 8))
    if min_dims == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min_dims):
        # Extract the i-th dimension from both datasets
        s1 = states1[:, i]
        s1_next = next_states1[:, i]
        s2 = states2[:, i]
        s2_next = next_states2[:, i]
        
        # Calculate common x and y ranges for both datasets
        x_min = min(np.min(s1), np.min(s2))
        x_max = max(np.max(s1), np.max(s2))
        y_min = min(np.min(s1_next), np.min(s2_next))
        y_max = max(np.max(s1_next), np.max(s2_next))
        
        # Create common bins for both datasets
        x_bins = np.linspace(x_min, x_max, 51)  # 51 edges = 50 bins
        y_bins = np.linspace(y_min, y_max, 51)  # 51 edges = 50 bins
        
        # Create heatmap for dataset 1
        ax1 = axes[0, i]
        h1, _, _ = np.histogram2d(s1, s1_next, bins=[x_bins, y_bins])
        im1 = ax1.imshow(h1.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                        cmap='Blues', aspect='auto')
        ax1.set_xlabel(f'Current State Dim {i+1}')
        ax1.set_ylabel(f'Next State Dim {i+1}')
        ax1.set_title(f'{dataset1_name}\n(s, s\') Distribution Dim {i+1}')
        plt.colorbar(im1, ax=ax1, label='Count')
        
        # Create heatmap for dataset 2
        ax2 = axes[1, i]
        h2, _, _ = np.histogram2d(s2, s2_next, bins=[x_bins, y_bins])
        im2 = ax2.imshow(h2.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                        cmap='Reds', aspect='auto')
        ax2.set_xlabel(f'Current State Dim {i+1}')
        ax2.set_ylabel(f'Next State Dim {i+1}')
        ax2.set_title(f'{dataset2_name}\n(s, s\') Distribution Dim {i+1}')
        plt.colorbar(im2, ax=ax2, label='Count')
        
        # Add statistics text
        stats1 = f'Dataset 1:\nN={len(s1)}\nμ_s={np.mean(s1):.3f}\nμ_s\'={np.mean(s1_next):.3f}'
        stats2 = f'Dataset 2:\nN={len(s2)}\nμ_s={np.mean(s2):.3f}\nμ_s\'={np.mean(s2_next):.3f}'
        
        ax1.text(0.02, 0.98, stats1, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.text(0.02, 0.98, stats2, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join('dataset_plots', 'state_transition_heatmaps.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    


def main():
    parser = argparse.ArgumentParser(description='Compare observation distributions and state transitions between two zarr datasets')
    parser.add_argument('--dataset1_path', type=str, help='Path to first zarr dataset')
    parser.add_argument('--dataset2_path', type=str, help='Path to second zarr dataset')
    parser.add_argument('--dataset1_name', type=str, default='Dataset 1', help='Name for first dataset')
    parser.add_argument('--dataset2_name', type=str, default='Dataset 2', help='Name for second dataset')
    parser.add_argument('--output_prefix', type=str, help='Output path prefix for the plots (will add suffixes)')
    parser.add_argument('--obs_output', type=str, help='Output path for observation distribution plot')
    parser.add_argument('--transition_output', type=str, help='Output path for state transition plot')
    
    args = parser.parse_args()
    
    # Load both datasets
    states1, next_states1, meta1 = load_zarr_dataset(args.dataset1_path)
    states2, next_states2, meta2 = load_zarr_dataset(args.dataset2_path)
    
    print(f"\nDataset 1 metadata: {meta1}")
    print(f"Dataset 2 metadata: {meta2}")
    
    # Set output paths - default to both plots if no specific output is provided
    if args.output_prefix:
        obs_output = f"{args.output_prefix}_observation_distributions.png"
        transition_output = f"{args.output_prefix}_state_transitions.png"
    elif args.obs_output or args.transition_output:
        # Use provided specific outputs, default the other to None (will show in window)
        obs_output = args.obs_output
        transition_output = args.transition_output
    else:
        # Default: save both plots with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_output = f"dataset_plots/observation_distributions_{timestamp}.png"
        transition_output = f"dataset_plots/state_transitions_{timestamp}.png"
    
    # Create the observation distribution comparison
    print("\nGenerating observation distribution plots...")
    plot_observation_distributions(
        states1, states2,  # Use states as observations for distribution analysis
        args.dataset1_name, args.dataset2_name,
        obs_output
    )
    
    # Create the state transition heatmap comparison
    print("\nGenerating state transition heatmaps...")
    plot_state_transition_heatmaps(
        states1, next_states1, states2, next_states2,
        args.dataset1_name, args.dataset2_name,
        transition_output
    )


if __name__ == "__main__":
    main()
