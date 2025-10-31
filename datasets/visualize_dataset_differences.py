import zarr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Dict, Any

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import load_transitions


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
    

def plot_object_states(states: np.ndarray, object_state_indices = [17,18], output_path: str = None):
    """
    Plot the object states.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(states[:, object_state_indices[0]], states[:, object_state_indices[1]], s=.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
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
    states1, _, next_states1, _, _ = load_transitions(args.dataset1_path)
    states2, _, next_states2, _, _ = load_transitions(args.dataset2_path)    

    
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

    plot_object_states(states1, [17,18], "dataset_plots/object_states_1.png")
    plot_object_states(states2, [17,18], "dataset_plots/object_states_2.png")


if __name__ == "__main__":
    main()
