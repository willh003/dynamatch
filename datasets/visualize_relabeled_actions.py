import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import get_relabeled_actions_path_from_config
from action_translation.train import load_action_translation_dataset
from action_translation.relabel_transition_dataset import plot_action_correlations


def main():
    parser = argparse.ArgumentParser(description='Train action translator from action translation dataset')
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--id_model_name', type=str, required=False, default=None,
                       help='Name of the inverse dynamics model')
    
    args = parser.parse_args()
    
    # Initialize wandb if not disabled
    # Extract config name for run name
    
    # Create paths based on config
    dataset_path = get_relabeled_actions_path_from_config(args.dataset_config, args.id_model_name)
    
    print(f"Dataset path: {dataset_path}")
    
    # Load action translation dataset
    states, original_actions, shifted_actions = load_action_translation_dataset(dataset_path)

    # Create output directory and path
    output_dir = 'dataset_plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'action_correlations_{args.id_model_name}.png')
    
    # Generate the correlation plot
    plot_action_correlations(original_actions, shifted_actions, output_path)



if __name__ == "__main__":
    main()
