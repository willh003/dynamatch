import argparse
import os
import sys

from dataset_creator import main as create_dataset
from trainer import main as train_model


def main():
    parser = argparse.ArgumentParser(description='Create action translation dataset and train action translator')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training')
    parser.add_argument('--skip_dataset_creation', action='store_true',
                       help='Skip dataset creation and use existing action translation dataset')
    
    args = parser.parse_args()
    
    # Step 1: Create action translation dataset (unless skipped)
    if not args.skip_dataset_creation:
        print("=== Step 1: Creating Action Translation Dataset ===")
        # Temporarily replace sys.argv to pass arguments to dataset_creator
        original_argv = sys.argv
        sys.argv = ['dataset_creator.py', '--config_path', args.config_path]
        if args.max_samples:
            sys.argv.extend(['--max_samples', str(args.max_samples)])
        
        try:
            create_dataset()
        finally:
            sys.argv = original_argv
    else:
        print("=== Skipping Dataset Creation ===")
        # Check if dataset exists by creating the expected path
        from dataset_creator import create_output_path_from_config
        dataset_path = create_output_path_from_config(args.config_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Action translation dataset not found at {dataset_path}")
    
    # Step 2: Train action translator
    print("\n=== Step 2: Training Action Translator ===")
    # Temporarily replace sys.argv to pass arguments to trainer
    original_argv = sys.argv
    sys.argv = ['trainer.py', '--config_path', args.config_path,
               '--num_epochs', str(args.num_epochs),
               '--learning_rate', str(args.learning_rate),
               '--batch_size', str(args.batch_size),
               '--device', args.device]
    
    try:
        train_model()
    finally:
        sys.argv = original_argv
    
    print("\n=== Complete Pipeline Finished Successfully! ===")


if __name__ == "__main__":
    main()