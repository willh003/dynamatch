#!/usr/bin/env python3
"""
Example script showing how to use collect_dataset.py to collect pendulum data.
This demonstrates the complete workflow from policy rollout to dataset creation.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
    env_id = "InvertedPendulum-v5"
    config_path = "/home/wph52/weird/dynamics/configs/dataset/pendulum.yaml"
    output_path = "/home/wph52/weird/dynamics/datasets/raw_data/pendulum_buffer.zarr"
    
    # Environment arguments (matching eval_rl.py)
    env_kwargs = {
        "action_add": 1.5
    }
    
    # Collection parameters
    num_episodes = 50
    max_steps_per_episode = 1000
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please update the model_path variable with a valid model file.")
        return
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "collect_dataset.py",
        "--model", model_path,
        "--env_id", env_id,
        "--config", config_path,
        "--output", output_path,
        "--episodes", str(num_episodes),
        "--max_steps", str(max_steps_per_episode),
        "--seed", "42",  # For reproducibility
    ]
    
    print("Collecting pendulum dataset...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the collection
    try:
        result = subprocess.run(cmd, check=True, cwd="/home/wph52/weird/dynamics/rl")
        print("\nDataset collection completed successfully!")
        print(f"Dataset saved to: {output_path}")
        
        # Verify the dataset was created
        if os.path.exists(output_path):
            print(f"Dataset file exists: {os.path.getsize(output_path)} bytes")
        else:
            print("Warning: Dataset file was not created")
            
    except subprocess.CalledProcessError as e:
        print(f"Error during dataset collection: {e}")
        return
    except FileNotFoundError:
        print("Error: collect_dataset.py not found. Make sure you're running from the correct directory.")
        return

if __name__ == "__main__":
    main()
