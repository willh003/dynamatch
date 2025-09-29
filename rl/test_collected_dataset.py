#!/usr/bin/env python3
"""
Test script to verify that collected datasets can be loaded by DroidDataset.
This ensures compatibility between collect_dataset.py and the dataset loader.
"""

import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Try to import zarr, fall back to alternative if not available
try:
    import zarr
except ImportError:
    print("Warning: zarr not available. Please install with: pip install zarr")
    zarr = None

# Add the dynamics directory to the path so we can import the dataset classes
sys.path.append('/home/wph52/weird/dynamics')

def test_zarr_format(zarr_path: str, config_path: str):
    """Test that the zarr file has the correct format."""
    print(f"Testing zarr format: {zarr_path}")
    
    if zarr is None:
        print("Error: zarr is required for testing. Please install with: pip install zarr")
        return False
    
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr file not found at {zarr_path}")
        return False
    
    # Load config to understand expected structure
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    shape_meta = config['shape_meta']
    
    try:
        # Open zarr file
        store = zarr.open(zarr_path, mode='r')
        
        # Check data group exists
        if 'data' not in store:
            print("Error: 'data' group not found in zarr file")
            return False
        
        data_group = store['data']
        
        # Check observations
        for key, attr in shape_meta['obs'].items():
            obs_key = f'obs.{key}'
            if obs_key not in data_group:
                print(f"Error: Observation '{obs_key}' not found in zarr file")
                return False
            
            obs_data = data_group[obs_key]
            expected_shape = tuple(attr['shape'])
            if len(obs_data.shape) != len(expected_shape) + 1:  # +1 for time dimension
                print(f"Error: Observation '{obs_key}' has wrong number of dimensions")
                print(f"Expected: {len(expected_shape) + 1}, Got: {len(obs_data.shape)}")
                return False
            
            print(f"✓ Observation '{obs_key}': shape {obs_data.shape}, dtype {obs_data.dtype}")
        
        # Check actions
        if 'action' not in data_group:
            print("Error: 'action' not found in zarr file")
            return False
        
        action_data = data_group['action']
        expected_action_shape = tuple(shape_meta['action']['shape'])
        if len(action_data.shape) != len(expected_action_shape) + 1:  # +1 for time dimension
            print(f"Error: Action has wrong number of dimensions")
            print(f"Expected: {len(expected_action_shape) + 1}, Got: {len(action_data.shape)}")
            return False
        
        print(f"✓ Action: shape {action_data.shape}, dtype {action_data.dtype}")
        
        # Check meta group
        if 'meta' not in store:
            print("Error: 'meta' group not found in zarr file")
            return False
        
        meta_group = store['meta']
        
        # Check episode_ends
        if 'episode_ends' not in meta_group:
            print("Error: 'episode_ends' not found in meta group")
            return False
        
        episode_ends = meta_group['episode_ends']
        print(f"✓ Episode ends: shape {episode_ends.shape}, dtype {episode_ends.dtype}")
        print(f"  Number of episodes: {episode_ends.shape[0]}")
        print(f"  Total timesteps: {episode_ends[-1] if episode_ends.shape[0] > 0 else 0}")
        
        # Verify episode_ends are consistent with data length
        if episode_ends.shape[0] > 0:
            total_timesteps = episode_ends[-1]
            action_length = action_data.shape[0]
            if total_timesteps != action_length:
                print(f"Error: Episode ends inconsistent with action data length")
                print(f"Episode ends total: {total_timesteps}, Action data length: {action_length}")
                return False
        
        print("✓ Zarr format validation passed!")
        return True
        
    except Exception as e:
        print(f"Error reading zarr file: {e}")
        return False

def test_droid_dataset_loading(zarr_path: str, config_path: str):
    """Test that the zarr file can be loaded by DroidDataset."""
    print(f"\nTesting DroidDataset loading: {zarr_path}")
    
    try:
        # Import the dataset classes
        from datasets.droid import DroidDataset
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create dataset
        dataset = DroidDataset(
            name=config['name'],
            buffer_path=zarr_path,
            shape_meta=config['shape_meta'],
            seq_len=config['seq_len'],
            history_len=config['history_len'],
            normalize_action=config['normalize_action'],
            normalize_lowdim=config['normalize_lowdim'],
            val_ratio=config['val_ratio'],
        )
        
        print(f"✓ Dataset created successfully!")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Buffer info: {dataset.buffer}")
        
        # Try to get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample loaded successfully!")
            print(f"  Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print("Warning: Dataset is empty")
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Test files
    zarr_path = "/home/wph52/weird/dynamics/datasets/sequence/pendulum_buffer.zarr"
    config_path = "/home/wph52/weird/dynamics/configs/dataset/pendulum.yaml"
    
    print("Testing collected dataset compatibility...")
    print("=" * 50)
    
    # Test 1: Zarr format
    zarr_ok = test_zarr_format(zarr_path, config_path)
    
    if zarr_ok:
        # Test 2: DroidDataset loading
        dataset_ok = test_droid_dataset_loading(zarr_path, config_path)
        
        if dataset_ok:
            print("\n" + "=" * 50)
            print("✓ All tests passed! Dataset is compatible with DroidDataset.")
        else:
            print("\n" + "=" * 50)
            print("✗ Dataset loading failed.")
    else:
        print("\n" + "=" * 50)
        print("✗ Zarr format validation failed.")

if __name__ == "__main__":
    main()
