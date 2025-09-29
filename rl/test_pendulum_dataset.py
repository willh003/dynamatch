#!/usr/bin/env python3
"""
Test script to verify that PendulumDataset works correctly with collected data.
"""

import sys
import os
import yaml
import numpy as np

# Add the dynamics directory to the path so we can import the dataset classes
sys.path.append('/home/wph52/weird/dynamics')

def test_pendulum_dataset_loading():
    """Test that PendulumDataset can load the collected pendulum data."""
    print("Testing PendulumDataset loading...")
    
    try:
        # Import the dataset classes
        from datasets.pendulum import make_pendulum_dataset
        
        # Load config
        config_path = "/home/wph52/weird/dynamics/configs/dataset/pendulum.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Dataset parameters
        zarr_path = "/home/wph52/weird/dynamics/datasets/sequence/pendulum_buffer.zarr"
        
        # Create dataset using the factory function
        train_set, val_set = make_pendulum_dataset(
            name=config['name'],
            buffer_path=zarr_path,
            shape_meta=config['shape_meta'],
            seq_len=config['seq_len'],
            history_len=config['history_len'],
            normalize_action=config['normalize_action'],
            normalize_lowdim=config['normalize_lowdim'],
            val_ratio=config['val_ratio'],
        )
        
        print(f"✓ PendulumDataset created successfully!")
        print(f"  Training set length: {len(train_set)}")
        print(f"  Validation set length: {len(val_set)}")
        print(f"  Buffer info: {train_set.buffer}")
        
        # Test that we can get samples from both datasets
        if len(train_set) > 0:
            print("\nTesting training set sample...")
            train_sample = train_set[0]
            print(f"  Training sample keys: {list(train_sample.keys())}")
            for key, value in train_sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        
        if len(val_set) > 0:
            print("\nTesting validation set sample...")
            val_sample = val_set[0]
            print(f"  Validation sample keys: {list(val_sample.keys())}")
            for key, value in val_sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        
        # Test that validation dataset is properly configured
        print(f"\n✓ Validation dataset is_validation: {val_set.is_validation}")
        print(f"✓ Training dataset is_validation: {train_set.is_validation}")
        
        return True
        
    except Exception as e:
        print(f"Error loading PendulumDataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_factory_function():
    """Test that the make_pendulum_dataset factory function works correctly."""
    print("\nTesting make_pendulum_dataset factory function...")
    
    try:
        from datasets.pendulum import make_pendulum_dataset
        
        # Load config
        config_path = "/home/wph52/weird/dynamics/configs/dataset/pendulum.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test with minimal parameters
        train_set, val_set = make_pendulum_dataset(
            name="test_pendulum",
            buffer_path="/home/wph52/weird/dynamics/datasets/pendulum/sequence/pendulum_buffer.zarr",
            shape_meta=config['shape_meta'],
            seq_len=10,  # Use smaller seq_len for testing
            history_len=1,
            normalize_action=False,
            normalize_lowdim=False,
            val_ratio=0.2,
        )
        
        print(f"✓ Factory function works correctly!")
        print(f"  Created training set: {type(train_set).__name__}")
        print(f"  Created validation set: {type(val_set).__name__}")
        print(f"  Both are PendulumDataset instances: {isinstance(train_set, type(val_set))}")
        
        return True
        
    except Exception as e:
        print(f"Error testing factory function: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing PendulumDataset implementation...")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Dataset loading
    if not test_pendulum_dataset_loading():
        all_passed = False
    
    # Test 2: Factory function
    if not test_dataset_factory_function():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! PendulumDataset is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
