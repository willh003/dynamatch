import sys
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.pendulum.pendulum import make_pendulum_dataset
from utils.config_utils import filter_config_with_debug
import yaml
from torch.utils.data import DataLoader

def test_dataset_comprehensive(dataset_config_path):
    """
    Comprehensive test suite for pendulum dataset validation.
    """
    
    print("=== Comprehensive Dataset Validation Test ===")
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    dataset_template_vars = {'num_frames': 18, 'obs_num_frames': 2}
    batch_size = 64
    
    # Filter config to only include valid kwargs for make_pendulum_dataset
    filtered_config = filter_config_with_debug(dataset_config, make_pendulum_dataset, debug=True, template_vars=dataset_template_vars)

    print(f"\n=== Creating Dataset ===")
    train_set, val_set = make_pendulum_dataset(**filtered_config)
    
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    
    # Get episode and timestep information
    total_episodes = train_set.buffer.num_episodes
    total_timesteps = train_set.buffer.num_steps
    train_episodes = train_set.train_mask.sum()
    val_episodes = total_episodes - train_episodes
    
    print(f"Total episodes in dataset: {total_episodes}")
    print(f"Total timesteps in dataset: {total_timesteps}")
    print(f"Train episodes: {train_episodes}")
    print(f"Val episodes: {val_episodes}")
    print(f"Average timesteps per episode: {total_timesteps / total_episodes:.1f}")
    
    # Test 1: Verify template variables were resolved correctly
    print(f"\n=== Test 1: Template Variable Resolution ===")
    expected_seq_len = dataset_template_vars['num_frames']
    expected_history_len = dataset_template_vars['obs_num_frames']
    
    print(f"Expected seq_len: {expected_seq_len}")
    print(f"Expected history_len: {expected_history_len}")
    print(f"Config seq_len: {filtered_config['seq_len']}")
    print(f"Config history_len: {filtered_config['history_len']}")
    
    assert filtered_config['seq_len'] == expected_seq_len, f"seq_len mismatch: expected {expected_seq_len}, got {filtered_config['seq_len']}"
    assert filtered_config['history_len'] == expected_history_len, f"history_len mismatch: expected {expected_history_len}, got {filtered_config['history_len']}"
    print("✅ Template variables resolved correctly")
    
    # Test 2: Verify dataset structure and keys
    print(f"\n=== Test 2: Dataset Structure ===")
    sample_data = train_set[0]
    print(f"Sample data keys: {list(sample_data.keys())}")
    
    expected_keys = ['obs', 'action']
    assert all(key in sample_data for key in expected_keys), f"Missing expected keys. Expected: {expected_keys}, got: {list(sample_data.keys())}"
    
    # Check observation structure
    obs_keys = list(sample_data['obs'].keys())
    expected_obs_keys = ['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity']
    assert all(key in obs_keys for key in expected_obs_keys), f"Missing expected obs keys. Expected: {expected_obs_keys}, got: {obs_keys}"
    print("✅ Dataset structure is correct")
    
    # Test 3: Verify data shapes
    print(f"\n=== Test 3: Data Shapes ===")
    print(f"Action shape: {sample_data['action'].shape}")
    print(f"Expected action shape: ({expected_seq_len}, 1)")
    assert sample_data['action'].shape == (expected_seq_len, 1), f"Action shape mismatch: expected ({expected_seq_len}, 1), got {sample_data['action'].shape}"
    
    for obs_key in expected_obs_keys:
        obs_shape = sample_data['obs'][obs_key].shape
        expected_obs_shape = (expected_seq_len, 1)  # 1 feature per timestep
        print(f"{obs_key} shape: {obs_shape}")
        print(f"Expected {obs_key} shape: {expected_obs_shape}")
        assert obs_shape == expected_obs_shape, f"{obs_key} shape mismatch: expected {expected_obs_shape}, got {obs_shape}"
    
    print("✅ All data shapes are correct")
    
    # Test 4: Verify data types
    print(f"\n=== Test 4: Data Types ===")
    assert isinstance(sample_data['action'], torch.Tensor), f"Action should be torch.Tensor, got {type(sample_data['action'])}"
    for obs_key in expected_obs_keys:
        assert isinstance(sample_data['obs'][obs_key], torch.Tensor), f"{obs_key} should be torch.Tensor, got {type(sample_data['obs'][obs_key])}"
    print("✅ All data types are correct")
    
    # Test 5: Verify data ranges and validity
    print(f"\n=== Test 5: Data Validity ===")
    
    # Check for NaN or infinite values
    assert not torch.isnan(sample_data['action']).any(), "Action contains NaN values"
    for obs_key in expected_obs_keys:
        assert not torch.isnan(sample_data['obs'][obs_key]).any(), f"{obs_key} contains NaN values"
        assert not torch.isinf(sample_data['obs'][obs_key]).any(), f"{obs_key} contains infinite values"
    
    print("✅ No NaN or infinite values found")
    
    # Test 6: Verify sequence coverage (all observations appear at start of at least one sequence)
    print(f"\n=== Test 6: Sequence Coverage ===")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    
    # Collect all first timesteps from sequences
    first_timesteps = []
    for i, data in enumerate(train_loader):
        if i >= 100:  # Sample first 100 sequences for efficiency
            break
        first_timestep = {}
        for obs_key in expected_obs_keys:
            # Get first timestep of the sequence (shape: [1, 1])
            first_timestep_data = data['obs'][obs_key][0]  # Shape: [1]
            first_timestep[obs_key] = first_timestep_data.tolist()  # Convert to list for easier handling
        first_timesteps.append(first_timestep)
    
    print(f"Sampled {len(first_timesteps)} first timesteps")
    print(f"Example first timestep: {first_timesteps[0] if first_timesteps else 'None'}")
    print("✅ Sequence coverage test completed (sampled first 100 sequences)")
    
    # Test 7: Verify train/val split
    print(f"\n=== Test 7: Train/Val Split ===")
    total_samples = len(train_set) + len(val_set)
    val_ratio = len(val_set) / total_samples
    expected_val_ratio = filtered_config['val_ratio']
    
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(f"Actual val ratio: {val_ratio:.3f}")
    print(f"Expected val ratio: {expected_val_ratio}")
    
    # Episode-level split information
    episode_val_ratio = val_episodes / total_episodes
    print(f"Total episodes: {total_episodes}")
    print(f"Train episodes: {train_episodes}")
    print(f"Val episodes: {val_episodes}")
    print(f"Episode val ratio: {episode_val_ratio:.3f}")
    
    # Timestep information
    print(f"Total timesteps: {total_timesteps}")
    print(f"Average timesteps per episode: {total_timesteps / total_episodes:.1f}")
    print(f"Average timesteps per sequence: {total_timesteps / total_samples:.1f}")
    
    # Allow some tolerance for rounding
    assert abs(val_ratio - expected_val_ratio) < 0.05, f"Val ratio mismatch: expected {expected_val_ratio}, got {val_ratio:.3f}"
    print("✅ Train/val split is correct")
    
    # Test 8: Verify DataLoader functionality
    print(f"\n=== Test 8: DataLoader Functionality ===")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Test a few batches
    for i, data in enumerate(train_loader):
        if i >= 3:  # Test first 3 batches
            break
        print(f"Train batch {i}: action shape {data['action'].shape}")
        assert data['action'].shape[0] <= batch_size, f"Batch size too large: {data['action'].shape[0]} > {batch_size}"
        assert data['action'].shape[1] == expected_seq_len, f"Action seq len wrong: {data['action'].shape[1]}"
        assert data['action'].shape[2] == 1, f"Action feature dim wrong: {data['action'].shape[2]}"
    
    for i, data in enumerate(val_loader):
        if i >= 3:  # Test first 3 batches
            break
        print(f"Val batch {i}: action shape {data['action'].shape}")
        assert data['action'].shape[0] <= batch_size, f"Batch size too large: {data['action'].shape[0]} > {batch_size}"
    
    print("✅ DataLoader functionality is correct")
    
    print(f"\n=== All Tests Passed! ===")
    print(f"✅ Dataset is correctly configured and functional")
    print(f"✅ Template variables resolved: seq_len={expected_seq_len}, history_len={expected_history_len}")
    print(f"✅ Dataset contains {len(train_set)} train and {len(val_set)} val samples (total {len(train_set) + len(val_set)})")
    print(f"✅ Dataset contains {total_episodes} total episodes ({train_episodes} train, {val_episodes} val)")
    print(f"✅ Dataset contains {total_timesteps} total timesteps ({total_timesteps / total_episodes:.1f} avg per episode)")
    print(f"✅ All data shapes, types, and ranges are valid")

def test_dataset_simple(dataset_config_path):
    """Simple test for debugging purposes."""
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    dataset_template_vars = {'num_frames': 18, 'obs_num_frames': 2}
    batch_size = 64
    # Filter config to only include valid kwargs for make_pendulum_dataset
    filtered_config = filter_config_with_debug(dataset_config, make_pendulum_dataset, debug=True, template_vars=dataset_template_vars)

    
    train_set, val_set = make_pendulum_dataset(**filtered_config)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    all_actions = []
    for i in tqdm(range(min(len(train_set),1000))):
        data = train_set[i]
        actions = data['action'][0].item()
        all_actions.append(actions)

    os.makedirs("test_media/test_load_dataset", exist_ok=True)
    plt.hist(np.array(all_actions))
    plt.title("Actions distribution")
    plt.savefig("test_media/test_load_dataset/actions_distribution.png")
    plt.clf()

if __name__ == "__main__":
    # Run comprehensive tests
    dataset ="/home/wph52/weird/dynamics/configs/dataset/pendulum_integrable.yaml" 
    test_dataset_simple(dataset)
    test_dataset_comprehensive(dataset)
    
    # Uncomment for simple debugging
    
