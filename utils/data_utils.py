import zarr
import numpy as np
import yaml
import os

def load_transition_dataset(dataset_path, state_indices=None, action_indices=None):
    """Load transition dataset from zarr file."""
    print("=== Loading Transition Dataset ===")
    
    store = zarr.open(dataset_path, mode='r')
    data_group = store['data']
    meta_group = store['meta']
    
    states = data_group['state'][:]
    actions = data_group['action'][:]
    next_states = data_group['next_state'][:]
    num_samples = meta_group['num_samples'][0]

    if state_indices is not None:
        state_indices = np.array(state_indices).astype(int)
        states = states[:, state_indices]
        next_states = next_states[:, state_indices]
    if action_indices is not None:
        action_indices = np.array(action_indices).astype(int)
        actions = actions[:, action_indices]
    
    print(f"Loaded dataset with {num_samples} samples")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Next state shape: {next_states.shape}")
    
    return states, actions, next_states


def get_transition_path_from_dataset_config(config_path):
    """Create output path by replacing 'sequence' with 'inverse_dynamics' in the buffer path."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    buffer_dir = config['buffer_dir']
    # Replace 'sequence' with 'inverse_dynamics' in the path
    output_dir = buffer_dir.replace('/sequence/', '/transitions/')

    buffer_path = os.path.join(output_dir, 'buffer.zarr')
    
    return buffer_path



def get_relabeled_actions_path_from_config(config_path, id_model_name=None):
    """Create output path by replacing 'sequence' with 'relabeled_actions' in the buffer path."""
    with open(config_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    buffer_dir = dataset_config['buffer_dir']
    output_dir = buffer_dir.replace('/sequence/', '/relabeled_actions/').replace('/transitions/', '/relabeled_actions/')

    if id_model_name is not None:
        output_dir = os.path.join(output_dir, id_model_name)
    buffer_path = os.path.join(output_dir,  'buffer.zarr')
    
    return buffer_path
