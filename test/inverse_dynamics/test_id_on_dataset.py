import sys
import os
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import gymnasium as gym
from datasets.trajectory_dataset import make_trajectory_dataset
from utils.config_utils import filter_config_with_debug, load_yaml_config
import yaml
from torch.utils.data import DataLoader
from inverse.physics_inverse_dynamics import gym_inverse_dynamics
from envs.register_envs import register_custom_envs
import mujoco 
import matplotlib.pyplot as plt
from cluster_utils import set_cluster_graphics_vars
from tqdm import tqdm
from envs.env_utils import modify_env_integrator

def test_id_on_dataset(test_env_id,dataset_config_path, state_keys, frame_skip=1, max_samples=10000):
    """
    On a dataset, check id(s,s') - a for all (s,a,s') pairs

    state_keys: list of strings, the keys for the state data in batch['obs']. MUST BE IN CORRECT ORDER
    """
    
    print("=== ID on Dataset Test ===")
    
    dataset_config = load_yaml_config(dataset_config_path)

    dataset_template_vars = {'num_frames': 18, 'obs_num_frames': 2}
    batch_size = 64
    
    # Filter config to only include valid kwargs for make_trajectory_dataset
    filtered_config = filter_config_with_debug(dataset_config, make_trajectory_dataset, debug=True, template_vars=dataset_template_vars)

    print(f"\n=== Creating Dataset ===")
    train_set, val_set = make_trajectory_dataset(**filtered_config)
    
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    
    # Get episode information
    total_episodes = train_set.buffer.num_episodes
    train_episodes = train_set.train_mask.sum()
    val_episodes = total_episodes - train_episodes

    inverse_dynamics_env = gym.make(test_env_id) # env used to compute inverse dynamics
    id_actions = []
    expert_actions = []
    errors = []

    for i in tqdm(range(min(max_samples, len(train_set)))):
        seq = train_set[i]

        expert_action = seq['action'][0]        
        state = []
        # reconstruct state from dataset obs
        for state_key in state_keys:
            state.append(seq['obs'][state_key][0])
        state = np.array(state).squeeze(axis=1)

        next_state = []
        for state_key in state_keys:
            next_state.append(seq['obs'][state_key][1])
        next_state = np.array(next_state).squeeze(axis=1)

        # Unnormalize the observations before passing to inverse dynamics
        if hasattr(train_set, 'lowdim_normalizer'):
            state_unnorm = []
            next_state_unnorm = []
            for i, state_key in enumerate(state_keys):
                state_unnorm.append(train_set.lowdim_normalizer[state_key].reconstruct(state[i:i+1]))
                next_state_unnorm.append(train_set.lowdim_normalizer[state_key].reconstruct(next_state[i:i+1]))
            state_unnorm = np.array(state_unnorm).squeeze()
            next_state_unnorm = np.array(next_state_unnorm).squeeze()
        else:
            state_unnorm = state
            next_state_unnorm = next_state

        id_action = gym_inverse_dynamics(inverse_dynamics_env, state_unnorm, next_state_unnorm)
        
        # Unnormalize the expert action from the dataset for fair comparison
        if hasattr(train_set, 'action_normalizer'):
            expert_action_unnorm = train_set.action_normalizer.reconstruct(expert_action.cpu().numpy())
        else:
            expert_action_unnorm = expert_action.cpu().numpy()
        
        id_actions.append(id_action.item())
        expert_actions.append(expert_action_unnorm.item())
        error = np.abs(id_action - expert_action_unnorm).item()
        errors.append(error)
        
        # Debug output for first few samples
        if i < 3:
            print(f"Sample {i}:")
            print(f"  State (normalized): {state}")
            print(f"  State (unnormalized): {state_unnorm}")
            print(f"  Next state (normalized): {next_state}")
            print(f"  Next state (unnormalized): {next_state_unnorm}")
            print(f"  Expert action (normalized): {expert_action.item():.4f}")
            print(f"  Expert action (unnormalized): {expert_action_unnorm.item():.4f}")
            print(f"  ID action: {id_action.item():.4f}")
            print(f"  Error: {error:.4f}")
            print()


    os.makedirs("test_media/id_on_dataset", exist_ok=True)
    errors = np.array(errors)
    print(f"Mean Error: {np.mean(errors):.4f}, Std Error: {np.std(errors):.4f}, Max Error: {np.max(errors):.4f}, Min Error: {np.min(errors):.4f}")
    plt.hist(errors)
    plt.title(f"Action Errors, N={len(errors)}, mse: {np.mean(errors**2):.4f}, var: {np.var(errors):.4f}, max: {np.max(errors):.4f}, min: {np.min(errors):.4f}")
    plt.savefig("test_media/id_on_dataset/action_errors.png")
    plt.clf()
    plt.hist(id_actions, color="orange", label="ID actions (unnormalized)", alpha=0.5)
    plt.hist(expert_actions, color="blue", label="Expert actions (unnormalized)", alpha=0.5)
    plt.title(f"Expert vs ID Actions")
    plt.legend()
    plt.savefig("test_media/id_on_dataset/actions.png")
    plt.clf()


        
            

if __name__ == "__main__":
    register_custom_envs()
    set_cluster_graphics_vars()

    state_keys = ['cart_position', 'pole_angle', 'cart_velocity', 'pole_velocity']
    test_env_id = 'InvertedPendulumIntegrable-v5'
    dataset_config_path = "/home/wph52/weird/dynamics/configs/dataset/pendulum_integrable_dynamics_shift.yaml"
    max_samples = 1000
    test_id_on_dataset(test_env_id, dataset_config_path, state_keys=state_keys, frame_skip=1, max_samples=max_samples)
    