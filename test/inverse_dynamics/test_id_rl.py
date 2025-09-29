import sys
import os
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import gymnasium as gym
from datasets.pendulum.pendulum import make_pendulum_dataset
utils.from config_utils import filter_config_with_debug
import yaml
from torch.utils.data import DataLoader
from envs.inverse.inverse_dynamics import gym_inverse_dynamics
from envs.register_envs import register_custom_envs
import mujoco 
import matplotlib.pyplot as plt
from cluster_utils import set_cluster_graphics_vars
from tqdm import tqdm
from envs.env_utils import modify_env_integrator



def test_id_rl(model_dynamics_1, model_dynamics_2, env_id, config_path, output_path, num_episodes, max_steps_per_episode):
    pass


if __name__ == "__main__":
    register_custom_envs()
    set_cluster_graphics_vars()

    model_dynamics_1 = PPO.load("/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip")
    model_dynamics_2 = PPO.load("/home/wph52/weird/dynamics/rl/runs/InvertedPendulumDynamicsShift-v5_20250923_164029/models/best_model.zip")
    
    
    test_id_rl(model_dynamics_1, model_dynamics_2, "InvertedPendulum-v5", "configs/dataset/pendulum_dynamics_shift.yaml", "test_media/id_rl.png", 10, 1000)