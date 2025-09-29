
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from cluster_utils import set_cluster_graphics_vars
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from envs.register_envs import register_custom_envs
from envs.env_utils import modify_env_integrator
from tqdm import tqdm
# Try to import zarr, fall back to alternative if not available

# Load the trained model
model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
expert_policy = PPO.load(model_path)

# Create environment
#env = gym.make(env_id, **env_kwargs)
env = gym.make('InvertedPendulum-v5')
#env = modify_env_integrator(env, integrator='euler', frame_skip=1)

actions = []

state, _ = env.reset()

for step in range(1000):
    if callable(expert_policy):
        policy_action = expert_policy(state)
    else:
        policy_action = expert_policy.predict(state)[0]

    state, reward, terminated, truncated, _ = env.step(policy_action)
    actions.append(policy_action.item())

    if terminated or truncated:
        print(f"Episode terminated or truncated at step {step}")
        break

os.makedirs("test_media/policy_test", exist_ok=True)
plt.hist(actions)
plt.savefig("test_media/policy_test/action_dist.png")
plt.clf()