from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from torchvision.utils import save_image
from cluster_utils import set_cluster_graphics_vars
import numpy as np
import torch
set_cluster_graphics_vars()

# Option 1: Using make_vec_env (simplest)
vec_env = make_vec_env(
    "Pusher-v5",
    n_envs=8,
    env_kwargs={"render_mode": "rgb_array"}
)


# To render/capture frames:
obs = vec_env.reset()
frames = np.array([env.render() for env in vec_env.envs]) # Returns RGB array for first env
print(frames.shape)
torch_frame = torch.as_tensor(frames).float().permute(0, 3, 1, 2) / 255.0
save_image(torch_frame, "test.png")
print(torch_frame.shape)
vec_env.close()