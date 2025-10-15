import numpy as np
import robosuite as suite
from cluster_utils import set_cluster_graphics_vars
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from envs.env_utils import make_env, make_vec_env
from envs.env_utils import VideoCallback

def test_policy():
    from stable_baselines3 import PPO
    env = make_vec_env("Robosuite-Lift-Panda", n_envs=4)

    model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
        )

    eval_env = make_env("Robosuite-Lift-Panda", eval_mode=True)
  

    video_callback = VideoCallback(eval_env,
        video_folder="test_outputs/",
        eval_freq=500,
        name_prefix="evaluation",
        deterministic=False,
        flip_vertical=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="test_outputs/",
        log_path="test_outputs/",
        eval_freq=1000,
        deterministic=True,
    )

    callbacks = CallbackList([eval_callback, video_callback])

    model.learn(total_timesteps=10000, callback=callbacks)


def test_single():
    env = make_env("Robosuite-Lift-Panda")
    env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)  # take action in the environment
        image = env.render()
        
        

def test_parallel():
    env = make_vec_env("Robosuite-Lift-Panda", n_envs=4)
    env.reset()

    for i in range(1000):
        action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, reward, done, info = env.step(action)  # take action in the environment
        

if __name__ == "__main__":
    set_cluster_graphics_vars()
    
    test_policy()
    #test_single()
    #test_parallel()
    