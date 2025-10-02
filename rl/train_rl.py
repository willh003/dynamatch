import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
from cluster_utils import set_cluster_graphics_vars
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime   
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.register_envs import register_custom_envs


def main(config):
    # Create the environment
    env_id = config.get('env_id', "InvertedPendulum-v5")
    env_kwargs = config.get('env_kwargs', {})
    env = make_vec_env(env_id, n_envs=4, env_kwargs=env_kwargs)  # Vectorized environment for faster training

    # Create unified run directory
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_runs_dir = config.get('runs_dir', './runs')
    run_dir = os.path.join(base_runs_dir, f"{env_id}_{run_stamp}")
    models_dir = os.path.join(run_dir, "models")
    videos_dir = os.path.join(run_dir, "videos", "train")
    tb_dir = os.path.join(run_dir, "tensorboard")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Create evaluation environment and local video recorder (not W&B)
    eval_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    eval_env = RecordVideo(
        eval_env,
        video_folder=videos_dir,
        episode_trigger=lambda ep_id: True,
        name_prefix="evaluation",
        disable_logger=True,
    )

    # Initialize Weights & Biases
    run = wandb.init(
        project="dynamics",
        entity="willhu003",
        config={**config, "run_dir": run_dir, "env_id": env_id},
        sync_tensorboard=True,
        mode=config.get('wandb_mode', 'online'),
        dir=run_dir,
        name=config.get('run_name', os.path.basename(run_dir)),
        tags=['rl']
    )

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tb_dir
    )



    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=10000,
        deterministic=False,
        render=False
    )

    # Set up W&B callback
    wandb_callback = WandbCallback()

    callbacks = CallbackList([eval_callback, wandb_callback])

    # Train the agent
    print("Training the PPO agent...")
    model.learn(
        total_timesteps=config.get('total_steps', 1000000),
        callback=callbacks,
        progress_bar=True
    )

    # Save the trained model
    final_model_path = os.path.join(models_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to '{final_model_path}.zip'")



if __name__ == "__main__":
    set_cluster_graphics_vars()
    register_custom_envs()

    # config = {
    #     'env_id': 'InvertedPendulumDynamicsShift-v5',
    #     'env_kwargs': {
    #         'action_add': .5
    #     },
    #     'wandb_mode': 'online'
    # }

    # config = {
    #     'env_id': 'InvertedPendulumIntegrableMLPShift-v5',
    #     'env_kwargs': {
    #         'checkpoint_path': '/home/wph52/weird/dynamics/envs/transformations/pendulum_shift_mlp/random_weights.pth'
    #     },
    #     'wandb_mode': 'online',
    #     'total_steps': 1000000
    # }

    config = {
        'env_id': 'Ant-v5',
        'env_kwargs': {},
        'wandb_mode': 'online',
        'total_steps': 5000000
    }

    main(config)