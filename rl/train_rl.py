import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO, DDPG, HerReplayBuffer, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import polyak_update
import wandb
from wandb.integration.sb3 import WandbCallback
from cluster_utils import set_cluster_graphics_vars
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime   
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.register_envs import register_custom_envs

import torch as th
from stable_baselines3.ddpg.ddpg import DDPG as SB3_DDPG

class ClippedTargetDDPG(SB3_DDPG):
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                target_q = self.critic_target(replay_data.next_observations, next_actions)
                # Handle tuple outputs (e.g., multiple critics) and tensors
                if isinstance(target_q, tuple):
                    if len(target_q) == 1:
                        target_q = target_q[0]
                    else:
                        target_q = th.min(th.stack(list(target_q), dim=0), dim=0).values
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

                # Clip critic targets to [−1/(1−γ), 0]
                min_target = -1.0 / (1.0 - self.gamma)
                target_q = th.clamp(target_q, min=min_target, max=0.0)

            current_q = self.critic(replay_data.observations, replay_data.actions)
            if isinstance(current_q, tuple):
                # Use first critic for loss or aggregate if needed
                current_q = current_q[0]
            critic_loss = th.nn.functional.mse_loss(current_q, target_q)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # actor + target updates remain identical to SB3
            self._n_updates += 1
            self.actor.optimizer.zero_grad()
            actor_actions = self.actor(replay_data.observations)
            actor_loss = -self.critic.q1_forward(replay_data.observations, actor_actions).mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            # Soft-update target networks with Polyak averaging
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

class ActionL2NormWrapper(gym.Wrapper):
    """Wrapper to apply L2 norm penalty to actions."""
    def __init__(self, env, l2_coeff=1.0):
        super().__init__(env)
        self.l2_coeff = l2_coeff
    
    def step(self, action):
        # Apply L2 penalty to reward
        obs, reward, terminated, truncated, info = self.env.step(action)
        l2_penalty = -self.l2_coeff * np.sum(action**2)
        reward += l2_penalty
        return obs, reward, terminated, truncated, info



def make_model(env,config, tb_dir):

    model_type = config.get('model_class', 'PPO')   
    if model_type == 'HER-DDPG':
        goal_selection_strategy = config.get('goal_selection_strategy', 'future')
        
        model = DDPG(
            "MultiInputPolicy", # her only works with MultiInputPolicy
            env,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
            ),
            tensorboard_log=tb_dir
        )
    elif model_type == 'SAC':
        model = SAC(
            "MultiInputPolicy",
            env,
            tensorboard_log=tb_dir
        )
    elif model_type == 'PPO':
        policy_type = config.get('policy_type', "MlpPolicy")
        model = PPO(
            policy_type,
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
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    return model

def main(config):
    # Create the environment
    env_id = config.get('env_id', "InvertedPendulum-v5")
    env_kwargs = config.get('env_kwargs', {})
    n_envs = config.get('n_envs', 4)
    video_freq = config.get('video_freq', 1)
    model_type = config.get('model_class', 'PPO')
    
    # Apply wrappers for DDPG
    if model_type == 'DDPG':
        # Create a wrapper function for DDPG-specific environment modifications
        def make_ddpg_env():
            env = gym.make(env_id, **env_kwargs)
            # Apply action L2 norm coefficient: 1.0
            env = ActionL2NormWrapper(env, l2_coeff=1.0)
            return env
        
        env = make_vec_env(make_ddpg_env, n_envs=n_envs)
    else:
        env = make_vec_env(env_id, n_envs=n_envs, env_kwargs=env_kwargs)  # Vectorized environment for faster training

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
        episode_trigger=lambda ep_id: ep_id % video_freq == 0,
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
        tags=['rl', config.get('model_class', 'PPO')]
    )

    # Initialize the model using the make_model function
    model = make_model(env, config, tb_dir)



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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RL policy")
    parser.add_argument("--env_id", type=str, default="FetchReachModifiedPhysics-v4", help="Environment ID")
    parser.add_argument("--model_class", type=str, default="PPO", help="Model class")
    return parser.parse_args()


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

    # config = {
    #     #'env_id': 'HandManipulateBlock-v1',
    #     'env_id': 'HandReach-v3',
    #     #'env_id': 'FetchReach-v4',
    #     #'env_id': 'HandManipulateBlockRotateZ-v1',
    #     'policy_type': 'MultiInputPolicy',
    #     'n_envs': 1,
    #     'env_kwargs': {},
    #     'wandb_mode': 'online',
    #     'total_steps': 10000000,
    #     'video_freq': 100,
    #     'model_class': 'DDPG',
    #     'use_her': True,
    #     'learning_starts': 10000
    # }

    args = parse_args()

    config = {
        'env_id': args.env_id,
        #'env_id': 'FetchPickAndPlaceDense-v4',
        #'env_id': 'AntModifiedPhysics-v1',
        #'env_id': 'HandReachObsDense-v3',
        'policy_type': 'MultiInputPolicy',
        'model_class': args.model_class,
        'n_envs': 64,
        #'policy_type': 'MlpPolicy',
        'env_kwargs': {},
        'wandb_mode': 'online',
        'total_steps': 100000000,
        'video_freq': 100
    }

    main(config)
