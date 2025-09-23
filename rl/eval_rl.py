import os
import argparse
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from cluster_utils import set_cluster_graphics_vars
from utils import modify_env_gravity
from register_envs import register_custom_envs

from pathlib import Path

def evaluate_and_record(
    model_path: str,
    env_id: str = "InvertedPendulum-v5",
    env_kwargs: dict = {},
    run_dir: str | None = None,
    num_episodes: int = 5,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    seed: int | None = None,
    config: dict = None,
):
    if run_dir is not None:
        video_dir = os.path.join(run_dir, "videos")
    else:
        # log to same run directory as model by default
        video_dir = Path(model_path).parent.parent / "videos" / "eval"

    os.makedirs(video_dir, exist_ok=True)

    # Ensure we get rgb frames for video generation
    video_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

    # Record every episode to a timestamped subfolder
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_run_dir = os.path.join(video_dir, f"{env_id}_{run_stamp}")
    video_env = RecordVideo(
        video_env,
        video_folder=video_run_dir,
        episode_trigger=lambda ep_id: True,
        name_prefix="evaluation",
        disable_logger=True,
    )

    # Load model
    model = PPO.load(model_path)

    # Quick quantitative evaluation (no video)
    eval_env = gym.make(env_id, **env_kwargs)


    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=max(5, num_episodes),
        deterministic=deterministic,
        render=False,
        warn=False,
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over {max(5, num_episodes)} episodes")

    # Rollout episodes with video recording
    for episode_idx in range(num_episodes):
        obs, _ = video_env.reset(seed=seed)
        episode_return = 0.0
        for step in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = video_env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        print(f"Episode {episode_idx + 1}/{num_episodes} return: {episode_return:.2f}")


    print(f"Saved videos to: {video_run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy and save videos.")
    parser.add_argument("--model", required=True, help="Path to the trained model .zip file")
    parser.add_argument("--env_id", default="InvertedPendulum-v5", help="Gymnasium env id")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for resets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    register_custom_envs()

    evaluate_and_record(
        model_path=args.model,
        env_id=args.env_id,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
