import os
import argparse
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from cluster_utils import set_cluster_graphics_vars
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from envs.register_envs import register_custom_envs
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from action_translation.utils.model_utils import load_action_translator_from_config, print_model_info


def evaluate_and_record(
    model_path: str = None,
    translator_config_path: str = None,
    env_id: str = "InvertedPendulum-v5",
    base_policy_checkpoint: str = None,
    action_translator_checkpoint: str = None,
    env_kwargs: dict = {},
    run_dir: str | None = None,
    num_episodes: int = 5,
    max_steps_per_episode: int = 1000,
    deterministic: bool = False,
    seed: int | None = None,
    config: dict = None,
):
    """
    Evaluate a policy (PPO or ActionTranslator) and record videos.
    
    Args:
        model_path: Path to PPO model .zip file (for regular PPO models)
        translator_config_path: Path to ActionTranslator config YAML file (for ActionTranslator models)
        base_policy_checkpoint: Path to base policy checkpoint (overrides config for ActionTranslator)
        action_translator_checkpoint: Path to action translator checkpoint (overrides config for ActionTranslator)
        env_id: Gymnasium environment ID
        env_kwargs: Environment keyword arguments
        run_dir: Directory to save videos (defaults to model directory)
        num_episodes: Number of episodes to record
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        config: Additional configuration dict
    """
    # Determine model type and validate arguments
    if model_path is not None and translator_config_path is not None:
        raise ValueError("Cannot specify both model_path and translator_config_path. Choose one model type.")
    elif model_path is None and translator_config_path is None:
        raise ValueError("Must specify either model_path (for PPO) or translator_config_path (for ActionTranslator).")
    
    is_action_translator = translator_config_path is not None
    
    if run_dir is not None:
        video_dir = os.path.join(run_dir, "videos")
    else:
        if is_action_translator:
            # For ActionTranslator, use current directory as default
            video_dir = "videos/eval"
        else:
            # For PPO, use model directory
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

    # Load model based on type
    if is_action_translator:
        print("Loading ActionTranslator model...")
        model = load_action_translator_from_config(
            translator_config_path,
            base_policy_checkpoint=base_policy_checkpoint,
            action_translator_checkpoint=action_translator_checkpoint
        )
        print("Model loaded successfully!")
        
        # Print parameter counts and model info
        print_model_info(model)
    else:
        print("Loading PPO model...")
        model = PPO.load(model_path)
        print("Model loaded successfully!")

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
    all_actions = []
    all_translated_actions = []
    all_returns = []
    
    for episode_idx in range(num_episodes):
        obs, _ = video_env.reset(seed=seed)
        episode_return = 0.0
        for step in range(max_steps_per_episode):
            if is_action_translator:
                # ActionTranslator returns (action, state) tuple
                translated_action, base_action = model.predict_base_and_translated(obs, deterministic=deterministic)
                action_to_step = translated_action
                all_actions.append(base_action)
                all_translated_actions.append(translated_action)
            else:
                # Regular PPO model
                action, _ = model.predict(obs, deterministic=deterministic)
                action_to_step = action
                all_actions.append(action)
            
            obs, reward, terminated, truncated, _info = video_env.step(action_to_step)
            episode_return += float(reward)
            if terminated or truncated:
                break
        print(f"Episode {episode_idx + 1}/{num_episodes} return: {episode_return:.2f}")
        all_returns.append(episode_return)

    # Plot results
    all_actions = np.array(all_actions)
    all_returns = np.array(all_returns)
    
    if is_action_translator:
        # ActionTranslator specific plotting
        all_translated_actions = np.array(all_translated_actions)
        
        plt.figure(figsize=(15, 5))
        
        # Action distributions
        plt.subplot(1, 3, 1)
        plt.hist(all_actions, alpha=0.7, label='Base Policy Actions', bins=50)
        plt.hist(all_translated_actions, alpha=0.7, label='Translated Actions', bins=50)
        plt.title('Action Distributions')
        plt.xlabel('Action Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Action comparison scatter plot
        plt.subplot(1, 3, 2)
        plt.scatter(all_actions, all_translated_actions, alpha=0.6)
        plt.plot([all_actions.min(), all_actions.max()], 
                 [all_actions.min(), all_actions.max()], 'r--', label='y=x')
        plt.title('Base Policy vs Translated Actions')
        plt.xlabel('Base Policy Action')
        plt.ylabel('Translated Action')
        plt.legend()
        
        # Returns distribution
        plt.subplot(1, 3, 3)
        plt.hist(all_returns, bins=20, alpha=0.7)
        plt.title('Episode Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(video_run_dir, "action_comparison.png"))
        plt.close()
    else:
        # PPO specific plotting
        plt.figure(figsize=(12, 5))
        
        # Actions distribution
        plt.subplot(1, 2, 1)
        plt.hist(all_actions, bins=50, alpha=0.7)
        plt.title('Action Distribution')
        plt.xlabel('Action Value')
        plt.ylabel('Frequency')
        
        # Returns distribution
        plt.subplot(1, 2, 2)
        plt.hist(all_returns, bins=20, alpha=0.7)
        plt.title('Episode Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(video_run_dir, "actions.png"))
        plt.close()
        
        # Save returns plot separately for PPO
        plt.figure()
        plt.hist(all_returns, bins=20, alpha=0.7)
        plt.title('Episode Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(video_run_dir, "returns.png"))
        plt.close()

    print(f"Saved videos to: {video_run_dir}")
    print(f"Mean return: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy (PPO or ActionTranslator) and save videos.")
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Path to the trained PPO model .zip file")
    model_group.add_argument("--translator_config", help="Path to the ActionTranslator config YAML file")
    
    # Environment arguments
    parser.add_argument("--env_id", default="InvertedPendulum-v5", help="Gymnasium env id")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions instead of deterministic")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for resets")
    
    # ActionTranslator specific arguments
    parser.add_argument("--base_policy_checkpoint", help="Path to base policy checkpoint (overrides config for ActionTranslator)")
    parser.add_argument("--action_translator_checkpoint", help="Path to action translator checkpoint (overrides config for ActionTranslator)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    register_custom_envs()

    evaluate_and_record(
        model_path=args.model,
        translator_config_path=args.translator_config,
        base_policy_checkpoint=args.base_policy_checkpoint,
        action_translator_checkpoint=args.action_translator_checkpoint,
        env_id=args.env_id,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
