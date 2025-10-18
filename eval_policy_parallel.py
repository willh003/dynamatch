import os
import argparse
import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from cluster_utils import set_cluster_graphics_vars
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from envs.register_envs import register_custom_envs
from utils.model_utils import load_action_translator_policy_from_config, load_source_policy_from_config
from utils.eval_utils import bootstrap_iqm_ci
from envs.env_utils import get_state_from_obs, make_vec_env




def get_state_from_vec_env(obs, infos, env_id: str) -> np.ndarray:
    """
    Get state from vectorized environment.
    """
    n_envs = len(obs)

    states = []
    for env_idx in range(n_envs):
        state = get_state_from_obs(obs[env_idx], infos[env_idx], env_id)
        states.append(state)
    return np.array(states)

def evaluate_policy_parallel(
    model,
    env_id: str,
    env_kwargs: dict = None,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    n_envs: int = 4,
    seed: int = None,
    max_steps_per_episode: int = 1000,
    is_action_translator: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """
    Evaluate a policy using parallel vectorized environments.
    
    Args:
        model: The trained model to evaluate
        env_id: Gymnasium environment ID
        env_kwargs: Environment keyword arguments
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        n_envs: Number of parallel environments
        seed: Random seed for environment resets
        is_action_translator: Whether the model is an ActionTranslator
        
    Returns:
        Tuple of (reward_iqm, reward_lower_ci, reward_upper_ci, x_displacement_iqm, x_displacement_lower_ci, x_displacement_upper_ci)
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Create vectorized environment
    vec_env = make_vec_env(env_id, n_envs=n_envs, **env_kwargs)
    
    episode_rewards = []
    episode_lengths = []
    episode_x_displacements = []
    
    # Check if this is an ant environment
    is_ant_env = "ant" in env_id.lower()
    
    # Initialize environments
    obs = vec_env.reset()
    infos = vec_env.reset_infos
    
    episode_steps = [0] * n_envs
    episode_dones = [False] * n_envs
    current_rewards = [0.0] * n_envs
    initial_x_positions = [None] * n_envs  # Track initial x positions for displacement calculation
    
    # Evaluate episodes in parallel
    with tqdm(total=n_eval_episodes, desc="Evaluating policy") as pbar:
        while len(episode_rewards) < n_eval_episodes:
            # Get actions from model
            if is_action_translator:
                # For ActionTranslator, we need to get full observations
                full_obs = get_state_from_vec_env(obs, infos, env_id)
                
                actions, _ = model.predict_base_and_translated(
                    policy_observation=obs, 
                    translator_observation=full_obs, 
                    deterministic=deterministic
                )
            else:
                # Regular PPO model
                actions, _ = model.predict(obs, deterministic=deterministic)
            
            # Step all environments
            next_obs, rewards, dones, infos = vec_env.step(actions)
            
            # Update episode tracking
            for env_idx in range(n_envs):
                current_rewards[env_idx] += rewards[env_idx]
                episode_steps[env_idx] += 1
                
                # Track initial x position for ant environments
                if is_ant_env and initial_x_positions[env_idx] is None:
                    x_pos = infos[env_idx].get('x_position', 0.0)
                    initial_x_positions[env_idx] = x_pos
                
                # Check if episode is done
                if dones[env_idx] and not episode_dones[env_idx] or episode_steps[env_idx] >= max_steps_per_episode:
                    episode_rewards.append(current_rewards[env_idx])
                    episode_lengths.append(episode_steps[env_idx])
                    
                    # Calculate x displacement for ant environments
                    if is_ant_env and initial_x_positions[env_idx] is not None:
                        final_x_pos = infos[env_idx].get('x_position', 0.0)
                        x_displacement = final_x_pos - initial_x_positions[env_idx]
                        episode_x_displacements.append(x_displacement)
                    else:
                        episode_x_displacements.append(0.0)
                    
                    episode_dones[env_idx] = True
                    pbar.update(1)
                    
                    # Reset this environment
                    if len(episode_rewards) < n_eval_episodes:
                        obs[env_idx], _ = vec_env.env_method('reset', indices=[env_idx])[0]
                        current_rewards[env_idx] = 0.0
                        episode_steps[env_idx] = 0
                        episode_dones[env_idx] = False
                        initial_x_positions[env_idx] = None  # Reset initial position tracking
            
            obs = next_obs
    
    vec_env.close()
    
    # Calculate bootstrap IQM and confidence interval for rewards
    reward_iqm, reward_lower_ci, reward_upper_ci = bootstrap_iqm_ci(np.array(episode_rewards))
    
    # Calculate bootstrap IQM and confidence interval for x displacements
    x_displacement_iqm, x_displacement_lower_ci, x_displacement_upper_ci = bootstrap_iqm_ci(np.array(episode_x_displacements))
    
    return reward_iqm, reward_lower_ci, reward_upper_ci, x_displacement_iqm, x_displacement_lower_ci, x_displacement_upper_ci


def evaluate_and_record_parallel(
    translator_policy_config_path: str = None,
    base_policy_config_path: str = None,
    env_id: str = "InvertedPendulum-v5",
    source_policy_checkpoint: str = None,
    action_translator_checkpoint: str = None,
    env_kwargs: dict = None,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    seed: int = None,
    max_steps_per_episode: int = 1000,
    n_envs: int = 4,
):
    """
    Evaluate a policy using parallel environments.
    
    Args:
        translator_policy_config_path: Path to ActionTranslator config YAML file
        base_policy_config_path: Path to source policy config YAML file
        source_policy_checkpoint: Path to source policy checkpoint
        action_translator_checkpoint: Path to action translator checkpoint
        env_id: Gymnasium environment ID
        env_kwargs: Environment keyword arguments
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment resets
        n_envs: Number of parallel environments
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Determine model type and validate arguments
    model_types = [translator_policy_config_path, base_policy_config_path]
    non_none_models = [m for m in model_types if m is not None]
    
    if len(non_none_models) > 1:
        raise ValueError("Cannot specify multiple model types. Choose one: translator_policy_config_path or base_policy_config_path.")
    elif len(non_none_models) == 0:
        raise ValueError("Must specify one model type: translator_policy_config_path (for ActionTranslator) or base_policy_config_path (for standalone source policy).")
    
    is_action_translator = translator_policy_config_path is not None
    
    # Load model based on type
    if is_action_translator:
        print("Loading ActionTranslator model...")
        model = load_action_translator_policy_from_config(
            translator_policy_config_path,
            source_policy_checkpoint=source_policy_checkpoint,
            action_translator_checkpoint=action_translator_checkpoint
        )
    else:
        print("Loading source policy from config...")
        model = load_source_policy_from_config(
            base_policy_config_path,
            source_policy_checkpoint=source_policy_checkpoint
        )
    
    print("Model loaded successfully!")
    
    # Evaluate policy in parallel
    print(f"Evaluating policy using {n_envs} parallel environments...")
    reward_iqm, reward_lower_ci, reward_upper_ci, x_displacement_iqm, x_displacement_lower_ci, x_displacement_upper_ci = evaluate_policy_parallel(
        model=model,
        env_id=env_id,
        max_steps_per_episode=max_steps_per_episode,
        env_kwargs=env_kwargs,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        n_envs=n_envs,
        seed=seed,
        is_action_translator=is_action_translator,
    )
    
    print(f"IQM reward: {reward_iqm:.2f} (95% CI: [{reward_lower_ci:.2f}, {reward_upper_ci:.2f}]) over {n_eval_episodes} episodes")
    
    # Report x displacement metrics for ant environments
    if "ant" in env_id.lower():
        print(f"IQM x displacement: {x_displacement_iqm:.2f} (95% CI: [{x_displacement_lower_ci:.2f}, {x_displacement_upper_ci:.2f}]) over {n_eval_episodes} episodes")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy using parallel environments.")
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--base_policy_config", help="Path to the source policy config YAML file")
    model_group.add_argument("--translator_policy_config", help="Path to the ActionTranslator config YAML file")
    
    # Environment arguments
    parser.add_argument("--env_id", default="InvertedPendulum-v5", help="Gymnasium env id")
    parser.add_argument("--episodes", type=int, default=64, help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for resets")
    parser.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per episode")
    
    # ActionTranslator specific arguments
    parser.add_argument("--source_policy_checkpoint", help="Path to base policy checkpoint")
    parser.add_argument("--action_translator_checkpoint", help="Path to action translator checkpoint")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    register_custom_envs()

    evaluate_and_record_parallel(
        translator_policy_config_path=args.translator_policy_config,
        base_policy_config_path=args.base_policy_config,
        source_policy_checkpoint=args.source_policy_checkpoint,
        action_translator_checkpoint=args.action_translator_checkpoint,
        env_id=args.env_id,
        max_steps_per_episode=args.max_steps_per_episode,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic,
        seed=args.seed,
        n_envs=args.n_envs,
    )
