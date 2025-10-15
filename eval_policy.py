import os
import argparse
import yaml
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor
from cluster_utils import set_cluster_graphics_vars
import warnings
from typing import Any, Callable, Optional, Union

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from envs.register_envs import register_custom_envs
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from utils.model_utils import load_action_translator_policy_from_config, load_source_policy_from_config, print_model_info
from omegaconf import OmegaConf
from envs.env_utils import get_state_from_obs

def resolve_hydra_config_and_get_source_checkpoint(config_path, source_policy_checkpoint=None):
    """
    Simple method to resolve a Hydra config and extract the source policy checkpoint path.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the config directory to resolve relative paths
        config_dir = os.path.dirname(config_path)
        
        # Load source policy config
        source_policy_name = config['defaults'][0]['source_policy']
        base_policy_config_path = os.path.join(config_dir, '..', 'source_policy', f'{source_policy_name}.yaml')
        base_policy_config_path = os.path.normpath(base_policy_config_path)
        
        with open(base_policy_config_path, 'r') as f:
            base_policy_config = yaml.safe_load(f)
        
        # Return checkpoint path (override if provided)
        if source_policy_checkpoint:
            return source_policy_checkpoint
        else:
            return base_policy_config.get('checkpoint_path')
            
    except Exception as e:
        print(f"Warning: Could not resolve Hydra config: {e}")
        return source_policy_checkpoint

def plot_ankle_forces(all_mj_ankle_actuator, all_mj_ankle_constraint, 
                     all_mj_ankle_actuator_world, all_mj_ankle_constraint_world, 
                     video_run_dir):
    """
    Create plots for ankle forces including scalar forces and world force magnitudes.
    
    Args:
        all_mj_ankle_actuator: Array of shape (N, 4) - scalar actuator forces
        all_mj_ankle_constraint: Array of shape (N, 4) - scalar constraint forces  
        all_mj_ankle_actuator_world: Array of shape (N, 4, 3) - actuator world forces
        all_mj_ankle_constraint_world: Array of shape (N, 4, 3) - constraint world forces
        video_run_dir: Directory to save plots
    """
    # Actuator forces plot
    print("Plotting ankle forces...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()
    
    for dim in range(4):
        ax = axes_flat[dim]
        force_data = all_mj_ankle_actuator[:, dim]
        mean_val = np.mean(force_data)
        std_val = np.std(force_data)
        ax.hist(force_data, alpha=0.7, bins=50, color='blue')
        ax.set_title(f'Ankle {dim} - Actuator Forces (μ={mean_val:.3f}, σ={std_val:.3f})')
        ax.set_xlabel('Force Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(video_run_dir, "mj_actuator_forces.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Constraint forces plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()
    
    for dim in range(4):
        ax = axes_flat[dim]
        force_data = all_mj_ankle_constraint[:, dim]
        mean_val = np.mean(force_data)
        std_val = np.std(force_data)
        ax.hist(force_data, alpha=0.7, bins=50, color='red')
        ax.set_title(f'Ankle {dim} - Constraint Forces (μ={mean_val:.3f}, σ={std_val:.3f})')
        ax.set_xlabel('Force Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(video_run_dir, "mj_constraint_forces.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Actuator world forces plot - lateral (x,y) and z magnitudes
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    for ankle_idx in range(4):
        # Lateral (x,y) magnitude
        lateral_magnitude = np.sqrt(all_mj_ankle_actuator_world[:, ankle_idx, 0]**2 + 
                                  all_mj_ankle_actuator_world[:, ankle_idx, 1]**2)
        mean_val = np.mean(lateral_magnitude)
        std_val = np.std(lateral_magnitude)
        axes[ankle_idx, 0].hist(lateral_magnitude, alpha=0.7, bins=50, color='blue')
        axes[ankle_idx, 0].set_title(f'Ankle {ankle_idx} - Actuator Lateral Force Magnitude (x,y) (μ={mean_val:.3f}, σ={std_val:.3f})')
        axes[ankle_idx, 0].set_xlabel('Force Magnitude')
        axes[ankle_idx, 0].set_ylabel('Frequency')
        axes[ankle_idx, 0].grid(True, alpha=0.3)
        
        # Z magnitude
        z_magnitude = np.abs(all_mj_ankle_actuator_world[:, ankle_idx, 2])
        mean_val = np.mean(z_magnitude)
        std_val = np.std(z_magnitude)
        axes[ankle_idx, 1].hist(z_magnitude, alpha=0.7, bins=50, color='blue')
        axes[ankle_idx, 1].set_title(f'Ankle {ankle_idx} - Actuator Z Force Magnitude (μ={mean_val:.3f}, σ={std_val:.3f})')
        axes[ankle_idx, 1].set_xlabel('Force Magnitude')
        axes[ankle_idx, 1].set_ylabel('Frequency')
        axes[ankle_idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(video_run_dir, "mj_actuator_world_forces.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Constraint world forces plot - lateral (x,y) and z magnitudes
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    for ankle_idx in range(4):
        # Lateral (x,y) magnitude
        lateral_magnitude = np.sqrt(all_mj_ankle_constraint_world[:, ankle_idx, 0]**2 + 
                                  all_mj_ankle_constraint_world[:, ankle_idx, 1]**2)
        mean_val = np.mean(lateral_magnitude)
        std_val = np.std(lateral_magnitude)
        axes[ankle_idx, 0].hist(lateral_magnitude, alpha=0.7, bins=50, color='red')
        axes[ankle_idx, 0].set_title(f'Ankle {ankle_idx} - Constraint Lateral Force Magnitude (x,y) (μ={mean_val:.3f}, σ={std_val:.3f})')
        axes[ankle_idx, 0].set_xlabel('Force Magnitude')
        axes[ankle_idx, 0].set_ylabel('Frequency')
        axes[ankle_idx, 0].grid(True, alpha=0.3)
        
        # Z magnitude
        z_magnitude = np.abs(all_mj_ankle_constraint_world[:, ankle_idx, 2])
        mean_val = np.mean(z_magnitude)
        std_val = np.std(z_magnitude)
        axes[ankle_idx, 1].hist(z_magnitude, alpha=0.7, bins=50, color='red')
        axes[ankle_idx, 1].set_title(f'Ankle {ankle_idx} - Constraint Z Force Magnitude (μ={mean_val:.3f}, σ={std_val:.3f})')
        axes[ankle_idx, 1].set_xlabel('Force Magnitude')
        axes[ankle_idx, 1].set_ylabel('Frequency')
        axes[ankle_idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(video_run_dir, "mj_constraint_world_forces.png"), dpi=150, bbox_inches='tight')
    plt.close()


def get_ant_ankle_force(qfrc, xmat, model) -> np.ndarray:
    """
    Get the qfrc of the ant's ankles.
    """
    ankle_actuator_indices = [7,9,11,13 ]

    all_ankle_force = []
    all_ankle_world_force = []
    for idx in ankle_actuator_indices:
        ankle_force = qfrc[idx]
        # To convert to 3D: you need the joint axis direction in world frame
        joint_id = model.dof_jntid[idx]
        axis_local = model.jnt_axis[joint_id]  # Axis in joint frame

        # Transform to world frame using body rotation
        body_id = model.jnt_bodyid[joint_id]
        axis_world = xmat[body_id].reshape(3, 3) @ axis_local 

        torque_vector_world = ankle_force * axis_world 

        all_ankle_world_force.append(torque_vector_world)
        all_ankle_force.append(ankle_force)

    return np.array(all_ankle_force), np.array(all_ankle_world_force)

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    is_monitor_wrapped: bool = False,
    is_action_translator: bool = False,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Taken from stable_baselines3.common.evaluation.evaluate_policy, but modified for non-vectorized environments.
    Also works with the ActionTranslator model.
    
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    if hasattr(env, "num_envs"):
        n_envs = env.num_envs
    else:
        n_envs = 1
    episode_rewards = []
    episode_lengths = []

    episode_count = 0
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observation, info = env.reset()
    with tqdm(total=n_eval_episodes, desc="Evaluating policy") as pbar:
        while episode_count < n_eval_episodes:
            if is_action_translator:
                # ActionTranslator returns (action, state) tuple
                full_observation = get_state_from_obs(observation, info, env.spec.id)

                actions, base_action = model.predict_base_and_translated(policy_observation=observation, translator_observation=full_observation, deterministic=deterministic)
            else:
                # Regular PPO model
                actions, _ = model.predict(observation=observation, deterministic=deterministic)

            # Only support single env eval for now
            if len(actions.shape) > 1:
                actions = actions[0]

            new_observation, reward, terminated, truncated, info = env.step(actions)
            current_rewards += reward
            current_lengths += 1

            # unpack values so that the callback can access the local variables
            reward = reward
            done = terminated or truncated
            info = info

            if callback is not None:
                callback(locals(), globals())

            if terminated or truncated:
                if is_monitor_wrapped:
                    # Atari wrapper can send a "done" signal when
                    # the agent loses a life, but it does not correspond
                    # to the true end of episode
                    if "episode" in info.keys():
                        # Do not trust "done" with episode endings.
                        # Monitor wrapper includes "episode" key in info if environment
                        # has been wrapped with it. Use those rewards instead.
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        # Only increment at the real end of an episode                        
                else:
                    episode_rewards.append(current_rewards)
                    episode_lengths.append(current_lengths)
                episode_count += 1
                pbar.update(1)
                observation, info = env.reset()
                current_rewards = 0
                current_lengths = 0

            observation = new_observation

            if render:
                env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def plot_episode_rewards(all_episode_rewards, video_run_dir):
    """
    Create plots of reward over time for each episode.
    
    Args:
        all_episode_rewards: List of lists, where each inner list contains rewards for one episode
        video_run_dir: Directory to save plots
    """
    print("Plotting episode rewards...")
    
    # Create individual plots for each episode
    for episode_idx, episode_rewards in enumerate(all_episode_rewards):
        if len(episode_rewards) == 0:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot reward over time
        steps = np.arange(len(episode_rewards))
        ax.plot(steps, episode_rewards, 'b-', linewidth=2, alpha=0.8)
        ax.fill_between(steps, episode_rewards, alpha=0.3)
        
        # Add cumulative reward line
        cumulative_rewards = np.cumsum(episode_rewards)
        ax2 = ax.twinx()
        ax2.plot(steps, cumulative_rewards, 'r--', linewidth=2, alpha=0.8, label='Cumulative Reward')
        
        # Formatting
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward', color='b')
        ax2.set_ylabel('Cumulative Reward', color='r')
        ax.set_title(f'Episode {episode_idx + 1} - Reward Over Time\nTotal Reward: {cumulative_rewards[-1]:.2f}')
        ax.grid(True, alpha=0.3)
        
        # Add legend for cumulative reward
        ax2.legend(loc='upper right')
        
        # Save plot
        plot_path = os.path.join(video_run_dir, f"reward_episode_{episode_idx}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create a combined plot showing all episodes
    if len(all_episode_rewards) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: All episode rewards overlaid
        ax1 = axes[0]
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(all_episode_rewards)))
        for episode_idx, (episode_rewards, color) in enumerate(zip(all_episode_rewards, colors)):
            if len(episode_rewards) == 0:
                continue
            steps = np.arange(len(episode_rewards))
            ax1.plot(steps, episode_rewards, color=color, linewidth=2, alpha=0.7, 
                    label=f'Episode {episode_idx + 1}')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('All Episodes - Reward Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode total rewards comparison
        ax2 = axes[1]
        episode_totals = [sum(rewards) for rewards in all_episode_rewards if len(rewards) > 0]
        episode_indices = list(range(1, len(episode_totals) + 1))
        
        bars = ax2.bar(episode_indices, episode_totals, color=colors[:len(episode_totals)], alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Episode Total Rewards Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, total in zip(bars, episode_totals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{total:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        combined_plot_path = os.path.join(video_run_dir, "all_episodes_rewards.png")
        plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined reward plot to: {combined_plot_path}")


def compute_frequency(action_values):
    """
    Compute the frequency of sign changes in the first action dimension.
    
    Args:
        action_values: List of action arrays for the first episode
    
    Returns:
        Frequency of sign changes (0 to 1)
    """
    if len(action_values) == 0:
        return 0.0
    
    # Convert to numpy array and get first dimension
    actions_array = np.array(action_values)
    if actions_array.ndim == 1:
        actions_array = actions_array.reshape(1, -1)
    
    if actions_array.shape[1] == 0:
        return 0.0
    
    action_values_dim_0 = actions_array[:, 0]
    
    if len(action_values_dim_0) < 2:
        return 0.0
    
    period_counter = 0
    for i in range(len(action_values_dim_0) - 1):
        current_action = action_values_dim_0[i]
        next_action = action_values_dim_0[i + 1]
        
        # Count sign changes from positive to negative
        if current_action > 0 and next_action < 0:
            period_counter += 1
    
    return period_counter / (len(action_values_dim_0) - 1)



def create_action_dim_overlay_frame(frame, action_values, step_idx, total_steps, action_dim=0):
    """
    Create an overlay frame with action dimension plot for a specific step.
    
    Args:
        frame: Video frame (numpy array)
        action_values: Array of action values for the first dimension
        step_idx: Current step index
        total_steps: Total number of steps
        action_dim: Which action dimension to plot (default: 0)
    
    Returns:
        Modified frame with action plot overlay
    """
    
    # Create a figure for the overlay - make it wider
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot the action dimension up to current step
    steps = np.arange(min(step_idx + 1, len(action_values)))
    current_actions = action_values[:step_idx + 1]
    
    ax.plot(steps, current_actions, 'b-', linewidth=3, alpha=0.8)
    ax.fill_between(steps, current_actions, alpha=0.3, color='blue')
    
    # Highlight current step
    if step_idx < len(action_values):
        ax.scatter([step_idx], [action_values[step_idx]], color='red', s=80, zorder=5)
    
    # Formatting
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(f'Action Dim {action_dim}', fontsize=12)
    ax.set_title(f'Action Dim {action_dim} Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_steps - 1)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Convert plot to image
    fig.canvas.draw()
    # Get the RGB data from the canvas
    buf = fig.canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    # Convert RGBA to RGB
    plot_img = plot_img[:, :, :3]
    plt.close(fig)
    
    # Resize plot to fit in video frame - make it wider
    target_height = min(150, frame.shape[0] // 3)
    target_width = min(400, frame.shape[1] // 2)
    
    # Simple resize using numpy (nearest neighbor)
    height_ratio = target_height / plot_img.shape[0]
    width_ratio = target_width / plot_img.shape[1]
    
    new_height = int(plot_img.shape[0] * height_ratio)
    new_width = int(plot_img.shape[1] * width_ratio)
    
    # Resize plot image
    resized_plot = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    for i in range(min(new_height, target_height)):
        for j in range(min(new_width, target_width)):
            src_i = int(i / height_ratio)
            src_j = int(j / width_ratio)
            if src_i < plot_img.shape[0] and src_j < plot_img.shape[1]:
                resized_plot[i, j] = plot_img[src_i, src_j]
    
    # Create overlay frame
    overlay_frame = frame.copy()
    
    # Position the plot in the top center area
    start_y = 10
    start_x = (frame.shape[1] - target_width) // 2  # Center horizontally
    
    # Add semi-transparent background
    overlay_frame[start_y:start_y + target_height, start_x:start_x + target_width] = \
        0.7 * overlay_frame[start_y:start_y + target_height, start_x:start_x + target_width] + \
        0.3 * resized_plot
    
    return overlay_frame



def plot_episode_trajectories(all_episode_positions, video_run_dir):
    """
    Create separate plots of x vs z trajectories for each episode and combined.
    
    Args:
        all_episode_positions: List of lists, where each inner list contains (x, z) tuples for one episode
        video_run_dir: Directory to save plots
    """
    print("Plotting episode trajectories...")
    
    # Create individual trajectory plots for each episode
    for episode_idx, episode_positions in enumerate(all_episode_positions):
        if len(episode_positions) == 0:
            continue
            
        # Extract x and z coordinates
        x_coords = [pos[0] for pos in episode_positions]
        z_coords = [pos[1] for pos in episode_positions]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot X vs Z trajectory
        ax.plot(x_coords, z_coords, 'g-', linewidth=3, alpha=0.8)
        ax.scatter(x_coords[0], z_coords[0], color='green', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(x_coords[-1], z_coords[-1], color='red', s=150, marker='x', label='End', zorder=5)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Z Position', fontsize=12)
        ax.set_title(f'Episode {episode_idx} - Trajectory (X vs Z)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        # Remove equal aspect ratio to prevent skinny plots when scales differ
        
        plt.tight_layout()
        plot_path = os.path.join(video_run_dir, f"trajectory_episode_{episode_idx}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create a combined trajectory plot showing all episodes
    if len(all_episode_positions) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(all_episode_positions)))
        for episode_idx, (episode_positions, color) in enumerate(zip(all_episode_positions, colors)):
            if len(episode_positions) == 0:
                continue
            x_coords = [pos[0] for pos in episode_positions]
            z_coords = [pos[1] for pos in episode_positions]
            ax.plot(x_coords, z_coords, color=color, linewidth=3, alpha=0.8, 
                    label=f'Episode {episode_idx}')
            # Mark start and end points
            ax.scatter(x_coords[0], z_coords[0], color=color, s=100, marker='o', zorder=5)
            ax.scatter(x_coords[-1], z_coords[-1], color=color, s=100, marker='x', zorder=5)
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Z Position', fontsize=12)
        ax.set_title('All Episodes - Trajectories (X vs Z)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        # Remove equal aspect ratio to prevent skinny plots when scales differ
        
        plt.tight_layout()
        combined_plot_path = os.path.join(video_run_dir, "all_episodes_trajectories.png")
        plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined trajectory plot to: {combined_plot_path}")


def plot_episode_positions(all_episode_positions, video_run_dir):
    """
    Create plots of x and z positions over time for each episode.
    
    Args:
        all_episode_positions: List of lists, where each inner list contains (x, z) tuples for one episode
        video_run_dir: Directory to save plots
    """
    print("Plotting episode positions...")
    
    # Create individual plots for each episode
    for episode_idx, episode_positions in enumerate(all_episode_positions):
        if len(episode_positions) == 0:
            continue
            
        # Extract x and z coordinates
        x_coords = [pos[0] for pos in episode_positions]
        z_coords = [pos[1] for pos in episode_positions]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: X position over time
        steps = np.arange(len(x_coords))
        axes[0].plot(steps, x_coords, 'b-', linewidth=2, alpha=0.8)
        axes[0].fill_between(steps, x_coords, alpha=0.3)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('X Position')
        axes[0].set_title(f'Episode {episode_idx} - X Position Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Z position over time
        axes[1].plot(steps, z_coords, 'r-', linewidth=2, alpha=0.8)
        axes[1].fill_between(steps, z_coords, alpha=0.3, color='red')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Z Position')
        axes[1].set_title(f'Episode {episode_idx} - Z Position Over Time')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Distance from origin over time
        distances = [np.sqrt(x**2 + z**2) for x, z in episode_positions]
        axes[2].plot(steps, distances, 'purple', linewidth=2, alpha=0.8)
        axes[2].fill_between(steps, distances, alpha=0.3, color='purple')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Distance from Origin')
        axes[2].set_title(f'Episode {episode_idx} - Distance from Origin')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(video_run_dir, f"position_episode_{episode_idx}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create a combined plot showing all episodes
    if len(all_episode_positions) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: All X positions overlaid
        ax1 = axes[0]
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(all_episode_positions)))
        for episode_idx, (episode_positions, color) in enumerate(zip(all_episode_positions, colors)):
            if len(episode_positions) == 0:
                continue
            x_coords = [pos[0] for pos in episode_positions]
            steps = np.arange(len(x_coords))
            ax1.plot(steps, x_coords, color=color, linewidth=2, alpha=0.7, 
                    label=f'Episode {episode_idx}')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('X Position')
        ax1.set_title('All Episodes - X Position Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All Z positions overlaid
        ax2 = axes[1]
        for episode_idx, (episode_positions, color) in enumerate(zip(all_episode_positions, colors)):
            if len(episode_positions) == 0:
                continue
            z_coords = [pos[1] for pos in episode_positions]
            steps = np.arange(len(z_coords))
            ax2.plot(steps, z_coords, color=color, linewidth=2, alpha=0.7, 
                    label=f'Episode {episode_idx}')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Z Position')
        ax2.set_title('All Episodes - Z Position Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final x-axis distances comparison
        ax3 = axes[2]
        final_x_distances = []
        episode_indices = []
        for episode_idx, episode_positions in enumerate(all_episode_positions):
            if len(episode_positions) == 0:
                continue
            final_pos = episode_positions[-1]
            final_x_distance = abs(final_pos[0])  # absolute x-axis distance
            final_x_distances.append(final_x_distance)
            episode_indices.append(episode_idx)
        
        bars = ax3.bar(episode_indices, final_x_distances, color=colors[:len(final_x_distances)], alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Final X-Axis Distance')
        ax3.set_title('Final X-Axis Distance by Episode')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, distance in zip(bars, final_x_distances):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{distance:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        combined_plot_path = os.path.join(video_run_dir, "all_episodes_positions.png")
        plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined position plot to: {combined_plot_path}")


def evaluate_and_record(
    translator_policy_config_path: str = None,
    base_policy_config_path: str = None,
    env_id: str = "InvertedPendulum-v5",
    source_policy_checkpoint: str = None,
    action_translator_checkpoint: str = None,
    env_kwargs: dict = {},
    run_dir: str | None = None,
    num_episodes: int = 5,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    seed: int | None = None,
    config: dict = None,
):
    """
    Evaluate a policy (PPO or ActionTranslator) and record videos.
    
    Args:
        translator_policy_config_path: Path to ActionTranslator config YAML file (for ActionTranslator models)
        base_policy_config_path: Path to source policy config YAML file (for standalone source policies)
        source_policy_checkpoint: Path to source policy checkpoint (overrides config checkpoint)
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
    model_types = [translator_policy_config_path, base_policy_config_path]
    non_none_models = [m for m in model_types if m is not None]
    
    if len(non_none_models) > 1:
        raise ValueError("Cannot specify multiple model types. Choose one: translator_policy_config_path or base_policy_config_path.")
    elif len(non_none_models) == 0:
        raise ValueError("Must specify one model type: translator_policy_config_path (for ActionTranslator) or base_policy_config_path (for standalone source policy).")
    
    is_action_translator = translator_policy_config_path is not None
    is_source_policy = base_policy_config_path is not None
    
    # Determine source policy checkpoint path for video directory
    source_checkpoint_path = None
    if is_action_translator:
        # For ActionTranslator, resolve the config to get the source policy checkpoint
        source_checkpoint_path = resolve_hydra_config_and_get_source_checkpoint(
            translator_policy_config_path, source_policy_checkpoint
        )
    elif is_source_policy:
        # For source policy, load the config to get the checkpoint path
        with open(base_policy_config_path, 'r') as f:
            config = yaml.safe_load(f)
        source_checkpoint_path = config.get('checkpoint_path')
        if source_policy_checkpoint:
            source_checkpoint_path = source_policy_checkpoint
    
    if run_dir is not None:
        video_dir = os.path.join(run_dir, "videos")
    elif source_checkpoint_path:
        # Extract directory from checkpoint path and create videos/eval/{env_id}-{time} subfolder
        checkpoint_dir = os.path.dirname(os.path.dirname(source_checkpoint_path))
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_stamp = f"{env_id}-{run_stamp}"
        if is_action_translator:
            run_stamp = f"{run_stamp}-action_translator"

        video_dir = os.path.join(checkpoint_dir, "videos", "eval", run_stamp)
    else:
        # Fallback to current directory
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join("videos", "eval", f"{env_id}-{run_stamp}")

    os.makedirs(video_dir, exist_ok=True)

    # Ensure we get rgb frames for video generation
    video_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    video_env = Monitor(video_env)

    # Record every episode to the video directory (already timestamped)
    video_run_dir = video_dir
    video_env = RecordVideo(
        video_env,
        video_folder=video_run_dir,
        episode_trigger=lambda ep_id: ep_id < 5,
        name_prefix="evaluation",
        disable_logger=True,
    )

    # Load model based on type
    if is_action_translator:
        print("Loading ActionTranslator model...")

        model = load_action_translator_policy_from_config(
            translator_policy_config_path,
            source_policy_checkpoint=source_policy_checkpoint,
            action_translator_checkpoint=action_translator_checkpoint
        )
        print("Model loaded successfully!")
        
        print_model_info(model)
    else:
        print("Loading source policy from config...")

        model = load_source_policy_from_config(
            base_policy_config_path,
            source_policy_checkpoint=source_policy_checkpoint
        )
        print("Model loaded successfully!")
        
        # Print parameter counts and model info
        print_model_info(model)

    # Quick quantitative evaluation (no video)
    eval_env = gym.make(env_id, **env_kwargs)
    eval_env = Monitor(eval_env)

    # mean_reward, std_reward = evaluate_policy(
    #     model,
    #     eval_env,
    #     n_eval_episodes=num_episodes,
    #     deterministic=deterministic,
    #     render=False,
    #     warn=False,
    #     is_monitor_wrapped=True,
    #     is_action_translator=is_action_translator
    # )

    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over {num_episodes} episodes")

    # Rollout episodes with video recording
    all_actions = []
    all_translated_actions = []
    all_returns = []
    all_mj_actuator = []
    all_mj_constraint = []
    all_mj_xmat = []
    all_episode_rewards = []  # Track rewards per episode
    all_episode_positions = []  # Track positions per episode (only for ant environments)
    is_ant_env = "ant" in env_id.lower()
    
    # Track actions for first episode only
    first_episode_actions = []
    first_episode_translated_actions = []  # For action translators
    first_episode_tracked = False
    
    for episode_idx in range(min(5, num_episodes)):
        obs, info = video_env.reset(seed=seed)
        episode_return = 0.0
        episode_rewards = []  # Track rewards for this episode
        episode_positions = [] if is_ant_env else None  # Track positions for this episode (only for ant environments)
        for step in range(max_steps_per_episode):
            if is_action_translator:
                # ActionTranslator returns (action, state) tuple
                full_observation = get_state_from_obs(obs, info, env_id)

                translated_action, base_action = model.predict_base_and_translated(policy_observation=obs, translator_observation=full_observation, deterministic=deterministic)
                
                
                if len(translated_action.shape) > 1:
                    translated_action = translated_action[0]
                    
                if len(base_action.shape) > 1:
                    base_action = base_action[0]
                    
                all_actions.append(base_action)
                all_translated_actions.append(translated_action)
                
                # Track actions for first episode
                if episode_idx == 0 and not first_episode_tracked:
                    first_episode_actions.append(base_action.copy())
                    first_episode_translated_actions.append(translated_action.copy())

                action_to_step = translated_action
            else:
                # Regular PPO model
                action, _ = model.predict(obs, deterministic=deterministic)
                action_to_step = action
                all_actions.append(action)
                
                # Track actions for first episode
                if episode_idx == 0 and not first_episode_tracked:
                    first_episode_actions.append(action.copy())
                    print(f"Collected action {len(first_episode_actions)}: action shape {action.shape}")

            if len(action_to_step.shape) > 1:
                action_to_step = action_to_step[0]

            obs, reward, terminated, truncated, info = video_env.step(action_to_step)

            # save the actuator forces, constraint forces, and xmat (body rotation matrix)   
            all_mj_actuator.append(video_env.unwrapped.data.qfrc_actuator.copy())
            all_mj_constraint.append(video_env.unwrapped.data.qfrc_constraint.copy())
            all_mj_xmat.append(video_env.unwrapped.data.xmat.copy())

            episode_return += float(reward)
            episode_rewards.append(float(reward))  # Track reward for this step
            
            # Track positions from info dictionary and observation (only for ant environments)
            if is_ant_env and episode_positions is not None:
                x_pos = info.get('x_position', 0.0)
                z_pos = obs[0] if len(obs) > 0 else 0.0  # z coordinate from first observation
                episode_positions.append((x_pos, z_pos))
            
            if terminated or truncated:
                break
        # Calculate final x position displacement for ant environments
        final_x_displacement = 0.0
        if is_ant_env and episode_positions is not None and len(episode_positions) > 0:
            final_x_displacement = episode_positions[-1][0]  # x coordinate of final position
            all_episode_positions.append(episode_positions)  # Store positions for this episode
        
        print(f"Episode {episode_idx + 1}/{min(5, num_episodes)} return: {episode_return:.2f}, final x displacement: {final_x_displacement:.2f}")
        all_returns.append(episode_return)
        all_episode_rewards.append(episode_rewards)  # Store rewards for this episode
        
        # Mark first episode as tracked
        if episode_idx == 0:
            first_episode_tracked = True

    # Plot results
    all_actions = np.array(all_actions)
    all_returns = np.array(all_returns)
    
    # Plot episode rewards
    plot_episode_rewards(all_episode_rewards, video_run_dir)
    
    # Plot episode positions (only for ant environments)
    if is_ant_env and len(all_episode_positions) > 0:
        plot_episode_positions(all_episode_positions, video_run_dir)
        plot_episode_trajectories(all_episode_positions, video_run_dir)

    if len(all_mj_actuator) > 0 and "Ant" in env_id:
        mj_model = video_env.unwrapped.model

        all_mj_ankle_actuator = []
        all_mj_ankle_constraint = []
        all_mj_ankle_actuator_world = []
        all_mj_ankle_constraint_world = []
        for qfrc,constraint, xmat in zip(all_mj_actuator, all_mj_constraint, all_mj_xmat):
            mj_ankle_actuator, mj_ankle_actuator_world = get_ant_ankle_force(qfrc, xmat, mj_model)
            mj_ankle_constraint, mj_ankle_constraint_world = get_ant_ankle_force(constraint, xmat, mj_model)

            all_mj_ankle_actuator.append(mj_ankle_actuator)
            all_mj_ankle_constraint.append(mj_ankle_constraint)
            all_mj_ankle_actuator_world.append(mj_ankle_actuator_world)
            all_mj_ankle_constraint_world.append(mj_ankle_constraint_world)

        all_mj_ankle_actuator = np.array(all_mj_ankle_actuator)
        all_mj_ankle_constraint = np.array(all_mj_ankle_constraint)
        all_mj_ankle_actuator_world = np.array(all_mj_ankle_actuator_world)
        all_mj_ankle_constraint_world = np.array(all_mj_ankle_constraint_world)

        # world are shape (N, 4, 3), and actuator are shape (N, 4)
        # Create plots for ankle forces
        # plot_ankle_forces(all_mj_ankle_actuator, all_mj_ankle_constraint, 
        #                  all_mj_ankle_actuator_world, all_mj_ankle_constraint_world, 
        #                  video_run_dir)
    
    
    # Export GIFs alongside MP4s, limiting GIFs to at most 100 frames
    mp4_files = sorted(Path(video_run_dir).glob("evaluation-episode-*.mp4"))
    for mp4_file in mp4_files:
        reader = imageio.get_reader(str(mp4_file))
        meta = reader.get_meta_data()
        fps = meta["fps"] if "fps" in meta else 30
        frames = []
        for idx, frame in enumerate(reader):
            if idx >= 100:
                break
            frames.append(frame)
        if len(frames) > 0:
            gif_path = mp4_file.with_suffix(".gif")
            imageio.mimsave(str(gif_path), frames, duration=1.0 / fps)
    

    freq = compute_frequency(first_episode_actions)
    print(f"METRIC: Frequency of first episode actions: {freq}")

    # Add action dimension overlay to the first video
    if len(first_episode_actions) > 0 and len(mp4_files) > 0:
        print("Adding action dimension overlay to first video...")
        first_video_path = mp4_files[0]  # First video (episode 0)
        
        # Convert actions to numpy array
        actions_array = np.array(first_episode_actions)
        if actions_array.ndim == 1:
            actions_array = actions_array.reshape(1, -1)
        
        # Get first action dimension
        first_action_dim = actions_array[:, 0] if actions_array.shape[1] > 0 else np.array([])
        
        if len(first_action_dim) > 0:
            # Read the original video
            reader = imageio.get_reader(str(first_video_path))
            meta = reader.get_meta_data()
            fps = meta["fps"] if "fps" in meta else 30
            
            # Create output video with overlay
            output_path = str(first_video_path).replace('.mp4', '_with_action_plot.mp4')
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
            
            total_frames = len(first_action_dim)
            for idx, frame in enumerate(reader):
                if idx >= total_frames:
                    break
                
                # Create overlay frame
                overlay_frame = create_action_dim_overlay_frame(
                    frame, first_action_dim, idx, total_frames, action_dim=0
                )
                
                writer.append_data(overlay_frame)
            
            writer.close()
            reader.close()
            print(f"Created video with action plot overlay: {output_path}")
            
            # Save the final action dimension plot as a separate PNG
            final_plot_path = os.path.join(video_run_dir, "first_episode_action_dim_0_plot.png")
            fig, ax = plt.subplots(figsize=(12, 6))
            steps = np.arange(len(first_action_dim))
            ax.plot(steps, first_action_dim, 'b-', linewidth=3, alpha=0.8)
            ax.fill_between(steps, first_action_dim, alpha=0.3, color='blue')
            ax.set_xlabel('Step', fontsize=14)
            ax.set_ylabel('Action Dim 0', fontsize=14)
            ax.set_title('First Episode - Action Dimension 0 Over Time', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved action dimension plot as PNG: {final_plot_path}")
            
            # Also create GIF version
            gif_output_path = output_path.replace('.mp4', '.gif')
            reader = imageio.get_reader(output_path)
            gif_frames = []
            for idx, frame in enumerate(reader):
                if idx >= 100:  # Limit GIF to 100 frames
                    break
                gif_frames.append(frame)
            if len(gif_frames) > 0:
                imageio.mimsave(gif_output_path, gif_frames, duration=1.0 / fps)
                print(f"Created GIF with action plot overlay: {gif_output_path}")
            reader.close()

    print(f"Saved videos to: {video_run_dir}")
    print(f"Mean return: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy (PPO or ActionTranslator) and save videos.")
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--base_policy_config", help="Path to the source policy config YAML file")
    model_group.add_argument("--translator_policy_config", help="Path to the ActionTranslator config YAML file")
    
    # Environment arguments
    parser.add_argument("--env_id", default="InvertedPendulum-v5", help="Gymnasium env id")
    parser.add_argument("--episodes", type=int, default=32, help="Number of episodes to eval (max 5 video recorded)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions instead of stochastic")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for resets")
    
    # ActionTranslator specific arguments
    parser.add_argument("--source_policy_checkpoint", help="Path to base policy checkpoint (overrides config for ActionTranslator)")
    parser.add_argument("--action_translator_checkpoint", help="Path to action translator checkpoint (overrides config for ActionTranslator)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_cluster_graphics_vars()
    register_custom_envs()

    evaluate_and_record(
        translator_policy_config_path=args.translator_policy_config,
        base_policy_config_path=args.base_policy_config,
        source_policy_checkpoint=args.source_policy_checkpoint,
        action_translator_checkpoint=args.action_translator_checkpoint,
        env_id=args.env_id,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
