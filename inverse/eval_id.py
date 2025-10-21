import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor
import torch
from tqdm import tqdm
import sys
import imageio.v2 as imageio
from pathlib import Path

from cluster_utils import set_cluster_graphics_vars


# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import load_source_policy_from_config, load_inverse_dynamics_model_from_config
from physics_inverse_dynamics import gym_inverse_dynamics
from envs.register_envs import register_custom_envs
from envs.env_utils import get_state_from_obs, make_env, VideoCallback


def rollout_policy(env, policy, deterministic=False, max_steps=1000):
    """
    Rollout a policy in the environment and collect trajectory data.
    
    Args:
        env: Gym environment
        policy: Policy to rollout
        max_steps: Maximum number of steps
    
    Returns:
        Dictionary containing states, actions, next_states, rewards, dones
    """
    print("Rolling out policy...")
    
    obs, info = env.reset()
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    
    for _ in tqdm(range(max_steps), desc="Rollout"):
        # Get action from policy
        action, _ = policy.predict(obs, deterministic=deterministic)
        
        # Store current state and action
        state = get_state_from_obs(obs, info, env.spec.id)
        states.append(state.copy())
        actions.append(action.copy())
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store next state and reward
        next_state = get_state_from_obs(next_obs, info, env.spec.id)
        next_states.append(next_state.copy())
        rewards.append(reward)
        dones.append(done)
        
        # Update observation
        obs = next_obs
        
        if done:
            break
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states),
        'rewards': np.array(rewards),
        'dones': np.array(dones)
    }


def evaluate_inverse_dynamics(trajectory, learned_model, physics_env, compute_physics_id=True, device='cpu'):
    """
    Evaluate both learned and physics-based inverse dynamics on the trajectory.
    
    Args:
        trajectory: Dictionary containing states, actions, next_states
        learned_model: Learned inverse dynamics model
        physics_env: Environment for physics-based inverse dynamics
        device: Device for learned model
    
    Returns:
        Dictionary containing errors and predicted actions for both methods
    """
    print("Evaluating inverse dynamics...")
    
    states = trajectory['states']
    true_actions = trajectory['actions']
    next_states = trajectory['next_states']
    
    learned_errors = []
    physics_errors = []
    learned_actions = []
    physics_actions = []
    
    # Move learned model to device
    learned_model.to(device)
    learned_model.eval()
    
    for i in tqdm(range(len(states)), desc="Computing inverse dynamics"):
        state = states[i]
        next_state = next_states[i]
        true_action = true_actions[i]

        
        # Learned inverse dynamics
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            predicted_action = learned_model.predict(state_tensor, next_state_tensor)
            predicted_action = predicted_action[0]  # Remove batch dimension

    
        
        learned_error = np.linalg.norm(predicted_action - true_action)
        learned_errors.append(learned_error)
        learned_actions.append(predicted_action)
        
        # Physics-based inverse dynamics
        if compute_physics_id:
            physics_action = gym_inverse_dynamics(physics_env, state, next_state)
            physics_error = np.linalg.norm(physics_action - true_action)
            physics_errors.append(physics_error)
            physics_actions.append(physics_action)
        else:
            physics_errors.append(0.0)
            physics_actions.append(np.zeros_like(true_action))
    
    return {
        'learned_errors': np.array(learned_errors),
        'physics_errors': np.array(physics_errors),
        'learned_actions': np.array(learned_actions),
        'physics_actions': np.array(physics_actions)
    }


def plot_errors(learned_errors, physics_errors, output_path='inverse_dynamics_errors.png'):
    """
    Plot the inverse dynamics errors over time.
    
    Args:
        learned_errors: Array of learned model errors
        physics_errors: Array of physics model errors
        output_path: Path to save the plot
    """
    print("Plotting errors...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Errors over time
    steps = np.arange(len(learned_errors))
    ax1.plot(steps, learned_errors, label='Learned ID', alpha=0.7)
    ax1.plot(steps, physics_errors, label='Physics ID', alpha=0.7)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('Inverse Dynamics Errors Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distributions
    ax2.hist(learned_errors, alpha=0.7, bins=30, label=f'Learned ID (μ={np.mean(learned_errors):.4f})')
    ax2.hist(physics_errors[~np.isnan(physics_errors)], alpha=0.7, bins=30, 
             label=f'Physics ID (μ={np.nanmean(physics_errors):.4f})')
    ax2.set_xlabel('L2 Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Error plots saved to", output_path)
    
    # Print summary statistics
    print("\n=== Error Summary ===")
    print(f"Learned ID - Mean: {np.mean(learned_errors):.4f}, Std: {np.std(learned_errors):.4f}")
    print(f"Physics ID - Mean: {np.nanmean(physics_errors):.4f}, Std: {np.nanstd(physics_errors):.4f}")


def plot_actions_over_time(trajectory, learned_actions, physics_actions, output_path='actions_over_time.png'):
    """
    Plot actions over time for any environment with multiple action dimensions.
    
    Args:
        trajectory: Dictionary containing true actions
        learned_actions: Array of learned model actions
        physics_actions: Array of physics model actions
        output_path: Path to save the plot
    """
    print("Plotting actions over time...")
    
    true_actions = trajectory['actions']
    steps = np.arange(len(true_actions))
    
    # Create figure with subplots for each action dimension
    action_dim = true_actions.shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 4 * action_dim))
    
    # Handle case where action_dim is 1
    if action_dim == 1:
        axes = [axes]
    
    for i in range(action_dim):
        ax = axes[i]
        
        # Plot actions over time
        ax.plot(steps, true_actions[:, i], label='True Actions', alpha=0.8, linewidth=2)
        ax.plot(steps, learned_actions[:, i], label='Learned ID', alpha=0.7, linestyle='--')
        ax.plot(steps, physics_actions[:, i], label='Physics ID', alpha=0.7, linestyle=':')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Action {i+1}')
        ax.set_title(f'Action {i+1} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Action plots saved to", output_path)


def record_trajectory_video(env_id, trajectory, policy, output_dir, deterministic=False):
    """
    Record a video and GIF of the trajectory using the policy.
    
    Args:
        env_id: Environment ID
        trajectory: Dictionary containing the trajectory data
        policy: Policy to use for actions
        output_dir: Directory to save the video
        deterministic: Whether to use deterministic actions
    """
    print("Recording trajectory video...")
    
    # Create video environment with RGB rendering
    video_env = make_env(env_id, render=True)
    
    
    # Record video
    video_path = os.path.join(output_dir, "trajectory_video")
    
    video_callback = VideoCallback(video_env,
        video_folder=video_path,
        eval_freq=1,
        name_prefix="trajectory",
        max_steps_per_episode=1000,
        deterministic=True,
        flip_vertical="robosuite" in env_id.lower()
    )
    policy.num_timesteps = 0 # hack to work with video callback
    video_callback.init_callback(policy)
    
    video_callback.on_step()
    video_callback.on_step()
    video_callback.on_step()
    
    video_env.close()
    
    # Convert MP4 to GIF
    mp4_files = sorted(Path(video_path).glob("trajectory-episode-*.mp4"))
    if mp4_files:
        mp4_file = mp4_files[0]
        reader = imageio.get_reader(str(mp4_file))
        meta = reader.get_meta_data()
        fps = meta["fps"] if "fps" in meta else 30
        
        # Limit GIF to reasonable number of frames
        max_frames = 200
        frames = []
        for idx, frame in enumerate(reader):
            if idx >= max_frames:
                break
            frames.append(frame)
        
        if len(frames) > 0:
            gif_path = os.path.join(output_dir, "trajectory.gif")
            imageio.mimsave(gif_path, frames, duration=1.0 / fps)
            print(f"Video saved to: {video_path}")
            print(f"GIF saved to: {gif_path}")
        else:
            print("No frames captured for GIF")
    else:
        print("No MP4 file found to convert to GIF")


def main():
    parser = argparse.ArgumentParser(description='Evaluate inverse dynamics model')
    parser.add_argument('--policy_config', type=str, required=True,
                       help='Path to policy config YAML file')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to inverse dynamics model config YAML file')
    parser.add_argument('--env_id', type=str, default='InvertedPendulumIntegrable-v5',
                       help='Environment ID for evaluation')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum number of rollout steps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for learned model')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save results')
    parser.add_argument('--physics_id', action='store_true', default=False,
                       help='Compute physics ID error')
    parser.add_argument('--deterministic', action='store_true', default=False,
                       help='Use deterministic actions instead of stochastic. Defaults to false for getting more state coverage')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Register custom environments
    register_custom_envs()
    set_cluster_graphics_vars()
    
    # Load policy
    print("Loading policy...")
    policy = load_source_policy_from_config(args.policy_config)
    
    # Load learned inverse dynamics model
    print("Loading inverse dynamics model...")

    learned_model = load_inverse_dynamics_model_from_config(args.model_config, load_checkpoint=True)
    
    # Create environments
    print("Creating environments...")
    rollout_env = make_env(args.env_id)
    physics_env = make_env(args.env_id)  # Separate env for physics ID
    
    # Rollout policy
    trajectory = rollout_policy(rollout_env, policy, deterministic=args.deterministic, max_steps=args.max_steps)
    print(f"Rollout completed: {len(trajectory['states'])} steps")
    
    # Evaluate inverse dynamics
    errors = evaluate_inverse_dynamics(trajectory, learned_model, physics_env, args.physics_id, args.device)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'inverse_dynamics_errors.png')
    plot_errors(errors['learned_errors'], errors['physics_errors'], plot_path)
    
    # Plot actions over time for any environment
    actions_plot_path = os.path.join(args.output_dir, 'actions_over_time.png')
    plot_actions_over_time(trajectory, errors['learned_actions'], errors['physics_actions'], actions_plot_path)
    
    # Record video and GIF of the trajectory
    record_trajectory_video(args.env_id, trajectory, policy, args.output_dir, args.deterministic)
    
    # Save trajectory data
    trajectory_path = os.path.join(args.output_dir, 'trajectory.npz')
    np.savez(trajectory_path, **trajectory, **errors)
    print(f"Trajectory and errors saved to {trajectory_path}")
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
