import os
import argparse
import yaml
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

# Add current directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from envs.register_envs import register_custom_envs
from utils.model_utils import load_action_translator_policy_from_config, load_source_policy_from_config, print_model_info
from envs.env_utils import get_state_from_obs
from cluster_utils import set_cluster_graphics_vars


def collect_trajectory(
    policy,
    env_id: str,
    env_kwargs: Optional[dict] = None,
    max_steps: int = 1000,
    deterministic: bool = True,
    seed: Optional[int] = None,
    is_action_translator: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Collect a single trajectory using the given policy.
    
    Args:
        policy: The policy to use for collecting the trajectory
        env_id: Gymnasium environment ID
        env_kwargs: Environment keyword arguments
        max_steps: Maximum number of steps in the trajectory
        deterministic: Whether to use deterministic actions
        seed: Random seed for environment reset
        is_action_translator: Whether the policy is an ActionTranslator
        
    Returns:
        Tuple of (states, actions, rewards, observations)
    """
    print(f"Collecting trajectory with {'ActionTranslator' if is_action_translator else 'base'} policy...")
    
    # Create environment
    if env_kwargs is None:
        env_kwargs = {}
    env = gym.make(env_id, **env_kwargs)
    env = Monitor(env)
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Storage for trajectory data
    states = []
    actions = []
    rewards = []
    observations = []
    
    for step in range(max_steps):
        # Get state from observation
        state = get_state_from_obs(obs, info, env_id)
        states.append(state.copy())
        observations.append(obs.copy())
        
        # Get action from policy
        if is_action_translator:
            # ActionTranslator returns (translated_action, base_action)
            translated_action, _ = policy.predict_base_and_translated(
                policy_observation=obs, 
                translator_observation=state, 
                deterministic=deterministic
            )
            # Use translated action for the trajectory
            action = translated_action
        else:
            # Regular PPO model
            action, _ = policy.predict(obs, deterministic=deterministic)
        
        # Ensure action is 1D
        if len(action.shape) > 1:
            action = action[0]
        
        actions.append(action.copy())
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        
        # Check if episode ended
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"Collected trajectory with {len(states)} steps")
    print(f"Total reward: {sum(rewards):.2f}")
    
    env.close()
    return states, actions, rewards, observations


def test_policies_on_states(
    policies: Dict[str, Any],
    states: List[np.ndarray],
    observations: List[np.ndarray],
    env_id: str,
    deterministic: bool = True
) -> Dict[str, List[np.ndarray]]:
    """
    Test multiple policies on the collected states.
    
    Args:
        policies: Dictionary mapping policy names to policy objects
        states: List of states from the trajectory
        observations: List of observations from the trajectory
        env_id: Environment ID for consistent state processing
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary mapping policy names to lists of predicted actions
    """
    print(f"Testing {len(policies)} policies on {len(states)} states...")
    print("Ensuring all policies act on the same states to avoid phase shifts...")
    
    # Debug: Print state shapes to verify consistency
    if len(states) > 0:
        print(f"State shape: {states[0].shape}, Observation shape: {observations[0].shape}")
        print(f"First state sample: {states[0][:5]}...")  # First 5 elements
        print(f"First observation sample: {observations[0][:5]}...")  # First 5 elements
    
    predicted_actions = {}
    
    for policy_name, policy in policies.items():
        print(f"Testing policy: {policy_name}")
        policy_actions = []
        
        is_action_translator = hasattr(policy, 'predict_base_and_translated')
        
        for i, (state, obs) in enumerate(zip(states, observations)):
            try:
                # Debug: Print input verification for first few steps
                if i < 3:
                    print(f"  Step {i}: State shape {state.shape}, Obs shape {obs.shape}")
                    print(f"  State sample: {state[:3]}..., Obs sample: {obs[:3]}...")
                
                if is_action_translator:
                    # ActionTranslator - use the exact same state and observation
                    # Ensure we're using the same state that was collected during trajectory
                    translated_action, _ = policy.predict_base_and_translated(
                        policy_observation=obs,
                        translator_observation=state,
                        deterministic=deterministic,
                        mask_source_action=False,
                        mask_observation=False,
                    )
                    # Use translated action
                    action = translated_action
                else:
                    # Regular PPO model - use the same observation
                    # For consistency, we could also use the state if the policy supports it
                    action, _ = policy.predict(obs, deterministic=deterministic)
                
                # Ensure action is 1D
                if len(action.shape) > 1:
                    action = action[0]
                
                policy_actions.append(action.copy())
                
            except (ValueError, RuntimeError, AttributeError) as e:
                print(f"Error predicting action for policy {policy_name} at step {i}: {e}")
                # Use zero action as fallback
                if len(policy_actions) > 0:
                    policy_actions.append(np.zeros_like(policy_actions[-1]))
                else:
                    # First action - need to determine action dimension
                    if is_action_translator:
                        # Try to get action dimension from a working step
                        try:
                            test_action, _ = policy.predict_base_and_translated(
                                policy_observation=obs,
                                translator_observation=state,
                                deterministic=deterministic
                            )
                            if len(test_action.shape) > 1:
                                test_action = test_action[0]
                            policy_actions.append(np.zeros_like(test_action))
                        except (ValueError, RuntimeError, AttributeError):
                            # Last resort - assume 8 dimensions (common for Ant)
                            policy_actions.append(np.zeros(8))
                    else:
                        try:
                            test_action, _ = policy.predict(obs, deterministic=deterministic)
                            if len(test_action.shape) > 1:
                                test_action = test_action[0]
                            policy_actions.append(np.zeros_like(test_action))
                        except (ValueError, RuntimeError, AttributeError):
                            policy_actions.append(np.zeros(8))
        
        predicted_actions[policy_name] = policy_actions
        print(f"Completed testing {policy_name}: {len(policy_actions)} actions predicted")
    
    return predicted_actions


def plot_action_comparisons(
    predicted_actions: Dict[str, List[np.ndarray]],
    original_actions: List[np.ndarray],
    output_dir: str,
    policy_names: List[str] = None
):
    """
    Plot comparisons of predicted actions across different policies.
    
    Args:
        predicted_actions: Dictionary mapping policy names to predicted actions
        original_actions: Original actions from the trajectory
        output_dir: Directory to save plots
        policy_names: Optional list of policy names to include in plots
    """
    print("Creating action comparison plots...")
    
    if policy_names is None:
        policy_names = list(predicted_actions.keys())
    
    # Convert to numpy arrays
    original_actions = np.array(original_actions)
    n_steps, n_dims = original_actions.shape
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Individual dimension plots - one plot per action dimension
    for dim in range(n_dims):
        _, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        steps = np.arange(n_steps)
        
        # Plot original actions
        ax.plot(steps, original_actions[:, dim], 'k-', linewidth=3, alpha=0.8, 
                label='Original Trajectory', marker='o', markersize=4)
        
        # Plot predicted actions for each policy
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(policy_names)))
        for i, policy_name in enumerate(policy_names):
            if policy_name in predicted_actions:
                policy_actions = np.array(predicted_actions[policy_name])
                ax.plot(steps, policy_actions[:, dim], color=colors[i], linewidth=2, 
                       alpha=0.7, label=policy_name, linestyle='--')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(f'Action Dimension {dim}', fontsize=12)
        ax.set_title(f'Action Dimension {dim} - Policy Comparison', fontsize=14)
        #ax.set_ylim(-1.1, 1.1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"action_dim_{dim}_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. All dimensions in one plot - subplot for each dimension
    n_cols = min(4, n_dims)  # Max 4 columns
    n_rows = (n_dims + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        pass  # axes is already a 1D array
    else:
        axes = axes.flatten()
    
    for dim in range(n_dims):
        ax = axes[dim]
        steps = np.arange(n_steps)
        
        # Plot original actions
        ax.plot(steps, original_actions[:, dim], 'k-', linewidth=2, alpha=0.8, 
                label='Original', marker='o', markersize=3)
        
        # Plot predicted actions for each policy
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(policy_names)))
        for i, policy_name in enumerate(policy_names):
            if policy_name in predicted_actions:
                policy_actions = np.array(predicted_actions[policy_name])
                ax.plot(steps, policy_actions[:, dim], color=colors[i], linewidth=1.5, 
                       alpha=0.7, label=policy_name, linestyle='--')
        
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel(f'Dim {dim}', fontsize=10)
        ax.set_title(f'Action Dim {dim}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first subplot
        if dim == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for dim in range(n_dims, len(axes)):
        axes[dim].set_visible(False)
    
    plt.tight_layout()
    all_dims_plot_path = os.path.join(output_dir, "all_action_dims_comparison.png")
    plt.savefig(all_dims_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Action magnitude comparison
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    steps = np.arange(n_steps)
    
    # Calculate action magnitudes
    original_magnitudes = np.linalg.norm(original_actions, axis=1)
    ax.plot(steps, original_magnitudes, 'k-', linewidth=3, alpha=0.8, 
            label='Original Trajectory', marker='o', markersize=4)
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(policy_names)))
    for i, policy_name in enumerate(policy_names):
        if policy_name in predicted_actions:
            policy_actions = np.array(predicted_actions[policy_name])
            policy_magnitudes = np.linalg.norm(policy_actions, axis=1)
            ax.plot(steps, policy_magnitudes, color=colors[i], linewidth=2, 
                   alpha=0.7, label=policy_name, linestyle='--')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Action Magnitude', fontsize=12)
    ax.set_title('Action Magnitude Comparison', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    magnitude_plot_path = os.path.join(output_dir, "action_magnitude_comparison.png")
    plt.savefig(magnitude_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Action difference heatmap - differences from original trajectory
    if len(policy_names) > 0:
        # Calculate differences between each policy and the original trajectory
        policy_arrays = {}
        for policy_name in policy_names:
            if policy_name in predicted_actions:
                policy_arrays[policy_name] = np.array(predicted_actions[policy_name])
        
        if len(policy_arrays) > 0:
            policy_list = list(policy_arrays.keys())
            n_policies = len(policy_list)
            
            # Create difference matrix - each policy vs original trajectory
            # Shape: (n_policies, n_dims) - each row is a policy, each col is an action dimension
            diff_matrix = np.zeros((n_policies, n_dims))
            for i, policy_name in enumerate(policy_list):
                policy_actions = policy_arrays[policy_name]
                for dim in range(n_dims):
                    # Calculate mean squared difference for this dimension
                    diff = np.mean((policy_actions[:, dim] - original_actions[:, dim])**2)
                    diff_matrix[i, dim] = diff
            
            # Plot heatmap
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            im = ax.imshow(diff_matrix, cmap='viridis', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(n_dims))
            ax.set_yticks(range(n_policies))
            ax.set_xticklabels([f'Dim {i}' for i in range(n_dims)], rotation=45, ha='right')
            ax.set_yticklabels(policy_list)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Squared Difference from Original', fontsize=12)
            
            # Add text annotations
            for i in range(n_policies):
                for j in range(n_dims):
                    ax.text(j, i, f'{diff_matrix[i, j]:.3f}',
                           ha="center", va="center", color="white" if diff_matrix[i, j] > diff_matrix.max()/2 else "black")
            
            ax.set_title('Policy Action Differences from Original Trajectory', fontsize=14)
            ax.set_xlabel('Action Dimension', fontsize=12)
            ax.set_ylabel('Policy', fontsize=12)
            plt.tight_layout()
            heatmap_plot_path = os.path.join(output_dir, "policy_vs_original_heatmap.png")
            plt.savefig(heatmap_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also create a summary plot showing overall differences per policy
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Calculate overall difference per policy (mean across all dimensions)
            overall_diffs = np.mean(diff_matrix, axis=1)
            
            bars = ax.bar(range(n_policies), overall_diffs, color=plt.cm.get_cmap('viridis')(np.linspace(0, 1, n_policies)))
            ax.set_xlabel('Policy', fontsize=12)
            ax.set_ylabel('Mean Squared Difference from Original', fontsize=12)
            ax.set_title('Overall Policy Differences from Original Trajectory', fontsize=14)
            ax.set_xticks(range(n_policies))
            ax.set_xticklabels(policy_list, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, diff) in enumerate(zip(bars, overall_diffs)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{diff:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            summary_plot_path = os.path.join(output_dir, "policy_vs_original_summary.png")
            plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    # 5. Error plot for first action dimension
    if len(policy_names) > 0 and n_dims > 0:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        steps = np.arange(n_steps)
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(policy_names)))
        
        for i, policy_name in enumerate(policy_names):
            if policy_name in predicted_actions:
                policy_actions = np.array(predicted_actions[policy_name])
                # Calculate error (difference) from original trajectory for first action dimension
                error = policy_actions[:, 0] - original_actions[:, 0]
                ax.plot(steps, error, color=colors[i], linewidth=2, 
                       alpha=0.8, label=f'{policy_name} Error', linestyle='-')
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Error (Predicted - Original)', fontsize=12)
        ax.set_title('Action Dimension 0 - Error from Original Trajectory', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.2, 2.2)
        
        plt.tight_layout()
        error_plot_path = os.path.join(output_dir, "action_dim_0_error.png")
        plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a combined error plot for all dimensions
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes_flat = axes.flatten()
        
        for dim in range(min(n_dims, 8)):  # Show up to 8 dimensions
            ax = axes_flat[dim]
            
            for i, policy_name in enumerate(policy_names):
                if policy_name in predicted_actions:
                    policy_actions = np.array(predicted_actions[policy_name])
                    # Calculate error for this dimension
                    error = policy_actions[:, dim] - original_actions[:, dim]
                    ax.plot(steps, error, color=colors[i], linewidth=1.5, 
                           alpha=0.8, label=policy_name, linestyle='-')
            
            # Add zero line for reference
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('Error', fontsize=10)
            ax.set_title(f'Action Dim {dim} Error', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Only show legend on first subplot
            if dim == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        # Hide unused subplots
        for dim in range(n_dims, len(axes_flat)):
            axes_flat[dim].set_visible(False)
        
        plt.tight_layout()
        all_errors_plot_path = os.path.join(output_dir, "all_action_dims_error.png")
        plt.savefig(all_errors_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 6. Individual trajectory plots for action dimension 0
        if len(policy_names) > 0:
            n_policies = len([name for name in policy_names if name in predicted_actions])
            if n_policies > 0:
                # Create subplots - arrange in a grid
                n_cols = min(3, n_policies)  # Max 3 columns
                n_rows = (n_policies + n_cols - 1) // n_cols  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    pass  # axes is already a 1D array
                else:
                    axes = axes.flatten()
                
                policy_idx = 0
                for i, policy_name in enumerate(policy_names):
                    if policy_name in predicted_actions:
                        ax = axes[policy_idx]
                        policy_actions = np.array(predicted_actions[policy_name])
                        
                        steps = np.arange(n_steps)
                        
                        # Plot original trajectory
                        ax.plot(steps, original_actions[:, 0], 'k-', linewidth=3, alpha=0.8, 
                               label='Original Trajectory', marker='o', markersize=2)
                        
                        # Plot policy trajectory
                        ax.plot(steps, policy_actions[:, 0], 'r--', linewidth=2, alpha=0.8, 
                               label=f'{policy_name}', marker='s', markersize=2)
                        
                        ax.set_xlabel('Step', fontsize=12)
                        ax.set_ylabel('Action Value', fontsize=12)
                        ax.set_title(f'Action Dim 0: {policy_name} vs Original', fontsize=14)
                        ax.set_ylim(-1.1, 1.1)
                        ax.legend(fontsize=10)
                        ax.grid(True, alpha=0.3)
                        
                        policy_idx += 1
                
                # Hide unused subplots
                for i in range(policy_idx, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                individual_traj_plot_path = os.path.join(output_dir, "action_dim_0_individual_trajectories.png")
                plt.savefig(individual_traj_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"Action comparison plots saved to: {output_dir}")


def load_policy_from_config(config_path: str, checkpoint_path: str = None) -> Tuple[Any, bool]:
    """
    Load a policy from config file, determining if it's an ActionTranslator or base policy.
    
    Args:
        config_path: Path to the config file
        checkpoint_path: Optional override for checkpoint path
        
    Returns:
        Tuple of (policy, is_action_translator)
    """
    print(f"Loading policy from config: {config_path}")
    
    # Check if it's an ActionTranslator config by looking for 'action_translator' in the config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check if it's a Hydra config with defaults
    if 'defaults' in config:
        # Check if action_translator is in defaults
        is_action_translator = any('action_translator' in str(default) for default in config['defaults'])
    else:
        # Check if action_translator key exists
        is_action_translator = 'action_translator' in config
    
    if is_action_translator:
        print("Loading ActionTranslator policy...")
        policy = load_action_translator_policy_from_config(
            config_path,
            source_policy_checkpoint=checkpoint_path
        )
    else:
        print("Loading base policy...")
        policy = load_source_policy_from_config(
            config_path,
            source_policy_checkpoint=checkpoint_path
        )
    
    print_model_info(policy)
    return policy, is_action_translator


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory and test multiple policies on those states")
    
    # Trajectory collection arguments
    parser.add_argument("--trajectory_policy_config", required=True, 
                       help="Path to policy config for collecting trajectory")
    parser.add_argument("--trajectory_policy_checkpoint", 
                       help="Override checkpoint path for trajectory policy")
    
    # Policy testing arguments
    parser.add_argument("--policy_configs", nargs='+', required=True,
                       help="Paths to policy configs to test on the trajectory states")
    parser.add_argument("--policy_checkpoints", nargs='*',
                       help="Override checkpoint paths for policy configs (same order)")
    parser.add_argument("--policy_names", nargs='*',
                       help="Names for the policies (defaults to config filenames)")
    
    # Environment arguments
    parser.add_argument("--env_id", default="Ant-v5", help="Gymnasium environment ID")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps in trajectory")
    parser.add_argument("--deterministic", action="store_true", default=True, 
                       help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False,
                       help="Use stochastic actions (overrides deterministic)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", help="Output directory for plots (defaults to timestamped dir)")
    
    args = parser.parse_args()
    
    # Handle deterministic/stochastic
    deterministic = args.deterministic and not args.stochastic
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"trajectory_eval_{args.env_id}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set up environment
    set_cluster_graphics_vars()
    register_custom_envs()
    
    # Load trajectory policy
    trajectory_policy, is_trajectory_translator = load_policy_from_config(
        args.trajectory_policy_config,
        args.trajectory_policy_checkpoint
    )
        
    # Collect trajectory
    states, actions, rewards, observations = collect_trajectory(
        trajectory_policy,
        args.env_id,
        max_steps=args.max_steps,
        deterministic=deterministic,
        seed=args.seed,
        is_action_translator=is_trajectory_translator
    )
    
    # Load test policies
    test_policies = {}
    policy_names = args.policy_names if args.policy_names else []
    
    for i, config_path in enumerate(args.policy_configs):
        # Get policy name
        if i < len(policy_names):
            policy_name = policy_names[i]
        else:
            policy_name = os.path.splitext(os.path.basename(config_path))[0]
        
        # Get checkpoint path
        checkpoint_path = None
        if args.policy_checkpoints and i < len(args.policy_checkpoints):
            checkpoint_path = args.policy_checkpoints[i]
        
        # Load policy
        policy, _ = load_policy_from_config(config_path, checkpoint_path)
        test_policies[policy_name] = policy
    
    # Test policies on trajectory states
    predicted_actions = test_policies_on_states(
        test_policies,
        states,
        observations,
        args.env_id,
        deterministic=deterministic
    )
    
    # Create plots
    plot_action_comparisons(
        predicted_actions,
        actions,
        output_dir,
        policy_names=list(test_policies.keys())
    )
    
    # Save trajectory data
    trajectory_data = {
        'states': [state.tolist() for state in states],
        'actions': [action.tolist() for action in actions],
        'rewards': rewards,
        'observations': [obs.tolist() for obs in observations],
        'env_id': args.env_id,
        'max_steps': args.max_steps,
        'deterministic': deterministic,
        'seed': args.seed
    }
    
    import json
    with open(os.path.join(output_dir, 'trajectory_data.json'), 'w', encoding='utf-8') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"Trajectory evaluation completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
