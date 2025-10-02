import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import sys
import os

# Add the parent directory to the path to import the custom environments
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from envs.register_envs import register_custom_envs

def test_mlp_action_transform():
    """Test the MLP action transformation by plotting original vs transformed actions."""
    
    # Register the custom environments
    register_custom_envs()
    
    # Create the environment with weight saving
    env = gym.make("InvertedPendulumIntegrableMLPShift-v5")
    
    # Test actions in the range [-3, 3]
    actions = np.linspace(-3, 3, 100)
    transformed_actions = []
    
    print("Testing MLP action transformation...")
    print("Original Action -> Transformed Action")
    print("-" * 40)
    
    # Find the MLPActionWrapper in the wrapper chain
    mlp_wrapper = None
    current_env = env
    while hasattr(current_env, 'env'):
        if hasattr(current_env, 'action_transform'):
            mlp_wrapper = current_env
            break
        current_env = current_env.env
    
    if mlp_wrapper is None:
        raise AttributeError("Could not find MLPActionWrapper in the environment wrapper chain")
    
    for i, action in enumerate(actions):
        # Get the transformed action directly from the MLP without stepping the environment
        # The MLPActionWrapper has an action_transform attribute
        with torch.no_grad():
            transformed_action = mlp_wrapper.action_transform(torch.tensor([action], dtype=torch.float32))
            transformed_actions.append(transformed_action.item())
        
        # Print a few sample transformations
        if i % 20 == 0:  # Print every 20th action
            print(f"{action:6.2f} -> {transformed_actions[-1]:8.4f}")
    
    # Convert to numpy array
    transformed_actions = np.array(transformed_actions)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(actions, actions, 'b--', label='Original Action (y=x)', linewidth=2)
    plt.plot(actions, transformed_actions, 'r-', label='Transformed Action (MLP output)', linewidth=2)
    plt.xlabel('Original Action')
    plt.ylabel('Action Value')
    plt.title('MLP Action Transformation: InvertedPendulumIntegrableMLPShift-v5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add some key points
    key_actions = [-3, -1, 0, 1, 3]
    for key_action in key_actions:
        idx = np.argmin(np.abs(actions - key_action))
        plt.plot(actions[idx], transformed_actions[idx], 'ko', markersize=8)
        plt.annotate(f'({key_action:.1f}, {transformed_actions[idx]:.3f})', 
                    (actions[idx], transformed_actions[idx]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.tight_layout()
    
    os.makedirs("test_media/mlp_test", exist_ok=True)

    plot_path = os.path.join("test_media/mlp_test", "mlp_action_transform.png")
    plt.savefig(plot_path)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Action range: [{actions.min():.1f}, {actions.max():.1f}]")
    print(f"Transformed range: [{transformed_actions.min():.3f}, {transformed_actions.max():.3f}]")
    print(f"Mean transformation: {np.mean(transformed_actions):.3f}")
    print(f"Std transformation: {np.std(transformed_actions):.3f}")
    
    env.close()

if __name__ == "__main__":
    test_mlp_action_transform()
