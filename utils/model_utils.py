import os
import yaml
import torch
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from action_translation import ActionTranslatorSB3Policy
from stable_baselines3 import PPO


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print detailed information about the ActionTranslator model."""
    base_policy_params = count_parameters(model.base_policy.policy)
    action_translator_params = count_parameters(model.action_translator)
    
    print(f"\n=== Model Parameter Counts ===")
    print(f"Base Policy Parameters: {base_policy_params:,}")
    print(f"Action Translator Parameters: {action_translator_params:,}")
    print(f"Total Parameters: {base_policy_params + action_translator_params:,}")
    
    # Detailed breakdown of action translator
    print(f"\n--- Action Translator Architecture ---")
    for name, module in model.action_translator.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                print(f"{name}: {param_count:,} parameters")
    
    print("=" * 40)


def load_action_translator_from_config(config_path, base_policy_checkpoint=None, action_translator_checkpoint=None):
    """
    Load an ActionTranslator from a config file.
    
    Args:
        config_path: Path to the translator config YAML file
        base_policy_checkpoint: Override path to base policy checkpoint
        action_translator_checkpoint: Override path to action translator checkpoint
    
    Returns:
        ActionTranslatorSB3Policy instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint paths if provided
    if base_policy_checkpoint:
        config['base_policy']['checkpoint_path'] = base_policy_checkpoint
    if action_translator_checkpoint:
        config['action_translator']['checkpoint_path'] = action_translator_checkpoint
    
    # Create base policy
    base_policy_config = config['base_policy'].copy()
    checkpoint_path = base_policy_config.pop('checkpoint_path', None)
    
    if checkpoint_path is None:
        raise ValueError("Base policy checkpoint path must be provided either in config or as argument")
    
    # Load the base policy from checkpoint
    base_policy = PPO.load(checkpoint_path)
    
    # Create action translator
    action_translator_config = config['action_translator'].copy()
    action_translator_checkpoint_path = action_translator_config.pop('checkpoint_path', None)
    
    # Remove Hydra-specific fields
    action_translator_config.pop('_target_', None)
    
    if action_translator_checkpoint_path is None:
        raise ValueError("Action translator checkpoint path must be provided either in config or as argument")
    
    # Handle Hydra template variables - try to infer from checkpoint or use defaults
    if 'action_dim' in action_translator_config and isinstance(action_translator_config['action_dim'], str):
        print(f"Warning: action_dim is a template variable '{action_translator_config['action_dim']}', using default value 1")
        action_translator_config['action_dim'] = 1  # Pendulum has 1D action space
    
    if 'obs_dim' in action_translator_config and isinstance(action_translator_config['obs_dim'], str):
        print(f"Warning: obs_dim is a template variable '{action_translator_config['obs_dim']}', using default value 4")
        action_translator_config['obs_dim'] = 4  # Pendulum has 4D observation space (cart_pos, pole_angle, cart_vel, pole_vel)
    
    # Validate that we have integer dimensions
    try:
        action_dim = int(action_translator_config['action_dim'])
        obs_dim = int(action_translator_config['obs_dim'])
        action_translator_config['action_dim'] = action_dim
        action_translator_config['obs_dim'] = obs_dim
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid dimensions in config: action_dim={action_translator_config.get('action_dim')}, obs_dim={action_translator_config.get('obs_dim')}. Error: {e}")
    
    # Debug: print the config being used
    print(f"Action translator config: {action_translator_config}")
    
    # Instantiate action translator
    from action_translation import SimpleActionTranslator
    action_translator = SimpleActionTranslator(**action_translator_config)
    
    # Load the action translator weights
    action_translator.load_state_dict(torch.load(action_translator_checkpoint_path, map_location='cpu'))
    action_translator.eval()
    
    # Create the combined policy
    combined_policy = ActionTranslatorSB3Policy(base_policy, action_translator)
    
    return combined_policy


def create_action_translator_from_hydra_config(config_path, base_policy_checkpoint=None, action_translator_checkpoint=None):
    """
    Alternative method using Hydra for more complex configs.
    This method is more robust for complex configurations.
    """
    # Clean up any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    
    # Initialize Hydra with the config directory
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    with initialize(config_path=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        
        # Override checkpoint paths if provided
        if base_policy_checkpoint:
            cfg.base_policy.checkpoint_path = base_policy_checkpoint
        if action_translator_checkpoint:
            cfg.action_translator.checkpoint_path = action_translator_checkpoint
        
        # Load base policy
        base_policy_checkpoint_path = cfg.base_policy.checkpoint_path
        if base_policy_checkpoint_path is None:
            raise ValueError("Base policy checkpoint path must be provided")
        
        base_policy = PPO.load(base_policy_checkpoint_path)
        
        # Create and load action translator
        action_translator_checkpoint_path = cfg.action_translator.checkpoint_path
        if action_translator_checkpoint_path is None:
            raise ValueError("Action translator checkpoint path must be provided")
        
        from action_translation import SimpleActionTranslator
        
        # Handle Hydra template variables - provide default values for pendulum environment
        action_dim = cfg.action_translator.action_dim
        obs_dim = cfg.action_translator.obs_dim
        
        if isinstance(action_dim, str):
            action_dim = 1  # Pendulum has 1D action space
        if isinstance(obs_dim, str):
            obs_dim = 4  # Pendulum has 4D observation space
        
        action_translator = SimpleActionTranslator(
            action_dim=action_dim,
            obs_dim=obs_dim
        )
        
        action_translator.load_state_dict(torch.load(action_translator_checkpoint_path, map_location='cpu'))
        action_translator.eval()
        
        # Create combined policy
        combined_policy = ActionTranslatorSB3Policy(base_policy, action_translator)
        
        return combined_policy
