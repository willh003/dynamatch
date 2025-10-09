import os
import yaml
import torch
from omegaconf import OmegaConf
import sys
from hydra.utils import instantiate, get_class
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import inspect
from generative_policies.action_translation import ActionTranslatorPolicy
from stable_baselines3 import PPO


# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print detailed information about the model (PPO or ActionTranslator)."""
    if hasattr(model, 'source_policy') and hasattr(model, 'action_translator'):
        # ActionTranslatorPolicy
        source_policy_params = count_parameters(model.source_policy.policy)
        action_translator_params = count_parameters(model.action_translator)
        
        print(f"\n=== ActionTranslator Model Parameter Counts ===")
        print(f"Base Policy Parameters: {source_policy_params:,}")
        print(f"Action Translator Parameters: {action_translator_params:,}")
        print(f"Total Parameters: {source_policy_params + action_translator_params:,}")
        
        # Detailed breakdown of action translator
        print(f"\n--- Action Translator Architecture ---")
        for name, module in model.action_translator.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if param_count > 0:
                    print(f"{name}: {param_count:,} parameters")
    else:
        # PPO model
        policy_params = count_parameters(model.policy)
        
        print(f"\n=== PPO Model Parameter Counts ===")
        print(f"Policy Parameters: {policy_params:,}")
        
        # Detailed breakdown of policy
        print(f"\n--- Policy Architecture ---")
        for name, module in model.policy.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if param_count > 0:
                    print(f"{name}: {param_count:,} parameters")
    
    print("=" * 40)


def build_action_translator_from_config(cfg_dict, obs_dim=None, action_dim=None, load_checkpoint=False):
    """
    Instantiate an action translator from a YAML config using Hydra's instantiate.

    Args:
        cfg_dict: Dictionary containing the translator config
        obs_dim: Optional override for observation dimension; if provided, overrides YAML
        action_dim: Optional override for action dimension; if provided, overrides YAML
        load_checkpoint: If True, tries to load 'checkpoint_path' weights if present

    Returns:
        nn.Module implementing ActionTranslatorInterface
    """

    # Pull out checkpoint path so it doesn't interfere with instantiation
    checkpoint_path = cfg_dict.pop('checkpoint_path', None)

    if '_target_' not in cfg_dict:
        raise ValueError("Model config must specify '_target_' class path")

    target_path = cfg_dict['_target_']

    # Introspect target class __init__ to filter kwargs (drop e.g. 'name')
    try:
        target_cls = get_class(target_path)
    except Exception as e:
        raise ImportError(f"Failed to resolve target '{target_path}': {e}")

    init_params = set(inspect.signature(target_cls.__init__).parameters.keys())
    init_params.discard('self')

    # Allow overrides for dims from data
    if action_dim is not None:
        cfg_dict['action_dim'] = int(action_dim)
    if obs_dim is not None:
        cfg_dict['obs_dim'] = int(obs_dim)

    # Keep only keys accepted by the constructor plus '_target_'
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k == '_target_' or k in init_params}

    # Convert to OmegaConf and instantiate via Hydra
    cfg = OmegaConf.create(filtered_cfg)
    model = instantiate(cfg, _convert_='all')

    if load_checkpoint and checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from checkpoint: {checkpoint_path}")

    return model

def load_action_translator_policy_from_config(config_path, source_policy_checkpoint=None, action_translator_checkpoint=None):
    """
    Load an ActionTranslator policy from a config file.
    
    Args:
        config_path: Path to the translator config YAML file
        source_policy_checkpoint: Override path to base policy checkpoint
        action_translator_checkpoint: Override path to action translator checkpoint
    
    Returns:
        ActionTranslatorPolicy instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if it's a Hydra config with defaults
    if 'defaults' in config:
        # Use OmegaConf to resolve the config
        return load_action_translator_from_hydra_config_simple(
            config_path, source_policy_checkpoint, action_translator_checkpoint
        )
    
    # Direct config handling
    # Override checkpoint paths if provided
    if source_policy_checkpoint:
        config['source_policy']['checkpoint_path'] = source_policy_checkpoint
    if action_translator_checkpoint:
        config['action_translator']['checkpoint_path'] = action_translator_checkpoint
    
    # Create base policy
    source_policy_config = config['source_policy'].copy()
    checkpoint_path = source_policy_config.pop('checkpoint_path', None)
    
    if checkpoint_path is None:
        raise ValueError("Base policy checkpoint path must be provided either in config or as argument")
    
    # Load the base policy from checkpoint
    source_policy = PPO.load(checkpoint_path)
    
    # Create action translator using the same approach as build_action_translator_from_config
    action_translator_config = config['action_translator'].copy()
    action_translator_checkpoint_path = action_translator_config.pop('checkpoint_path', None)
    
    if action_translator_checkpoint_path is None:
        raise ValueError("Action translator checkpoint path must be provided either in config or as argument")
    
    # Use build_action_translator_from_config for consistent parameter filtering
    action_translator = build_action_translator_from_config(
        action_translator_config, 
        load_checkpoint=False
    )
    
    # Load the action translator weights
    action_translator.load_state_dict(torch.load(action_translator_checkpoint_path, map_location='cpu'))
    action_translator.eval()
    
    # Create the combined policy
    combined_policy = ActionTranslatorPolicy(source_policy, action_translator)
    
    return combined_policy


def load_action_translator_from_hydra_config_simple(config_path, source_policy_checkpoint=None, action_translator_checkpoint=None):
    """
    Simple method to load ActionTranslator from Hydra config by manually resolving defaults.
    Requires config_path for relative path resolution.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the config directory to resolve relative paths
    config_dir = os.path.dirname(config_path)
    
    # Load source policy config
    source_policy_name = config['defaults'][0]['source_policy']
    source_policy_config_path = os.path.join(config_dir, '..', 'source_policy', f'{source_policy_name}.yaml')
    source_policy_config_path = os.path.normpath(source_policy_config_path)
    
    with open(source_policy_config_path, 'r') as f:
        source_policy_config = yaml.safe_load(f)
    
    # Override checkpoint path if provided
    if source_policy_checkpoint:
        source_policy_config['checkpoint_path'] = source_policy_checkpoint
    
    # Load source policy
    source_checkpoint_path = source_policy_config.get('checkpoint_path')
    if source_checkpoint_path is None:
        raise ValueError("Source policy checkpoint path must be provided")
    
    source_policy = PPO.load(source_checkpoint_path)
    
    # Load action translator config
    action_translator_name = config['defaults'][1]['action_translator']
    action_translator_config_path = os.path.join(config_dir, '..', 'action_translator', f'{action_translator_name}.yaml')
    action_translator_config_path = os.path.normpath(action_translator_config_path)
    
    with open(action_translator_config_path, 'r') as f:
        action_translator_config = yaml.safe_load(f)
    
    # Get checkpoint path before removing it from config
    action_translator_checkpoint_path = action_translator_config.get('checkpoint_path')
    if action_translator_checkpoint:
        action_translator_checkpoint_path = action_translator_checkpoint
    
    # Use build_action_translator_from_config for consistent parameter filtering
    action_translator = build_action_translator_from_config(
        action_translator_config, 
        load_checkpoint=False
    )
    
    # Load action translator weights
    if action_translator_checkpoint_path is None:
        raise ValueError("Action translator checkpoint path must be provided")
    
    action_translator.load_state_dict(torch.load(action_translator_checkpoint_path, map_location='cpu'))
    action_translator.eval()
    
    # Create combined policy
    combined_policy = ActionTranslatorPolicy(source_policy, action_translator)
    
    return combined_policy


def load_source_policy_from_config(config_path, source_policy_checkpoint=None):
    """
    Load a source policy from a config file.
    
    Args:
        config_path: Path to the source policy config YAML file
        source_policy_checkpoint: Override path to source policy checkpoint
    
    Returns:
        PPO policy instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint path if provided
    if source_policy_checkpoint:
        config['checkpoint_path'] = source_policy_checkpoint
    
    # Get checkpoint path
    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path is None:
        raise ValueError("Source policy checkpoint path must be provided either in config or as argument")
    
    # Load the source policy from checkpoint
    source_policy = PPO.load(checkpoint_path)
    
    return source_policy


def load_inverse_dynamics_model_from_config(config_path, load_checkpoint=False):
    """
    Load inverse dynamics model from config file.
    
    Args:
        config_path: Path to model config YAML file
        load_checkpoint: If True, loads the checkpoint from the config
    
    Returns:
        Loaded inverse dynamics model
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Import the model class
    target_path = config['_target_']
    module_path, class_name = target_path.rsplit('.', 1)
    
    # Import the module and get the class
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    # Introspect target class __init__ to filter kwargs (drop e.g. 'checkpoint_path')
    init_params = set(inspect.signature(model_class.__init__).parameters.keys())
    init_params.discard('self')
    
    # Get model parameters from config, filtering out unused ones
    model_params = {k: v for k, v in config.items() if k in init_params}
    
    # Create model instance
    model = model_class(**model_params)
    
    # Load checkpoint if provided
    if load_checkpoint:
        checkpoint_path = config.get('checkpoint_path')
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    
    
    return model
