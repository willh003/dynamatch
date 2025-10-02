#!/usr/bin/env python3
"""
Utility functions for handling configuration files and filtering parameters.
"""

import inspect
import re
from typing import Dict, Any, Callable
import yaml

def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def resolve_template_variables(config: Dict[str, Any], template_vars: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Resolve template variables in the format ${variable_name} in a config dictionary.
    
    Args:
        config: Configuration dictionary that may contain template variables
        template_vars: Dictionary of template variable values
        
    Returns:
        Configuration dictionary with template variables resolved
    """
    if template_vars is None:
        # Default values for common template variables
        template_vars = {
            'num_frames': 18,
            'obs_num_frames': 1
        }
    
    resolved_config = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            # Extract variable name from ${variable_name}
            var_name = value[2:-1]
            if var_name in template_vars:
                resolved_config[key] = template_vars[var_name]
            else:
                # Keep original value if template variable not found
                resolved_config[key] = value
        else:
            resolved_config[key] = value
    
    return resolved_config


def filter_config_for_function(config: Dict[str, Any], target_function: Callable, 
                             template_vars: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Filter a configuration dictionary to only include parameters that are valid
    for the target function.
    
    Args:
        config: Configuration dictionary to filter
        target_function: Function to get valid parameters from
        template_vars: Dictionary of template variable values to resolve
        
    Returns:
        Filtered configuration dictionary with only valid parameters
    """
    # First resolve template variables
    resolved_config = resolve_template_variables(config, template_vars)
    
    # Get the valid parameter names for the target function
    sig = inspect.signature(target_function)
    valid_params = set(sig.parameters.keys())
    
    # Filter config to only include valid kwargs
    filtered_config = {k: v for k, v in resolved_config.items() if k in valid_params}
    
    return filtered_config


def filter_config_with_debug(config: Dict[str, Any], target_function: Callable, 
                           debug: bool = False, template_vars: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Filter a configuration dictionary with optional debug output.
    
    Args:
        config: Configuration dictionary to filter
        target_function: Function to get valid parameters from
        debug: Whether to print debug information
        template_vars: Dictionary of template variable values to resolve
        
    Returns:
        Filtered configuration dictionary with only valid parameters
    """
    # First resolve template variables
    resolved_config = resolve_template_variables(config, template_vars)
    
    # Get the valid parameter names for the target function
    sig = inspect.signature(target_function)
    valid_params = set(sig.parameters.keys())
    
    # Filter config to only include valid kwargs
    filtered_config = {k: v for k, v in resolved_config.items() if k in valid_params}
    
    if debug:
        print(f"Original config keys: {list(config.keys())}")
        print(f"Resolved config keys: {list(resolved_config.keys())}")
        print(f"Valid parameter keys: {list(valid_params)}")
        print(f"Filtered config keys: {list(filtered_config.keys())}")
        
        # Show what was filtered out
        filtered_out = set(resolved_config.keys()) - set(filtered_config.keys())
        if filtered_out:
            print(f"Filtered out keys: {list(filtered_out)}")
        else:
            print("No keys were filtered out")
    
    return filtered_config
