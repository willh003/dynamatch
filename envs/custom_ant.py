"""
Custom Ant factory.

Provides a minimal factory that creates an environment identical to
Gymnasium's Ant-v5 but loading a copied local XML, so the Gym installation
remains unmodified.
"""

import os
import gymnasium as gym
from .env_transforms import ModifyFrictionWrapper, IntegrableEnvWrapper

def make_integrable_ant_standard_friction(**kwargs):
    """Exact Ant-v5 copy using local copied XML, with frame_skip and integrater modified."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, "assets", "ant_custom.xml")
    
    base_env = gym.make("Ant-v5", xml_file=xml_file, **kwargs)
    integrable_env = IntegrableEnvWrapper(base_env)
    
    return integrable_env

def make_integrable_ant_high_friction(**kwargs):
    """Exact Ant-v5 copy using local copied XML (no changes)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, "assets", "ant_custom.xml")

    ant_env = gym.make("Ant-v5", xml_file=xml_file, **kwargs)
    #return ModifyFrictionWrapper(ant_env, friction_coeffs=(2.0, 1.0, 1.0))
    mod_friction = ModifyFrictionWrapper(ant_env, friction_coeffs=(4.0, 2.0, 2.0))
    return IntegrableEnvWrapper(mod_friction)

def make_integrable_ant_low_friction(**kwargs):
    """Exact Ant-v5 copy using local copied XML (no changes)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, "assets", "ant_custom.xml")

    ant_env = gym.make("Ant-v5", xml_file=xml_file, **kwargs)
    return ModifyFrictionWrapper(ant_env, friction_coeffs=(0.1, 0.05, 0.05))
