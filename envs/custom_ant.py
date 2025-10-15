"""
Custom Ant factory.

Provides a minimal factory that creates an environment identical to
Gymnasium's Ant-v5 but loading a copied local XML, so the Gym installation
remains unmodified.
"""

import os
import gymnasium as gym
from .env_transforms import ModifyFrictionWrapper, IntegrableEnvWrapper, ModifyPhysicsWrapper

def make_ant(**kwargs):
    """
    Same as Ant-v5, but fixed healthy min z
    """
    ant_env = gym.make("Ant-v5", healthy_z_range=(0.25, 1.0), **kwargs)
    return ant_env

def make_ant_high_friction(**kwargs):

    ant_env = gym.make("Ant-v5", healthy_z_range=(0.25, 1.0), **kwargs)
    #return ModifyFrictionWrapper(ant_env, friction_coeffs=(2.0, 1.0, 1.0))
    mod_friction = ModifyFrictionWrapper(ant_env, friction_coeffs=(4.0, 2.0, 2.0))
    return mod_friction

def make_ant_modified_physics(**kwargs):
    """Exact Ant-v5 copy using local copied XML, with modified physics."""

    ant_env = gym.make("Ant-v5", healthy_z_range=(0.25, 1.0), **kwargs)
    mod_physics = ModifyPhysicsWrapper(ant_env, friction_mult=0.3, damping_mult=0.4, mass_mult=5.0, gear_mult=0.3, armature_mult=1.6)
    return mod_physics

def make_integrable_ant_standard_friction(**kwargs):
    """Exact Ant-v5 copy using local copied XML, with frame_skip and integrater modified."""
    ant_env = gym.make("Ant-v5", healthy_z_range=(0.25, 1.0), **kwargs)
    integrable_env = IntegrableEnvWrapper(ant_env)
    
    return integrable_env

def make_integrable_ant_high_friction(**kwargs):
    """Exact Ant-v5 copy using local copied XML (no changes)."""
    ant_env = gym.make("Ant-v5", healthy_z_range=(0.25, 1.0), **kwargs)
    mod_friction = ModifyFrictionWrapper(ant_env, friction_coeffs=(4.0, 2.0, 2.0))
    return IntegrableEnvWrapper(mod_friction)

def make_integrable_ant_low_friction(**kwargs):
    """Exact Ant-v5 copy using local copied XML (no changes)."""
    ant_env = gym.make("Ant-v5", healthy_z_range=(0.25, 1.0), **kwargs)
    return ModifyFrictionWrapper(ant_env, friction_coeffs=(0.1, 0.05, 0.05))
