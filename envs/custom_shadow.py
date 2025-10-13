

import gymnasium as gym
from .env_transforms import ModifyPhysicsWrapper, ObsFromDictWrapper

def make_shadow_obs(**kwargs):
    """Factory for the shadow actuator environment.
    """
    base_env = gym.make("HandReachDense-v3", **kwargs)
    return ObsFromDictWrapper(base_env, obs_key='observation')

def make_shadow_actuator_friction(**kwargs):
    """Factory for the integrable InvertedPendulum.
    """
    base_env = gym.make("HandReachDense-v3", **kwargs)
    return ModifyPhysicsWrapper(base_env, friction_mult=0.9, damping_mult=1.2, mass_mult=1.1)
