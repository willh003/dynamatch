import gymnasium as gym
import mujoco

from .env_transforms import ObsFromDictWrapper, ModifyPhysicsWrapper

def make_fetch_reach_dense(**kwargs):
    """Factory for the FetchReachDense environment.
    """
    base_env = gym.make("FetchReachDense-v4", **kwargs)
    return ObsFromDictWrapper(base_env, obs_key='observation')


def make_fetch_reach_modified_physics_dense(**kwargs):
    """Factory for the FetchReachDense environment.
    """
    base_env = gym.make("FetchReachDense-v4", **kwargs)
    return ModifyPhysicsWrapper(base_env, friction_mult=1.0, damping_mult=10.0, mass_mult=0.1, gear_mult=0.1, armature_mult=1.0)


def make_fetch_reach_modified_physics_sparse(**kwargs):
    """Factory for the FetchReachDense environment.
    """
    base_env = gym.make("FetchReach-v4", **kwargs)
    return ModifyPhysicsWrapper(base_env, friction_mult=1.0, damping_mult=10.0, mass_mult=0.1, gear_mult=0.1, armature_mult=1.0)
