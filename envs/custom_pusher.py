import gymnasium as gym
from .env_transforms import ModifyPhysicsWrapper

def make_pusher_mod_physics(**kwargs):    
    pusher_env = gym.make("Pusher-v5", **kwargs)
    mod_friction = ModifyPhysicsWrapper(pusher_env, friction_mult=1.5, mass_mult=1.5, gear_mult=0.9, damping_mult=1.1)
    return mod_friction