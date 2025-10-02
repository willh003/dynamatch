import gymnasium as gym
import mujoco

from .env_utils import IntegrableEnvWrapper, ActionAddWrapper, MLPActionWrapper

def make_inverted_pendulum_integrable(**kwargs):
    """Factory for the integrable InvertedPendulum.
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    
    return IntegrableEnvWrapper(base_env)

def make_inverted_pendulum_integrable_dynamics_shift(action_add=.5, **kwargs):
    """Factory for the integrable InvertedPendulum.
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    integrable_env = IntegrableEnvWrapper(base_env)
    return ActionAddWrapper(integrable_env, action_add=action_add)

def make_inverted_pendulum_dynamics_shift(action_add=.5, **kwargs):
    """Factory for the dynamics-shifted InvertedPendulum.

    Accepts the same kwargs as the underlying `InvertedPendulum-v5` env
    (e.g., render_mode, max_episode_steps via Gym wrappers, etc.).
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    return ActionAddWrapper(base_env, action_add=action_add)


def make_inverted_pendulum_integrable_mlp_shift(save_weights_path=None, **kwargs):
    """Factory for the dynamics-shifted InvertedPendulum.
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    integrable_env = IntegrableEnvWrapper(base_env)
    
    # If save_weights_path is not provided, use the default weights path
    # Otherwise, create a new mlp and save the weights to the path
    checkpoint_path = None
    if save_weights_path is None:
        checkpoint_path = "/home/wph52/weird/dynamics/envs/transformations/pendulum_shift_mlp/random_weights.pth"
    return MLPActionWrapper(integrable_env, state_conditional=False, checkpoint_path=checkpoint_path, save_weights_path=save_weights_path)


def make_inverted_pendulum_integrable_mlp_shift_state_conditional(save_weights_path=None, **kwargs):
    """Factory for the dynamics-shifted InvertedPendulum.
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    integrable_env = IntegrableEnvWrapper(base_env)
    
    # If save_weights_path is not provided, use the default weights path
    # Otherwise, create a new mlp and save the weights to the path
    checkpoint_path = None
    if save_weights_path is None:
        checkpoint_path = "/home/wph52/weird/dynamics/envs/transformations/pendulum_shift_mlp/random_weights.pth"
    return MLPActionWrapper(integrable_env, state_conditional=True, checkpoint_path=checkpoint_path, save_weights_path=save_weights_path)
