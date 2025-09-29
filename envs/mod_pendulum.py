import gymnasium as gym
import mujoco
class ModPendulumWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip=None,integrator=None, action_add=0):
        """
        Wrapper for the pendulum environment to shift the action space.
        Args:
            env: The environment to wrap.
            action_add: The amount to add to the actions.
        """
        super().__init__(env)
        self.action_add = action_add

        if frame_skip is not None:
            self.unwrapped.frame_skip = frame_skip
        if integrator is not None:
            self.unwrapped.model.opt.integrator = integrator

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        print(f"Action: {action}")
        action = action + self.action_add
        print(f"Modified Action: {action}")
        return super().step(action)


def make_inverted_pendulum_integrable(**kwargs):
    """Factory for the integrable InvertedPendulum.
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    frame_skip = 1
    integrator = mujoco.mjtIntegrator.mjINT_EULER
    return ModPendulumWrapper(base_env, frame_skip=frame_skip, integrator=integrator)


def make_inverted_pendulum_integrable_dynamics_shift(action_add=.5, **kwargs):
    """Factory for the integrable InvertedPendulum.
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    frame_skip = 1
    integrator = mujoco.mjtIntegrator.mjINT_EULER
    return ModPendulumWrapper(base_env, action_add=action_add, frame_skip=frame_skip, integrator=integrator)


def make_inverted_pendulum_dynamics_shift(action_add=.5, **kwargs):
    """Factory for the dynamics-shifted InvertedPendulum.

    Accepts the same kwargs as the underlying `InvertedPendulum-v5` env
    (e.g., render_mode, max_episode_steps via Gym wrappers, etc.).
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    return ModPendulumWrapper(base_env, action_add=action_add)



