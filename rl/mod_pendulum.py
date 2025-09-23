import gymnasium as gym

class ModPendulumWrapper(gym.Wrapper):
    def __init__(self, env, action_add=1.5):
        """
        Wrapper for the pendulum environment to shift the action space.
        Args:
            env: The environment to wrap.
            action_add: The amount to add to the actions.
        """
        super().__init__(env)
        self.action_add = action_add

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        print(f"Action: {action}")
        action = action + self.action_add
        print(f"New Action: {action}")
        return super().step(action)


def make_inverted_pendulum_dynamics_shift(action_add=.5, **kwargs):
    """Factory for the dynamics-shifted InvertedPendulum.

    Accepts the same kwargs as the underlying `InvertedPendulum-v5` env
    (e.g., render_mode, max_episode_steps via Gym wrappers, etc.).
    """
    base_env = gym.make("InvertedPendulum-v5", **kwargs)
    return ModPendulumWrapper(base_env, action_add=action_add)
