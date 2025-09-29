from gymnasium.envs.registration import register, registry
from .mod_pendulum import make_inverted_pendulum_dynamics_shift, make_inverted_pendulum_integrable, make_inverted_pendulum_integrable_dynamics_shift

def register_custom_envs():
    """Register custom env IDs backed by local factory functions.

    Safe to call multiple times; skips IDs already present in the Gymnasium registry.
    """
    env_specs = [
        (
            "InvertedPendulumDynamicsShift-v5",
            make_inverted_pendulum_dynamics_shift
        ),
        (
            "InvertedPendulumIntegrable-v5",
            make_inverted_pendulum_integrable
        ),
        (
            "InvertedPendulumIntegrableDynamicsShift-v5",
            make_inverted_pendulum_integrable_dynamics_shift
        ),
    ]

    for env_id, entry_point in env_specs:
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point)


