from gymnasium.envs.registration import register, registry
from .custom_pendulum import (
    make_inverted_pendulum_dynamics_shift, 
    make_inverted_pendulum_integrable, 
    make_inverted_pendulum_integrable_dynamics_shift,
    make_inverted_pendulum_integrable_mlp_shift,
    make_inverted_pendulum_integrable_mlp_shift_state_conditional
)

from .custom_ant import (
    make_integrable_ant_standard_friction,
    make_integrable_ant_high_friction,
    make_integrable_ant_low_friction,
)

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
        (
            "InvertedPendulumIntegrableMLPShift-v5",
            make_inverted_pendulum_integrable_mlp_shift
        ),
        (
            "InvertedPendulumIntegrableMLPShiftStateConditional-v5",
            make_inverted_pendulum_integrable_mlp_shift_state_conditional
        ),
        # Custom Ant environments with different contact parameters
        (
            "AntIntegrable-v1",
            make_integrable_ant_standard_friction
        ),
        (
            "AntIntegrableHighFriction-v1",
            make_integrable_ant_high_friction
        ),
        (
            "AntIntegrableLowFriction-v1",
            make_integrable_ant_low_friction
        ),
        # Only exact-copy variant to avoid behavioral differences / OOMs

    ]

    for env_id, entry_point in env_specs:
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point)


