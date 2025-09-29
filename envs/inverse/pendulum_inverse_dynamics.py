import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO

class PendulumInverseDynamics:
    def __init__(self, env):
        self.env = env

    def sample(self, state, next_state):
        """
        Compute action using MuJoCo's inverse dynamics.
        
        Args:
            env: InvertedPendulum environment
            state: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
            next_state: target next state in same format
        
        Returns:
            action: [force] that produces the state transition
        """
        unwrapped = self.env.unwrapped

        # Set current state in MuJoCo
        qpos = np.array([state[0], state[1]])  # [cart_pos, pole_angle]
        qvel = np.array([state[2], state[3]])  # [cart_vel, pole_angular_vel]

        # Compute desired accelerations
        target_qpos = np.array([next_state[0], next_state[1]])
        target_qvel = np.array([next_state[2], next_state[3]])
        dt = unwrapped.dt
        qacc_desired = (target_qvel - qvel) #/ dt
        
        # Use MuJoCo inverse dynamics
        unwrapped.set_state(qpos, qvel)
        unwrapped.data.qacc[:] = qacc_desired
        mujoco.mj_inverse(unwrapped.model, unwrapped.data)
        
        # Return the control force
        action = unwrapped.data.qfrc_inverse[:unwrapped.model.nu]

        return action

def test_inverse_dynamics(model_path=None, num_episodes=10, episode_length=200):
    """
    Test inverse dynamics accuracy by rolling out random actions
    and comparing predicted vs actual actions.
    
    Returns:
        dict: Statistics about prediction errors
    """

    if model_path is not None:
        model = PPO.load(model_path)

    rollout_env = gym.make('InvertedPendulum-v5')
    id_env = gym.make('InvertedPendulum-v5')
    errors = []
    id_model = PendulumInverseDynamics(id_env)
    for episode in range(num_episodes):
        state, _ = rollout_env.reset()
        
        for step in range(episode_length):
            true_action = model.predict(state)[0]

            # Take step to get next state
            next_state, _, terminated, truncated, _ = rollout_env.step(true_action)
            # mujoco.mj_compareFwdInv(rollout_env.unwrapped.model, rollout_env.unwrapped.data)
            # print("x: ", rollout_env.unwrapped.data.solver_fwdinv)

            # print("qfrc_inverse: ", rollout_env.unwrapped.data.qfrc_inverse)

            print([i for i in rollout_env.unwrapped.data.__dir__() if "q" in i])

            breakpoint()


            # Sample random action
            true_action = model.predict(state)[0]
            
            print(f"True action: {true_action}")
            next_state = rollout_env.unwrapped.data.qpos[:2]
            terminated = False
            truncated = False
            
            # Predict action using inverse dynamics
            predicted_action = id_model.sample(state, next_state)
            print(f"Predicted action: {predicted_action}")
            
            # Compute error
            error = np.abs(predicted_action - true_action).item()
            errors.append(error)
            
            state = next_state
            
            if terminated or truncated:
                break
    
    rollout_env.close()
    id_env.close()
    
    errors = np.array(errors)
    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'median_error': np.median(errors),
        'num_samples': len(errors)
    }

if __name__ == "__main__":
    # Run test
    model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
    results = test_inverse_dynamics(model_path=model_path)
    print(f"Mean error: {results['mean_error']:.6f}")
    print(f"Max error: {results['max_error']:.6f}")
    print(f"Samples: {results['num_samples']}")