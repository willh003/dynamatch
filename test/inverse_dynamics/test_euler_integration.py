import numpy as np
import mujoco
import copy
import gymnasium as gym
from stable_baselines3 import PPO


def test(model_path, episode_length=100, frame_skip=None):
    """
    Ensure that mujoco.mj_Euler is the same as initial_qvel + commanded_qacc * dt
    """
    
    if model_path is not None:
        action_model = PPO.load(model_path)
    else:
        action_model = lambda x: np.array([0.0])

    rollout_env = gym.make('InvertedPendulum-v5')
    # NOTE: NOT BY DEFAULT IN MUJOCO! BUT MAKES INVERSE DYNAMICS WORK
    rollout_env.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    if frame_skip is not None:
        rollout_env.unwrapped.frame_skip = frame_skip

    # Use the exact same model instance as Gym to avoid any discrepancies
    model = rollout_env.unwrapped.model    
    state, _ = rollout_env.reset()

    for step in range(episode_length):
        action = action_model.predict(state)[0]

        initial_data = copy.deepcopy(rollout_env.unwrapped.data)
        initial_data.ctrl = action
        state, _, _, _, _ = rollout_env.step(action)

        # Run forward dynamics first
        initial_qvel = initial_data.qvel.copy()
        print(f"Initial qvel: {initial_qvel}")

        mujoco.mj_forward(model, initial_data) 
        dt = model.opt.timestep

        commanded_qacc = initial_data.qacc.copy()
        expected_qvel = initial_qvel + commanded_qacc * dt
        print(f"Commanded qacc: {commanded_qacc}")
        print(f"Expected qvel: {expected_qvel}")

        # Integrate
        mujoco.mj_Euler(model, initial_data)
        
        actual_qvel = initial_data.qvel
        print(f"Actual qvel: {actual_qvel}")
        print(f"Error: {np.linalg.norm(actual_qvel - expected_qvel)}")
        
    return np.linalg.norm(actual_qvel - expected_qvel) < 1e-12




if __name__ == "__main__":
    model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
    results = test(model_path=model_path, frame_skip=1)