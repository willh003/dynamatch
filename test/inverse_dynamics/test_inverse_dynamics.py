import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
import copy
from envs.inverse.inverse_dynamics import inverse_acceleration_integration, get_ctrl_from_qfrc_actuator
import matplotlib.pyplot as plt



def mj_step_custom(model, data, nstep=1):
    """
    Complete implementation of mj_step that matches MuJoCo's internal implementation exactly.
    
    Based on the actual MuJoCo source code implementation:
    - mj_checkPos/mj_checkVel for state validation
    - mj_forward for all forward dynamics computation
    - mj_checkAcc for acceleration validation  
    - Integration using the selected integrator
    
    Args:
        model: MuJoCo model (mjModel)
        data: MuJoCo data (mjData) 
        nstep: Number of simulation steps (frame skip count)
    """
    
    for step in range(nstep):

        mujoco.mj_forward(model, data)
        
        # Step 5: Integration using the selected integrator
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            print("Using RK4 integrator")
            # Use MuJoCo's built-in RK4 integrator (4 substeps)
            mujoco.mj_RungeKutta(model, data, 4)
        else:
            print("Using Euler integrator")
            # Default to Euler integrator (includes mjINT_EULER and mjINT_IMPLICIT)
            mujoco.mj_Euler(model, data)
    
    return data


def test_qacc_inverse(model, data, nstep=1):
    """
    we know that the qacc inverse recovers the initial qacc very closely

    we also know that mj_forward + qacc_forward should be the same as mj_step

    and we know that mj_inverse inverts mj_forward (unless there are additional forces it doesn't account for)

    so qacc_inverse + mj_inverse should invert mj_step
    """
        
    for step in range(nstep):

        mujoco.mj_forward(model, data)

        initial_data = copy.deepcopy(data)
        
        # Step 5: Integration using the selected integrator
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            print("Using RK4 integrator")
            # Use MuJoCo's built-in RK4 integrator (4 substeps)
            mujoco.mj_RungeKutta(model, data, 4)
        else:
            print("Using Euler integrator")
            # Default to Euler integrator (includes mjINT_EULER and mjINT_IMPLICIT)
            mujoco.mj_Euler(model, data)

        qacc_inverse = inverse_acceleration_integration(model, initial_data, data)
        print(f"initial data qacc: {initial_data.qacc}")
        print(f"qacc inverse: {qacc_inverse}")
        print(f"error (should be 0): {initial_data.qacc - qacc_inverse}")

        finite_difference = (data.qvel - initial_data.qvel) / model.opt.timestep
        
        print(f"finite difference vs estimate (should be basically 0 for euler): {np.linalg.norm(finite_difference - qacc_inverse)}" )
    return data


def check_custom_mj_step(model_path=None, num_episodes=10, episode_length=200, frame_skip=None):
    """
    Align the MuJoCo and Gym environments.
    
    Returns:
        None
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
        mj_custom_data =copy.deepcopy(rollout_env.unwrapped.data)
        mj_gt_data = copy.deepcopy(rollout_env.unwrapped.data)

        print(f"frame skip: {frame_skip}")
        action = action_model.predict(state)[0]
        mj_custom_data.ctrl[:] = action
        mj_gt_data.ctrl[:] = action

        # Sync state from Gym to our separate MjData and run forward to compute passive forces
        mj_custom_data = mj_step_custom(model, mj_custom_data, nstep=rollout_env.unwrapped.frame_skip)
        mujoco.mj_step(model, mj_gt_data, nstep=rollout_env.unwrapped.frame_skip)
        
        next_state, _, _, _, _ = rollout_env.step(action)
        print(f"mujoco custom qpos: {mj_custom_data.qpos}")
        print(f"mujoco custom qvel: {mj_custom_data.qvel}")
        print(f"mujoco custom qacc: {mj_custom_data.qacc}")
        print(f"mujoco custom ctrl: {mj_custom_data.ctrl}")

        print(f"mujoco qpos: {mj_gt_data.qpos}")
        print(f"mujoco qvel: {mj_gt_data.qvel}")
        print(f"mujoco qacc: {mj_gt_data.qacc}")
        print(f"mujoco ctrl: {mj_gt_data.ctrl}")

        print(f"gym qpos: {rollout_env.unwrapped.data.qpos}")
        print(f"gym qvel: {rollout_env.unwrapped.data.qvel}")
        print(f"gym qacc: {rollout_env.unwrapped.data.qacc}")
        print(f"gym ctrl: {rollout_env.unwrapped.data.ctrl}")

        print(f"diff qpos: {np.abs(mj_custom_data.qpos - rollout_env.unwrapped.data.qpos)}")
        print(f"diff qvel: {np.abs(mj_custom_data.qvel - rollout_env.unwrapped.data.qvel)}")
        print(f"diff qacc: {np.abs(mj_custom_data.qacc - rollout_env.unwrapped.data.qacc)}")
        print(f"diff ctrl: {np.abs(mj_custom_data.ctrl - rollout_env.unwrapped.data.ctrl)}")


def align_mujoco_and_gym(model_path=None, num_episodes=10, episode_length=200, frame_skip=None):
    """
    Align the MuJoCo and Gym environments.
    
    Returns:
        None
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

    all_qacc_percent_errors = []
    all_force_percent_errors = []
    all_control_percent_errors = []
    all_qaccs = []
    all_pred_qaccs = []

    for rollout in range(num_episodes):
        state, _ = rollout_env.reset()
        for step in range(episode_length):
            cur_data = copy.deepcopy(rollout_env.unwrapped.data)

            action = action_model.predict(state)[0]
            state, _, _, _, _ = rollout_env.step(action)

            #test_qacc_inverse(model, cur_data, nstep=rollout_env.unwrapped.frame_skip)
            
            next_data = copy.deepcopy(rollout_env.unwrapped.data)
            acceleration = inverse_acceleration_integration(model, cur_data, next_data)

            print(f"qacc error: {(next_data.qacc - acceleration)}")
            percent_error = np.abs(next_data.qacc - acceleration) / next_data.qacc
            print(f"qacc percent error: {percent_error}")

            all_qacc_percent_errors.append(percent_error)
            all_qaccs.append(next_data.qacc)
            all_pred_qaccs.append(acceleration)

            original_data_with_acceleration = copy.deepcopy(next_data)
            original_data_with_acceleration.qacc[:] = acceleration

            print(f"original data force actuator: {original_data_with_acceleration.qfrc_actuator}")
            print(f"original data force total: {original_data_with_acceleration.qfrc_actuator + original_data_with_acceleration.qfrc_passive + original_data_with_acceleration.qfrc_constraint}")
            mujoco.mj_inverse(model, original_data_with_acceleration)

            print(f"predicted force from inverse: {original_data_with_acceleration.qfrc_inverse}")
            all_force_percent_errors.append(np.abs(original_data_with_acceleration.qfrc_inverse - original_data_with_acceleration.qfrc_actuator) / (original_data_with_acceleration.qfrc_actuator + 1e-12))
            
            predicted_control = get_ctrl_from_qfrc_actuator(original_data_with_acceleration.qfrc_inverse, model)

            print(f"commanded control: {next_data.ctrl} == action {action}")
            print(f"predicted control: {predicted_control}")

            all_control_percent_errors.append(np.abs(action - predicted_control) / (action + 1e-12))

    return all_qacc_percent_errors, all_qaccs, all_pred_qaccs, all_force_percent_errors, all_control_percent_errors



def plot_errors(all_qacc_percent_errors, path):
    """
    Plot the errors
    """
    plt.hist(all_qacc_percent_errors, bins=100)
    plt.savefig(path)
    plt.clf()

def plot_qaccs_scatter(all_qaccs, all_pred_qaccs, path):
    """
    Plot the qaccs scatter
    """
    plt.scatter(all_qaccs[:,0], all_qaccs[:,1], label="True qaccs")
    plt.scatter(all_pred_qaccs[:,0], all_pred_qaccs[:,1], label="Predicted qaccs")
    plt.savefig(path)
    plt.clf()

if __name__ == "__main__":
    model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
    errors, all_qaccs, all_pred_qaccs, all_force_percent_errors, all_control_percent_errors = align_mujoco_and_gym(model_path=model_path, frame_skip=1)
    all_force_percent_errors = np.vstack(all_force_percent_errors)
    all_control_percent_errors = np.vstack(all_control_percent_errors)
    errors = np.vstack(errors)
    all_qaccs = np.vstack(all_qaccs)
    all_pred_qaccs = np.vstack(all_pred_qaccs)

    print(f"mean percent acceleration error: {np.abs(errors).mean(axis=0)}")
    print(f"mean force error: {np.abs(all_force_percent_errors).mean(axis=0)}")
    
    plot_errors(errors[:,0], "accel_errors_dof0.png")
    plot_errors(errors[:, 1], "accel_errors_dof1.png")
    plot_errors(all_force_percent_errors[:,0], "force_errors_dof0.png")
    plot_errors(all_force_percent_errors[:,1], "force_errors_dof1.png")
    plot_errors(all_control_percent_errors[:,0], "action_errors.png")

    plot_qaccs_scatter(all_qaccs, all_pred_qaccs, "qaccs_scatter.png")