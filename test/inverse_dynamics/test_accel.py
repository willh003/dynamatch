import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO

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
    mj_data = mujoco.MjData(model)

    nq = model.nq
    nv = model.nv
    
    state, _ = rollout_env.reset()
    for step in range(episode_length):
        
        # Sync state from Gym to our separate MjData and run forward to compute passive forces
        mj_data.qpos[:] = rollout_env.unwrapped.data.qpos
        mj_data.qvel[:] = rollout_env.unwrapped.data.qvel
        mj_data.qacc[:] = rollout_env.unwrapped.data.qacc
        model.opt.timestep = rollout_env.unwrapped.model.opt.timestep

        
        print(f"gym passive forces: {rollout_env.unwrapped.data.qfrc_passive}")
        print(f"mujoco passive forces: {mj_data.qfrc_passive}")

        print(f"gym bias forces: {rollout_env.unwrapped.data.qfrc_bias}")
        print(f"mujoco bias forces: {mj_data.qfrc_bias}")

        print("-------------- STEPPING ------------------")
        action = action_model.predict(state)[0]
        mj_data.ctrl[:] = action
        frame_skip = getattr(rollout_env.unwrapped, "frame_skip", 1)
        print(f"frame_skip: {frame_skip}")

        #mujoco.mj_step(model, mj_data, nstep=frame_skip)
        init_qvel = mj_data.qvel.copy()
        assert np.allclose(init_qvel, state[2:])
        for _ in range(frame_skip):

            # Kinematics: positions -> global coordinates and orientations  
            mujoco.mj_kinematics(model, mj_data)
            
            # Compute center of mass quantities
            mujoco.mj_comPos(model, mj_data)
            
            # Collision detection
            mujoco.mj_collision(model, mj_data)
            
            # Constraint setup and solving
            mujoco.mj_makeConstraint(model, mj_data)  
            mujoco.mj_projectConstraint(model, mj_data)
            
            # Compute forces (passive, applied, constraint forces)
            mujoco.mj_rne(model, mj_data, 1, mj_data.qfrc_bias)  # Coriolis/centrifugal
            mujoco.mj_rne(model, mj_data, 0, mj_data.qfrc_passive) # Passive forces
            

            mujoco.mj_forwardSkip(model, mj_data, mujoco.mjtStage.mjSTAGE_VEL, 1) 
            
            dt = model.opt.timestep
            mj_data.qvel += mj_data.qacc * dt
            mujoco.mj_integratePos(model, mj_data.qvel, mj_data.qpos, dt)
            mj_data.time += dt
            #mj_data.qpos += mj_data.qvel * dt

            # mujoco.mj_inverse(model, mj_data)
            # # print(f"mj_data qfrc_inverse: {mj_data.qfrc_inverse}")
            # # print(f"mj_data qfrc_actuator: {mj_data.qfrc_actuator}")
            # # print(f"mj_data qfrc_applied: {mj_data.qfrc_applied}")
            # # print(f"mj_data qfrc_passive: {mj_data.qfrc_passive}")
            # # print(f"mj_data qfrc_constraint: {mj_data.qfrc_constraint}")

            # print(f"fwd inverse actuator error: {np.linalg.norm(mj_data.qfrc_actuator - mj_data.qfrc_inverse)}")
            # print(f"fwd inverse all error: {np.linalg.norm(mj_data.qfrc_actuator + mj_data.qfrc_passive + mj_data.qfrc_constraint - mj_data.qfrc_inverse)}")
            # print(f"pred control: {get_ctrl_from_qfrc_actuator(mj_data.qfrc_actuator, model)}")
            # print(f"true control: {mj_data.ctrl}")

        mujoco.mj_rnePostConstraint(model, mj_data)


        next_state, _, _, _, _ = rollout_env.step(action)
        

        # Debug: Compare Gym's internal state after step
        print(f"=== GYM INTERNAL STATE AFTER STEP ===")
        print(f"Gym unwrapped qpos: {rollout_env.unwrapped.data.qpos}")
        print(f"Gym unwrapped qvel: {rollout_env.unwrapped.data.qvel}")
        print(f"Gym unwrapped qacc: {rollout_env.unwrapped.data.qacc}")
        print(f"Gym unwrapped ctrl: {rollout_env.unwrapped.data.ctrl}")
        print(f"Gym unwrapped qfrc_actuator: {rollout_env.unwrapped.data.qfrc_actuator}")
        
        print(f"=== MUJOCO STATE AFTER INTEGRATION ===")
        print(f"Mujoco qpos: {mj_data.qpos}")
        print(f"Mujoco qvel: {mj_data.qvel}")
        print(f"Mujoco qacc: {mj_data.qacc}")
        print(f"Mujoco ctrl: {mj_data.ctrl}")
        print(f"Mujoco qfrc_actuator: {mj_data.qfrc_actuator}")
        
        # mujoco_qacc = mj_data.qacc
        # print(f"gym_qacc: {gym_qacc}")
        # print(f"mujoco_qacc: {mujoco_qacc}")
        # print(f"acceleration diff: {np.abs(gym_qacc - mujoco_qacc)}")
        

        # print("-------------- MUJOCO STEPPED ------------------")
        # print(f"mj_data.qpos: {mj_data.qpos}")
        # print(f"mj_data.vel: {mj_data.qvel}")
        # print(f"mj_data.acc: {mj_data.qacc}")
        # print(f"mj_data.ctrl: {mj_data.ctrl}")

        # print(f"mj_data fwdinv: {mj_data.solver_fwdinv}")
        # print(f"mj_data qfrc_inverse: {mj_data.qfrc_inverse}")
        # print(f"mj_data qfrc_actuator: {mj_data.qfrc_actuator}")
        # print(f"mj_data qfrc_applied: {mj_data.qfrc_applied}")
        # print(f"mj_data qfrc_passive: {mj_data.qfrc_passive}")
        # print(f"mj_data qfrc_constraint: {mj_data.qfrc_constraint}")

        # print("------------ GYM STEPPED ------------------")
        # print(f"GYM data.qpos: {rollout_env.unwrapped.data.qpos}")
        # print(f"GYM data.vel: {rollout_env.unwrapped.data.qvel}")
        # print(f"GYM data.acc: {rollout_env.unwrapped.data.qacc}")
        # print(f"GYM data.ctrl: {rollout_env.unwrapped.data.ctrl}")
        # print(f"GYM qfrc_actuator: {rollout_env.unwrapped.data.qfrc_actuator}")
        # print(f"GYM qfrc_applied: {rollout_env.unwrapped.data.qfrc_applied}")
        # print(f"GYM qfrc_passive: {rollout_env.unwrapped.data.qfrc_passive}")
        # print(f"GYM qfrc_constraint: {rollout_env.unwrapped.data.qfrc_constraint}")

        print(f"---------DIFF---------")

        print(f"Error qpos: {np.abs(mj_data.qpos - rollout_env.unwrapped.data.qpos)}")
        print(f"Error qvel: {np.abs(mj_data.qvel - rollout_env.unwrapped.data.qvel)}")
        print(f"Error qacc: {np.abs(mj_data.qacc - rollout_env.unwrapped.data.qacc)}")
        print(f"Error ctrl: {np.abs(mj_data.ctrl - rollout_env.unwrapped.data.ctrl)}")

        print(f"GYM Error qfrc_actuator: {np.abs(mj_data.qfrc_actuator - rollout_env.unwrapped.data.qfrc_actuator)}")
        print(f"GYM Error qfrc_applied: {np.abs(mj_data.qfrc_applied - rollout_env.unwrapped.data.qfrc_applied)}")
        print(f"GYM Error qfrc_passive: {np.abs(mj_data.qfrc_passive - rollout_env.unwrapped.data.qfrc_passive)}")
        
        state = next_state
        # print("-------------- INVERSE ------------------")

        
        


if __name__ == "__main__":
    model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
    results = align_mujoco_and_gym(model_path=model_path, frame_skip=1)