import numpy as np
import mujoco
import copy

from scipy.optimize import minimize_scalar, minimize
from envs.inverse.set_state import set_state


def get_ctrl_from_qfrc_actuator(qfrc_actuator, model, debug=False):

    # Debug: print model structure
    if debug:
        print(f"Model nu (actuators): {model.nu}")
        print(f"Model nv (DOFs): {model.nv}")
        print(f"actuator_trnid shape: {model.actuator_trnid.shape}")
        print(f"actuator_gear shape: {model.actuator_gear.shape}")
        print(f"qfrc_actuator shape: {qfrc_actuator.shape}")
        print(f"qfrc_actuator shape: {qfrc_actuator.shape}")
        
    # The relationship is: qfrc_actuator = actuator_trnid.T @ (actuator_gear * ctrl)
    # So: ctrl = (qfrc_actuator @ pinv(actuator_trnid.T)) / actuator_gear
    
    # Build the actuator force matrix manually
    actuator_force = np.zeros((model.nv, model.nu))
    for i in range(model.nu):
        for j in range(model.actuator_trnid.shape[1]):
            dof_idx = model.actuator_trnid[i, j]
            if dof_idx >= 0:  # Valid DOF index
                # Handle both 1D and 2D gear arrays
                gear_value = model.actuator_gear[i, j] if model.actuator_gear.ndim > 1 else model.actuator_gear[i]
                actuator_force[dof_idx, i] = gear_value
    
    if debug:
        print(f"Built actuator_force shape: {actuator_force.shape}")
        print(f"actuator_force:\n{actuator_force}")
        
    # Use pseudoinverse (svd, basically least squares) to solve for ctrl
    #ctrl, residuals, rank, s = np.linalg.lstsq(actuator_force, qfrc_actuator, rcond=None)
    ctrl = np.linalg.pinv(actuator_force) @ qfrc_actuator
    
    if debug:
        print(f"Computed ctrl: {ctrl}")
    
    return ctrl

def rk4_inverse_iterative(model, data_old, data_new, dt, qacc_initial_guess, max_iter=10, tol=1e-6):
    """
    Iteratively solve for the acceleration that produces the observed RK4 step.
    
    This uses optimization to find the initial acceleration that, when integrated
    with RK4, produces the observed final velocity.
    """
    
    def rk4_forward_simulation(qacc_test):
        """Simulate one RK4 step with given initial acceleration and return final velocity"""
        # Create temporary data
        
        data_temp = copy.deepcopy(data_old)
        
        # Set the test acceleration 
        data_temp.qacc[:] = qacc_test
        
        # Perform RK4 integration (this is approximate since we don't know intermediate accelerations)
        # For a more exact solution, we'd need to implement the full RK4 with dynamics calls
        
        # RK4 approximation: use the same acceleration for all substeps
        # This is not perfect but better than simple Euler
        h = dt
        k1 = h * qacc_test
        k2 = h * qacc_test  # Should be h * accel_at_intermediate_state
        k3 = h * qacc_test  # Should be h * accel_at_intermediate_state  
        k4 = h * qacc_test  # Should be h * accel_at_intermediate_state
        
        # RK4 velocity update
        qvel_predicted = data_old.qvel + (k1 + 2*k2 + 2*k3 + k4)/6
        
        return qvel_predicted
    
    def objective(qacc_test):
        """Objective function: minimize difference between predicted and actual final velocity"""
        qvel_predicted = rk4_forward_simulation(qacc_test)
        error = np.linalg.norm(qvel_predicted - data_new.qvel)
        return error
    
    # Use optimization to find the best acceleration
    try:
        # For single DOF, use scalar optimization
        if model.nv == 1:
            result = minimize_scalar(lambda a: objective(np.array([a])), 
                                   bounds=(-1000, 1000), method='bounded')
            if result.success and result.fun < tol:
                return np.array([result.x])
        else:
            # For multiple DOF, use general optimization
            result = minimize(objective, qacc_initial_guess, 
                            method='BFGS', options={'gtol': tol, 'maxiter': max_iter})
            if result.success and result.fun < tol:
                return result.x
                
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    return None



def inverse_acceleration_integration(model, data_old, data_new):
    """
    Inverse the mj_step process to recover applied forces.
    
    Given the state before and after mj_step, this function:
    1. Reverses the integration to compute the acceleration that occurred
    2. Uses mj_inverse to compute the forces that would produce that acceleration
    
    Args:
        model: MuJoCo model (mjModel)
        data_old: MuJoCo data before the step (mjData)
        data_new: MuJoCo data after the step (mjData) 
        
    Returns:
        dict containing:
        - 'qacc_recovered': The recovered acceleration vector
        - 'qfrc_inverse': The forces that would produce this acceleration
        - 'qfrc_applied_recovered': Estimated externally applied forces
        - 'ctrl_recovered': Estimated control inputs (if actuators present)
    """
    
    # Get time step
    dt = model.opt.timestep

    # Create a working copy of the data to avoid modifying inputs
    data_work = copy.deepcopy(data_old)
    
    # Step 1: Reverse the integration to recover acceleration
    # The integration formula depends on which integrator was used
    
    if model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
        # Semi-implicit Euler: 
        # qvel_new = qvel_old + qacc * dt
        # qpos_new = qpos_old + qvel_new * dt
        
        # Reverse velocity integration: qacc = (qvel_new - qvel_old) / dt
        qacc_recovered = (data_new.qvel - data_old.qvel) / dt
        
    elif model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        # For RK4, we can either approximate or use an iterative solver
        # Option 1: Simple approximation (fast but approximate)
        print("Attempting iterative RK4 reversal...")
        qacc_approx = (data_new.qvel - data_old.qvel) / dt
        
        # Option 2: Iterative refinement (more accurate)
        # This tries to find the initial acceleration that would produce the observed state change
        
        qacc_recovered = rk4_inverse_iterative(model, data_old, data_new, dt, qacc_approx)
        
        if qacc_recovered is None:
            print("Warning: RK4 iterative reversal failed, using euler instead")
            qacc_recovered = qacc_approx
        
    elif model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
        # For implicit integration, we also approximate 
        print("Warning: Implicit integration reversal is approximate")
        qacc_recovered = (data_new.qvel - data_old.qvel) / dt
        
    else:
        # Default to Euler-like reversal
        qacc_recovered = (data_new.qvel - data_old.qvel) / dt
    
    # Step 2: Set up the working data with the old state for inverse dynamics

    return qacc_recovered


def mujoco_inverse_dynamics(env, state_data, next_state_data, min_acc = -np.inf, max_acc = np.inf):
    """
    Given (state_data, next_state_data) from mujoco, compute the action that produced the next state

    REQUIRES (for minimal error): 
    - time integrator to be Euler
    - frame skip to be 1

    TODO: implement set_state
    """

    model = env.unwrapped.model

    # assert env.unwrapped.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER, "ERROR: integrators other than Euler are not supported"
    # assert env.unwrapped.frame_skip == 1, "ERROR: frame skip other than 1 are not supported"
    
    if env.unwrapped.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        print("WARNING: integrators other than Euler are dangerous")
    if env.unwrapped.frame_skip != 1:
        print("WARNING: frame skip other than 1 are dangerous")
    data = copy.deepcopy(state_data)
    next_data = copy.deepcopy(next_state_data)

    # get the acceleration that produced the next state
    acceleration = inverse_acceleration_integration(model, data, next_data)
        
    # get the forces that produced the acceleration
    data.qacc[:] = acceleration    
    data.qacc[0] = np.clip(acceleration[0], min_acc, max_acc)
    data.qacc[1] = np.clip(acceleration[1], min_acc, max_acc)
        
    mujoco.mj_inverse(model, data)

    qfrc = data.qfrc_inverse

    ctrl = get_ctrl_from_qfrc_actuator(qfrc, model)

    return ctrl 
    


def estimate_velocity_fd(state, next_state):
    return (next_state - state) / env.unwrapped.dt

def gym_inverse_dynamics(env, state, next_state, min_acc = -np.inf, max_acc = np.inf):
    """
    Given (state, next_state) from gym env.step(), compute the action that produced the next state

    REQUIRES:
    - env is a gym environment NOT CURRENTLY IN USE (or it may cause issues with the rollout)
    - set_state to be implemented for the environment

    REQUIRES (for minimal error): 
    - time integrator to be Euler
    - frame skip to be 1

    """ 
    # assert env.unwrapped.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER, "ERROR: integrators other than Euler are not supported"
    # assert env.unwrapped.frame_skip == 1, "ERROR: frame skip other than 1 are not supported"
    env.reset()
    set_state(env, next_state)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    next_data = copy.deepcopy(env.unwrapped.data)

    env.reset()
    set_state(env, state)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    data = copy.deepcopy(env.unwrapped.data)

    next_data.qvel = (next_data.qpos - data.qpos) / env.unwrapped.dt
    return mujoco_inverse_dynamics(env, data, next_data, min_acc, max_acc)