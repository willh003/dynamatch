

def test_mujoco_fwdinv_error():
    """
    Check the inverse dynamics model error against the forward dynamics under actions from an expert policy.
    This error is inherent to the simulation, and cannot possibly be avoided using physics.
    i.e.,
    ctrl = pi(s)
    qacc, qfrc_actuator = fwd(qpos, qvel, ctrl)
    qfrc_inverse = id(qpos, qvel, qacc)
    error = qfrc_inverse - qfrc_actuator

    fwd is the simulation dynamics model (from env.step)
    id is the inverse dynamics model    
    """
    pass

test_mujoco_integration_error():
    """
    Check the inverse dynamics model error against the forward dynamics under actions from an expert policy,
    but using qvel finite difference to estimate the qacc, instead of the ground truth qacc
    i.e.,
    ctrl = pi(s)
    _, qfrc_actuator = fwd(qpos, qvel, ctrl)
    next_qpos, next_qvel = step(qpos, qvel, qfrc_actuator) 
    qacc_approx = (next_qvel - qvel) / dt
    qfrc_inverse = id(qpos, qvel, qacc_approx)
    error = qfrc_inverse - qfrc_actuator

    This should be > mujoco_fwdinv_error, since it adds acceleration approximation error
    """
    pass

    
def test_full_id_error():
    """
    Check the error of the full inverse dynamics, 
    including the inverse of the function mapping actuator ctrl to dof force 

    ctrl = pi(s)
    _, qfrc_actuator = fwd(qpos, qvel, ctrl)
    next_qpos, next_qvel = step(qpos, qvel, qfrc_actuator) 
    qacc_approx = (next_qvel - qvel) / dt
    qfrc_inverse = id(qpos, qvel, qacc_approx)
    ctrl_inverse = A_inverse @ qfrc_inverse
    error = ctrl_inverse - ctrl

    This should be > mujoco_integration_error, since it adds acceleration approximation error and dof approximation error

    """
    pass

def test_approximate_id_error():
    """
    Check the inverse dynamics model error against the forward dynamics under actions from an expert policy
    i.e.,
    error = id_approx(s,s') - a
    where a ~ pi(a|s), and s' = fwd(s,a)

    fwd is the simulation dynamics model (from env.step)
    id is the inverse dynamics model
    """
    pass

