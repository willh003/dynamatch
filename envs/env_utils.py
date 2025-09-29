import mujoco 
def modify_env_integrator(env, integrator=None, frame_skip=None):
    if integrator is not None:
        assert integrator in ['euler', 'rk4'], "ERROR: integrator must be euler or rk4"
        if integrator == 'euler':
            env.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        elif integrator == 'rk4':
            env.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    else:
        print(f"WARNING: using default integrator for forward, {env.unwrapped.model.opt.integrator}")
    
    if frame_skip is not None:
        env.unwrapped.frame_skip = frame_skip
    else:
        print(f"WARNING: using default frame skip for forward, {env.unwrapped.frame_skip}")

    return env