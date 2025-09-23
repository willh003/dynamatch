
def modify_env_gravity(env, new_gravity_vec=[0,0,-9.81]):

    if hasattr(env, 'envs'):
        # vectorized environment
        for i in range(env.num_envs):
            env.envs[i].unwrapped.model.opt.gravity = new_gravity_vec
            env.reset()
    else:
        env.unwrapped.model.opt.gravity = new_gravity_vec
        env.reset()