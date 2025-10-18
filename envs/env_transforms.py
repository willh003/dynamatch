import mujoco 
import torch
import torch.nn as nn
import gymnasium as gym
from robosuite.utils.mjmod import DynamicsModder
import numpy as np

class ObsFromDictWrapper(gym.Wrapper):
    def __init__(self, env, obs_key='observation'):
        super().__init__(env)
        self.obs_key = obs_key
        
        # Update observation space to match the extracted observation
        if isinstance(env.observation_space, gym.spaces.Dict):
            if obs_key not in env.observation_space.spaces:
                raise KeyError(f"Key '{obs_key}' not found in observation space keys: {list(env.observation_space.spaces.keys())}")
            self.observation_space = env.observation_space.spaces[obs_key]
        else:
            raise ValueError("Environment observation space must be a Dict space")
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[self.obs_key], reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self.obs_key], info


class ModifyFrictionWrapper(gym.Wrapper):
    def __init__(self, env, friction_coeffs=(1.0, 0.5, 0.5)):
        super().__init__(env)

        for i in range(self.unwrapped.model.ngeom):
            self.unwrapped.model.geom_friction[i] = list(friction_coeffs)

class ModifyPhysicsWrapper(gym.Wrapper):
    """
    Wrapper to modify MuJoCo physics parameters for sim2real transfer testing.
    
    Args:
        friction_mult: Multiplier for geom_friction [tangential, torsional, rolling].
                      Controls sliding and rotational resistance at contacts.
        damping_mult: Multiplier for dof_damping. Joint velocity damping coefficient.
        mass_mult: Multiplier for body_mass. Inertial mass of each body.
        armature_mult: Multiplier for dof_armature. Rotor inertia reflected to joint.
        gear_mult: Multiplier for actuator_gear. Actuator force/torque transmission ratio.
        solref_timeconst_mult: Multiplier for solref[0]. Contact stiffness time constant 
                               (lower = stiffer, higher = softer/more compliant).
        solref_dampratio_mult: Multiplier for solref[1]. Contact damping ratio 
                               (1.0 = critical, <1.0 = bouncy, >1.0 = overdamped).
        solimp_dmin_mult: Multiplier for solimp[0]. Min penetration distance for 
                          corrective force activation.
        solimp_dmax_mult: Multiplier for solimp[1]. Max penetration distance for 
                          full corrective force.
    """
    def __init__(self, env, friction_mult=1.0, damping_mult=1.0, mass_mult=1.0,
                 armature_mult=1.0, gear_mult=1.0, solref_timeconst_mult=1.0,
                 solref_dampratio_mult=1.0, solimp_dmin_mult=1.0, solimp_dmax_mult=1.0):
        super().__init__(env)
        
        # Get the MuJoCo model - handle both regular MuJoCo envs and Robosuite envs
        model = self.unwrapped.model

        # Modify friction
        for i in range(model.ngeom):
            model.geom_friction[i] *= friction_mult
        
        # Modify damping
        for i in range(model.njnt):
            model.dof_damping[i] *= damping_mult
        
        # Modify masses
        for i in range(model.nbody):
            model.body_mass[i] *= mass_mult
        
        # Modify armature (rotor inertia)
        for i in range(model.njnt):
            model.dof_armature[i] *= armature_mult
        
        # Modify actuator gear ratios
        for i in range(model.nu):
            model.actuator_gear[i] *= gear_mult
        
        # Modify contact solver reference (solref)
        for i in range(model.ngeom):
            model.geom_solref[i, 0] *= solref_timeconst_mult
            model.geom_solref[i, 1] *= solref_dampratio_mult
        
        # Modify contact solver impedance (solimp)
        for i in range(model.ngeom):
            model.geom_solimp[i, 0] *= solimp_dmin_mult
            model.geom_solimp[i, 1] *= solimp_dmax_mult

class IntegrableEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.unwrapped.frame_skip = 1
        self.unwrapped.model.opt.integrator = 0  # mjINT_EULER

class ActionAddWrapper(gym.Wrapper):
    def __init__(self, env, action_add=0):
        """
        Wrapper for the pendulum environment to shift the action space.
        Args:
            env: The environment to wrap.
            action_add: The amount to add to the actions.
        """
        super().__init__(env)
        self.action_add = action_add

    def step(self, action):
        action = action + self.action_add
        return super().step(action)

class ActionNonlinearWrapper(gym.Wrapper):
    def __init__(self, env, action_nonlinear_transformation=lambda x: x):
        super().__init__(env)
        self.action_nonlinear_transformation = action_nonlinear_transformation
    
    def step(self, action):
        return super().step(self.action_nonlinear_transformation(action))


class ActionTransformMLP(nn.Module):
    def __init__(self, action_dim=1, state_dim=None, hidden_dim=32):
        super().__init__()
        self.state_conditional = state_dim is not None
        if self.state_conditional:
            in_dim = state_dim + action_dim
        else:
            in_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights with larger variance and add some bias
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with larger variance to create more interesting transformations."""
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                # Use different initialization strategies for different layers
                if i == 0:  # First layer
                    nn.init.xavier_uniform_(layer.weight, gain=3.0)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -1.0, 1.0)
                elif i == len([l for l in self.net if isinstance(l, nn.Linear)]) - 1:  # Last layer
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -0.2, 0.2)
                else:  # Hidden layers
                    nn.init.xavier_uniform_(layer.weight, gain=2.5)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -0.3, 0.3)
        
    def forward(self, a_source):
        return self.net(a_source)


class MLPActionWrapper(gym.Wrapper):
    def __init__(self, env, state_conditional=False, checkpoint_path=None, save_weights_path=None):
        super().__init__(env)
        assert checkpoint_path is None or save_weights_path is None, "ERROR: cannot provide both checkpoint_path and save_weights_path"
        
        self.action_transform = ActionTransformMLP(
            action_dim=env.action_space.shape[0], 
            state_dim=env.observation_space.shape[0] if state_conditional else None
            )
            
        if checkpoint_path is not None:
            self.action_transform.load_state_dict(torch.load(checkpoint_path))
        
        # Save the initialized weights if save_weights_path is provided
        if save_weights_path is not None:
            import os
            os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
            torch.save(self.action_transform.state_dict(), save_weights_path)
            print(f"Saved MLP weights to: {save_weights_path}")
    
    def step(self, action, debug=False):

        action_torch = torch.as_tensor(action, dtype=torch.float32)
        with torch.no_grad():   
            action_transformed = self.action_transform(action_torch)
        action_transformed_numpy = action_transformed.detach().cpu().numpy()

        if debug:
            print(f"Action: {action}, Transformed: {action_transformed_numpy}")

        return super().step(action_transformed_numpy)




def modify_suite_door_physics(env, door_mass=8.0, hinge_friction=10.0, door_inertia=None, hinge_damping=5.0, hinge_stiffness=2.0):
    """
    Modify the door environment's physics properties to make it much harder to open.
    
    Args:
        env: Door environment instance
        door_mass (float): New mass for the door panel (default: 8.0 kg, original: 2.43455 kg)
        hinge_friction (float): New friction for the hinge joint (default: 10.0, original: 1.0)
        door_inertia (list or None): New inertia tensor for door [ixx, iyy, izz] (default: None, keeps original)
        hinge_damping (float): New damping for the hinge joint (default: 5.0, original: 1.0)
        hinge_stiffness (float): New stiffness for the hinge joint (default: 2.0, original: 0.0)
    """
    # Get door body and joint references
    door_body_id = env.object_body_ids["door"]
    hinge_joint_id = env.sim.model.joint_name2id(env.door.joints[0])
    
    # Get door geom IDs for friction modification
    door_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.door.contact_geoms]
    
    def print_params():
        print(f"Door mass: {env.sim.model.body_mass[door_body_id]}")
        print(f"Door inertia: {env.sim.model.body_inertia[door_body_id]}")
        print(f"Hinge friction: {env.sim.model.dof_frictionloss[hinge_joint_id]}")
        print(f"Hinge damping: {env.sim.model.dof_damping[hinge_joint_id]}")
        print(f"Hinge stiffness: {env.sim.model.dof_armature[hinge_joint_id]}")
        print(f"Door geom frictions: {env.sim.model.geom_friction[door_geom_ids]}")
        print()
    
    print("Before modification:")
    print_params()
    
    # Modify door mass (make it much heavier)
    env.sim.model.body_mass[door_body_id] = door_mass
    
    # Modify door inertia if specified (make it harder to rotate)
    if door_inertia is not None:
        env.sim.model.body_inertia[door_body_id] = np.array(door_inertia)
    else:
        # Increase inertia to make door harder to rotate
        original_inertia = env.sim.model.body_inertia[door_body_id].copy()
        env.sim.model.body_inertia[door_body_id] = original_inertia * 3.0
    
    # Modify hinge joint friction (make it much more resistant)
    env.sim.model.dof_frictionloss[hinge_joint_id] = hinge_friction
    
    # Modify hinge joint damping (add more resistance to movement)
    env.sim.model.dof_damping[hinge_joint_id] = hinge_damping
    
    # Add stiffness to the hinge joint (spring-like resistance)
    env.sim.model.dof_armature[hinge_joint_id] = hinge_stiffness
    
    # Modify door panel friction (affects grasping and contact)
    for geom_id in door_geom_ids:
        # Increase friction significantly
        original_friction = env.sim.model.geom_friction[geom_id].copy()
        env.sim.model.geom_friction[geom_id] = original_friction * 3.0  # Increase friction by 200%
    
    print("After modification:")
    print_params()
    env.sim.forward()

    return env

def modify_suite_cube_physics(env, mass=5.0, friction=[2.0, 0.2, 0.04]):

    cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
    cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]

    def print_params():
        print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
        print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
        print()

    print("Before modification:")
    print_params()

    geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]
    for geom_id in geom_ids:
        env.sim.model.geom_friction[geom_id] = np.array(friction)

    body_id = env.sim.model.body_name2id(env.cube.root_body)
    env.sim.model.body_mass[body_id] = mass

    # joint_names = ['robot0_joint1', 'robot0_joint2', 'robot0_joint3', 'robot0_joint4', 'robot0_joint5', 'robot0_joint6', 'robot0_joint7', 'gripper0_right_finger_joint1', 'gripper0_right_finger_joint2']
    # jnt_ids = [env.sim.model.joint_name2id(jnt) for jnt in joint_names]
    # for jnt_id in jnt_ids:
    #     dof_idx = [i for i, v in enumerate(env.sim.model.dof_jntid) if v == jnt_id]    
    # env.sim.model.dof_damping = 0.000000001
    # env.sim.model.dof_armature = 1000000.0
    # env.sim.model.jnt_stiffness = 0.00000001
    env.sim.forward()

    print("After modification:")
    print_params()



    return env

def modify_suite_slide_physics(env, cube_mass=10.0, cube_friction=None, 
                              table_friction=None, cube_inertia=None):
    """
    Modify the slide environment's physics properties to make it much harder to slide the cube.
    This will break policies trained on the original environment.
    
    Args:
        env: Slide environment instance
        cube_mass (float): New mass for the cube (default: 10.0 kg, original: ~0.1 kg)
        cube_friction (list): New friction for cube geoms [tangential, torsional, rolling] 
                             (default: [0.1, 0.01, 0.001], original: [1.0, 0.5, 0.5])
        table_friction (list): New friction for table geoms [tangential, torsional, rolling]
                              (default: [0.1, 0.01, 0.001], original: [1.0, 0.005, 0.0001])
        cube_inertia (list or None): New inertia tensor for cube [ixx, iyy, izz] (default: None, keeps original)
    """
    if cube_friction is None:
        cube_friction = [0.1, 0.01, 0.001]
    if table_friction is None:
        table_friction = [0.1, 0.01, 0.001]
    # Get cube body and geom references
    cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
    cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]
    
    # Find table geoms by searching for table-related geometry
    table_geom_ids = []
    for i in range(env.sim.model.ngeom):
        geom_name = env.sim.model.geom(i).name
        if 'table' in geom_name.lower() and 'collision' in geom_name.lower():
            table_geom_ids.append(i)
    
    def print_params():
        print(f"Cube mass: {env.sim.model.body_mass[cube_body_id]}")
        print(f"Cube inertia: {env.sim.model.body_inertia[cube_body_id]}")
        print(f"Cube geom frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
        print(f"Table geom frictions: {env.sim.model.geom_friction[table_geom_ids]}")
        print()
    
    print("Before modification:")
    print_params()
    
    # Modify cube mass (make it much heavier)
    env.sim.model.body_mass[cube_body_id] = cube_mass
    
    # Modify cube inertia if specified (make it harder to rotate)
    if cube_inertia is not None:
        env.sim.model.body_inertia[cube_body_id] = np.array(cube_inertia)
    else:
        # Increase inertia to make cube harder to rotate
        original_inertia = env.sim.model.body_inertia[cube_body_id].copy()
        env.sim.model.body_inertia[cube_body_id] = original_inertia * 5.0
    
    # Modify cube friction (make it much more slippery)
    for geom_id in cube_geom_ids:
        env.sim.model.geom_friction[geom_id] = np.array(cube_friction)
    
    # Modify table friction (make it much more slippery)
    for geom_id in table_geom_ids:
        env.sim.model.geom_friction[geom_id] = np.array(table_friction)
    
    print("After modification:")
    print_params()
    env.sim.forward()
    
    return env
