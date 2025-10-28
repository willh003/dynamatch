import gymnasium as gym
import numpy as np
import mujoco
from typing import Dict, Any
from robosuite.wrappers import Wrapper, GymWrapper
import logging
import os
import imageio

from stable_baselines3.common.env_util import make_vec_env as make_vec_env_sb3
from stable_baselines3.common.callbacks import BaseCallback

from .env_transforms import ModifyPhysicsWrapper, modify_suite_door_physics, modify_suite_cube_physics, modify_suite_slide_physics
from .robosuite_controllable_gripper import PandaControllableGripper


def make_vec_env(env_id: str, n_envs: int, **kwargs):
    """
    Make a vectorized environment for gym or robosuite
    """
    env_make_fn = lambda: make_env(env_id, **kwargs)
    return make_vec_env_sb3(env_make_fn, n_envs)

def parse_robosuite_env_id(env_id: str) -> dict:
    """
    Parse the robosuite environment id into a dictionary of information
    """
    env_names = ['Lift', 'Door', 'Slide']
    robot_names =['Panda', 'Sawyer']
    wrapper_types = ['Sparse', 'Dense']
    physics_types = ['HighFriction', 'LowFriction', 'Normal', 'Extreme']

    env_name = None
    reward_shaping = True
    physics_type = 'Normal'
    robots = None

    env_info = env_id.split("-")
    for info in env_info:
        if info in env_names:
            env_name = info
        elif info in robot_names:
            robots = info
        elif info in wrapper_types:
            reward_shaping = info != 'Sparse'
        elif info in physics_types:
            physics_type = info

    assert env_name is not None and robots is not None and physics_type is not None, "ERROR: could not parse environment id"
    return env_name, robots, reward_shaping, physics_type

def make_env(env_id: str, render_mode: str = None, render: bool = False, **kwargs):
    """
    For robosuite, we expect format: Robosuite-<env_id>-<robot>
    Only uses render_mode for gym environments. Otherwise, uses RobosuiteImageWrapper to get images
    """
    if "Robosuite" in env_id:    
        import robosuite as suite
        
        logging.getLogger("robosuite").setLevel(logging.WARNING)
        
        env_name, robots, reward_shaping, physics_type = parse_robosuite_env_id(env_id)
        gripper_types = ["PandaControllableGripper"] if env_name == "Lift" else ["PandaGripper"]

        env = suite.make(env_name = env_name, robots=robots,    
                        gripper_types=gripper_types,
                        has_renderer=False,
                        has_offscreen_renderer=render,
                        use_camera_obs=False,
                        camera_names="frontview",
                        hard_reset=False,
                        reward_shaping=reward_shaping,
                        **kwargs)

        env = GymWrapper(env, flatten_obs=False)
        env = RobosuiteImageWrapper(env)

        if physics_type == "LowFriction":  
            if env_name == "Lift":
                env = modify_suite_cube_physics(env, mass=2, friction=[.005, .0001, .00001])
            elif env_name == "Slide":
                env = modify_suite_slide_physics(env, cube_mass=2.0, cube_friction=[0.005, 0.0001, 0.00001], table_friction=[0.1, 0.0001, 0.00001]) 
        elif physics_type == "HighFriction":
            if env_name == "Door":
                env = modify_suite_door_physics(env, door_mass=12.0, hinge_friction=15.0, hinge_damping=8.0, hinge_stiffness=3.0)
            elif env_name == "Slide":
                env = modify_suite_slide_physics(env, cube_mass=0.01, cube_friction=[10.0, 5e-2, 1e-3], table_friction=[5.0, 0.01, 0.001]) 

        print(f"Made env with name: {env_name}, robots: {robots}, reward_shaping: {reward_shaping}, physics_type: {physics_type}")
    else:
        render_mode = "rgb_array" if render else None

        if "Pusher" in env_id:
            kwargs['max_episode_steps'] = 100
        env = gym.make(env_id, render_mode=render_mode, **kwargs)


    return env

class RobosuiteImageWrapper(Wrapper, gym.Env):
    """
    Renders using mj render function instead of the env render function (which is broken for robosuite)
    """
    def __init__(self, env: GymWrapper, render_width: int = 224, render_height: int = 224):
        super().__init__(env)
        render_cam = self.env.camera_names[0]
        self.render_cam = render_cam
        self.render_width = render_width
        self.render_height = render_height

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Forward Gymnasium-compatible reset signature
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        # Ensure renderer returns a contiguous uint8 RGB array
        frame = self.env.sim.render(width = self.render_width, height = self.render_height, camera_name = self.render_cam)

        return frame

class VideoCallback(BaseCallback):
    """
    Callback for recording videos of agent episodes.

    :param eval_env: The environment used for video recording
    :param video_folder: Path to the folder where videos will be saved
    :param eval_freq: Record video every ``eval_freq`` call of the callback
    :param name_prefix: Common prefix to the saved videos
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when recording video
    """

    def __init__(
        self,
        eval_env: gym.Env,
        video_folder: str,
        eval_freq: int = 10000,
        name_prefix: str = "evaluation",
        verbose: int = 0,
        deterministic: bool = True,
        flip_vertical: bool = False,
        max_steps_per_episode: int = 1000,
        addl_noise_std: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.eval_freq = eval_freq
        self.name_prefix = name_prefix
        self.episode_count = 0
        self.deterministic = deterministic
        self.addl_noise_std = addl_noise_std
        self.flip_vertical = flip_vertical
        self.max_steps_per_episode = max_steps_per_episode
        self.seed = seed
        self.env_id = eval_env.spec.id if eval_env.spec is not None else eval_env.name
        
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.video_folder is not None:
            os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._record_video()
        self.seed += 1
        return True

    def _record_video(self) -> None:
        """Record a single episode and save it as a video."""
        if self.verbose >= 1:
            print(f"Recording video episode {self.episode_count}...")
        
        if self.seed is not None:
            np.random.seed(self.seed)
        # Reset environment
        obs, info = self.eval_env.reset(seed=self.seed)

        state = get_state_from_obs(obs, info, self.env_id)
        frames = []
        done = False
        num_steps = 0

        # Record episode
        while not done and num_steps < self.max_steps_per_episode:
            # Get action from model
            if hasattr(self.model, 'predict_base_and_translated'):
                # action translator
                action, _ = self.model.predict_base_and_translated(policy_observation=obs, translator_observation=state, deterministic=self.deterministic)
                
                action = action[0]
            else:
                # regular sb3 model
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            if self.addl_noise_std > 0:
                action = action + np.random.normal(loc=0.0, scale=self.addl_noise_std, size=action.shape)
            # Render and store frame
            frame = self.eval_env.render()
            
            if frame is not None:
                if self.flip_vertical:
                    frame = np.flipud(frame)
                frames.append(frame)
            
            # Step environment
            obs, _, done, truncated, info = self.eval_env.step(action)
            state = get_state_from_obs(obs, info, self.env_id)
            done = done or truncated
            num_steps += 1
        # Save video
        if frames:
            video_path = os.path.join(
                self.video_folder, 
                f"{self.name_prefix}-episode-{self.episode_count}.mp4"
            )
            imageio.mimsave(video_path, frames, fps=30)
            
            if self.verbose >= 1:
                print(f"Video saved to {video_path}")
        
        self.episode_count += 1

def modify_env_integrator(env, integrator=None, frame_skip=None):
    if integrator is not None:
        assert integrator in ['euler', 'rk4'], "ERROR: integrator must be euler or rk4"
        if integrator == 'euler':
            env.unwrapped.model.opt.integrator = 0  # mjINT_EULER
        elif integrator == 'rk4':
            env.unwrapped.model.opt.integrator = 1  # mjINT_RK4
    else:
        print(f"WARNING: using default integrator for forward, {env.unwrapped.model.opt.integrator}")
    
    if frame_skip is not None:
        env.unwrapped.frame_skip = frame_skip
    else:
        print(f"WARNING: using default frame skip for forward, {env.unwrapped.frame_skip}")

    return env

def set_state(env: gym.Env, state: np.ndarray):
    """
    Set the state of the environment to the given state

    Args:
        env: The environment to set the state of
        state: The state to set the environment to (shape: [cart_pos, pole_angle, cart_vel, pole_angular_vel])
        info: The info from stepping (often includes full state variables, in case they are masked in the env observation)
    
    Returns:
        np.ndarray: The state of the environment after setting it
    """
    env_id = env.unwrapped.spec.id
    if "InvertedPendulum" in env_id:
        set_state_pendulum(env, state)
    elif "Ant" in env_id:
        set_state_ant(env, state)
    else:
        raise ValueError(f"Environment {env_id} not supported for state setting - implement set_state for this environment") 

def set_state_pendulum(env: gym.Env, state: np.ndarray):
    """
    Set the state of the environment to the given state

    Args:
        env: The environment to set the state of
        state: The state to set the environment to (shape: [cart_pos, pole_angle, cart_vel, pole_angular_vel])
    
    Returns: None
    """
    # Extract position and velocity from the state vector
    # State format: [cart_pos, pole_angle, cart_vel, pole_angular_vel]
    qpos = state[:2]  # [cart_pos, pole_angle]
    qvel = state[2:]  # [cart_vel, pole_angular_vel]
    
    # Use the environment's set_state method
    env.unwrapped.set_state(qpos, qvel)
    
    # Return the current state of the environment
    current_qpos = env.unwrapped.data.qpos[:2]
    current_qvel = env.unwrapped.data.qvel[:2]
    current_state = np.concatenate([current_qpos, current_qvel])

    assert np.allclose(current_state, state, atol=1e-10), "ERROR: state not set correctly"


def set_state_ant(env: gym.Env, state: np.ndarray):
    """
    Set the MuJoCo state from Gym observation and info.
    
    Args:
        env: Ant-v5 gym environment
        obs: observation array (105 elements: 13 qpos + 14 qvel + 78 cfrc_ext)
        info: info dict containing 'x_position' and 'y_position'
    """
    # Extract components from observation
    qpos= state[:15]  # x,y,z, quat(4), joints(8)
    qvel = state[15:29]   # all 14 velocities
    
    # Set the state
    env.unwrapped.set_state(qpos, qvel)


def parse_raw_observations_pendulum(obs_array: np.ndarray, _obs_shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Parse raw observation array into dictionary format expected by the dataset.
    
    For InvertedPendulum-v5, the observation array typically contains:
    [x,theta, x_dot, theta_dot] where:
    - x: cart position
    - x_dot: cart velocity  
    - theta: pole angle
    - theta_dot: pole velocity
    
    Args:
        obs_array: Raw observation array from environment
        obs_shape_meta: Shape metadata for observations
        
    Returns:
        Dictionary with parsed observations
    """
    obs_dict = {}
    
    # For InvertedPendulum-v5, map the 4-element array to the expected keys
    # Environment observation order: [cart_pos, pole_angle, cart_vel,  pole_vel]
    if len(obs_array.shape) == 1 and obs_array.shape[0] == 4:
        # Single observation - reorder to match set_state expectation
        obs_dict["cart_position"] = np.array([obs_array[0]])  # cart_pos
        obs_dict["pole_angle"] = np.array([obs_array[1]])     # pole_angle  
        obs_dict["cart_velocity"] = np.array([obs_array[2]])  # cart_vel
        obs_dict["pole_velocity"] = np.array([obs_array[3]])  # pole_vel
    elif len(obs_array.shape) == 2 and obs_array.shape[1] == 4:
        # Multiple observations (trajectory) - reorder to match set_state expectation
        obs_dict["cart_position"] = obs_array[:, 0:1]  # cart_pos
        obs_dict["pole_angle"] = obs_array[:, 1:2]     # pole_angle
        obs_dict["cart_velocity"] = obs_array[:, 2:3]  # cart_vel
        obs_dict["pole_velocity"] = obs_array[:, 3:4]  # pole_vel
    
    return obs_dict

def parse_raw_observations_ant(obs_array: np.ndarray, info:dict, _obs_shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Parse raw observation array into dictionary format expected by the dataset.
    
    For Ant-v5, the observation array typically contains:
    """
    x_pos = np.array([d["x_position"] for d in info])[:,None]
    y_pos = np.array([d["y_position"] for d in info])[:,None]

    full_obs = np.concatenate([x_pos, y_pos,obs_array], axis=1)
    obs_dict = {"full_obs": full_obs}
    return obs_dict

def parse_raw_observations(obs_array: np.ndarray, info:dict, obs_shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Parse raw observation array into dictionary format expected by the dataset.
    """
    if "cart_position" in obs_shape_meta:
        # TODO: better check for env type
        return parse_raw_observations_pendulum(obs_array, obs_shape_meta)
    else:
        return parse_raw_observations_ant(obs_array, info, obs_shape_meta)

def get_state_from_obs_pendulum(obs_array: np.ndarray) -> np.ndarray:
    """
    Get state from observation array for pendulum environment.
    """
    return obs_array

def get_state_from_obs_ant(obs_array: np.ndarray, info:dict) -> np.ndarray:
    """
    Get state from observation array for ant environment.
    """
    x_pos = info['x_position']
    y_pos = info['y_position']

    full_obs = np.concatenate([[x_pos], [y_pos],obs_array], axis=-1)
    return full_obs

def get_state_from_obs_ant_nopos(obs_array: np.ndarray, _info:dict) -> np.ndarray:
    """
    Get state from observation array for ant environment.
    """
    return obs_array


def get_state_from_obs_fetch(obs: dict, _info: dict) -> np.ndarray:
    """
    Get state from observation array for fetch environment.
    """

    obs_array = obs['observation']
    goal_array = obs['desired_goal']
    full_obs = np.concatenate([obs_array, goal_array], axis=-1)
    return full_obs


def get_state_from_obs_door(obs: dict, _info: dict) -> np.ndarray:
    """
    Get state from observation array for door environment.
    """
    proprio_array = obs['robot0_proprio-state']
    object_state_array = obs['object-state']
    full_obs = np.concatenate([proprio_array, object_state_array], axis=-1)
    
    return full_obs

def get_state_from_obs_pusher(obs: dict, _info: dict, state_indices: list = None) -> np.ndarray:
    """
    Get state from observation array for pusher environment.
    """
    if state_indices is not None:
        if len(obs.shape) > 1:
            state = obs[:, state_indices]
        else:
            state = obs[state_indices]
    else:
        state = obs
    
    return state

def get_state_from_obs(obs: np.ndarray, info:dict, env_id: str, state_indices: list = None) -> np.ndarray:
    """
    Get state from observation array for environment.
    """
    if "Pendulum" in env_id:
        return get_state_from_obs_pendulum(obs)
    elif "Ant" in env_id:
        if "NoPos" in env_id: # don't append (x,y) position to state
            return get_state_from_obs_ant_nopos(obs, info)
        return get_state_from_obs_ant(obs, info)
    elif "Fetch" in env_id:
        return get_state_from_obs_fetch(obs, info)
    elif "Door" in env_id:
        return get_state_from_obs_door(obs, info)
    elif "Pusher" in env_id:
        return get_state_from_obs_pusher(obs, info, state_indices)
    else:
        raise ValueError(f"Environment {env_id} not supported for state getting - implement get_state_from_obs for this environment")


def get_reward_from_obs_pusher(reward, info) -> float:
    """
    Get reward from observation array for pusher environment.
    """
    if isinstance(info, dict):
        reward = info['reward_dist']
    else: 
        reward = [i['reward_dist'] for i in info]
    return reward

def get_reward_from_obs(reward: float, obs: dict, info:dict, env_id: str) -> float:
    """
    Get reward from observation array for environment.
    """
    if "Pusher" in env_id:
        return get_reward_from_obs_pusher(reward, info)
    else:
        return reward

def get_success_from_obs_pusher(obs: dict, info:dict) -> bool:
    """
    Get success from observation array for pusher environment.
    """
    PUSHER_THRESHOLD = 0.1
    if len(obs.shape) > 1:
        goal = obs[:, 20:23]
        object_pos = obs[:, 14:17]
    else:
        goal = obs[20:23]
        object_pos = obs[14:17]

    dist_to_goal = np.linalg.norm(object_pos - goal, axis=-1)

    return dist_to_goal < PUSHER_THRESHOLD

def get_success_from_obs_pendulum(obs: dict, _info: dict) -> bool:
    """
    Get success from observation array for pendulum environment.
    """
    return np.zeros(obs.shape[0], dtype=bool)

def get_success_from_obs(obs: dict, info:dict, env_id: str) -> bool:
    """
    Get success from observation array for environment.

    """
    if "Pusher" in env_id:
        return get_success_from_obs_pusher(obs, info)
    elif "Pendulum" in env_id:
        return get_success_from_obs_pendulum(obs, info)
    else:
        print(f"WARNING: no success function implemented for environment {env_id}")
        return False