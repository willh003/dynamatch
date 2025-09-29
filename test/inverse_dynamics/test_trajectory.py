import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
import copy
from envs.inverse.inverse_dynamics import mujoco_inverse_dynamics, gym_inverse_dynamics
from envs.inverse.set_state import set_state
import matplotlib.pyplot as plt
import os
import imageio
from envs.env_utils import modify_env_integrator
from cluster_utils import set_cluster_graphics_vars



def plot_states_with_action_arrows(original_states, id_states, original_actions, id_actions, out_dir, state_idx=1):
    """
    Plot states with action arrows for both original and ID trajectories.
    
    Args:
        original_states: numpy array of original trajectory states
        id_states: numpy array of ID trajectory states  
        original_actions: numpy array of original trajectory actions
        id_actions: numpy array of ID trajectory actions
        out_dir: output directory to save the plot
    """
    min_len_states = min(len(original_states), len(id_states))
    if min_len_states > 0:
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate scaling factor based on max absolute state value
        max_state_value = max(np.max(np.abs(original_states[:min_len_states, 0])), 
                             np.max(np.abs(id_states[:min_len_states, 0])))
        
        # Original trajectory with action arrows
        time_steps = np.arange(min_len_states)
        ax1.plot(time_steps, original_states[:min_len_states, state_idx], linewidth=2, color="blue", label="Original traj theta")
        
        # Add action arrows for original trajectory
        original_actions_np = np.array(original_actions)
        min_len_actions_orig = min(len(original_actions_np), min_len_states)
        for i in range(min_len_actions_orig):
            if i < len(original_actions_np):
                action = original_actions_np[i]
                # Scale arrow length to max state value
                arrow_length = abs(action) * max_state_value * 0.1  # Scale to max state value
                if action > 0:
                    ax1.arrow(i, original_states[i, state_idx], 0, arrow_length, 
                             head_width=0.2, head_length=0.02, fc='blue', ec='blue', alpha=0.7)
                elif action < 0:
                    ax1.arrow(i, original_states[i, state_idx], 0, -arrow_length, 
                             head_width=0.2, head_length=0.02, fc='blue', ec='blue', alpha=0.7)
        
        ax1.set_xlabel("time")
        ax1.set_ylabel("theta")
        ax1.set_title("Original vs ID trajectory with action arrows")
        ax1.grid(True, alpha=0.3)
        
        # ID trajectory with action arrows
        ax1.plot(time_steps, id_states[:min_len_states, state_idx], color="orange", linewidth=2, label="ID traj theta")
        
        # Add action arrows for ID trajectory
        id_actions_np = np.array(id_actions)
        min_len_actions_id = min(len(id_actions_np), min_len_states)
        for i in range(min_len_actions_id):
            if i < len(id_actions_np):
                action = id_actions_np[i]
                # Scale arrow length to max state value
                arrow_length = abs(action) * max_state_value * 0.1  # Scale to max state value
                if action > 0:
                    ax1.arrow(i, id_states[i, state_idx], 0, arrow_length, 
                             head_width=0.2, head_length=0.02, fc='orange', ec='orange', alpha=0.7)
                elif action < 0:
                    ax1.arrow(i, id_states[i, state_idx], 0, -arrow_length, 
                             head_width=0.2, head_length=0.02, fc='orange', ec='orange', alpha=0.7)

        ax1.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"states_{state_idx}_with_action_arrows.png"))
        plt.clf()



def test_mujoco_id(model_path=None, num_episodes=10, episode_length=200, integrator=None, frame_skip=None):
    """
    Align the MuJoCo and Gym environments.
    
    Returns:
        None
    """
    env = gym.make('InvertedPendulum-v5')
    # NOTE: NOT BY DEFAULT IN MUJOCO! BUT MAKES INVERSE DYNAMICS WORK
    env = modify_env_integrator(env, integrator, frame_skip)

    if model_path is not None:
        action_model = PPO.load(model_path)
    else:
        # use random actions if no model
        action_model = lambda x: env.action_space.sample()

    # Use the exact same model instance as Gym to avoid any discrepancies
    mujoco_model = env.unwrapped.model    

    all_error_on_expert = []
    all_pred_controls = []
    all_real_controls = []

    for rollout in range(num_episodes):
        state, _ = env.reset()

        traj_error_on_expert = []
        traj_pred_controls = []
        traj_real_controls = []
       
        for step in range(episode_length):
            mujoco_state = copy.deepcopy(env.unwrapped.data)
            policy_action = action_model.predict(state)[0]
            next_state, _, _, _, _ = env.step(policy_action)
            mujoco_next_state = copy.deepcopy(env.unwrapped.data)
            
            id_action = mujoco_inverse_dynamics(env, mujoco_state, mujoco_next_state)
            error = policy_action - id_action
            traj_error_on_expert.append(error.item())

            print(f"expert action: {policy_action}, id action: {id_action}, error on expert: {error}")
            traj_pred_controls.append(id_action.squeeze())
            traj_real_controls.append(policy_action.squeeze())
            
            state = next_state
            
        all_error_on_expert.append(traj_error_on_expert)
        all_pred_controls.append(traj_pred_controls)
        all_real_controls.append(traj_real_controls)

    all_error_on_expert = np.array(all_error_on_expert)
    all_pred_controls = np.array(all_pred_controls)
    all_real_controls = np.array(all_real_controls)

    plt.plot(all_error_on_expert.T)
    plt.savefig("test_media/traj_test/error_on_expert.png")
    plt.clf()
    plt.plot(all_pred_controls.T, color="red", label="Predicted controls")
    plt.plot(all_real_controls.T, color="blue", label="Real controls")
    plt.legend()
    plt.savefig("test_media/traj_test/controls.png")
    plt.clf()

    print(f"mean absolute error on expert over {num_episodes} episodes and {len(all_error_on_expert[0])} steps: {np.abs(all_error_on_expert).mean():.6f}")
    print(f"mean absolute error as percent of action: {np.abs(all_error_on_expert / all_real_controls).mean() * 100:.3f}%")

    return all_error_on_expert, all_pred_controls, all_real_controls




def test_gym_id(model_path=None, num_episodes=10, episode_length=200, integrator=None, frame_skip=None, deterministic=True):
    """
    Align the MuJoCo and Gym environments.
    
    Returns:
        None
    """
    # Output directory for plots and videos
    out_dir = "test_media/id_traj_following_test"
    os.makedirs(out_dir, exist_ok=True)

    # Envs for recording videos
    env_original = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    env_original = gym.wrappers.RecordVideo(env_original, video_folder=out_dir, name_prefix='original')
    env_original = modify_env_integrator(env_original, integrator, frame_skip)

    env_id_rollout = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
    env_id_rollout = gym.wrappers.RecordVideo(env_id_rollout, video_folder=out_dir, name_prefix='id_follow')
    env_id_rollout = modify_env_integrator(env_id_rollout, integrator, frame_skip)

    if model_path is not None:
        expert_policy = PPO.load(model_path)
    else:
        # use random actions if no model
        expert_policy = lambda obs: env_original.action_space.sample()

    # Use the exact same model instance as Gym to avoid any discrepancies
    mujoco_model = env_original.unwrapped.model    

    state, _ = env_original.reset()
    original_traj_actions = []
    original_traj_states = []
    original_traj_rewards = []

    for step in range(episode_length):
        original_traj_states.append(state.squeeze())
        if callable(expert_policy):
            policy_action = expert_policy(state)
        else:
            policy_action = expert_policy.predict(state, deterministic=deterministic)[0]

        next_state, reward, terminated, truncated, _ = env_original.step(policy_action)

        original_traj_actions.append(np.squeeze(policy_action))
        original_traj_rewards.append(np.squeeze(reward))

        state = next_state

        if terminated or truncated:
            print(f"Episode terminated or truncated at step {step}")
            break

    id_traj_actions = []
    id_traj_states = []
    id_traj_rewards = []

    expert_corrective_actions = []
    expert_corrective_rewards = []
    acc_experts = []
    acc_ids_from_fd = []

    # dummy environment for inverse dynamics (no rendering needed)
    id_env = gym.make('InvertedPendulum-v5')
    id_env = modify_env_integrator(id_env, integrator, frame_skip)

    # Roll the ID follower over the recorded original trajectory
    env_id_rollout.reset()
    set_state(env_id_rollout, original_traj_states[0])
    state = np.copy(original_traj_states[0])
    for step in range(len(original_traj_states) - 1):
        id_traj_states.append(np.squeeze(state))
        # trajectory following: get id(cur_state, target_state), where target_state is the next state in the trajectory        

        #accel_from_id = (target_state[2:] - state[2:]) / id_env.unwrapped.dt
        # accel_in_traj = (target_state[2:] - original_traj_states[step][2:]) / id_env.unwrapped.dt
        # print(f'step {step}: id qpos: {state[:2]}, id qvel: {state[2:]}, id qacc: {accel_from_id})')
        # print(f'step {step}: traj qpos: {original_traj_states[step][:2]}, traj qvel: {original_traj_states[step][2:]}, traj qacc: {accel_in_traj}')
        
        state_id = np.copy(state)
        state_expert = np.copy(original_traj_states[step])
        next_state_expert = np.copy(original_traj_states[step + 1])
        next_state_fd_vel = np.copy(next_state_expert)
        next_state_fd_vel[2:] = (next_state_expert[:2] - state_id[:2]) / id_env.unwrapped.dt
        
        id_action_from_expert_traj = gym_inverse_dynamics(id_env, state_expert, next_state_expert)
        id_action_from_id_traj = gym_inverse_dynamics(id_env, state_id, next_state_expert)
        id_action_from_fd_vel = gym_inverse_dynamics(id_env, state_id, next_state_fd_vel)

        print(f"--------------------------------STATES AND ACTIONS--------------------------------")

        print(f"pos expert: {state_expert[:2]}, vel expert: {state_expert[2:]}, acc expert: {(next_state_expert[2:] - state_expert[2:]) / id_env.unwrapped.dt}")
        print(f"state id: {state_id[:2]}, vel id: {state_id[2:]}, acc id: {(next_state_expert[2:] - state_id[2:]) / id_env.unwrapped.dt}")
        print(f"state fd vel: {state_id[:2]}, vel fd vel: {state_id[2:]}, acc fd vel: {(next_state_fd_vel[2:] - state_id[2:]) / id_env.unwrapped.dt}")

        acc_experts.append((next_state_expert[2:] - state_expert[2:]) / id_env.unwrapped.dt)
        acc_ids_from_fd.append((next_state_fd_vel[2:] - state_id[2:]) / id_env.unwrapped.dt)

        print(f"actual action: {original_traj_actions[step]}")
        print(f"id action from expert traj: {id_action_from_expert_traj}")
        print(f"id action from id traj: {id_action_from_id_traj}")
        print(f"id action from fd vel: {id_action_from_fd_vel}")

        state, reward, terminated, truncated, _ = env_id_rollout.step(id_action_from_fd_vel)
        print(f"--------------------------------Ending id computation--------------------------------")
        
        
        
        id_traj_actions.append(np.squeeze(id_action_from_fd_vel))
        id_traj_rewards.append(np.squeeze(reward))

        if terminated or truncated:
            break
    
    print(f"min acc experts: {np.min(np.array(acc_experts), axis=0)}")
    print(f"max acc experts: {np.max(np.array(acc_experts), axis=0)}")

    plt.clf()
    plt.hist(original_traj_actions, color="blue", alpha=0.5, label="Expert actions")
    plt.hist(id_traj_actions, color="orange", alpha=0.5, label="ID actions")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "actions_distribution.png"))
    plt.clf()


    # plot the direction of acc_expert, acc_ids_from_fd, and ctrl at each timestep
    acc_experts_np = np.array(acc_experts)
    acc_ids_from_fd_np = np.array(acc_ids_from_fd)
    id_actions_np = np.array(id_traj_actions)
    
    # Plot for acceleration component 0
    min_len_acc = min(len(acc_experts_np), len(acc_ids_from_fd_np), len(id_actions_np))
    if min_len_acc > 0:
        plt.figure(figsize=(12, 6))
        time_steps = np.arange(min_len_acc)
        
        #plt.plot(time_steps, acc_experts_np[:min_len_acc, 0], 'b-', linewidth=2, label='Expert acc[0]')
        plt.plot(time_steps, acc_ids_from_fd_np[:min_len_acc, 0], color='orange', linewidth=2, label='ID from FD vel acc[0]')
        
        # Plot control direction as arrows - scale based on data range
        data_range = max(np.max(np.abs(acc_experts_np[:min_len_acc, 0])), np.max(np.abs(acc_ids_from_fd_np[:min_len_acc, 0])))
        arrow_scale = data_range * 0.3  # Scale arrows to 30% of data range
        arrow_width = 0.5
        arrow_head_length = arrow_scale * 0.1
        
        for i in range(min_len_acc):
            action = id_actions_np[i]
            if action > 0:
                plt.arrow(i, 0, 0, arrow_scale, head_width=arrow_width, head_length=arrow_head_length, fc='green', ec='green', alpha=0.8)
            elif action < 0:
                plt.arrow(i, 0, 0, -arrow_scale, head_width=arrow_width, head_length=arrow_head_length, fc='green', ec='green', alpha=0.8)
        
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.title('Acceleration component 0 and Control direction over time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "acc_0_and_control.png"))
        plt.clf()
        
        # Plot for acceleration component 1
        plt.figure(figsize=(12, 6))
        #plt.plot(time_steps, acc_experts_np[:min_len_acc, 1], 'b-', linewidth=2, label='Expert acc[1]')
        plt.plot(time_steps, acc_ids_from_fd_np[:min_len_acc, 1], color='orange', linewidth=2, label='ID from FD vel acc[1]')
        
        # Plot control direction as arrows - scale based on data range
        data_range = max(np.max(np.abs(acc_experts_np[:min_len_acc, 1])), np.max(np.abs(acc_ids_from_fd_np[:min_len_acc, 1])))
        arrow_scale = data_range * 0.3  # Scale arrows to 30% of data range
        arrow_width = 0.5
        arrow_head_length = arrow_scale * 0.1
        
        for i in range(min_len_acc):
            action = id_actions_np[i]
            if action > 0:
                plt.arrow(i, 0, 0, arrow_scale, head_width=arrow_width, head_length=arrow_head_length, fc='green', ec='green', alpha=0.8)
            elif action < 0:
                plt.arrow(i, 0, 0, -arrow_scale, head_width=arrow_width, head_length=arrow_head_length, fc='green', ec='green', alpha=0.8)
        
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.title('Acceleration component 1 and Control direction over time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "acc_1_and_control.png"))
        plt.clf()

    # From each ID state, compute what the expert would do (corrective) and the resulting reward
    id_env.reset()
    for s in id_traj_states:
        set_state(id_env, s)
        if callable(expert_policy):
            expert_action = expert_policy(s)
        else:
            expert_action = expert_policy.predict(s, deterministic=deterministic)[0]
        _, r, _, _, _ = id_env.step(expert_action)
        expert_corrective_actions.append(np.squeeze(expert_action))
        expert_corrective_rewards.append(np.squeeze(r))

    # Ensure videos are written to disk
    env_original.close()
    env_id_rollout.close()

    # Convert MP4s to GIFs
    try:
        # Find the generated MP4 files
        original_mp4 = os.path.join(out_dir, "original-episode-0.mp4")
        id_follow_mp4 = os.path.join(out_dir, "id_follow-episode-0.mp4")
        
        if os.path.exists(original_mp4):
            # Convert original video to GIF
            reader = imageio.get_reader(original_mp4)
            frames = [frame for frame in reader]
            imageio.mimsave(os.path.join(out_dir, "original-episode-0.gif"), frames, duration=0.1)
            print(f"Converted {original_mp4} to GIF")
            
        if os.path.exists(id_follow_mp4):
            # Convert ID follow video to GIF
            reader = imageio.get_reader(id_follow_mp4)
            frames = [frame for frame in reader]
            imageio.mimsave(os.path.join(out_dir, "id_follow-episode-0.gif"), frames, duration=0.1)
            print(f"Converted {id_follow_mp4} to GIF")
            
    except Exception as e:
        print(f"Warning: Could not convert videos to GIF: {e}")

    # Plotting

    # 1) States: original vs ID (each component vs time)
    original_states_np = np.array(original_traj_states)
    id_states_np = np.array(id_traj_states)
    min_len_states = min(len(original_states_np), len(id_states_np))
    if min_len_states > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # State component 0 vs time
        ax1.plot(original_states_np[:min_len_states, 0], label="Original traj")
        ax1.plot(id_states_np[:min_len_states, 0], label="ID traj")
        ax1.set_xlabel("time")
        ax1.set_ylabel("state[0]")
        ax1.legend()
        ax1.set_title("State component 0: Original vs ID")
        
        # State component 1 vs time
        ax2.plot(original_states_np[:min_len_states, 1], label="Original traj")
        ax2.plot(id_states_np[:min_len_states, 1], label="ID traj")
        ax2.set_xlabel("time")
        ax2.set_ylabel("state[1]")
        ax2.legend()
        ax2.set_title("State component 1: Original vs ID")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "states_original_vs_id.png"))
        plt.clf()


    # 1b) States with action arrows: original and ID trajectories with action direction indicators
    plot_states_with_action_arrows(original_states_np, id_states_np, original_traj_actions, id_traj_actions, out_dir, state_idx=1)
    plot_states_with_action_arrows(original_states_np, id_states_np, original_traj_actions, id_traj_actions, out_dir, state_idx=0)

    # 2) Rewards over time: original, ID, expert corrective (aligned by min length)
    original_rewards_np = np.array(original_traj_rewards)
    id_rewards_np = np.array(id_traj_rewards)
    corrective_rewards_np = np.array(expert_corrective_rewards)
    min_len_rewards = min(len(original_rewards_np), len(id_rewards_np), len(corrective_rewards_np))
    if min_len_rewards > 0:
        plt.figure()
        plt.plot(original_rewards_np[:min_len_rewards], label="Original policy rewards")
        plt.plot(id_rewards_np[:min_len_rewards], label="ID rewards")
        plt.plot(corrective_rewards_np[:min_len_rewards], label="Expert corrective rewards")
        plt.xlabel("t")
        plt.ylabel("reward")
        plt.legend()
        plt.title("Rewards over trajectory")
        plt.savefig(os.path.join(out_dir, "rewards.png"))
        plt.clf()

    # 3) Actions over time: original expert vs ID vs expert corrective
    original_actions_np = np.array(original_traj_actions)
    id_actions_np = np.array(id_traj_actions)
    corrective_actions_np = np.array(expert_corrective_actions)
    min_len_actions = min(len(original_actions_np), len(id_actions_np), len(corrective_actions_np))
    
    if min_len_actions > 0:
        plt.figure()
        plt.plot(original_actions_np[:min_len_actions], label="Original expert actions")
        plt.plot(id_actions_np[:min_len_actions], label="ID actions")
        #plt.plot(corrective_actions_np[:min_len_actions], label="Expert corrective actions")
        plt.xlabel("t")
        plt.ylabel("action")
        plt.legend()
        plt.title("Actions over trajectory")
        plt.savefig(os.path.join(out_dir, "actions.png"))
        plt.clf()

    # 4) Error per time step: (original expert action - ID action)
    min_len_err = min(len(original_actions_np), len(id_actions_np))
    if min_len_err > 0:
        err = original_actions_np[:min_len_err] - id_actions_np[:min_len_err]
        plt.figure()
        plt.plot(err, label="expert - id")
        plt.xlabel("t")
        plt.ylabel("action error")
        plt.legend()
        plt.title("Expert vs ID action error over time")
        plt.savefig(os.path.join(out_dir, "error_on_expert.png"))
        plt.clf()
    


def test_single_state():
    id_env = gym.make('InvertedPendulum-v5')
    id_env = modify_env_integrator(id_env, integrator='euler', frame_skip=1)
    state_expert = np.array([ 0.01591522,  0.02704899, -0.20948179,  0.30704767])
    state_id = np.array([ 0.04854799,  0.12820882, -0.11944098,  0.73436591])
    next_state_expert =np.array([ 0.01218804,  0.032207,   -0.18635916,  0.25790006])
    expert_action = np.array([0.1402])
    #next_state_id = np.array([ 0.04760707,  0.1403369, -0.04704569,  0.60640354])

    next_state_fd_vel = np.copy(next_state_expert)
    next_state_fd_vel[2:] = (next_state_expert[:2] - state_id[:2]) / id_env.unwrapped.dt
    
    
    
    id_action_from_expert_traj = gym_inverse_dynamics(id_env, state_expert, next_state_expert)
    id_action_from_id_traj = gym_inverse_dynamics(id_env, state_id, next_state_expert)
    id_action_from_fd_vel = gym_inverse_dynamics(id_env, state_id, next_state_fd_vel)
    print(f'--------------------------------')
    print(f"state expert: {state_expert}")
    print(f"state id: {state_id}")
    print(f"next state expert: {next_state_expert}")
    print(f"next state fd vel: {next_state_fd_vel}")
    
    

    print(f"id action from expert traj: {id_action_from_expert_traj}")
    print(f"id action from id traj: {id_action_from_id_traj}")
    print(f"id action from id traj + fd vel: {id_action_from_fd_vel}")
    print(f"original action: {expert_action}")


if __name__ == "__main__":
    set_cluster_graphics_vars()
    model_path = "/home/wph52/weird/dynamics/rl/runs/InvertedPendulum-v5_20250923_150900/models/best_model.zip"
    #test_single_state()
    #test_mujoco_id(model_path=model_path, frame_skip=1)
    test_gym_id(model_path=model_path, integrator='euler', frame_skip=1, deterministic=False)