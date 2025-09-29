#!/usr/bin/env python3
"""
Test script for the set_state function implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import gymnasium as gym
from envs.inverse.set_state import set_state

def test_set_state():
    """Test that set_state correctly sets the environment state and returns it."""
    
    # Create environment
    env = gym.make('InvertedPendulum-v5')
    
    # Reset to get initial state
    initial_state, _ = env.reset()
    print(f"Initial state: {initial_state}")
    print(f"Initial state shape: {initial_state.shape}")
    
    # Create a test state vector
    test_state = np.array([0.1, 0.2, 0.3, 0.4])  # [cart_pos, pole_angle, cart_vel, pole_angular_vel]
    print(f"Test state: {test_state}")
    
    # Set the environment to the test state and get the returned state
    returned_state = set_state(env, test_state)
    print(f"Returned state: {returned_state}")
    
    # Verify the state was set correctly by checking the internal MuJoCo data
    unwrapped = env.unwrapped
    actual_qpos = unwrapped.data.qpos[:2]  # [cart_pos, pole_angle]
    actual_qvel = unwrapped.data.qvel[:2]  # [cart_vel, pole_angular_vel]
    actual_state = np.concatenate([actual_qpos, actual_qvel])
    
    print(f"Actual qpos: {actual_qpos}")
    print(f"Actual qvel: {actual_qvel}")
    print(f"Expected qpos: {test_state[:2]}")
    print(f"Expected qvel: {test_state[2:]}")
    
    # Check if the state was set correctly
    qpos_match = np.allclose(actual_qpos, test_state[:2], atol=1e-10)
    qvel_match = np.allclose(actual_qvel, test_state[2:], atol=1e-10)
    returned_state_match = np.allclose(returned_state, test_state, atol=1e-10)
    
    print(f"qpos match: {qpos_match}")
    print(f"qvel match: {qvel_match}")
    print(f"returned state match: {returned_state_match}")
    
    if qpos_match and qvel_match and returned_state_match:
        print("✅ set_state test PASSED!")
        return True
    else:
        print("❌ set_state test FAILED!")
        return False

def test_state_roundtrip():
    """Test that we can set a state and then retrieve it correctly."""
    
    env = gym.make('InvertedPendulum-v5')
    
    # Get initial state
    initial_state, _ = env.reset()
    
    # Set a specific state and get the returned state
    test_state = np.array([0.5, -0.3, 0.1, -0.2])
    returned_state = set_state(env, test_state)
    
    # Verify the state was set correctly by checking the internal MuJoCo data
    unwrapped = env.unwrapped
    actual_qpos = unwrapped.data.qpos[:2]
    actual_qvel = unwrapped.data.qvel[:2]
    actual_state = np.concatenate([actual_qpos, actual_qvel])
    
    print(f"Test state: {test_state}")
    print(f"Returned state: {returned_state}")
    print(f"Actual state after set_state: {actual_state}")
    
    # The states should match exactly since we just set them
    state_diff = np.abs(test_state - actual_state)
    returned_diff = np.abs(test_state - returned_state)
    print(f"State difference (test vs actual): {state_diff}")
    print(f"State difference (test vs returned): {returned_diff}")
    
    if np.allclose(test_state, actual_state, atol=1e-10) and np.allclose(test_state, returned_state, atol=1e-10):
        print("✅ State roundtrip test PASSED!")
        return True
    else:
        print("❌ State roundtrip test FAILED!")
        return False

if __name__ == "__main__":
    print("Testing set_state function...")
    print("=" * 50)
    
    test1_passed = test_set_state()
    print("\n" + "=" * 50)
    
    test2_passed = test_state_roundtrip()
    print("\n" + "=" * 50)
    
    if test1_passed and test2_passed:
        print("🎉 All tests PASSED!")
    else:
        print("💥 Some tests FAILED!")
