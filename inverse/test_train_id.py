#!/usr/bin/env python3
"""
Simple test script to verify the inverse dynamics training works.
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from learned_inverse_dynamics import FlowInverseDynamics


def test_flow_inverse_dynamics():
    """Test FlowInverseDynamics model with dummy data."""
    print("Testing FlowInverseDynamics...")
    
    # Create dummy data
    batch_size = 32
    obs_dim = 4
    action_dim = 2
    
    obs = torch.randn(batch_size, obs_dim)
    next_obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    
    # Create model
    model = FlowInverseDynamics(
        action_dim=action_dim,
        obs_dim=obs_dim,
        device='cpu'
    )
    
    # Test forward pass
    print("Testing forward pass...")
    loss = model(obs, next_obs, actions)
    print(f"Loss: {loss.item():.6f}")
    assert loss.requires_grad, "Loss should require gradients"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    # Test prediction
    print("Testing prediction...")
    model.eval()
    with torch.no_grad():
        predicted_actions = model.predict(obs, next_obs)
        print(f"Predicted actions shape: {predicted_actions.shape}")
        assert predicted_actions.shape == (batch_size, action_dim), f"Expected shape {(batch_size, action_dim)}, got {predicted_actions.shape}"
    
    print("✓ FlowInverseDynamics test passed!")


def test_training_loop():
    """Test a simple training loop."""
    print("\nTesting training loop...")
    
    # Create dummy data
    batch_size = 16
    obs_dim = 4
    action_dim = 2
    num_samples = 100
    
    states = torch.randn(num_samples, obs_dim)
    actions = torch.randn(num_samples, action_dim)
    next_states = torch.randn(num_samples, obs_dim)
    
    # Create model
    model = FlowInverseDynamics(
        action_dim=action_dim,
        obs_dim=obs_dim,
        device='cpu'
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Simple training loop
    model.train()
    initial_loss = None
    final_loss = None
    
    for epoch in range(5):
        epoch_loss = 0.0
        num_batches = 0
        
        # Simple batch processing
        for i in range(0, num_samples, batch_size):
            batch_states = states[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            batch_next_states = next_states[i:i+batch_size]
            
            optimizer.zero_grad()
            loss = model(batch_states, batch_next_states, batch_actions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if initial_loss is None:
                initial_loss = loss.item()
        
        avg_loss = epoch_loss / num_batches
        if epoch == 4:
            final_loss = avg_loss
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    assert final_loss < initial_loss, "Loss should decrease during training"
    
    print("✓ Training loop test passed!")


if __name__ == "__main__":
    test_flow_inverse_dynamics()
    test_training_loop()
    print("\n🎉 All tests passed!")
