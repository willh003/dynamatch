import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt

from action_translator import SimpleActionTranslator
from model_utils import print_model_info


def load_action_translation_dataset(dataset_path):
    """Load action translation dataset from zarr file."""
    print("=== Loading Action Translation Dataset ===")
    
    store = zarr.open(dataset_path, mode='r')
    data_group = store['data']
    meta_group = store['meta']
    
    states = data_group['state'][:]
    original_actions = data_group['original_action'][:]
    shifted_actions = data_group['shifted_action'][:]
    num_samples = meta_group['num_samples'][0]
    
    print(f"Loaded dataset with {num_samples} samples")
    print(f"State shape: {states.shape}")
    print(f"Original action shape: {original_actions.shape}")
    print(f"Shifted action shape: {shifted_actions.shape}")
    
    return states, original_actions, shifted_actions


def train_action_translator(states, original_actions, shifted_actions, 
                          obs_dim, action_dim, num_epochs=100, learning_rate=1e-3, 
                          batch_size=64, device='cpu'):
    """Train the SimpleActionTranslator model."""
    print("=== Training Action Translator ===")
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    original_actions_tensor = torch.FloatTensor(original_actions).unsqueeze(1)  # Add action dimension
    shifted_actions_tensor = torch.FloatTensor(shifted_actions).unsqueeze(1)  # Add action dimension
    
    # Create dataset and dataloader
    dataset = TensorDataset(states_tensor, original_actions_tensor, shifted_actions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SimpleActionTranslator(action_dim, obs_dim)
    model.to(device)
    
    # Print model info
    print(f"\n--- Action Translator Model Info ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    
    # Detailed breakdown
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                print(f"{name}: {param_count:,} parameters")
    print("=" * 40)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_original_actions, batch_shifted_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_original_actions = batch_original_actions.to(device)
            batch_shifted_actions = batch_shifted_actions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: predict shifted action given state and original action
            predicted_shifted_actions = model(batch_states, batch_original_actions)
            
            # Compute loss
            loss = criterion(predicted_shifted_actions, batch_shifted_actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    print(f"Training completed. Final loss: {train_losses[-1]:.6f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Action Translator Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('action_translator_training_curve.png')
    plt.close()
    
    return model, train_losses

def create_output_path_from_config(config_path):
    """Create output path by replacing 'sequence' with 'relabeled_actions' in the buffer path."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    buffer_path = config['buffer_path']
    # Replace 'sequence' with 'relabeled_actions' in the path
    output_path = buffer_path.replace('/sequence/', '/relabeled_actions/')
    
    return output_path


def create_model_output_path_from_config(config_path):
    """Create model output path based on config path."""
    # Extract the dataset name from the config path
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    # Create model path in the same directory as the dataset
    model_output_path = f"/home/wph52/weird/dynamics/datasets/pendulum/relabeled_actions/{config_name}/action_translator_model.pth"
    return model_output_path


def main():
    parser = argparse.ArgumentParser(description='Train action translator from action translation dataset')
    parser.add_argument('--dataset_config_path', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create paths based on config
    dataset_path = create_output_path_from_config(args.dataset_config_path)
    model_output_path = create_model_output_path_from_config(args.dataset_config_path)
    plot_output_path = os.path.join(os.path.dirname(model_output_path), 'action_distributions.png')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Model output path: {model_output_path}")
    print(f"Plot output path: {plot_output_path}")
    
    # Load action translation dataset
    states, original_actions, shifted_actions = load_action_translation_dataset(dataset_path)
    
    # Determine dimensions from data
    obs_dim = states.shape[1]
    action_dim = 1  # Assuming 1D action space
    
    # Train action translator
    model, _ = train_action_translator(
        states, original_actions, shifted_actions,
        obs_dim, action_dim, args.num_epochs, args.learning_rate, 
        args.batch_size, args.device
    )
    
    # Save trained model
    torch.save(model.state_dict(), model_output_path)
    print(f"Trained model saved to {model_output_path}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
