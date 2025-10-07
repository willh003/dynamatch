import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import build_action_translator_from_config

# Removed unused import: from utils.model_utils import print_model_info


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
                          batch_size=64, device='cpu', val_split=0.2, model=None):
    """Train the Action Translator model."""
    print("=== Training Action Translator ===")
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    original_actions_tensor = torch.FloatTensor(original_actions)
    shifted_actions_tensor = torch.FloatTensor(shifted_actions)

    # Add action dimension if not present
    if len(original_actions_tensor.shape) == 1:
        original_actions_tensor = original_actions_tensor.unsqueeze(1)
    if len(shifted_actions_tensor.shape) == 1:
        shifted_actions_tensor = shifted_actions_tensor.unsqueeze(1)
    
    # Create dataset and split into train/val
    dataset = TensorDataset(states_tensor, original_actions_tensor, shifted_actions_tensor)
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    model.to(device)
    
    # Print model info
    print("\n--- Action Translator Model Info ---")
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
    train_losses = []
    val_losses = []
    
    # Create outer progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0, leave=True)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        # Create inner progress bar for training batches
        train_pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs}", 
                         position=1, leave=False, ncols=100)
        
        for batch_states, batch_original_actions, batch_shifted_actions in train_pbar:
            batch_states = batch_states.to(device)
            batch_original_actions = batch_original_actions.to(device)
            batch_shifted_actions = batch_shifted_actions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: predict shifted action given state and original action
            loss = model(obs=batch_states, 
                        action_prior=batch_original_actions,
                        action=batch_shifted_actions)
                         
                        
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            # Update training progress bar with current loss
            train_pbar.set_postfix({
                'batch_loss': f'{loss.item():.6f}',
                'avg_loss': f'{epoch_train_loss/num_train_batches:.6f}'
            })
        
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        
        # Create inner progress bar for validation batches
        val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}", 
                       position=1, leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_states, batch_original_actions, batch_shifted_actions in val_pbar:
                batch_states = batch_states.to(device)
                batch_original_actions = batch_original_actions.to(device)
                batch_shifted_actions = batch_shifted_actions.to(device)
                
                # Forward pass
                loss = model(batch_states, batch_original_actions, batch_shifted_actions)
                
                                
                epoch_val_loss += loss.item()
                num_val_batches += 1
                
                # Update validation progress bar with current loss
                val_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_val_loss/num_val_batches:.6f}'
                })
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Update epoch progress bar with epoch summary
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}'
        })
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Close progress bars
    epoch_pbar.close()
    
    print(f"Training completed. Final train loss: {train_losses[-1]:.6f}, Final val loss: {val_losses[-1]:.6f}")

    
    return model, train_losses, val_losses

def create_output_path_from_config(config_path):
    """Create output path by replacing 'sequence' with 'relabeled_actions' in the buffer path."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    buffer_path = config['buffer_path']
    # Replace 'sequence' with 'relabeled_actions' in the path
    output_path = buffer_path.replace('/sequence/', '/relabeled_actions/')
    
    return output_path

def create_model_path_from_data_path(model_config_path, data_path):
    """Create model output path based on config path."""
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    model_name = model_config['name']
    output_dir = os.path.dirname(data_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f'{model_name}_{timestamp}_translator.pth')
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Train action translator from action translation dataset')
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--model_config', type=str, default=None,
                       help='Path to model config YAML file (e.g., dynamics/configs/action_translator/ant_mlp.yaml)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--wandb', default='online',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Initialize wandb if not disabled
    # Extract config name for run name
    config_name = os.path.splitext(os.path.basename(args.dataset_config))[0]
    wandb.init(
        project="dynamics",
        entity="willhu003",
        name=f"action_translator_{config_name}",
        mode=args.wandb,
        config={
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "device": args.device,
            "val_split": args.val_split,
            "config_path": args.dataset_config,
        }
    )
    
    # Create paths based on config
    dataset_path = create_output_path_from_config(args.dataset_config)
    model_output_path = create_model_path_from_data_path(args.model_config, dataset_path)
    plot_output_path = os.path.join(os.path.dirname(model_output_path), 'action_distributions.png')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Model output path: {model_output_path}")
    print(f"Plot output path: {plot_output_path}")
    
    # Load action translation dataset
    states, original_actions, shifted_actions = load_action_translation_dataset(dataset_path)
    
    # Determine dimensions from data
    obs_dim = states.shape[1]
    action_dim = original_actions.shape[1]
    
    # Optionally build model from config for flexible architectures
    model_from_config = None
    if args.model_config is not None:
        print(f"Building action translator from model config: {args.model_config}")
        model_from_config = build_action_translator_from_config(args.model_config, obs_dim, action_dim, load_checkpoint=False)

    # Train action translator
    model, train_losses, val_losses = train_action_translator(
        states, original_actions, shifted_actions,
        obs_dim, action_dim, args.num_epochs, args.learning_rate, 
        args.batch_size, args.device, args.val_split, model=model_from_config
    )
    
    # Save trained model
    torch.save(model.state_dict(), model_output_path)
    print(f"Trained model saved to {model_output_path}")
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
