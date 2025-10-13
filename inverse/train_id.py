import argparse
import os
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import zarr
from tqdm import tqdm
import wandb
import sys
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import load_inverse_dynamics_model_from_config
from utils.data_utils import load_transition_dataset, get_transition_path_from_dataset_config

# For physics ID sanity check
import gymnasium as gym
from inverse.physics_inverse_dynamics import gym_inverse_dynamics
from envs.register_envs import register_custom_envs


def quick_validate_id(dataset, env, max_samples=2000):
    """Quickly validate the ID model on a dataset."""
    print("=== Quick Validating Physics ID Model ===")
    errors = []
    for i in tqdm(range(min(max_samples, len(dataset)))):
        state = dataset[i][0]
        action = dataset[i][1]
        next_state = dataset[i][2]
        error = np.linalg.norm(gym_inverse_dynamics(env, state.cpu().numpy(), next_state.cpu().numpy()) - action.cpu().numpy())
        
        errors.append(error)

    print(f"Mean Physics ID error: {np.mean(errors):.3e}, Std Physics ID error: {np.std(errors):.3e}, Max ID error: {np.max(errors):.3e}, Min ID error: {np.min(errors):.3e}")


def train_inverse_dynamics(states, actions, next_states, model, 
                          num_epochs=100, learning_rate=1e-3, 
                          batch_size=64, device='cpu', val_split=0.2, env_id=None, 
                          model_output_path=None, validate_physics_id=False):
    """Train the Inverse Dynamics model."""
    print("=== Training Inverse Dynamics ===")

    
    # Use the same environment that was used for data collection
    if env_id is None:
        env_id = "InvertedPendulumIntegrable-v5"  # fallback for backward compatibility
    physics_env = gym.make(env_id)

    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    next_states_tensor = torch.FloatTensor(next_states)

    # Add action dimension if not present
    if len(actions_tensor.shape) == 1:
        actions_tensor = actions_tensor.unsqueeze(1)
    
    # Create dataset and split into train/val
    dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor)

    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    if validate_physics_id:   
        quick_validate_id(train_dataset, physics_env)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    model.to(device)
    
    # Print model info
    print("\n--- Inverse Dynamics Model Info ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    
    # Detailed breakdown
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                print(f"{name}: {param_count:,} parameters")
    print("=" * 40)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    # Checkpoint tracking
    best_val_loss = float('inf')
    
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
        
        for batch_states, batch_actions, batch_next_states in train_pbar:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_next_states = batch_next_states.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: compute loss for inverse dynamics p(a | s, s')
            loss = model(obs=batch_states, 
                        next_obs=batch_next_states,
                        action=batch_actions)
                         
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
        epoch_val_physics_id_loss = 0.0
        
        # Create inner progress bar for validation batches
        val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}", 
                       position=1, leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states in val_pbar:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_next_states = batch_next_states.to(device)
                
                # Forward pass
                #loss = model(batch_states, batch_next_states, batch_actions)

                preds = model.predict(batch_states, batch_next_states)
                loss = torch.nn.functional.mse_loss(torch.as_tensor(preds).to(device), batch_actions)

                if validate_physics_id:
                    physics_id_actions = [gym_inverse_dynamics(physics_env, state.cpu().numpy(), next_state.cpu().numpy()) for state, next_state in zip(batch_states, batch_next_states)]

                    physics_id_actions = torch.as_tensor(physics_id_actions).to(device)
                    physics_id_error = torch.norm(physics_id_actions - batch_actions, dim=1)
                    physics_id_error = physics_id_error.mean()
                    print(f"Phys ID loss: {physics_id_error.item():.3e}")
                    epoch_val_physics_id_loss += physics_id_error.item()

                epoch_val_loss += loss.item()
                num_val_batches += 1.
                
                # Update validation progress bar with current loss
                val_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_val_loss/num_val_batches:.6f}'
                })
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        avg_val_physics_id_error = epoch_val_physics_id_loss / num_val_batches

        # Save checkpoint if validation loss improved
        if model_output_path is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_output_path)
            print(f"\nNew best validation loss: {avg_val_loss:.6f}. Checkpoint saved to {model_output_path}")

        # Update epoch progress bar with epoch summary
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}',
            'best_val_loss': f'{best_val_loss:.6f}'
        })
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_physics_id_error": avg_val_physics_id_error,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Close progress bars
    epoch_pbar.close()
    
    print(f"Training completed. Final train loss: {train_losses[-1]:.6f}, Final val loss: {val_losses[-1]:.6f}")

    return model, train_losses, val_losses

def get_env_id_from_config(config_path):
    """Extract environment ID from dataset config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('env_id', 'InvertedPendulumIntegrable-v5')


def create_model_path_from_data_path(model_config_path, data_path):
    """Create model output path based on config path."""
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    model_name = model_config['name']
    output_dir = os.path.dirname(data_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_name = f'id_{model_name}_{timestamp}'
    model_path = os.path.join(output_dir, f'{path_name}.pth')
    return model_path, path_name


def main():
    parser = argparse.ArgumentParser(description='Train inverse dynamics model from inverse dynamics dataset')
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML file (e.g., dynamics/configs/inverse_dynamics/ant_flow.yaml)')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--validate_physics_id', type=bool, default=False, help="Whether to validate physics ID (default: False)")
    parser.add_argument('--wandb', default='online',
                       help='Disable wandb logging')

    register_custom_envs()
    
    args = parser.parse_args()
    
    # Create paths based on config
    dataset_path = get_transition_path_from_dataset_config(args.dataset_config)
    model_output_path, model_path_name = create_model_path_from_data_path(args.model_config, dataset_path)
    plot_output_path = os.path.join(os.path.dirname(model_output_path), 'training_curves.png')
    
    # Get environment ID from dataset config
    env_id = get_env_id_from_config(args.dataset_config)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Model output path: {model_output_path}")
    print(f"Plot output path: {plot_output_path}")
    print(f"Environment ID: {env_id}")
    
    # Load inverse dynamics dataset
    states, actions, next_states = load_transition_dataset(dataset_path)
    
    # Build model from config
    print(f"Building inverse dynamics model from model config: {args.model_config}")
    model_from_config = load_inverse_dynamics_model_from_config(args.model_config, load_checkpoint=False)

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config_dict = yaml.safe_load(f)
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    train_config_dict = {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "device": args.device,
            "val_split": args.val_split,
            "config_path": args.dataset_config,
        }
    train_config_dict.update(model_config_dict)
    model_name = model_config_dict['name']
    dataset_name = dataset_config['name']
    wandb_tags = [model_name, dataset_name, 'id']
    config_name = os.path.splitext(os.path.basename(args.dataset_config))[0]
    wandb.init(
        project="dynamics",
        entity="willhu003",
        name=model_path_name,
        mode=args.wandb,
        config=train_config_dict,
        tags=wandb_tags
    )
    
    # Train inverse dynamics model
    model, train_losses, val_losses = train_inverse_dynamics(
        states, actions, next_states, model_from_config,
        args.num_epochs, args.learning_rate, 
        args.batch_size, args.device, args.val_split, env_id, model_output_path, args.validate_physics_id
    )
    
    # Save final model (overwrites the best checkpoint with the final model)
    print(f"Final training loss: {train_losses[-1]:.6f}, Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best val loss checkpoint saved to {model_output_path}")
    # Finish wandb run
    wandb.finish()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
