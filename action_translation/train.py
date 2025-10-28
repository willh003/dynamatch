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
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import build_action_translator_from_config
from utils.data_utils import get_relabeled_actions_path_from_config
from eval_policy_parallel import evaluate_policy_parallel
from utils.model_utils import load_action_translator_policy_from_config
from envs.register_envs import register_custom_envs
from utils.config_utils import load_yaml_config

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


def train_action_translator(states,model, original_actions, shifted_actions, 
                        num_epochs=100, learning_rate=1e-3, 
                          batch_size=64, device='cpu', val_split=0.2, weight_decay=0.0, 
                          model_output_path=None,dataset_config=None, translator_policy_config=None, id_config=None):
    """
    Train the Action Translator model.
    
    dataset_config and translator_policy_config are used for eval (not necessary to define)
    """
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    success_rate = 0.0
    best_success_rate = 0.0
    reward_iqm = 0.0
    save_best = False
    best_model_output_path = os.path.splitext(model_output_path)[0] + '_best' + os.path.splitext(model_output_path)[1]
    
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
        epoch_val_path_length = 0.0
        epoch_val_straight_path_length = 0.0
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
                path_length, straight_path_length, preds = model.compute_path_length(batch_states, batch_original_actions, batch_shifted_actions)
                loss = torch.nn.functional.mse_loss(torch.as_tensor(preds).to(device), batch_shifted_actions)

                mean_path_length = path_length.mean()
                mean_straight_length = straight_path_length.mean()
                                
                epoch_val_loss += loss.item()
                num_val_batches += 1

                epoch_val_path_length += mean_path_length.item()
                epoch_val_straight_path_length += mean_straight_length.item()
                
                # Update validation progress bar with current loss
                val_pbar.set_postfix({
                    'batch_loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_val_loss/num_val_batches:.6f}'
                })
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        avg_val_path_length = epoch_val_path_length / num_val_batches
        avg_val_straight_path_length = epoch_val_straight_path_length / num_val_batches
        
        # Save current model before eval on every epoch
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        torch.save(model.state_dict(), model_output_path)
            
        if translator_policy_config is not None and epoch % 5 == 0:
            # Save model if translator policy success rate improved
            # Only do this every few epochs to save time
            print("Evaluating translator policy...")
            translator_policy = load_action_translator_policy_from_config(
                config_path=translator_policy_config,
                action_translator_checkpoint=model_output_path
            )

            dataset_config_dict = load_yaml_config(dataset_config)
            env_id = dataset_config_dict['env_id']

            id_config_dict = load_yaml_config(id_config)
            state_indices = id_config_dict.get('state_indices', None)
            model_kwargs = {'state_indices': state_indices}

            success_rate, reward_iqm , _, _, _, _, _ = evaluate_policy_parallel(
                model=translator_policy,
                is_action_translator=True,
                env_id=env_id,
                n_eval_episodes=128,
                deterministic=True,
                n_envs=16,
                seed=None,
                model_kwargs=model_kwargs,
            )

            save_best = model_output_path is not None and success_rate > best_success_rate
        elif translator_policy_config is None:
            # If no eval in the loop, just save if validation loss improved
            save_best = model_output_path is not None and avg_val_loss < best_val_loss
        
        best_success_rate = max(best_success_rate, success_rate)
        best_val_loss = min(best_val_loss, avg_val_loss)

        if save_best:
            torch.save(model.state_dict(), best_model_output_path)
            print(f"\nModel saved to {best_model_output_path}, Validation loss: {best_val_loss:.6f}, Success rate: {best_success_rate:.4f}. ")

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Success Rate: {success_rate:.2f}, Reward IQM: {reward_iqm:.2f}")
            
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}',
                'best_val_loss': f'{best_val_loss:.6f}',
                'success_rate': success_rate,
                'reward_iqm': reward_iqm,
            })
        

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_path_length": avg_val_path_length,
            "val_straight_path_length": avg_val_straight_path_length,
            "val_path_length_diff": avg_val_straight_path_length - avg_val_path_length,
            "best_val_loss": best_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "eval/success_rate": success_rate,
            "eval/best_success_rate": best_success_rate,
            "eval/reward_iqm": reward_iqm,
        })
    
    # Close progress bars
    epoch_pbar.close()
    
    print(f"Training completed. Final train loss: {train_losses[-1]:.6f}, Final val loss: {val_losses[-1]:.6f}")

    
    return model, train_losses, val_losses


def create_model_path_from_data_path(model_config_path, data_path):
    """Create model output path based on config path."""
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    model_name = model_config['name']
    output_dir = os.path.dirname(data_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_name = f'translator_{model_name}_{timestamp}'
    model_path = os.path.join(output_dir, f'{path_name}.pth')
    return model_path, path_name

def main():
    parser = argparse.ArgumentParser(description='Train action translator from action translation dataset')
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to dataset config YAML file (e.g., pendulum_integrable_dynamics_shift.yaml)')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML file (e.g., dynamics/configs/action_translator/ant_mlp.yaml)')
    parser.add_argument('--translator_policy_config', type=str, default=None, required=False, help='Path to translator policy config YAML file, used for eval. Overrides the model config for eval.')
    parser.add_argument('--id_config', type=str, required=True,
                       help='Path to inverse dynamics model config YAML file')
    parser.add_argument('--num_epochs', type=int, default=1500,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--wandb', default='online',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    register_custom_envs()
    # Initialize wandb if not disabled
    # Extract config name for run name
    
    # Create paths based on config
    idm_config = load_yaml_config(args.id_config)
    idm_name = idm_config['name']
    dataset_path = get_relabeled_actions_path_from_config(args.dataset_config, idm_name)
    model_output_path, model_path_name = create_model_path_from_data_path(args.model_config, dataset_path)
    plot_output_path = os.path.join(os.path.dirname(model_output_path), 'action_distributions.png')

    
    
    print(f"Dataset path: {dataset_path}")
    print(f"Model output path: {model_output_path}")
    print(f"Plot output path: {plot_output_path}")
    
    # Load action translation dataset
    states, original_actions, shifted_actions = load_action_translation_dataset(dataset_path)
    
    # Optionally build model from config for flexible architectures
    print(f"Building action translator from model config: {args.model_config}")
    # Load the YAML config first
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config_dict = yaml.safe_load(f)
    model_from_config = build_action_translator_from_config(model_config_dict, load_checkpoint=False)

    train_config_dict = {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "device": args.device,
            "val_split": args.val_split,
            "dataset_config_path": args.dataset_config,
            "model_config_path": args.model_config,
            "id_config_path": args.id_config,
        }
    train_config_dict.update(model_config_dict)
    model_name = model_config_dict['name']
    wandb_tags = [model_name, 'translator']
    config_name = os.path.splitext(os.path.basename(args.dataset_config))[0]



    wandb.init(
        project="dynamics",
        entity="willhu003",
        name=model_path_name,
        mode=args.wandb,
        config=train_config_dict,
        tags=wandb_tags
    )

    weight_decay = float(train_config_dict.get('weight_decay', 0.0))
    
    # Train action translator
    model, train_losses, val_losses = train_action_translator(
        states, model_from_config, original_actions, shifted_actions,
        args.num_epochs, args.learning_rate, 
        args.batch_size, args.device, args.val_split, weight_decay, model_output_path,
        args.dataset_config, args.translator_policy_config, args.id_config
    )
    
    print(f"Training completed. Best model was saved during training to {model_output_path}")
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
