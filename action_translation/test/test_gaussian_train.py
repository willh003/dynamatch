import sys
import os
import numpy as np
import wandb
import torch
import yaml
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from action_translation.train import train_action_translator
from utils.model_utils import build_action_translator_from_config
from generative_policies.flow_model import ConditionalFlowModel 



def test_gaussian(model_config, test_name):

    print(f"Building action translator from model config: {model_config}")

    with open(model_config, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f) or {}

    model_from_config = build_action_translator_from_config(cfg_dict)

    num_samples = 1000
    num_inference_steps = cfg_dict['num_inference_steps']

    states = np.ones((num_samples, 2))
    original_actions = np.random.randn(num_samples, 2)
    shifted_actions_mode1 = np.random.randn(num_samples // 2, 2) - 3
    shifted_actions_mode2 = np.random.randn(num_samples - num_samples // 2, 2) + 3
    shifted_actions = np.concatenate([shifted_actions_mode1, shifted_actions_mode2], axis=0)
    
    
    obs_dim = states.shape[-1]
    action_dim = original_actions.shape[-1]
    num_epochs = 100
    learning_rate = 1e-3
    batch_size = 16
    device = 'cuda'
    val_split = 0.2

    wandb.init(
        project="dynamics",
        entity="willhu003",
        name=f"dummy",
        mode='disabled'
    )


    # Train action translator
    model, train_losses, val_losses = train_action_translator(
        states, original_actions, shifted_actions,
        obs_dim, action_dim, num_epochs, learning_rate, 
        batch_size, device, val_split, model=model_from_config
    )

    num_eval_samples = 1000 
    states_eval = np.ones((num_eval_samples, 2))
    original_actions_eval = np.random.randn(num_eval_samples, 2)
    shifted_actions_mode1 = np.random.randn(num_eval_samples // 2, 2) - 3
    shifted_actions_mode2 = np.random.randn(num_eval_samples - num_eval_samples // 2, 2) + 3
    shifted_actions_eval = np.concatenate([shifted_actions_mode1, shifted_actions_mode2], axis=0)
    
    model.eval()
    with torch.inference_mode():
        
        translated_actions = model.predict(states_eval, original_actions_eval, num_steps=num_inference_steps)
    
    states_to_plot = states_eval.squeeze()  
    original_actions_to_plot = original_actions_eval.squeeze()
    shifted_actions_to_plot = shifted_actions_eval.squeeze()
    translated_actions_to_plot = translated_actions.squeeze()
    
    plt.figure(figsize=(10, 8))
    plt.title(f'{test_name} Action Translator Evaluation - 2D Plot')
    # Plot original actions (prior) in green
    plt.scatter(original_actions_to_plot[:, 0], original_actions_to_plot[:, 1], 
                label='Original actions (Prior)', alpha=0.6, s=20, c='green')
    # Plot shifted actions in blue
    plt.scatter(shifted_actions_to_plot[:, 0], shifted_actions_to_plot[:, 1], 
                label='Ground Truth Shifted actions', alpha=0.6, s=20, c='blue')
    # Plot translated actions in red
    plt.scatter(translated_actions_to_plot[:, 0], translated_actions_to_plot[:, 1], 
                label='Translated actions', alpha=0.6, s=20, c='red')
    plt.xlabel('Action Dimension 0')
    plt.ylabel('Action Dimension 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'test_media/{test_name}_2d_translated_actions.png')
    plt.close()

    plt.title(f'{test_name} Action Translator Loss')
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    
    plt.savefig(f'test_media/{test_name}_2d_loss.png')
    plt.close()


def test_flow_action_translator():
    model_config = '/home/wph52/weird/dynamics/configs/action_translator/2d_flow_act_cond.yaml'
    test_gaussian(model_config, test_name='flow')
    #test_action_translator_1d(model_config, test_name='flow', use_prior=False)
if __name__ == "__main__":
    test_flow_action_translator()
    
    
    
