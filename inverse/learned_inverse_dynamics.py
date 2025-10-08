import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

from generative_policies.flow_model import ConditionalFlowModel
from generative_policies.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder
from generative_policies.prior import GaussianPrior

class MlpInverseDynamics(nn.Module):
    def __init__(self, action_dim, obs_dim, net_arch=[32, 32], device='cuda'):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.net_arch = net_arch
        self.device = device

        self.cond_encoder = IdentityTwoInputEncoder(input_1_dim=obs_dim, input_2_dim=obs_dim, device=device)
        self.action_encoder = IdentityEncoder(action_dim, device=device)

        in_feat_dim = self.cond_encoder.output_dim

        layers = []
        prev_dim = in_feat_dim
        for hidden_units in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_units))
            layers.append(nn.ReLU())
            prev_dim = hidden_units
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self.to(device)

    def forward(self, obs, next_obs, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For SimpleActionTranslator, this computes MSE loss between predicted and target actions.
        """ 
        # Predict translated action given observation and action_prior
        cond = self.cond_encoder(obs, next_obs)
        action = self.action_encoder(action)

        sample = self.network(cond)
        # Compute MSE loss between predicted and target action
        loss = torch.nn.functional.mse_loss(sample, action)
        return loss
    
    def predict(self, obs, next_obs):
        """Predict translated action given observation and original action."""
        with torch.no_grad():
            cond = self.cond_encoder(obs, next_obs)
            action = self.network(cond)

        # Remove batch dimension if needed
        if len(obs.shape) == 1:
            action = action.squeeze(dim=0)  # type: ignore[assignment]
        return action.cpu().numpy()

    def to(self, device):
        self.device = device
        self.network.to(device)
        return self

class FlowInverseDynamics(nn.Module):
    """
    Model p(a | s, s') by flowing from N(0,1) to a, conditioned on s,s'.
    """

    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super().__init__()
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.cond_encoder = IdentityTwoInputEncoder(obs_dim, obs_dim) # encode s,s'
        self.action_encoder = IdentityEncoder(action_dim)

        action_prior = GaussianPrior(action_dim)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.cond_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               source_sampler=action_prior
                                               )
        self.to(device)

    def forward(self, obs, next_obs, action):
        cond = self.cond_encoder(obs, next_obs)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, device=self.device)
        return loss

    def predict(self, obs, next_obs):
        cond = self.cond_encoder(obs, next_obs)
        batch_size = cond.shape[0]
        sample = self.flow_model.predict(
            batch_size=batch_size, 
            condition=cond,
            num_steps=self.num_inference_steps,
            device=self.device
            )

        return sample.cpu().numpy()

    def to(self, device):
        self.device = device
        self.flow_model.to(device)
        self.cond_encoder.to(device)
        self.action_encoder.to(device)
        return self