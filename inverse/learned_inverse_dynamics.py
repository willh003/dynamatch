import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

from generative_policies.flow_model import ConditionalFlowModel
from generative_policies.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder
from generative_policies.prior import GaussianPrior

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