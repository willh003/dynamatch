import torch.nn as nn
import torch
from generative_policies.flow_model import ConditionalFlowModel
from generative_policies.unet import UnetNoisePredictionNet
from generative_policies.obs_encoder import IdentityObservationEncoder
from generative_policies.prior import GaussianPrior

class FlowActionTranslator(ActionTranslatorInterface):

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        global_cond = self.obs_encoder(obs)
        return self.flow_model(global_cond, action_prior, action)


class FlowInverseDynamics(nn.Module):

    def __init__(self,
                action_dim, 
                obs_dim,
                num_train_steps=100, 
                num_inference_steps=10, 
                timeshift=1.0):
        self.action_dim = action_dim
        self.obs_encoder = IdentityObservationEncoder(obs_dim)
        global_cond_dim = obs_dim + obs_dim # obs + next_obs
        noise_pred_net = UnetNoisePredictionNet(action_dim, global_cond_dim=global_cond_dim)

        self.action_prior = GaussianPrior(action_dim)
        self.flow_model = ConditionalFlowModel(noise_pred_net, num_train_steps, num_inference_steps, timeshift)

    def sample(self, obs, next_obs):
        obs_cond = self.obs_encoder(obs)
        next_obs_cond = self.obs_encoder(next_obs)
        global_cond = torch.cat([obs_cond, next_obs_cond], dim=-1)
        prior_sample = self.action_prior.sample(n_samples=obs.shape[0], device=obs.device)
        return self.flow_model.sample(global_cond=global_cond, prior_sample=prior_sample)

    def forward(self, obs, next_obs, action):
        obs_cond = self.obs_encoder(obs)
        next_obs_cond = self.obs_encoder(next_obs)
        global_cond = torch.cat([obs_cond, next_obs_cond], dim=-1)
        prior_sample = self.action_prior.sample(n_samples=obs.shape[0], device=obs.device)
        return self.flow_model(global_cond=global_cond, prior_sample=prior_sample, action=action)