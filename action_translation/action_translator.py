import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

from generative_policies.flow_model import ConditionalFlowModel
from generative_policies.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder
from generative_policies.prior import GaussianPrior

class ActionTranslatorSB3Policy:
    def __init__(self, source_policy, action_translator):
        self.source_policy = source_policy
        self.action_translator = action_translator
        self.translator = action_translator

    def predict_base_and_translated(        
        self,
        policy_observation: Union[np.ndarray, dict[str, np.ndarray]],
        translator_observation: Optional[dict[str, np.ndarray]] = None,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        ):
        """
        Predict a base action and then translate it, returning both
        """
        base_prediction = self.source_policy.predict(policy_observation, state, episode_start, deterministic)
        # Extract just the action from the base policy prediction (which is a tuple of (action, state))
        base_action = base_prediction[0] if isinstance(base_prediction, tuple) else base_prediction
        translated_action = self.action_translator.predict(translator_observation, base_action)
        return translated_action, base_action


class ActionTranslatorInterface(nn.Module):
    def __init__(self):
        super(ActionTranslatorInterface, self).__init__()
    
    def predict(self, obs, action_prior):
        """
        Predict an action given the observation and the action_prior
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def to(self, device):
        """
        Move the model to the specified device
        """
        raise NotImplementedError("Subclasses must implement to method")


class SimpleActionTranslator(ActionTranslatorInterface):
    def __init__(self, action_dim, obs_dim, device='cuda', net_arch=None):
        super(SimpleActionTranslator, self).__init__()
        self.action_dim = action_dim
        self.cond_encoder = IdentityTwoInputEncoder(input_1_dim=obs_dim, input_2_dim=action_dim, device=device)

        in_feat_dim = self.cond_encoder.output_dim
        # Default to two hidden layers of 32 units each for backward compatibility
        if net_arch is None:
            net_arch = [32, 32]

        layers = []
        prev_dim = in_feat_dim
        for hidden_units in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_units))
            layers.append(nn.ReLU())
            prev_dim = hidden_units
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self.to(device)

    def to(self, device):
        self.device = device
        self.network.to(device)
        return self

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For SimpleActionTranslator, this computes MSE loss between predicted and target actions.
        """ 
        # Predict translated action given observation and action_prior
        cond = self.cond_encoder(obs, action_prior)

        predicted_action = self.network(cond)
        # Compute MSE loss between predicted and target action
        if len(action.shape) == 3:
            action = action.squeeze(dim=1)
        loss = torch.nn.functional.mse_loss(predicted_action, action)
        return loss
        
    def is_vectorized_observation(self, obs):
        return len(obs.shape) > 1
    
    def predict(self, obs, action_prior):
        """Predict translated action given observation and original action."""
        with torch.no_grad():
            cond = self.cond_encoder(obs, action_prior)
            translated_action = self.network(cond)

        # Remove batch dimension if needed
        if not self.is_vectorized_observation(obs):
            translated_action = translated_action.squeeze(dim=0)  # type: ignore[assignment]
        return translated_action.cpu().numpy()



class FlowActionPriorTranslator(ActionTranslatorInterface):
    """
    Action translator that learns p(a_trg | o), using a_src as prior.
    Does NOT condition explicitly on a_src
    """
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super(FlowActionPriorTranslator, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_encoder = IdentityEncoder(obs_dim, device=device)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.obs_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               )
        self.num_inference_steps = num_inference_steps

        self.to(device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionPriorTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        cond = self.obs_encoder(obs)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, prior_samples=action_prior, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        cond = self.obs_encoder(obs)
        action_prior = self.action_encoder(action_prior)
        batch_size = obs.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         prior_samples=action_prior,
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()

    def to(self, device):
        self.device = device
        self.flow_model.to(device)
        self.obs_encoder.to(device)
        return self


class FlowActionConditionedTranslator(ActionTranslatorInterface):
    """
    Action translator that learns p(a_trg | o, a_src), conditioned explicitly on a_src
    Uses N(0,1) as prior
    """
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super(FlowActionConditionedTranslator, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_inference_steps = num_inference_steps

        self.cond_encoder = IdentityTwoInputEncoder(obs_dim, action_dim, device=device)
        self.action_encoder = IdentityEncoder(action_dim, device=device)
        
        action_prior = GaussianPrior(action_dim)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.cond_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               source_sampler=action_prior,
                                               )

        self.to(device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionPriorTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        cond = self.cond_encoder(obs, action_prior)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        cond = self.cond_encoder(obs, action_prior)
        batch_size = cond.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()

    def to(self, device):
        self.device = device
        self.flow_model.to(device)
        self.cond_encoder.to(device)
        self.action_encoder.to(device)
        return self

    
class FlowActionPriorConditionedTranslator(FlowActionConditionedTranslator):
    """
    Acion translator that learns p(a_trg | o, a_src), conditioned explicitly and implicitly on a_src
    Uses a_src as the flow prior instead of N(0,1), and unet on a_src
    So basically this is FlowActionConditionedTranslator and FlowActionPriorTranslator combined
    """
    
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super().__init__(action_dim, obs_dim, diffusion_step_embed_dim, down_dims, num_inference_steps, device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionPriorConditionedTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, prior_samples=action_prior, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        batch_size = cond.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         prior_samples=action_prior,
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()