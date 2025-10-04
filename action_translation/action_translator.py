import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

from generative_policies.flow_policy import FlowPolicy
from generative_policies.unet import UnetNoisePredictionNet
from generative_policies.obs_encoder import IdentityObservationEncoder

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
        deterministic: bool = False,
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
    
    def sample(self, obs, action_prior):
        """
        Sample an action given the observation and the action_prior
        """
        raise NotImplementedError("Subclasses must implement sample method")

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
class SimpleActionTranslator(ActionTranslatorInterface):
    def __init__(self, action_dim, obs_dim, net_arch=None):
        super(SimpleActionTranslator, self).__init__()
        self.action_dim = action_dim
        
        self.obs_encoder = IdentityObservationEncoder(obs_dim)
        in_feat_dim = action_dim + self.obs_encoder.output_dim
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

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For SimpleActionTranslator, this computes MSE loss between predicted and target actions.
        """ 
        # Predict translated action given observation and action_prior
        encoded_obs = self.obs_encoder(obs)
        predicted_action = self.network(torch.cat([encoded_obs, action_prior], dim=-1))
        # Compute MSE loss between predicted and target action
        loss = torch.nn.functional.mse_loss(predicted_action, action)
        return loss

    def sample(self, obs, action_prior):
        """
        Sample an action given the observation and the action_prior.
        For SimpleActionTranslator, this is deterministic prediction.
        """
        with torch.no_grad():

            # Predict translated action
            encoded_obs = self.obs_encoder(obs)
            translated_action = self.network(torch.cat([encoded_obs, action_prior], dim=-1))
            
            return translated_action

    def is_vectorized_observation(self, obs):
        return len(obs.shape) > 1
    
    def predict(self, obs, action):
        """Predict translated action given observation and original action."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) if len(obs.shape) == 1 else torch.FloatTensor(obs)
            action_tensor = torch.FloatTensor(action).unsqueeze(0) if len(action.shape) == 1 else torch.FloatTensor(action)

            translated_action = self.network(torch.cat([obs_tensor, action_tensor], dim=-1))
        
        # Remove batch dimension if needed
        if not self.is_vectorized_observation(obs):
            translated_action = translated_action.squeeze(dim=0)  # type: ignore[assignment]
        
        
        return translated_action.cpu().numpy()



class FlowActionTranslator(ActionTranslatorInterface):
    def __init__(self, action_dim, obs_dim):
        super(FlowActionTranslator, self).__init__()
        self.action_dim = action_dim
        obs_encoder = IdentityObservationEncoder(obs_dim)
        noise_pred_net = UnetNoisePredictionNet(action_dim, global_cond_dim=obs_encoder.output_dim)
        
        self.flow_policy = FlowPolicy(obs_encoder, noise_pred_net)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionTranslator, this would need to be implemented based on the flow policy's loss function.
        """
        # This is a placeholder implementation - would need to be implemented based on flow policy
        raise NotImplementedError("FlowActionTranslator forward method needs to be implemented based on flow policy loss")

    def sample(self, obs, action_prior):
        return self.flow_policy.sample(obs=obs, prior_sample=action_prior)