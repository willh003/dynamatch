import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

class ActionTranslatorSB3Policy:
    def __init__(self, base_policy, action_translator):
        self.base_policy = base_policy
        self.action_translator = action_translator
        self.translator = action_translator

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        """
        Predict actions, following StableBaselines PolicyPredictor interface.
        """ 
        translated_action, base_action = self.predict_base_and_translated(observation, state, episode_start, deterministic)
        return translated_action, state

    def predict_base_and_translated(        
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        ):
        """
        Predict a base action and then translate it, returning both
        """
        base_prediction = self.base_policy.predict(observation, state, episode_start, deterministic)
        # Extract just the action from the base policy prediction (which is a tuple of (action, state))
        base_action = base_prediction[0] if isinstance(base_prediction, tuple) else base_prediction
        translated_action = self.action_translator.predict(observation, base_action)
        return translated_action, base_action

class SimpleActionTranslator(nn.Module):
    def __init__(self, action_dim, obs_dim):
        super(SimpleActionTranslator, self).__init__()
        in_feat_dim = action_dim + obs_dim
        self.network = nn.Sequential(
            nn.Linear(in_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))
    

    def is_vectorized_observation(self, obs):
        return len(obs.shape) > 1
    
    def predict(self, obs, action):
        """Predict translated action given observation and original action."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) if len(obs.shape) == 1 else torch.FloatTensor(obs)
            action_tensor = torch.FloatTensor(action).unsqueeze(0) if len(action.shape) == 1 else torch.FloatTensor(action)
            translated_action = self.forward(obs_tensor, action_tensor)
        
        # Remove batch dimension if needed
        if not self.is_vectorized_observation(obs):
            translated_action = translated_action.squeeze(dim=0)  # type: ignore[assignment]
        
        
        return translated_action.cpu().numpy()