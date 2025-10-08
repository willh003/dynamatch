import mujoco 
import torch
import torch.nn as nn
import gymnasium as gym

class IntegrableEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.unwrapped.frame_skip = 1
        self.unwrapped.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

class ActionAddWrapper(gym.Wrapper):
    def __init__(self, env, action_add=0):
        """
        Wrapper for the pendulum environment to shift the action space.
        Args:
            env: The environment to wrap.
            action_add: The amount to add to the actions.
        """
        super().__init__(env)
        self.action_add = action_add

    def step(self, action):
        print(f"Action: {action}")
        action = action + self.action_add
        print(f"Modified Action: {action}")
        return super().step(action)

class ActionNonlinearWrapper(gym.Wrapper):
    def __init__(self, env, action_nonlinear_transformation=lambda x: x):
        super().__init__(env)
        self.action_nonlinear_transformation = action_nonlinear_transformation
    
    def step(self, action):
        return super().step(self.action_nonlinear_transformation(action))

class ModifyFrictionWrapper(gym.Wrapper):
    def __init__(self, env, friction_coeffs=(1.0, 0.5, 0.5)):
        super().__init__(env)

        for i in range(self.unwrapped.model.ngeom):
            self.unwrapped.model.geom_friction[i] = list(friction_coeffs)


class ActionTransformMLP(nn.Module):
    def __init__(self, action_dim=1, state_dim=None, hidden_dim=32):
        super().__init__()
        self.state_conditional = state_dim is not None
        if self.state_conditional:
            in_dim = state_dim + action_dim
        else:
            in_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights with larger variance and add some bias
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with larger variance to create more interesting transformations."""
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                # Use different initialization strategies for different layers
                if i == 0:  # First layer
                    nn.init.xavier_uniform_(layer.weight, gain=3.0)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -1.0, 1.0)
                elif i == len([l for l in self.net if isinstance(l, nn.Linear)]) - 1:  # Last layer
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -0.2, 0.2)
                else:  # Hidden layers
                    nn.init.xavier_uniform_(layer.weight, gain=2.5)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -0.3, 0.3)
        
    def forward(self, a_source):
        return self.net(a_source)


class MLPActionWrapper(gym.Wrapper):
    def __init__(self, env, state_conditional=False, checkpoint_path=None, save_weights_path=None):
        super().__init__(env)
        assert checkpoint_path is None or save_weights_path is None, "ERROR: cannot provide both checkpoint_path and save_weights_path"
        
        self.action_transform = ActionTransformMLP(
            action_dim=env.action_space.shape[0], 
            state_dim=env.observation_space.shape[0] if state_conditional else None
            )
            
        if checkpoint_path is not None:
            self.action_transform.load_state_dict(torch.load(checkpoint_path))
        
        # Save the initialized weights if save_weights_path is provided
        if save_weights_path is not None:
            import os
            os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
            torch.save(self.action_transform.state_dict(), save_weights_path)
            print(f"Saved MLP weights to: {save_weights_path}")
    
    def step(self, action, debug=False):

        action_torch = torch.as_tensor(action, dtype=torch.float32)
        with torch.no_grad():   
            action_transformed = self.action_transform(action_torch)
        action_transformed_numpy = action_transformed.detach().cpu().numpy()

        if debug:
            print(f"Action: {action}, Transformed: {action_numpy}")

        return super().step(action_transformed_numpy)