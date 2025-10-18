from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.panda_gripper import PandaGripperBase
import numpy as np

@register_gripper
class PandaControllableGripper(PandaGripperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_action(self, action):
        """
        Apply even action to both fingers (but in opposite directions)
        """
        
        self.current_action = np.array([-1.0, 1.0]) * action
        return self.current_action

    @property
    def dof(self):
        return 1