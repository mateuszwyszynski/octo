import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete


class UR5ActionMode(MoveArmThenGripper):
    def __init__(self):
        super(UR5ActionMode, self).__init__(JointPosition(absolute_mode = True), Discrete())

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(6 * [-1] + [0.0]), np.array(6 * [1] + [1.0])