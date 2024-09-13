import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    JointPosition,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.scene import Scene
from scipy.spatial.transform import Rotation as R


class UR5ActionMode(MoveArmThenGripper):
    def __init__(self):
        super(UR5ActionMode, self).__init__(
            JointPosition(absolute_mode=True), Discrete()
        )

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        # return np.array(6 * [-1] + [0.0]), np.array(6 * [1] + [1.0])
        return np.array(7 * [-1] + [0.0]), np.array(7 * [1] + [1.0])


class DeltaEndEffectorPose(MoveArmThenGripper):
    def __init__(self):
        super(DeltaEndEffectorPose, self).__init__(
            EndEffectorPoseViaPlanning(absolute_mode=False), Discrete()
        )

    def action(self, scene: Scene, action: np.ndarray):
        delta_rotation = R.from_euler("xyz", action[3:-1])
        rotation_quat = delta_rotation.as_quat()

        action = np.concatenate([action[:3], rotation_quat, action[-1:]], axis=-1)

        return super().action(scene, action)

    def action_bounds(self):
        """Returns the min and max of the action mode.
        TODO: These values are chosen arbitrary without any specific reason.
        Probably it should be investigated what are correct values."""
        return np.array(6 * [-1] + [0.0]), np.array(6 * [1] + [1.0])

    def action_shape(self, scene: Scene):
        return 7
