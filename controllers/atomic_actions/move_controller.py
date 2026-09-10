from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class MoveController(BaseController):
    """A simple controller for moving the robot to a target position and orientation.

    This controller moves the robot's end effector to a specified position and orientation
    using a Cartesian space controller.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that returns ArticulationAction.
        position_threshold (float, optional): Position threshold for considering the target reached. Defaults to 0.01.
        orientation_threshold (float, optional): Orientation threshold for considering the target reached. Defaults to 0.1.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        position_threshold: float = 0.02,
        orientation_threshold: float = 0.1,
    ) -> None:
        super().__init__(name=name)
        self._cspace_controller = cspace_controller
        self._position_threshold = position_threshold
        self._orientation_threshold = orientation_threshold
        self._target_position = None
        self._target_orientation = None
        self._is_done = False
        self._waypoints = []
        self._current_waypoint_index = 0
        self._multi_segment_mode = False

    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        target_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Computes the joint positions to move to the target position and orientation.

        Args:
            target_position (np.ndarray): Target position for the end effector.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            gripper_position (np.ndarray): Current position of the gripper.
            gripper_orientation (np.ndarray): Current orientation of the gripper.
            target_orientation (np.ndarray, optional): Target orientation for the end effector. 
                Defaults to [0, pi, 0] Euler angles.

        Returns:
            ArticulationAction: Joint positions for the robot to execute.
        """
        if target_orientation is None:
            target_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._target_position = target_position
        self._target_orientation = target_orientation

        target_joint_positions = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation
        )

        position_distance = np.linalg.norm(gripper_position - target_position)
        
        if position_distance < self._position_threshold:
            self._is_done = True
        else:
            self._is_done = False

        return target_joint_positions

    def forward_multi_segment(
        self,
        waypoints: typing.List[np.ndarray],
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        target_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Multi-segment movement: move to multiple path points in sequence.

        Args:
            waypoints (List[np.ndarray]): Path point list, executed in sequence
            current_joint_positions (np.ndarray): Current joint positions
            gripper_position (np.ndarray): Current gripper position
            target_orientation (np.ndarray, optional): Target orientation

        Returns:
            ArticulationAction: Joint position action
        """
        if target_orientation is None:
            target_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        if not self._multi_segment_mode:
            self._waypoints = waypoints.copy()
            self._current_waypoint_index = 0
            self._multi_segment_mode = True
            self._is_done = False

        if self._current_waypoint_index >= len(self._waypoints):
            self._is_done = True
            return self._cspace_controller.forward(
                target_end_effector_position=self._waypoints[-1],
                target_end_effector_orientation=target_orientation
            )
        current_target = self._waypoints[self._current_waypoint_index]
        self._target_position = current_target
        self._target_orientation = target_orientation
        
        position_distance = np.linalg.norm(gripper_position - current_target)
        print(position_distance, gripper_position, current_target)
        if position_distance < 0.08:
            self._current_waypoint_index += 1
            if self._current_waypoint_index < len(self._waypoints):
                current_target = self._waypoints[self._current_waypoint_index]
                self._target_position = current_target

        target_joint_positions = self._cspace_controller.forward(
            target_end_effector_position=current_target,
            target_end_effector_orientation=target_orientation
        )

        if self._current_waypoint_index >= len(self._waypoints):
            self._is_done = True
        
        return target_joint_positions

    def forward_two_points(
        self,
        first_position: np.ndarray,
        final_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        target_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """A convenient method for vertical movement followed by horizontal movement.

        Args:
            final_position (np.ndarray): Final target position
            current_joint_positions (np.ndarray): Current joint positions
            gripper_position (np.ndarray): Current gripper position
            target_orientation (np.ndarray, optional): Target orientation
            vertical_offset (float): Vertical offset, default 0.2 meters

        Returns:
            ArticulationAction: Joint position action
        """
        waypoint1 = first_position.copy()
        
        waypoint2 = final_position.copy()
        
        waypoints = [waypoint1, waypoint2]
        
        return self.forward_multi_segment(
            waypoints=waypoints,
            current_joint_positions=current_joint_positions,
            gripper_position=gripper_position,
            target_orientation=target_orientation
        )

    def reset(self) -> None:
        """Reset controller state.

        Args:
            None

        Returns:
            None
        """
        super().reset()
        self._cspace_controller.reset()
        self._target_position = None
        self._target_orientation = None
        self._is_done = False
        # Reset multi-segment movement state
        self._waypoints = []
        self._current_waypoint_index = 0
        self._multi_segment_mode = False

    def is_done(self) -> bool:
        """Check if the target position and angle have been reached.

        Returns:
            bool: True if the target has been reached, False otherwise.
        """
        return self._is_done

    def get_target_position(self) -> typing.Optional[np.ndarray]:
        """Get current target position.

        Returns:
            np.ndarray or None: Current target position, None if not set.
        """
        return self._target_position

    def get_target_orientation(self) -> typing.Optional[np.ndarray]:
        """Get current target orientation.

        Returns:
            np.ndarray or None: Current target orientation, None if not set.
        """
        return self._target_orientation

    def set_position_threshold(self, threshold: float) -> None:
        """Set position threshold.

        Args:
            threshold (float): New position threshold.

        Returns:
            None
        """
        self._position_threshold = threshold

    def set_orientation_threshold(self, threshold: float) -> None:
        """Set orientation threshold.

        Args:
            threshold (float): New orientation threshold.

        Returns:
            None
        """
        self._orientation_threshold = threshold