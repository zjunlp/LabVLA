from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class PickController(BaseController):
    """A state machine controller for picking up objects.

    Manages the process of picking an object through multiple phases:
    - Phase 0: Move end effector above the object.
    - Phase 1: Lower end effector closer to the object.
    - Phase 2: Position end effector for grasping.
    - Phase 3: Wait for robot dynamics to settle.
    - Phase 4: Close gripper to grasp the object.
    - Phase 5: Lift the object.
    - Phase 6: Complete the sequence.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that returns ArticulationAction.
        events_dt (List[float], optional): Duration for each phase. Defaults to [0.004, 0.002, 0.01, 0.2, 0.05, 0.004, 0.006].

    Raises:
        Exception: If events_dt is not a list or numpy array, or if its length is not 7.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0

        if events_dt is None:
            self._events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._cspace_controller = cspace_controller
        self._start = True
        self.object_size = None
        self._position_threshold = position_threshold
        self._robot_position = None

    def set_robot_position(self, position: np.ndarray):
        self._robot_position = position

    def _calculate_approach_direction(self, picking_position: np.ndarray) -> np.ndarray:
        if self._robot_position is None:
            return np.array([-1, 0, 0])  

        relative_pos = picking_position - self._robot_position
        
        horizontal_vec = relative_pos.copy()
        horizontal_vec[2] = 0

        if np.linalg.norm(horizontal_vec) > 0:
            horizontal_vec = -horizontal_vec / np.linalg.norm(horizontal_vec)
        else:
            horizontal_vec = np.array([-1, 0, 0])
        return horizontal_vec

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_name: str,
        object_size: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        pre_offset_z: float = 0.12,
        after_offset_z: float = 0.15,
        pre_offset_x: float = 0.1,
        gripper_distances: float = None
    ) -> ArticulationAction:
        """Computes the joint positions for the current picking phase.

        Args:
            picking_position (np.ndarray): Target position for picking.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            object_name (str): Name of the object to pick.
            object_size (np.ndarray): Size of the object.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current position of the gripper.
            end_effector_orientation (np.ndarray, optional): Target orientation for the end effector. Defaults to [0, pi, 0] Euler angles.

        Returns:
            ArticulationAction: Joint positions for the robot to execute.
        """
        self.object_size = object_size

        if self._start:
            return self._handle_start_state(current_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self.pre_offset_z = pre_offset_z
        self.after_offset_z = after_offset_z
        self.pre_offset_x = pre_offset_x
        
        target_joint_positions = self._execute_phase(
            picking_position,
            end_effector_orientation,
            current_joint_positions,
            object_name,
            gripper_control,
            gripper_position,
            gripper_distances
        )

        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
            
        return target_joint_positions

    def _handle_start_state(self, current_joint_positions):
        """Handles the initial state by opening the gripper.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.

        Returns:
            ArticulationAction: Joint positions with gripper opened.
        """
        self._start = False
        target_joint_positions = [None] * current_joint_positions.shape[0]
        target_joint_positions[7] = 0.04 / get_stage_units()
        target_joint_positions[8] = 0.04 / get_stage_units()
        return ArticulationAction(joint_positions=target_joint_positions)

    def _execute_phase(self, picking_position, end_effector_orientation, current_joint_positions, object_name, gripper_control, gripper_position, gripper_distances):
        """Executes the current phase of the picking sequence.

        Args:
            picking_position (np.ndarray): Target position for picking.
            end_effector_orientation (np.ndarray): Target orientation for end effector.
            current_joint_positions (np.ndarray): Current robot joint positions.
            object_name (str): Name of the target object.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current gripper position.

        Returns:
            ArticulationAction: Joint position targets for robot control.
        """
        
        approach_dir = self._calculate_approach_direction(picking_position)
        
        if self._event == 0:
            picking_position = picking_position + approach_dir * (self.pre_offset_x / get_stage_units())
            picking_position[2] += self.object_size[2] + self.pre_offset_z
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 1:
            picking_position = picking_position + approach_dir * (0.1 / get_stage_units())
            picking_position[2] += self.get_pickprez_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 2:
            picking_position[2] += self.get_pickz_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            z_distance = abs(gripper_position[2] - picking_position[2])
            if xy_distance < 0.005 and z_distance < 0.005:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 3:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            if gripper_distances is None:
                gripper_distances = self.get_gripper_distance(object_name) / get_stage_units()
            target_joint_positions[7] = gripper_distances
            target_joint_positions[8] = gripper_distances
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = picking_position
            self.target_position[2] += self.after_offset_z / get_stage_units()
            if "glass" in object_name:
                gripper_control.add_object_to_gripper("/World/glass_rod/Cylinder", "/World/Franka/panda_hand/tool_center")
            return target_joint_positions

        elif self._event == 5:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - self.target_position[:2])
            z_distance = abs(gripper_position[2] - self.target_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions
        else:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the controller to the initial phase.

        Args:
            events_dt (List[float], optional): New phase durations. Defaults to None.

        Raises:
            Exception: If events_dt is not a list or numpy array, or if its length is not 7.
        """
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0

        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._start = True
        self.object_size = None
        self._robot_position = None

    def is_done(self) -> bool:
        """Checks if the picking sequence is complete.

        Returns:
            bool: True if the final phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)

    def get_gripper_distance(self, item_name):
        """Determines the gripper opening distance for the specified object.

        Args:
            item_name (str): Name of the object to be gripped.

        Returns:
            float: Gripper finger distance in meters.
        """
        gripper_distances = {
            "rod": 0.003,
            "tube": 0.01,
            "beaker": 0.022,
            "beaker2": 0.028,
            "beaker_l": 0.03,
            "beaker_04": 0.025,
            "beaker_05": 0.025,
            "beaker_03": 0.025,
            "Erlenmeyer flask": 0.018,
            "Petri dish": 0.005,
            "pipette": 0.008,
            "microscope slide": 0.002,
            "conical_bottle01": 0.01,
            "conical_bottle02": 0.023,
            "conical_bottle03": 0.03,
            "conical_bottle04": 0.03,
            "graduated_cylinder_01": 0.005,
            "graduated_cylinder_02": 0.018,
            "graduated_cylinder_03": 0.024,
            "graduated_cylinder_04": 0.030,
        }

        for key in gripper_distances:
            if key == item_name.lower():
                return gripper_distances[key]

        return 0.02

    def get_pickz_offset(self, item_name):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.04,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker": 0.0,
            "beaker_04": 0.0,
            "beaker_05": 0.0,
            "beaker_03": 0.0,
            "beaker2": 0.0,
            "beaker_2": 0.0,
            "beaker_l": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "glass_rod": 0.02,
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5

    def get_pickprez_offset(self, item_name):
        """Calculates the vertical offset for the pre-grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "volume_flask": 0,
            "beaker2": 0.05,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "graduated_cylinder_01": 0.05,
            "graduated_cylinder_02": 0.03,
            "graduated_cylinder_03": 0.03,
            "graduated_cylinder_04": 0.03
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 3