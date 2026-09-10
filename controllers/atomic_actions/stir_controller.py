from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class StirController(BaseController):
    """
    A position-based controller for performing stirring actions with time-based fallback.

    Uses position thresholds for responsive transitions with time-based timeout as backup:
    - Phase 0: Lift the glass rod
    - Phase 1: Move above the beaker
    - Phase 2: Lower into the beaker
    - Phase 3: Perform stirring motion
    - Phase 4: Lift out of beaker

    Args:
        name (str): Controller identifier
        cspace_controller (BaseController): Cartesian space controller
        events_dt (List[float], optional): Duration for each phase as backup
        position_threshold (float): Distance threshold for phase transitions
        stir_radius (float): Radius of stirring motion in meters
        stir_speed (float): Angular velocity for stirring
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.005,
        stir_radius: float = 0.009,
        stir_speed: float = 3.0,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0
        
        if events_dt is None:
            self._events_dt = [0.004, 0.004, 0.005, 0.001, 0.004]  # 5 phases
        else:
            if not isinstance(events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            else:
                self._events_dt = events_dt
            if len(self._events_dt) != 5:
                raise Exception(f"events_dt length must be 5, got {len(self._events_dt)}")
        
        self._cspace_controller = cspace_controller
        self._position_threshold = position_threshold
        self._stir_radius = stir_radius / get_stage_units()
        self._stir_speed = stir_speed
        self._start = True
        self._current_stir_angle = 0.0

    def forward(
        self,
        center_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute current phase with position threshold and time-based backup.

        Args:
            center_position (np.ndarray): Reference position for stirring
            current_joint_positions (np.ndarray): Current robot joint positions
            gripper_position (np.ndarray): Current gripper position
            end_effector_orientation (np.ndarray, optional): End effector orientation

        Returns:
            ArticulationAction: Joint positions for robot execution
        """
        if self._start:
            self._start = False
            self._event = 0
            self._t = 0

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if self._event >= len(self._events_dt):
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        target_joint_positions = self._execute_phase(
            center_position, gripper_position, end_effector_orientation, current_joint_positions
        )

        # Time-based progression as backup
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event] 
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def _execute_phase(self, center_position, gripper_position, end_effector_orientation, current_joint_positions):
        """Execute current phase and handle transitions."""
        
        if self._event == 0:
            # Lift phase
            target_position = center_position.copy()
            target_position[2] += 0.3 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            distance = np.linalg.norm(gripper_position - target_position)
            if distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 1:
            # Move above beaker
            target_position = center_position.copy()
            target_position[2] += 0.3 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 2:
            # Lower into beaker
            target_position = center_position.copy()
            target_position[2] += 0.12 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 3:
            # Stirring motion
            angle_increment = self._stir_speed * 0.01
            self._current_stir_angle += angle_increment
            
            x_offset = self._stir_radius * np.cos(self._current_stir_angle)
            y_offset = self._stir_radius * np.sin(self._current_stir_angle)
            target_position = center_position.copy()
            target_position[0] += x_offset
            target_position[1] += y_offset
            target_position[2] += 0.1 / get_stage_units()
            
            return self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 4:
            # Lift out of beaker
            target_position = center_position.copy()
            target_position[2] += 0.2 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        else:
            return ArticulationAction(joint_positions=[None] * len(current_joint_positions))

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """Reset controller to initial state."""
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self._current_stir_angle = 0.0
        
        if events_dt is not None:
            if not isinstance(events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            else:
                self._events_dt = events_dt
            if len(self._events_dt) != 5:
                raise Exception(f"events_dt length must be 5, got {len(self._events_dt)}")

    def is_done(self) -> bool:
        """Check if stirring sequence is complete."""
        return self._event >= len(self._events_dt)

