from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from isaacsim.robot.manipulators.grippers.gripper import Gripper

class PressController(BaseController):
    """
    A pressing state machine controller.
    
    This controller handles the process of pressing a button, including the following phases:
    - Phase 0: Move the end effector in front of the target object (along X-axis).
    - Phase 1: Close the gripper.
    - Phase 2: Press forward to the target position.
    
    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that returns ArticulationAction.
        gripper (Gripper): Controller for opening/closing the gripper.
        initial_offset (float, optional): Initial offset distance (along X-axis), defaults to 0.1 meters.
        events_dt (list of float, optional): Duration for each phase, defaults to [0.01, 0.01, 0.01].
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        end_effector_initial_height: typing.Optional[float] = None,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        # Initialize parent BaseController
        BaseController.__init__(self, name=name)
        
        self._event = 0  # Current phase number
        self._t = 0  # Current phase time counter
        self._initial_offset = initial_offset if initial_offset is not None else 0.2 / get_stage_units()
        # Initial offset distance, default 0.1 meters (adjusted by stage units)
        
        if events_dt is None:
            self._events_dt = [0.005, 0.1, 0.01]  # Default phase durations
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events_dt length must be exactly 3")
        
        self._cspace_controller = cspace_controller  # Store Cartesian space controller
        self._start = True

    def get_current_event(self) -> int:
        """
        Get the current phase/event of the state machine.

        Returns:
            int: Current phase/event number.
        """
        return self._event
    
    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        press_distance: float = 0.04
    ) -> ArticulationAction:
        """
        Execute one step of the pressing action.
        
        Args:
            target_position (np.ndarray): Target pressing position.
            current_joint_positions (np.ndarray): Current robot joint positions.
            gripper_control: Gripper controller.
            end_effector_orientation (np.ndarray, optional): End effector orientation.
        
        Returns:
            ArticulationAction: Robot control action.
        """
        
        if self._start:
            # Initial state: Open the gripper
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = 0.04 / get_stage_units()  # Open the gripper
            target_joint_positions[8] = 0.04 / get_stage_units()  # Open the gripper
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if self.is_done():
            # Pause or done state: Maintain current joint positions
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # Execute the current phase action
        if self._event == 0:
            # Phase 0: Move in front of the target object
            target_position[0] -= self._initial_offset  # Offset forward along X-axis
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 1:
            # Phase 1: Close the gripper
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.0015 / get_stage_units()  # Default gripper close distance (adjustable)
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 2:
            # Phase 2: Press forward to the target position
            target_position[0]+= press_distance/ get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        
        return target_joint_positions

    
    def reset(
        self,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None
    ) -> None:
        """
        Reset the state machine to initial state.
        
        Args:
            initial_offset (float, optional): New initial offset distance.
            events_dt (list of float, optional): New list of phase durations.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if initial_offset is not None:
            self._initial_offset = initial_offset
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events_dt length must be exactly 3")
        self._start = True
    
    def is_done(self) -> bool:
        # Check if the state machine is done
        return self._event >= len(self._events_dt)
    
