from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction

import numpy as np
import typing
from scipy.spatial.transform import Rotation as R
class PourController(BaseController):
    """
    PourController implements a state machine for pouring liquid. The state transitions are as follows:

    State 0: Move above the target position (with random height offset). When XY distance is close, proceed to next state.
    State 1: Further adjust height and position (considering object size and offset). When XY distance is close, proceed to next state.
    State 2: Switch joint 7 to velocity mode, start pouring (positive velocity).
    State 3: Hold joint 7 velocity at 0, pause pouring.
    State 4: Switch joint 7 to velocity mode, pour in reverse (negative velocity).
    State 5: Hold joint 7 velocity at 0, finish pouring.

    Each state's duration is controlled by self._events_dt. State transitions are managed by self._event and self._t.
    The control process is advanced step-by-step via the forward() method, and can be reset with the reset() method.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1,
        position_threshold: float = 0.006
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [dt / speed for dt in [0.002, 0.01, 0.009, 0.005, 0.009, 0.5]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            assert len(self._events_dt) == 6, "events dt need have length of 6 or less"
        self._cspace_controller = cspace_controller

        self._pour_default_speed = - 120.0 / 180.0 * np.pi
        self._position_threshold = position_threshold

        self._height_range_1 = (0.3, 0.4)
        self._height_range_2 = (0.1, 0.2)
        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        
        return

    def forward(
        self,
        articulation_controller: ArticulationController,
        source_size: np.ndarray,
        target_position: np.ndarray,
        current_joint_velocities: np.ndarray,
        gripper_position: np.ndarray,
        source_name: str = None,
        pour_speed: float = None,
        target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            articulation_controller (ArticulationController): The articulation controller for the robot.
            source_size (np.ndarray): Size of the source object being poured.
            current_joint_velocities (np.ndarray): Current joint velocities of the robot.
            pour_speed (float, optional): Speed for the pouring action. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        self.object_size = source_size
        
        if pour_speed is None:
            self._pour_speed = self._pour_default_speed
        else:
            self._pour_speed = pour_speed
            
        if  self._event >= len(self._events_dt):
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            return ArticulationAction(joint_velocities=target_joint_velocities)
        
        if self._event == 0:
            target_position[2] += self._random_height_1
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            self._random_height_1 = np.random.uniform(*self._height_range_1)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < 0.08:
                self._event += 1
                self._t = 0
                return target_joints
                
        elif self._event == 1:
            target_position[2] += self._random_height_2 + self.object_size[2] / 2 + self.get_pickz_offset(source_name)
            target_position[1] -= self.object_size[2] / 2 - self.get_pickz_offset(source_name)
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            self._random_height_2 = np.random.uniform(*self._height_range_2)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joints
        elif self._event == 2:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 3:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 4:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 5:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 3.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self.object_size = None
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events dt need have length of 3 or less")

        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)
    
    def get_pickz_offset(self, item_name):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.03,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker2": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "beaker": 0.02,
            "beaker_l": 0.02,
            
        }

        for key in offsets:
            if key in item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5