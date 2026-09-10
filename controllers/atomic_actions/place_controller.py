from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from isaacsim.robot.manipulators.grippers.gripper import Gripper

class PlaceController(BaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0

        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt must be numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt 6")
        self._position_threshold = _position_threshold
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._start = True
        self.target_position = None  
        return

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_position: typing.Optional[np.ndarray] = None,
        pre_place_z: float = 0.2,
        place_offset_z: float = 0.05,
    ) -> ArticulationAction:
        
        if self._start:
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        if self._event == 0:
            self.target_position = place_position.copy()
            self.target_position[2] += pre_place_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        elif self._event == 1:
            self.target_position = place_position.copy()
            self.target_position[2] += place_offset_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position - gripper_position)
                if xy_distance < 0.02:
                    self._event += 1
                    self._t = 0
        elif self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position.copy()
            self.target_position[2] += 0.15 / get_stage_units()
            self.target_position[0] -= 0.15 / get_stage_units()
            gripper_control.release_object()
        elif self._event == 4:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities  numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt  6")
        return

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)
