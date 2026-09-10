from controllers.robot_controllers.grapper_manager import Gripper
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
import numpy as np
import typing
from isaacsim.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

class OpenController(BaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        furniture_type: str = "drawer",
        door_width: float = 0.3,
        door_open_direction: str = "counterclockwise"
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self.furniture_type = furniture_type
        self.door_width = door_width
        self.door_open_direction = door_open_direction
        self.position_rotation_interp_iter = None
        self._start = True
        
        if events_dt is None:
            self._events_dt = [0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 8:
                raise Exception(f"events_dt length must be 8, got {len(self._events_dt)}")

        self._position_threshold = 0.01 / get_stage_units()
        
    def forward(
        self,
        handle_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        revolute_joint_position: np.ndarray = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        angle: float = 50.0,
        close_gripper_distance: float = 0.023
    ) -> ArticulationAction:
        """
        Execute one step of opening control.

        Args:
            handle_position (np.ndarray): Handle position
            current_joint_positions (np.ndarray): Current robot joint positions
            revolute_joint_position (np.ndarray): Revolute joint position
            gripper_position (np.ndarray): Gripper position
            end_effector_orientation (np.ndarray, optional): End effector orientation (quaternion).
                Defaults to None, which uses [0, np.pi, 0].

        Returns:
            ArticulationAction: Control action
        """

        if self._start:
            return self._handle_start_state(current_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat([0, 110, 0], degrees=True, extrinsic=False)
            
        self._t += self._events_dt[self._event]
        
        target_joint_positions = self._execute_phase(
            handle_position, 
            end_effector_orientation, 
            current_joint_positions,
            gripper_position,
            revolute_joint_position,
            angle,
            close_gripper_distance
        )
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
    
    def _execute_phase(self, handle_position, end_effector_orientation, current_joint_positions, gripper_position, revolute_joint_position = None, angle = 50, close_gripper_distance = 0.023):
        """Execute current phase of grasping action"""
        if self.furniture_type == "drawer":
            return self._execute_drawer_phase(handle_position, end_effector_orientation, current_joint_positions, gripper_position)
        else:
            return self._execute_door_phase(handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle, close_gripper_distance)
    
    def _execute_drawer_phase(self, handle_position, end_effector_orientation, current_joint_positions, gripper_position):
        """Execute drawer opening action"""
        if self._event == 0:
            handle_position[0] -= 0.08 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=handle_position, target_end_effector_orientation=end_effector_orientation
                )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 1:
            handle_position[0] -= 0.015
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=handle_position, target_end_effector_orientation=end_effector_orientation
                )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold / 3:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 2:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.01 / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = handle_position
            self.target_position[0] -= 0.1 / get_stage_units()
        elif self._event == 3:
            handle_position[0] -= 0.04 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            ) 
        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 5:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.04 / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 6:
            handle_position[0] -= 0.12 / get_stage_units()
            handle_position[2] += 0.06
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        return target_joint_positions

    def _execute_door_phase(self, handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle = 50, close_gripper_distance = 0.023):
        """Execute door opening action"""
        if self._event == 0:
            handle_position[0] -= 0.08
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 1:
            handle_position[0] -= 0.015
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold / 3:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 2:
            handle_position[0] -= 0.015
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = close_gripper_distance
            target_joint_positions[8] = close_gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.start_position = handle_position.copy()
        elif self._event == 3:
            if self.position_rotation_interp_iter is None:
                if revolute_joint_position[1] > self.start_position[1]:
                    angle = -angle
                self.target_position = self.rotate_around_z_axis(self.start_position, revolute_joint_position, angle)
                num_interpolation = int(600 * np.linalg.norm(self.start_position - self.target_position))
                alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
                position_rotation_interp_list = self.action_interpolation(
                    self.start_position, end_effector_orientation, self.target_position, self.rotate_quaternion_around_x(end_effector_orientation, angle), alphas, joint_pos=revolute_joint_position
                )
                self.position_rotation_interp_iter = iter(position_rotation_interp_list)
            try:
                self.trans_interp, self.rotation_interp = next(self.position_rotation_interp_iter)
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=self.trans_interp,
                    target_end_effector_orientation=self.rotation_interp
                )
            except:
                self._event += 1
                self._t = 0
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=self.trans_interp, 
                    target_end_effector_orientation=self.rotation_interp
                )
        elif self._event == 4:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 5:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.04
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 6:
            handle_position = self.trans_interp.copy()
            handle_position[0] -= 0.06
            if revolute_joint_position[1] > self.start_position[1]:
                handle_position[1] += 0.04
            else:
                handle_position[1] -= 0.04

            # handle_position[2] += 0.4
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=self.rotation_interp
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joint_positions
        else :
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        return target_joint_positions
    
    def reset(self) -> None:
        """Reset controller state"""
        BaseController.reset(self)
        self._event = 0
        self._t = 0
        self.position_rotation_interp_iter = None
        self._start = True

    def is_done(self) -> bool:
        """Check if controller has completed all states"""
        return self._event >= len(self._events_dt)
        
    def action_interpolation(self, trans_previous, rotation_previous, trans_target, rotation_target, alphas, joint_pos=None):
        """
        Interpolate between two poses for smooth motion.

        Args:
            trans_previous: Starting position
            rotation_previous: Starting orientation (quaternion)
            trans_target: Target position
            rotation_target: Target orientation (quaternion)
            alphas: Interpolation points
            joint_pos: Joint position for door rotation center

        Returns:
            List of interpolated (position, orientation) pairs
        """
        action_list = []
        rotation_previous_xyzw = rotation_previous[[1, 2, 3, 0]]
        rotation_target_xyzw = rotation_target[[1, 2, 3, 0]]
        key_rots = R.from_quat(np.stack([rotation_previous_xyzw, rotation_target_xyzw]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(alphas).as_quat()
        interp_rots = interp_rots[:, [3, 0, 1, 2]]  # back to wxyz

        # Calculate polar coordinates relative to joint_pos
        r_0 = trans_previous[:2] - joint_pos[:2]
        r_1 = trans_target[:2] - joint_pos[:2]

        # Calculate radii
        radius_0 = np.linalg.norm(r_0)
        radius_1 = np.linalg.norm(r_1)
        radii = np.linspace(radius_0, radius_1, len(alphas) + 1)[1:]

        # Calculate start and target angles
        theta_0 = np.arctan2(r_0[1], r_0[0])
        theta_1 = np.arctan2(r_1[1], r_1[0])

        # Ensure interpolation angle range is within 1/4 circle (90°)
        # Choose shortest path and limit to π/2
        delta_theta = theta_1 - theta_0
        if delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        elif delta_theta < -np.pi:
            delta_theta += 2 * np.pi
        # Limit interpolation to 1/4 circle
        delta_theta = np.clip(delta_theta, -np.pi / 2, np.pi / 2)
        thetas = np.linspace(theta_0, theta_0 + delta_theta, len(alphas) + 1)[1:]

        for alpha, radius, theta, interp_rot in zip(alphas, radii, thetas, interp_rots):
            # Correct Cartesian coordinate calculation, remove np.pi offset
            trans_interp = np.array([
                joint_pos[0] + radius * np.cos(theta),  # Directly use theta
                joint_pos[1] + radius * np.sin(theta),  # Directly use theta
                alpha * trans_target[2] + (1 - alpha) * trans_previous[2],  # Linear interpolation for Z axis
            ])
            action_list.append((trans_interp, interp_rot))

        return action_list
    
    def rotate_around_z_axis(self, p1, p2, angle_deg):
        """
        Rotate point p1 around point p2 by angle_deg degrees around Z axis.

        Args:
            p1: Point to rotate
            p2: Center of rotation
            angle_deg: Rotation angle in degrees

        Returns:
            Rotated point
        """
        angle_rad = np.deg2rad(angle_deg)
        # Translate: Move p1 to coordinate system with p2 as origin
        p1_relative = p1 - p2
        # Counterclockwise rotation matrix around Z axis by angle_deg
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                1]
        ])
        # Apply rotation matrix
        p1_rotated_relative = rotation_matrix @ p1_relative
        # Translate back: Add p2 coordinates
        p1_rotated = p1_rotated_relative + p2
        return p1_rotated

    def rotate_quaternion_around_x(self, q, angle_deg):
        """
        Rotate quaternion around X axis by specified angle.

        Args:
            q: Input quaternion
            angle_deg: Rotation angle in degrees

        Returns:
            Rotated quaternion
        """
        angle_rad = np.deg2rad(angle_deg)
        q_rot = np.array([
            -np.sin(angle_rad / 2), 0, 0, np.cos(angle_rad / 2)
        ])
        r = R.from_quat(q)
        r_rot = R.from_quat(q_rot)
        r_new = r * r_rot
        return r_new.as_quat()