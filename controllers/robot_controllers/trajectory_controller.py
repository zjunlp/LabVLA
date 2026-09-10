import os
import numpy as np
from typing import List, Optional

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.prims.impl import Articulation
from isaacsim.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController


class FrankaTrajectoryController(RMPFlowController):
    """Franka robotic arm trajectory controller with support for continuous trajectory generation and execution"""

    def __init__(
        self, 
        name: str, 
        robot_articulation: Articulation, 
        physics_dt: float = 1.0/60.0,
        use_interpolation: bool = False
    ) -> None:
        super().__init__(name=name, robot_articulation=robot_articulation, physics_dt=physics_dt)
        
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        
        self._c_space_trajectory_generator = mg.LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        self._kinematics_solver = mg.LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        self._action_sequence = []
        self._action_sequence_index = 0
        self._end_effector_name = "panda_hand"
        self._physics_dt = physics_dt
        self._use_interpolation = use_interpolation

    def generate_trajectory(
        self, 
        waypoints: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> None:
        """Generate joint space trajectory using direct waypoints or interpolated trajectory
        
        Args:
            waypoints (np.ndarray): Array of shape (N, 8) containing N waypoints with joint angles and gripper positions
            timestamps (Optional[np.ndarray]): Array of timestamps for waypoints when using interpolation
        """
        joint_waypoints = waypoints[:, :7]
        self.gripper_positions = waypoints[:, 7]

        if np.allclose(joint_waypoints, joint_waypoints[0]):
            self._action_sequence = []
            for i in range(len(joint_waypoints)):
                action = ArticulationAction(
                    joint_positions=np.concatenate([joint_waypoints[0], [self.gripper_positions[i], self.gripper_positions[i]]]),
                    joint_velocities=None,
                    joint_efforts=None
                )
                self._action_sequence.append(action)
            total_actions = len(self._action_sequence)
            self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
            self._action_sequence_index = 0
            return
        
        if self._use_interpolation:
            joint_limits = self._c_space_trajectory_generator.get_c_space_position_limits()
            joint_min, joint_max = joint_limits[0], joint_limits[1]
            joint_waypoints = np.clip(joint_waypoints, joint_min, joint_max)
            if timestamps is not None:
                trajectory = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(
                    joint_waypoints, timestamps
                )
            else:
                trajectory = self._c_space_trajectory_generator.compute_c_space_trajectory(joint_waypoints)

            if trajectory is not None:
                articulation_trajectory = mg.ArticulationTrajectory(
                    self._articulation_motion_policy._robot_articulation,
                    trajectory,
                    self._physics_dt
                )
                self._action_sequence = articulation_trajectory.get_action_sequence()
                total_actions = len(self._action_sequence)
                self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
                self._action_sequence_index = 0
            else:
                print("Warning: Failed to generate trajectory")
                self._action_sequence = []
                self.gripper_positions = []
                self.gripper_indices = []
        else:
            # Direct waypoint output without interpolation
            joint_limits = self._c_space_trajectory_generator.get_c_space_position_limits()
            joint_min, joint_max = joint_limits[0], joint_limits[1]
            joint_waypoints = np.clip(joint_waypoints, joint_min, joint_max)
            
            self._action_sequence = []
            for i in range(len(joint_waypoints)):
                action = ArticulationAction(
                    joint_positions=np.concatenate([joint_waypoints[i], [self.gripper_positions[i], self.gripper_positions[i]]]),
                    joint_velocities=None,
                    joint_efforts=None
                )
                self._action_sequence.append(action)
                
            total_actions = len(self._action_sequence)
            self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
            self._action_sequence_index = 0

    def get_next_action(self) -> Optional[ArticulationAction]:
        """Get the next action in the sequence
        
        Returns:
            Optional[ArticulationAction]: Next action to execute, or None if sequence is complete
        """
        if not self._action_sequence or self._action_sequence_index >= len(self._action_sequence):
            return None
            
        action = self._action_sequence[self._action_sequence_index]
        
        if self._use_interpolation:
            # Add gripper position to interpolated trajectory actions
            if hasattr(self, 'gripper_positions') and len(self.gripper_positions) > 0:
                gripper_idx = self.gripper_indices[self._action_sequence_index]
                gripper_position = self.gripper_positions[gripper_idx]
                
                joint_positions = np.concatenate([
                    action.joint_positions,
                    np.array([gripper_position, gripper_position], dtype=np.float32)
                ])
                if action.joint_velocities is not None:
                    joint_velocities = np.concatenate([
                        action.joint_velocities,
                        np.array([0.0, 0.0], dtype=np.float32)
                    ])
                else:
                    joint_velocities = None
                
                action = ArticulationAction(
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities,
                    joint_efforts=action.joint_efforts
                )
        
        self._action_sequence_index += 1
        return action

    def is_trajectory_complete(self) -> bool:
        """Check if trajectory execution is complete
        
        Returns:
            bool: True if trajectory is complete, False otherwise
        """
        return len(self._action_sequence) == 0 or self._action_sequence_index >= len(self._action_sequence)

    def reset(self) -> None:
        """Reset controller state"""
        super().reset()
        self._action_sequence = []
        self._action_sequence_index = 0
        self.gripper_positions = []
        self.gripper_indices = [] 