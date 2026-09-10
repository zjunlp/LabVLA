import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.utils.types import ArticulationAction

from .base_controller import BaseController
from .robot_controllers.ridgebase.ridgebase_controller import RidgebaseController
from .atomic_actions.pick_controller import PickController
from lab_utils.object_utils import ObjectUtils


class MobilePickController(BaseController):
    """
    Mobile pick controller for controlling the Ridgebase robot to navigate and pick objects.
    
    Combines navigation and pick operations in two phases:
    1. Navigation phase: Navigate to target position
    2. Pick phase: Pick the target object
    
    Supports two modes:
    - collect mode: Collect trajectory data for both navigation and pick
    - infer mode: Use learned policies (reserved interface)
    
    Attributes:
        ridgebase_controller: Ridgebase low-level motion controller
        pick_controller: Pick operation controller
        franka_subset: Articulation subset for Franka arm
        waypoints_set: Whether the path points have been set
        pick_started: Whether the pick phase has started
        navigation_done: Whether the navigation phase is done
        initial_object_z: Initial z position of the target object
    """
    
    def __init__(self, cfg, robot):
        """
        Initialize the mobile pick controller.
        
        Args:
            cfg: Configuration object
            robot: Robot instance
        """
        # Initialize base controller
        super().__init__(cfg, robot, use_default_config=True)
        
        # Initialize Ridgebase controller for navigation
        self.ridgebase_controller = RidgebaseController(
            robot_articulation=robot,
            max_linear_speed=cfg.task.max_linear_speed if hasattr(cfg.task, 'max_linear_speed') else 0.02,
            max_angular_speed=cfg.task.max_angular_speed if hasattr(cfg.task, 'max_angular_speed') else 1.5,
            position_threshold=cfg.task.position_threshold if hasattr(cfg.task, 'position_threshold') else 0.05,
            angle_threshold=cfg.task.angle_threshold if hasattr(cfg.task, 'angle_threshold') else 0.02
        )
        
        # Initialize Pick controller
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )
        
        # Create Franka arm subset for pick operations
        self.franka_subset = ArticulationSubset(
            robot,
            ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
             'panda_joint5', 'panda_joint6', 'panda_joint7', 
             'panda_finger_joint1', 'panda_finger_joint2']
        )
        
        # State tracking
        self.waypoints_set = False
        self.pick_started = False
        self.navigation_done = False
        self.initial_object_z = None
        self.object_utils = ObjectUtils.get_instance()
        
        # Final angle for navigation (face the pick target)
        self.final_nav_angle = cfg.task.final_nav_angle if hasattr(cfg.task, 'final_nav_angle') else np.pi/2
        
    def reset(self) -> None:
        """Reset the controller state"""
        super().reset()
        self.waypoints_set = False
        self.pick_started = False
        self.navigation_done = False
        self.initial_object_z = None
        self.pick_controller.reset()
    
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Perform one control step.
        
        Args:
            state: The current state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Control step in collect mode.
        
        Handles both navigation and pick phases, collecting data throughout.
        
        Args:
            state: The state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        # Record initial object z position
        if self.initial_object_z is None and state.get('object_position') is not None:
            self.initial_object_z = state['object_position'][2]
        
        # Phase 1: Navigation
        if not self.navigation_done:
            return self._navigation_phase(state)
        
        # Phase 2: Pick
        else:
            return self._pick_phase(state)
    
    def _navigation_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Handle navigation phase.
        
        Args:
            state: The state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        # Set waypoints on first call
        if not self.waypoints_set and state.get('waypoints') is not None:
            self.ridgebase_controller.set_waypoints(state['waypoints'], self.final_nav_angle)
            self.waypoints_set = True
        
        current_pose = state['current_pose']
        action, done = self.ridgebase_controller.get_action(current_pose)
        
        # Collect navigation data
        if 'camera_data' in state and not done:
            # For navigation, only record base movement (first 3 joints: x, y, theta)
            nav_joint_positions = np.array([
                current_pose[0],
                current_pose[1],
                current_pose[2]
            ])
            
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=nav_joint_positions,
                language_instruction="Navigate to the pick location"
            )
        
        # Check if navigation is complete
        if done or self.ridgebase_controller.is_path_complete():
            print("Navigation completed, starting pick task!")
            self.navigation_done = True
            self.pick_started = True
            
            # Save navigation trajectory data
            if hasattr(self, 'data_collector'):
                final_nav_positions = np.array([
                    current_pose[0],
                    current_pose[1],
                    current_pose[2]
                ])
                self.data_collector.write_cached_data(final_nav_positions)
        
        return action, False, False
    
    def _pick_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Handle pick phase.
        
        Args:
            state: The state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        # Update RMP controller with robot base pose
        robot_base_path = "/World/Ridgebase/panda_link0"
        pose = self.object_utils.get_object_xform_position(object_path=robot_base_path)
        quat = self.object_utils.get_transform_quat(object_path=robot_base_path, w_first=True)
        self.rmp_controller.rmp_flow.set_robot_base_pose(pose, quat)
        self.pick_controller.set_robot_position(pose)
        
        # Get Franka arm joint positions
        joint_positions = self.franka_subset.get_joint_positions()
        
        # Get end effector position
        end_effector_position = self.object_utils.get_object_xform_position(
            object_path="/World/Ridgebase/endeffector"
        )
        
        # Execute pick action
        action = self.pick_controller.forward(
            picking_position=state['object_position'],
            current_joint_positions=joint_positions,
            object_name=state['object_name'],
            object_size=state['object_size'] if state['object_size'] is not None else np.array([0.06, 0.06, 0]),
            gripper_control=self.gripper_control,
            gripper_position=end_effector_position,
            end_effector_orientation=R.from_euler('xyz', np.radians([-90, 90, 30])).as_quat(),
        )
        
        # Collect pick data (full joint positions including arm)
        if 'camera_data' in state and not self.pick_controller.is_done():
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-2],  # Exclude gripper joints
                language_instruction=self.get_language_instruction()
            )
        
        if action is not None:
            action = ArticulationAction(
                joint_positions=action.joint_positions, 
                joint_velocities=action.joint_velocities, 
                joint_indices=self.franka_subset.joint_indices[:len(action.joint_positions)]
            )
        
        # Check if pick is complete
        if self.pick_controller.is_done():
            print("Pick task completed!")
            
            # Check if pick was successful (object lifted)
            current_z = state['object_position'][2]
            pick_success = (current_z - self.initial_object_z) > 0.1
            
            if pick_success:
                print("Pick successful - object lifted!")
                # Save pick trajectory data
                if hasattr(self, 'data_collector'):
                    self.data_collector.write_cached_data(state['joint_positions'][:-2])
            else:
                print("Pick failed - object not lifted enough")
                self.data_collector.clear_cache()
            
            self._last_success = pick_success
            self.reset_needed = True
            return action, True, pick_success
        
        return action, False, False
    
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Control step in inference mode (reserved interface).
        
        Args:
            state: The state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        # Similar structure to collect mode but using inference engine
        # This is a placeholder for future implementation
        if not self.navigation_done:
            return self._navigation_phase(state)
        else:
            return self._pick_phase(state)
    
    def get_language_instruction(self) -> Optional[str]:
        """
        Get the language instruction for the task.

        Returns:
            str: The language instruction
        """
        if not self.navigation_done:
            self._language_instruction = "Navigate to the pick location"
        else:
            self._language_instruction = f"Pick up the object"
        return self._language_instruction

