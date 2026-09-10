import numpy as np
import yaml
from typing import Dict, Any
from .base_task import BaseTask
from lab_utils.a_star import plan_navigation_path, real_to_grid, load_grid
from isaacsim.core.utils.rotations import quat_to_euler_angles


class MobilePickTask(BaseTask):
    """
    Mobile pick task class, combining navigation and pick operations.
    
    This task is responsible for:
    - Generating random start positions
    - Planning navigation path to target position
    - Managing pick target object
    - Managing task state and reset
    
    Attributes:
        navigation_assets (list): Navigation scene configuration list
        grid: Obstacle grid map
        current_start: Current start position
        target_position: Target position for navigation
        current_path: Current planned path
        target_object_path: Path to the object to pick
        initial_object_position: Initial position of the target object
    """
    
    def __init__(self, cfg, world, stage, robot):
        """
        Initialize the mobile pick task.
        
        Args:
            cfg: Task configuration object
            world: Isaac Sim world instance
            stage: USD scene stage
            robot: Robot instance
        """
        self.navigation_assets = []
        self.grid = None
        self.current_start = None
        self.target_position = None
        self.current_path = None
        self.target_object_path = None
        self.initial_object_position = None
        self.navigation_done = False
        
        super().__init__(cfg, world, stage, robot)
        
    def setup_objects(self) -> None:
        """
        Load the navigation scene configuration and pick target object.
        """
        super().setup_objects()
        
        # Load navigation scene configuration
        if hasattr(self.cfg.task, 'navigation_config_path'):
            with open(self.cfg.task.navigation_config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.navigation_assets = config.get('assets', [])
        
        # Load obstacle grid if navigation scene exists
        if self.navigation_assets:
            nav_scene = self.navigation_assets[0]
            self.grid, self.W, self.H = load_grid(nav_scene['barrier_image_path'])
        
        # Load pick target object
        if hasattr(self.cfg.task, 'pick_object_usd') and hasattr(self.cfg.task, 'pick_object_prim_path'):
            from isaacsim.core.utils.stage import add_reference_to_stage
            import os
            add_reference_to_stage(
                usd_path=os.path.abspath(self.cfg.task.pick_object_usd),
                prim_path=self.cfg.task.pick_object_prim_path
            )
            self.target_object_path = self.cfg.task.pick_object_path
            
            # Set object position if specified
            if hasattr(self.cfg.task, 'object_position'):
                import numpy as np
                self.object_utils.set_object_position(
                    object_path=self.cfg.task.pick_object_path,
                    position=np.array(self.cfg.task.object_position)
                )
        elif hasattr(self.cfg.task, 'pick_object_path'):
            self.target_object_path = self.cfg.task.pick_object_path
    
    def reset(self) -> None:
        """
        Reset the task, generate new start position and plan path to target.
        """
        super().reset()
        self.robot.initialize()
        self.navigation_done = False
        self.initial_object_position = None
        
        # Generate new navigation task
        if self.navigation_assets and len(self.navigation_assets) > 0:
            success = self._generate_navigation_task()
            if not success:
                print("Warning: Unable to generate a valid navigation path")
    
    def _generate_navigation_task(self) -> bool:
        """
        Generate a new navigation task (start position and path to target).
        
        Returns:
            bool: Whether the path is successfully generated
        """
        nav_scene = self.navigation_assets[0]
        
        # Get target position from configuration
        if hasattr(self.cfg.task, 'target_position'):
            self.target_position = self.cfg.task.target_position
        else:
            self.target_position = [-3.0, -0.46]  # Default target position
        
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Generate random start position
            start_point = self._generate_random_start_position(
                nav_scene['x_bounds'],
                nav_scene['y_bounds'],
                self.grid
            )
            
            if start_point is None:
                continue
            
            # Plan path from start to target
            task_info = {
                "asset": nav_scene,
                "start": start_point,
                "end": self.target_position
            }
            
            path_result = plan_navigation_path(task_info)
            
            if path_result is not None:
                self.current_start = start_point
                merged_path_real, _ = path_result
                
                # Add orientation to each path point
                waypoints = []
                for i in range(len(merged_path_real)):
                    x, y, r = merged_path_real[i]
                    
                    if i < len(merged_path_real) - 1:
                        next_x, next_y, _ = merged_path_real[i + 1]
                        theta = np.arctan2(next_y - y, next_x - x)
                    else:
                        theta = waypoints[-1][2] if waypoints else 0.0
                    waypoints.append([x, y, theta])
                
                self.current_path = waypoints
                
                # Set robot initial position
                initial_position = np.array([start_point[0], start_point[1], 0.0])
                self.robot.set_world_pose(position=initial_position)
                
                return True
        
        return False
    
    def _generate_random_start_position(self, x_bounds, y_bounds, grid, attempts=100):
        """
        Generate random start position in free space.
        
        Args:
            x_bounds: X-axis boundary [min, max]
            y_bounds: Y-axis boundary [min, max]
            grid: Obstacle grid map
            attempts: Maximum number of attempts
            
        Returns:
            list: [x, y] start position coordinates, return None if failed
        """
        W = len(grid[0])
        H = len(grid)
        
        for _ in range(attempts):
            start_x = np.random.uniform(x_bounds[0], x_bounds[1])
            start_y = np.random.uniform(y_bounds[0], y_bounds[1])
            
            i_start, j_start = real_to_grid(
                start_x, start_y, x_bounds, y_bounds, (W, H)
            )
            
            # Check if start position is in free space
            if grid[i_start][j_start] == 0:
                return [start_x, start_y]
        
        return None
    
    def step(self) -> Dict[str, Any]:
        """
        Execute a simulation step and return the current state.

        Returns:
            dict: The state dictionary containing robot pose, object info, path info, etc.
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
        
        # Get robot current position and orientation
        position, orientation = self.robot.get_world_pose()
        euler_angles = quat_to_euler_angles(orientation, extrinsic=False)
        current_pose = np.array([position[0], position[1], euler_angles[2]])
        
        # Get camera data
        camera_data, display_data = self.get_camera_data()
        
        # Get object position
        object_position = None
        object_size = None
        if self.target_object_path:
            object_position = self.object_utils.get_geometry_center(object_path=self.target_object_path)
            object_size = self.object_utils.get_object_size(object_path=self.target_object_path)
            
            # Record initial object position
            if self.initial_object_position is None:
                self.initial_object_position = object_position.copy()
        
        # Get robot joint positions
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None
        
        state = {
            'current_pose': current_pose,
            'start_point': self.current_start,
            'target_position': self.target_position,
            'waypoints': self.current_path,
            'navigation_done': self.navigation_done,
            'object_position': object_position,
            'object_size': object_size,
            'object_path': self.target_object_path,
            'object_name': self.target_object_path.split("/")[-1] if self.target_object_path else "",
            'initial_object_position': self.initial_object_position,
            'joint_positions': joint_positions,
            'gripper_position': self.robot.get_gripper_position(),
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'frame_idx': self.frame_idx,
            'robot_world_position': position
        }
        
        return state
    
    def set_navigation_done(self, done: bool) -> None:
        """
        Set navigation done flag.
        
        Args:
            done: Whether navigation is done
        """
        self.navigation_done = done
    
    def on_task_complete(self, success: bool) -> None:
        """
        Handle the task completion.
        
        Args:
            success: Whether the task is successfully completed
        """
        self.reset_needed = True

