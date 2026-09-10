import numpy as np
import yaml
from typing import Dict, Any
from .base_task import BaseTask
from lab_utils.a_star import plan_navigation_path, real_to_grid, load_grid
from isaacsim.core.utils.rotations import quat_to_euler_angles

class NavigationTask(BaseTask):
    """
    Navigation task class, used for path planning and navigation of mobile robots in the scene.
    
    This task is responsible for:
    - Generating random start and end points
    - Using A* algorithm to plan paths
    - Managing task state and reset
    
    Attributes:
        navigation_assets (list): Navigation scene configuration list
        grid: Obstacle grid map
        current_start: Current start point
        current_end: Current end point
        current_path: Current planned path
    """
    
    def __init__(self, cfg, world, stage, robot):
        """
        Initialize the navigation task.
        
        Args:
            cfg: Task configuration object
            world: Isaac Sim world instance
            stage: USD scene stage
            robot: Robot instance
        """
        self.navigation_assets = []
        self.grid = None
        self.current_start = None
        self.current_end = None
        self.current_path = None
        
        super().__init__(cfg, world, stage, robot)
        
    def setup_objects(self) -> None:
        """
        Load the navigation scene configuration.
        Load the navigation scene configuration from the configuration file (boundary, obstacle image, etc.).
        """
        super().setup_objects()
        
        # Load the navigation scene configuration
        if hasattr(self.cfg.task, 'navigation_config_path'):
            with open(self.cfg.task.navigation_config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.navigation_assets = config.get('assets', [])
        
        # If there is a navigation scene in the configuration, load the obstacle grid
        if self.navigation_assets:
            nav_scene = self.navigation_assets[0]
            self.grid, self.W, self.H = load_grid(nav_scene['barrier_image_path'])
    
    def reset(self) -> None:
        """
        Reset the task, generate new start and end points, and plan the path.
        """
        super().reset()
        self.robot.initialize()
        
        # Generate a new navigation task.
        if self.navigation_assets and len(self.navigation_assets) > 0:
            success = self._generate_navigation_task()
            if not success:
                print("Warning: Unable to generate a valid navigation path")
    
    def _generate_navigation_task(self) -> bool:
        """
        Generate a new navigation task (start point, end point, and path).  
        
        Returns:
            bool: Whether the path is successfully generated
        """
        nav_scene = self.navigation_assets[0]
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Randomly generate start and goal positions.
            start_point, end_point = self._generate_random_points(
                nav_scene['x_bounds'],
                nav_scene['y_bounds'],
                self.grid
            )
            
            if start_point is None or end_point is None:
                continue
            
            # Plan the path.
            task_info = {
                "asset": nav_scene,
                "start": start_point,
                "end": end_point
            }
            
            path_result = plan_navigation_path(task_info)
            
            if path_result is not None:
                self.current_start = start_point
                self.current_end = end_point
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
                
                # Set the initial position of the robot
                initial_position = np.array([start_point[0], start_point[1], 0.0])
                self.robot.set_world_pose(position=initial_position)
                
                return True
        
        return False
    
    def _generate_random_points(self, x_bounds, y_bounds, grid, attempts=100):
        """
        Generate random start and end points in free space.
        
        Args:
            x_bounds: X-axis boundary [min, max]
            y_bounds: Y-axis boundary [min, max]
            grid: Obstacle grid map
            attempts: Maximum number of attempts
            
        Returns:
            tuple: (start_point, end_point) The start and end point coordinates, return (None, None) if failed
        """
        W = len(grid[0])
        H = len(grid)
        
        for _ in range(attempts):
            start_x = np.random.uniform(x_bounds[0], x_bounds[1])
            start_y = np.random.uniform(y_bounds[0], y_bounds[1])
            end_x = np.random.uniform(x_bounds[0], x_bounds[1])
            end_y = np.random.uniform(y_bounds[0], y_bounds[1])
            
            i_start, j_start = real_to_grid(
                start_x, start_y, x_bounds, y_bounds, (W, H)
            )
            i_end, j_end = real_to_grid(
                end_x, end_y, x_bounds, y_bounds, (W, H)
            )
            
            # Check that start and goal are in free space.
            if grid[i_start][j_start] == 0 and grid[i_end][j_end] == 0:
                return [start_x, start_y], [end_x, end_y]
        
        return None, None
    
    def step(self) -> Dict[str, Any]:
        """
        Execute a simulation step and return the current state.

        Returns:
            dict: The state dictionary containing the robot position, path information, etc.
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
        
        # Get the current position and orientation of the robot
        position, orientation = self.robot.get_world_pose()

        euler_angles = quat_to_euler_angles(orientation, extrinsic=False)
        current_pose = np.array([position[0], position[1], euler_angles[2]])
        
        # Get the camera data
        camera_data, display_data = self.get_camera_data()
        
        state = {
            'current_pose': current_pose,
            'start_point': self.current_start,
            'end_point': self.current_end,
            'waypoints': self.current_path,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'frame_idx': self.frame_idx
        }
        
        return state
    
    def on_task_complete(self, success: bool) -> None:
        """
        Handle the task completion.
        
        Args:
            success: Whether the task is successfully completed
        """
        self.reset_needed = True

