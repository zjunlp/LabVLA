import numpy as np
from typing import List, Tuple, Optional
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.prims.impl import Articulation
from isaacsim.core.utils.types import ArticulationAction

class RidgebaseController:
    def __init__(
        self,
        robot_articulation: Articulation,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        position_threshold: float = 0.1,
        angle_threshold: float = 0.1,
        dt: float = 0.01,
        final_angle: float = None  
    ):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold
        self.dt = 0.02
        self.final_angle = final_angle  

        self.k_p_linear = 1
        self.k_p_angular = 4
        
        self.waypoints = None
        self.current_waypoint_idx = 0
        
        self._joints_subset = ArticulationSubset(
            robot_articulation,
            ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]
        )

    def set_waypoints(self, waypoints: List[Tuple[float, float, float]], final_angle: Optional[float] = None) -> None:
        self.waypoints = np.array(waypoints)
        self.current_waypoint_idx = 0
        self.final_angle = final_angle  

    def compute_control(self, current_pose: np.ndarray) -> Tuple[float, float, float]:
        if self.waypoints is None or self.current_waypoint_idx >= len(self.waypoints):
            return 0.0, 0.0, 0.0, 0.0
        joint_positions = self._joints_subset.get_joint_positions()
        target = self.waypoints[self.current_waypoint_idx]
        current_pose[0] += joint_positions[0]
        current_pose[1] += joint_positions[1]
        current_pose[2] += joint_positions[2]
        
        dx = target[0] - current_pose[0]
        dy = target[1] - current_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        target_angle = np.arctan2(dy, dx)
        angle_diff = (target_angle - current_pose[2]) % (2 * np.pi)
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        if distance < self.position_threshold:
            if self.current_waypoint_idx == len(self.waypoints) - 1:
                if self.final_angle is not None:
                    final_angle_diff = (self.final_angle - current_pose[2]) % (2 * np.pi)
                    if final_angle_diff > np.pi:
                        final_angle_diff -= 2 * np.pi
                else:
                    final_angle_diff = (target[2] - current_pose[2]) % (2 * np.pi)
                    if final_angle_diff > np.pi:
                        final_angle_diff -= 2 * np.pi
                
                if abs(final_angle_diff) < self.angle_threshold:
                    return 0.0, 0.0, 0.0, final_angle_diff
                return 0.0, 0.0, self.k_p_angular * final_angle_diff, final_angle_diff
            else:
                self.current_waypoint_idx += 1
                return self.compute_control(current_pose)

        speed = min(distance * 0.2, self.max_linear_speed)
        x_vel = speed * np.cos(target_angle)
        y_vel = speed * np.sin(target_angle)
        
        theta_vel = self.k_p_angular * angle_diff
        
        return x_vel, y_vel, theta_vel, angle_diff

    def get_action(self, current_pose: np.ndarray) -> Tuple[Optional[ArticulationAction], bool]:
        x_vel, y_vel, theta_vel, angle_diff = self.compute_control(current_pose)
        x_vel = np.clip(abs(x_vel), 0, self.max_linear_speed) * np.sign(x_vel)
        y_vel = np.clip(abs(y_vel), 0, self.max_linear_speed) * np.sign(y_vel)
        theta_vel = np.clip(theta_vel, -self.max_angular_speed, self.max_angular_speed)
        
        joint_positions = self._joints_subset.get_joint_positions()
        
        next_x = joint_positions[0] + x_vel
        next_y = joint_positions[1] + y_vel
        next_theta = joint_positions[2] + theta_vel
        
        position = np.array([next_x, next_y, next_theta])
        action = self._joints_subset.make_articulation_action(
            joint_positions=position,
            joint_velocities=None
        )
        
        if self.final_angle is None:
            done = (self.waypoints is not None and 
                    self.current_waypoint_idx == len(self.waypoints) - 1 and 
                    abs(theta_vel) < self.angle_threshold)
        else:
            done = (self.waypoints is not None and 
                    self.current_waypoint_idx == len(self.waypoints) - 1 and 
                    abs(theta_vel) < self.angle_threshold and 
                    abs(angle_diff) < self.angle_threshold)

        return action, done

    def is_path_complete(self) -> bool:
        return (self.waypoints is not None and 
                self.current_waypoint_idx >= len(self.waypoints))
