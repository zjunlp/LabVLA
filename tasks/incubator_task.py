# ============================================================================
# incubator_task.py: incubator task.
# Place a Petri dish in the incubator.
# Level 3 task of moderate complexity.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class IncubatorLoadTask(BaseTask):
    """Level 3 incubator loading task. Open the door, pick and place the Petri dish, and close the door."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the incubator loading task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.incubator_path = cfg.incubator_path
        self.incubator_door_path = cfg.incubator_door_path
        self.shelf_path = cfg.shelf_path
        
    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
        
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx, far_distance=10.0
        )
        
    def step(self):
        """Step the simulation and return the current state."""
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2000):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        # Get positions for all relevant objects.
        incubator_pos = self.object_utils.get_geometry_center(object_path=self.incubator_path)
        dish_pos = self.object_utils.get_geometry_center(object_path=self.current_obj_path)
        shelf_pos = self.object_utils.get_geometry_center(object_path=self.shelf_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'object_position': dish_pos,
            'object_name': self.current_obj_path.split("/")[-1],
            'incubator_position': incubator_pos,
            'shelf_position': shelf_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
