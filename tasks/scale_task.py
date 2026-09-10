# ============================================================================
# scale_task.py: weighing task.
# Measure an object's mass with an electronic scale.
# Level 2 combined task.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class ScaleMeasureTask(BaseTask):
    """Level 2 weighing task derived from BaseTask. Pick up an object and place it gently at the center of the scale for an accurate reading."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the weighing task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.scale_path = cfg.scale_path
        
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
        
        if not self.check_frame_limits():
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        # Get scale and object positions.
        scale_pos = self.object_utils.get_geometry_center(object_path=self.scale_path)
        obj_pos = self.object_utils.get_geometry_center(object_path=self.current_obj_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'object_position': obj_pos,
            'object_name': self.current_obj_path.split("/")[-1],
            'scale_position': scale_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
