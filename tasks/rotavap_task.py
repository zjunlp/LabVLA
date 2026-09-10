# ============================================================================
# rotavap_task.py: rotary evaporator task.
# Mount a round-bottom flask on the rotary evaporator.
# Level 3 task of moderate complexity.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class RotavapSetupTask(BaseTask):
    """Level 3 rotary evaporator setup task. Pick up the flask, align it with the holder, and secure it at the required angle."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the rotary evaporator task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.rotavap_path = cfg.rotavap_path
        self.rotavap_holder_path = cfg.rotavap_holder_path
        
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
        
        if not self.check_frame_limits(max_steps=1800):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        # Get positions for all relevant objects.
        rotavap_pos = self.object_utils.get_geometry_center(object_path=self.rotavap_path)
        flask_pos = self.object_utils.get_geometry_center(object_path=self.current_obj_path)
        holder_pos = self.object_utils.get_geometry_center(object_path=self.rotavap_holder_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'object_position': flask_pos,
            'object_name': self.current_obj_path.split("/")[-1],
            'rotavap_position': rotavap_pos,
            'holder_position': holder_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
