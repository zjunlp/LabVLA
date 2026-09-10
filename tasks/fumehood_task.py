# ============================================================================
# fumehood_task.py: fume hood task.
# Operate inside the fume hood.
# Level 4 long-sequence task.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class FumehoodOperationTask(BaseTask):
    """Level 4 fume hood task. Open the sash, pick and place a flask, perform the operation, retrieve it, and close the sash."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the fume hood task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.fumehood_path = cfg.fumehood_path
        self.fumehood_sash_path = cfg.fumehood_sash_path
        self.work_surface_path = cfg.work_surface_path
        
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
        fumehood_pos = self.object_utils.get_geometry_center(object_path=self.fumehood_path)
        flask_pos = self.object_utils.get_geometry_center(object_path=self.current_obj_path)
        surface_pos = self.object_utils.get_geometry_center(object_path=self.work_surface_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'object_position': flask_pos,
            'object_name': self.current_obj_path.split("/")[-1],
            'fumehood_position': fumehood_pos,
            'surface_position': surface_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
