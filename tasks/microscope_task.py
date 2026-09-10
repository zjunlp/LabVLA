# ============================================================================
# microscope_task.py: microscope sample task.
# Prepare and place a microscope sample.
# Level 4 long-sequence task.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class MicroscopeSampleTask(BaseTask):
    """Level 4 microscope sample task: prepare a slide, add a sample with a dropper, apply a coverslip, and place the slide on the microscope stage."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the microscope sample task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.microscope_path = cfg.microscope_path
        self.microscope_stage_path = cfg.microscope_stage_path
        self.dropper_path = cfg.dropper_path
        self.slide_path = cfg.slide_path
        self.coverslip_path = cfg.coverslip_path
        
    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        # Initialize positions for all objects.
        for obj_cfg in self.cfg.task.obj_paths:
            obj_path = obj_cfg['path']
            pos_range = obj_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=obj_path, position=position)
        
    def step(self):
        """Step the simulation and return the current state."""
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2500):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        # Get positions for all relevant objects.
        microscope_pos = self.object_utils.get_geometry_center(object_path=self.microscope_path)
        slide_pos = self.object_utils.get_geometry_center(object_path=self.slide_path)
        dropper_pos = self.object_utils.get_geometry_center(object_path=self.dropper_path)
        coverslip_pos = self.object_utils.get_geometry_center(object_path=self.coverslip_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'microscope_position': microscope_pos,
            'slide_position': slide_pos,
            'dropper_position': dropper_pos,
            'coverslip_position': coverslip_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
