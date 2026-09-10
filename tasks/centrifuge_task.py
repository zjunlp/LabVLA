# ============================================================================
# centrifuge_task.py: centrifuge tasks.
# Handle sample loading and the full centrifuge workflow.
# Level 2 and Level 5 tasks.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class CentrifugeLoadTask(BaseTask):
    """Level 2 centrifuge loading task. Inherit BaseTask and pick a test tube from its rack, then place it in a centrifuge slot."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the centrifuge loading task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.centrifuge_path = cfg.centrifuge_path
        self.tube_rack_path = cfg.tube_rack_path
        
    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        # Read object paths from the configuration.
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
        
        # Get centrifuge and test tube positions.
        centrifuge_pos = self.object_utils.get_geometry_center(object_path=self.centrifuge_path)
        tube_pos = self.object_utils.get_geometry_center(object_path=self.current_obj_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'object_position': tube_pos,
            'object_name': self.current_obj_path.split("/")[-1],
            'centrifuge_position': centrifuge_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }


class FullCentrifugeTask(BaseTask):
    """Level 5 full centrifuge task. Load multiple sample tubes, balance the rotor, close the lid, start the run, wait for completion, and remove samples."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the full centrifuge task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.centrifuge_path = cfg.centrifuge_path
        self.centrifuge_lid_path = cfg.centrifuge_lid_path
        self.tube_rack_path = cfg.tube_rack_path
        self.sample_tubes = cfg.sample_tubes
        
    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        # Initialize positions for all sample tubes.
        for i, tube_cfg in enumerate(self.cfg.task.obj_paths):
            tube_path = tube_cfg['path']
            pos_range = tube_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=tube_path, position=position)
        
    def step(self):
        """Step the simulation and return the current state."""
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=3000):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        # Collect positions for all relevant objects.
        sample_positions = []
        for tube in self.sample_tubes:
            pos = self.object_utils.get_geometry_center(object_path=tube['path'])
            sample_positions.append(pos)
        
        centrifuge_pos = self.object_utils.get_geometry_center(object_path=self.centrifuge_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'sample_positions': sample_positions,
            'centrifuge_position': centrifuge_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
