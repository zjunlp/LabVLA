# ============================================================================
# synthesis_task.py: chemical synthesis task.
# Run the chemical synthesis experiment.
# Level 5 complex-sequence task.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class ChemicalSynthesisTask(BaseTask):
    """Level 5 synthesis task covering flask preparation, reagent A and B addition, condenser setup, heating, stirring, and product extraction with a separatory funnel."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the synthesis task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
        self.round_flask_path = cfg.round_flask_path
        self.condenser_path = cfg.condenser_path
        self.heating_mantle_path = cfg.heating_mantle_path
        self.stirrer_path = cfg.stirrer_path
        self.separatory_funnel_path = cfg.separatory_funnel_path
        self.reagent_bottles = cfg.reagent_bottles
        
    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        # Initialize all object positions.
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
        
        if not self.check_frame_limits(max_steps=4000):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        # Get positions for all relevant objects.
        flask_pos = self.object_utils.get_geometry_center(object_path=self.round_flask_path)
        condenser_pos = self.object_utils.get_geometry_center(object_path=self.condenser_path)
        heater_pos = self.object_utils.get_geometry_center(object_path=self.heating_mantle_path)
        funnel_pos = self.object_utils.get_geometry_center(object_path=self.separatory_funnel_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'flask_position': flask_pos,
            'condenser_position': condenser_pos,
            'heater_position': heater_pos,
            'funnel_position': funnel_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
