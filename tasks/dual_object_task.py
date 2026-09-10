import numpy as np
from typing import Dict, Any
from .base_task import BaseTask

class DualObjectTask(BaseTask):
    """
    Base class for handling dual object tasks.
    Suitable for tasks like place, pour, etc. that require both source and target objects.
    """
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        
    def reset(self):
        """
        Reset task state.
        Initialize robot position and randomize positions of source and target objects.
        """
        super().reset()
        self.robot.initialize()
        
        self.source_obj = self.cfg.task.obj_paths[0]['path']
        self.target_obj = self.cfg.task.obj_paths[1]['path']
        
        source_position_range = self.cfg.task.obj_paths[0]['position_range']
        self.randomize_object_position(self.source_obj, source_position_range)
        
        target_position_range = self.cfg.task.obj_paths[1]['position_range']
        self.randomize_object_position(self.target_obj, target_position_range)
        
    def step(self):
        """
        Execute one simulation step.
        
        Returns:
            dict: Dictionary containing simulation state data, returns None if not ready
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
        
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj
        ) 