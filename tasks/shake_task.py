import numpy as np
from .base_task import BaseTask

class ShakeTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.source_beaker = self.cfg.obj_path

    def reset(self):
        super().reset()
        self.robot.initialize()
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx, far_distance=10.0
        )
        
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2000):
            return None
        
        return self.get_basic_state_info(
            object_path=self.source_beaker)
