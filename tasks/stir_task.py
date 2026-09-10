import numpy as np
from .base_task import BaseTask

class StirTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.glass_rod = self.cfg.obj_path
        self.target_beaker = self.cfg.target_path
        self.glass_rod_mesh = self.cfg.sub_obj_path
            
    def reset(self):
        super().reset()
        self.robot.initialize()

        # target_beaker
        self.object_utils.set_object_position(object_path=self.target_beaker, position=np.array([0.24125 + np.random.uniform(-0.075, 0.075), -0.31358 + np.random.uniform(-0.075, 0.075), 0.77]))

        # test_tube_rack
        rack_position = np.array([0.28421, 0.30755, 0.82291])
        self.object_utils.set_object_position(object_path="/World/test_tube_rack", position=rack_position)

        # glass_rod
        self.object_utils.set_object_position(object_path=self.glass_rod, position=rack_position + np.array([-0.01152, -0.1125, 0.03197]))
        self.object_utils.set_object_position(object_path=self.glass_rod_mesh, position=[0, 0, 0])
            
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2000):
            return None
        
        return self.get_basic_state_info(
            object_path=self.glass_rod,
            target_path=self.target_beaker,
            additional_info={
                'target_beaker': self.target_beaker,
                'object_position': self.object_utils.get_object_xform_position(self.glass_rod),
                'glass_rod_position':self.object_utils.get_object_xform_position(object_path=self.cfg.sub_obj_path)
            }
        )
