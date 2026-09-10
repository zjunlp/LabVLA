import numpy as np
from .base_task import BaseTask
from lab_utils.Material_utils import bind_material_to_object

class CleanBeakerTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.world.reset()
        self.table_material_paths = self.cfg.table_material_paths

        self.table1_surface_1 = self.cfg.table1_surface_1
        self.table1_surface_2 = self.cfg.table1_surface_2
        self.table2_surface_1 = self.cfg.table2_surface_1
        self.table2_surface_2 = self.cfg.table2_surface_2
            
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        self.target_beaker = "/World/target_beaker"

        self.beaker_1 = "/World/beaker_hard_1"
        self.beaker_2 = "/World/beaker_hard_2"

        self.plat_1 = "/World/target_plat_1"
        self.plat_2 = "/World/target_plat_2"

        beaker_1_position = np.array([np.random.uniform(0.20, 0.25), np.random.uniform(-0.05, -0.1), 0.77])
        self.object_utils.set_object_position(object_path=self.beaker_1, position=beaker_1_position)

        plat_1_position = beaker_1_position + [0.03, 0, -0.057]
        self.object_utils.set_object_position(object_path=self.plat_1, position=plat_1_position)

        beaker_2_position = np.array([np.random.uniform(0.20, 0.25), np.random.uniform(0.20, 0.25), 0.77])
        self.object_utils.set_object_position(object_path=self.beaker_2, position=beaker_2_position)

        plat_2_position = np.array([0.056, np.random.uniform(0.27, 0.32), 0.713])
        self.object_utils.set_object_position(object_path=self.plat_2, position=plat_2_position)

        bind_material_to_object(stage=self.stage,
                                obj_path=self.table1_surface_1,
                                material_path=self.table_material_paths[0])
        
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table1_surface_2,
                                material_path=self.table_material_paths[0])
        
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table2_surface_1,
                                material_path=self.table_material_paths[0])
        
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table2_surface_2,
                                material_path=self.table_material_paths[0])
            
    def step(self):
        self.frame_idx += 1

        if not self.check_frame_limits(max_steps=4000):
            return None
        
        joint_positions = self.robot.get_joint_positions()

        beaker_1_position = self.object_utils.get_geometry_center(object_path=self.beaker_1)
        beaker_2_position = self.object_utils.get_geometry_center(object_path=self.beaker_2)

        plat_1_position = self.object_utils.get_geometry_center(object_path=self.plat_1)
        plat_2_position = self.object_utils.get_geometry_center(object_path=self.plat_2)

        target_size = self.object_utils.get_object_size(object_path=self.target_beaker)
        target_position = self.object_utils.get_geometry_center(object_path=self.target_beaker)
        
        camera_data, camera_display = self.get_camera_data()
                
        return {
            'joint_positions': joint_positions,
            'target_size': target_size,
            'target_position': target_position,
            'target_name': self.target_beaker,
            'beaker_1_position': beaker_1_position,
            'beaker_2_position': beaker_2_position,
            'plat_1_position': plat_1_position,
            'plat_2_position': plat_2_position,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'beaker_2': self.beaker_2,
            'beaker_1': self.beaker_1,
            'object_size': self.object_utils.get_object_size(object_path=self.beaker_2),
            'target_beaker': self.target_beaker,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
