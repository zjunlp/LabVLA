import numpy as np
import random
from tasks.base_task import BaseTask
from lab_utils.Material_utils import bind_material_to_object

class PickPlaceTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)

        self.table_path = self.cfg.table_path

        self.table_material_paths = self.cfg.table_material_paths
        self.button_material_paths = self.cfg.button_material_paths
        self.source_beaker = self.cfg.task.obj_paths[0]['path']
        self.target_plat = self.cfg.task.obj_paths[1]['path']

        self.num_episode = 0
        self.per_episode = self.cfg.max_episodes // self.cfg.material_types
        self.material_types = self.cfg.material_types
        self.button_types = self.cfg.button_types

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        obj_position_range = self.cfg.task.obj_paths[0]['position_range'] 
        object_position = np.array([
                        np.random.uniform(obj_position_range['x'][0], obj_position_range['x'][1]),
                        np.random.uniform(obj_position_range['y'][0], obj_position_range['y'][1]),
                        obj_position_range['z'][0]
                    ])
        self.object_utils.set_object_position(object_path=self.source_beaker, position=object_position)

        target_position_range = self.cfg.task.obj_paths[1]['position_range']
        target_position = np.array([
                        np.random.uniform(target_position_range['x'][0], target_position_range['x'][1]),
                        np.random.uniform(target_position_range['y'][0], target_position_range['y'][1]),
                        target_position_range['z'][0]
                    ])
        self.object_utils.set_object_position(object_path=self.target_plat, position=target_position)

        random_material_path = random.choice(self.button_material_paths[:self.button_types])
        bind_material_to_object(stage=self.stage,
                                obj_path=self.cfg.target_sub_path,
                                material_path=random_material_path)
        
        table_material_index = self.num_episode % self.material_types
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table_path,
                                material_path=self.table_material_paths[table_material_index])
        
        self.num_episode += 1
   

    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        
        return self.get_basic_state_info(
            object_path=self.source_beaker,
            target_path=self.target_plat,
        )
