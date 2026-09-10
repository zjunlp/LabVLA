import numpy as np
from .base_task import BaseTask
from typing import Dict, Any

class OpenTransportPourTask(BaseTask):
    """Level 4 composite task: Open Door + Transfer Beaker + Stir
    
    This task consists of three stages:
    1. Door Opening Stage: Open the specified device door (drying oven or muffle furnace)
    2. Transfer Stage: Transfer the beaker from the table to the target platform
    3. Stirring Stage: Use a glass rod to stir in the beaker
    """
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the composite task
        
        Args:
            cfg: Task configuration
            world: Simulation world instance
            stage: USD stage
            robot: Robot instance
        """
        super().__init__(cfg, world, stage, robot)

        self.beaker_path = cfg.task.obj_paths[0]["path"]  # "/World/beaker2"
        self.target_plat_path = cfg.task.obj_paths[1]["path"]  # "/World/target_plat"
        
    def reset(self):
        """Reset task state and scene"""
        super().reset()
        self.robot.initialize()
        
        object_position = np.array([
                np.random.uniform(0.04, 0.05),
                np.random.uniform(0.32, 0.33),
                0.86,
            ])
        self.object_utils.set_object_position(object_path="/World/beaker2", position=object_position)

        object_position = np.array([
                np.random.uniform(0.11, 0.12),
                np.random.uniform(-0.41, -0.40),
                0.86,
            ])
        self.object_utils.set_object_position(object_path="/World/conical_bottle02", position=object_position)

        object_position = np.array([
                np.random.uniform(0.17, 0.18),
                np.random.uniform(-0.155, -0.145),
                0.86,
            ])
        self.object_utils.set_object_position(object_path="/World/beaker1", position=object_position)
        
        object_position = np.array([
                np.random.uniform(0.10, 0.11),
                np.random.uniform(-0.52, -0.51),
                0.775,
            ])
        self.object_utils.set_object_position(object_path="/World/target_plat2", position=object_position)
        
        object_position = np.array([
                np.random.uniform(0.03, 0.04),
                np.random.uniform(0.54, 0.55),
                0.775,
            ])
        self.object_utils.set_object_position(object_path="/World/target_plat", position=object_position)
        
        object_position = np.array([
                np.random.uniform(0.69, 0.70),
                np.random.uniform(0.09, 0.1),
                0.78,
            ])
        self.object_utils.set_object_position(object_path="/World/MuffleFurnace", position=object_position)
        
    def step(self) -> Dict[str, Any]:
        """Execute one step of the task
        
        Returns:
            Dict[str, Any]: Dictionary containing current state information
        """
        self.frame_idx += 1

        if not self.check_frame_limits(max_steps=self.cfg.task.max_steps):
            return None
        
        return self.get_basic_state_info(
            object_path=self.beaker_path,
            target_path=self.target_plat_path,
            additional_info={
            }
        ) 