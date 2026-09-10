from .base_task import BaseTask
from isaacsim.core.utils.prims import set_prim_visibility

class PickPourTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
        self.target_path = cfg.target_path

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        self.current_obj_path = self.place_objects_with_visibility_management(self.current_obj_idx)
        self.randomize_object_position(
            self.target_path,
            self.cfg.task.left_pos
        )

    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        source_quaternion = self.object_utils.get_transform_quat(object_path=self.current_obj_path+"/mesh")
        
        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            target_path=self.target_path,
            additional_info={
                'object_quaternion': source_quaternion,
                'source_beaker': self.current_obj_path,
            }
        )
        
