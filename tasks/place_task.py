from .dual_object_task import DualObjectTask

class PlaceTask(DualObjectTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Place task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
