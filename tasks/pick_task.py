from .single_object_task import SingleObjectTask

class PickTask(SingleObjectTask):
    """
    A task class for robotic picking operations.
    Manages object placement, material switching, and task state transitions.
    """
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
