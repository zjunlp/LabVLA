# ============================================================================
# pipette_pick_task.py: pipette picking task.
# Pick a pipette from its rack.
# Level 1 atomic task.
# ============================================================================

from .single_object_task import SingleObjectTask


class PipettePickTask(SingleObjectTask):
    """Pipette picking task derived from SingleObjectTask. This Level 1 atomic task picks a pipette from its rack using a precise handle grasp and accounts for the object's elongated shape."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the pipette picking task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)


class TestTubePickTask(SingleObjectTask):
    """Test tube picking task derived from SingleObjectTask. This Level 1 atomic task extracts a fragile tube vertically from its rack with gentle handling."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the test tube picking task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)
