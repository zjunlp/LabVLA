# ============================================================================
# single_object_task.py: single-object task base class.
# Handle tasks involving one target object.
# Used by pick, open, and close tasks.
# ============================================================================

from .base_task import BaseTask  # Base task import.


class SingleObjectTask(BaseTask):
    """Base class for single-object tasks such as pick, open, and close. It simplifies object management and provides standard reset and step implementations."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize a single-object task from cfg, world, stage, and robot."""
        super().__init__(cfg, world, stage, robot)  # Initialize the base class.
        
    def on_task_complete(self, success):
        """Update object and material indices when an episode completes. success indicates whether the task succeeded."""
        self.update_object_and_material_indices(success)  # Advance the object and material cycle indices.
        
    def reset(self):
        """Reset the world, initialize the robot, apply the configured material, place the active object, and hide other objects before a new episode."""
        super().reset()  # Reset the world through the base task.
        self.robot.initialize()  # Initialize the robot state.
        
        # Apply the configured material to the target surface.
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
        
        # Place the active object and hide the others.
        # Return the path of the active object.
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx, far_distance=10.0
        )
        
    def step(self):
        """Advance one simulation frame, enforce frame limits, and return a state dictionary with joint positions, camera data, object position, and object name. Return None while the initial state is invalid."""
        self.frame_idx += 1  # Increment the frame counter.
        
        # Enforce the five-frame warmup and max_steps limit.
        if not self.check_frame_limits():
            return None
            
        # Return the base state.
        return self.get_basic_state_info(
            object_path=self.current_obj_path,  # Active object path.
            additional_info={
                'object_name': self.current_obj_path.split("/")[-1]  # Active object name.
            }
        )
