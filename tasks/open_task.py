from .single_object_task import SingleObjectTask

class OpenTask(SingleObjectTask):
    """
    A task class for robotic opening operations.
    Manages door/drawer opening, material switching, and task state transitions.
    """
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        
    def reset(self):
        """
        Resets the task state.
        Initializes robot position, updates materials, and places objects.
        """
        super().reset()
        
        # Set sub-object path (handle)
        if self.cfg.get("handle_path"):
            self.current_sub_obj_path = self.cfg.get("handle_path")
        else:
            self.current_sub_obj_path = self.current_obj_path + "/handle"
        
    def step(self):
        """
        Executes one simulation step and returns current state.
        
        Returns:
            dict: Current state dictionary containing:
                - joint_positions: Robot joint positions
                - object_position: Target object position
                - object_size: Target object dimensions
                - camera_data: Camera image data
                - done: Whether episode is complete
                - object_name: Name of current target object
                - gripper_position: End effector position
                - revolute_joint_position: Joint angle of the opening mechanism
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits():
            return None
        
        # Get position and size of sub-object (handle)
        object_position = self.object_utils.get_geometry_center(object_path=self.current_sub_obj_path)
        object_size = self.object_utils.get_object_size(object_path=self.current_sub_obj_path)
        
        # Get close_gripper_distance from current object config
        close_gripper_distance = self.obj_configs[self.current_obj_idx].get('close_gripper_distance', 0.023)
        
        if self.cfg.task.get("operate_type") == "door":
            return self.get_basic_state_info(
                object_path=self.current_obj_path,
                additional_info={
                    'object_position': object_position,
                    'object_size': object_size,
                    'revolute_joint_position': self.object_utils.get_revolute_joint_positions(
                        joint_path=self.current_obj_path+"/RevoluteJoint"
                    ),
                    'close_gripper_distance': close_gripper_distance
                }
            )
        else:
            return self.get_basic_state_info(
                object_path=self.current_obj_path,
                additional_info={
                    'object_position': object_position,
                    'object_size': object_size,
                    'close_gripper_distance': close_gripper_distance
                }
            )
