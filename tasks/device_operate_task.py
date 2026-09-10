import numpy as np
from .base_task import BaseTask

class DeviceOperateTask(BaseTask):
    """
    Task class for operating a laboratory device with multiple steps:
    1. Open device door
    2. Pick up beaker
    3. Place beaker inside device
    4. Close device door
    5. Press device button
    """
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.device_path = cfg.task.device_path
        self.beaker_path = cfg.task.beaker_path
        self.beaker3_path = cfg.task.beaker3_path
        self.button_path = cfg.task.button_path
        self.interior_position = cfg.task.device_interior_position
        self.beaker3_target_position = cfg.task.beaker3_target_position
        self.initial_beaker_height = None
        
    def reset(self):
        """Reset task state and object positions."""
        self.world.reset()
        self.reset_needed = False
        self.frame_idx = 0
        self.robot.initialize()
        
        self.source_obj = self.cfg.task.obj_paths[0]['path']
        self.beaker3_obj = self.cfg.task.obj_paths[1]['path']
        self.target_obj = self.cfg.task.obj_paths[2]['path']
        
        source_position_range = self.cfg.task.obj_paths[0]['position_range']
        self.randomize_object_position(self.source_obj, source_position_range)
        
        beaker3_position_range = self.cfg.task.obj_paths[1]['position_range']
        self.randomize_object_position(self.beaker3_obj, beaker3_position_range)
        
        target_position_range = self.cfg.task.obj_paths[2]['position_range']
        self.randomize_object_position(self.target_obj, target_position_range)

    def step(self):
        """Execute one simulation step."""
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > self.cfg.task.max_steps:
            self.on_task_complete(True)
            self.reset_needed = True
            
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None

        door_handle_position = self.object_utils.get_geometry_center(
            object_path=f"{self.device_path}/handle"
        )
        beaker_position = self.object_utils.get_geometry_center(
            object_path=self.beaker_path
        )
        beaker_size = self.object_utils.get_object_size(
            object_path=self.beaker_path
        )
        beaker3_position = self.object_utils.get_geometry_center(
            object_path=self.beaker3_path
        )
        beaker3_size = self.object_utils.get_object_size(
            object_path=self.beaker3_path
        )
        button_position = self.object_utils.get_geometry_center(
            object_path=self.button_path
        )
        
        revolute_joint_position = self.object_utils.get_revolute_joint_positions(
            joint_path=f"{self.device_path}/RevoluteJoint"
        )
        
        button_pressed = self._check_button_pressed()
        
        camera_data, display_data = self.get_camera_data()

        basic_info = self.get_basic_state_info(
            joint_positions=joint_positions,
            object_path=self.beaker_path,
            additional_info={
                'object_name': self.beaker_path.split("/")[-1],
                'door_handle_position': door_handle_position,
                'beaker_position': beaker_position,
                'beaker_size': beaker_size,
                'beaker3_position': beaker3_position,
                'beaker3_size': beaker3_size,
                'beaker3_target_position': self.beaker3_target_position,
                'device_interior_position': self.interior_position,
                'button_position': button_position,
                'button_pressed': button_pressed,
                'revolute_joint_position': revolute_joint_position,
                'initial_beaker_height': self.initial_beaker_height,
                'gripper_position': self.robot.get_gripper_position(),
                'camera_display': display_data,
                'camera_data': camera_data,
                'done': self.reset_needed
            }
        )
        
        return basic_info
        
    def _check_button_pressed(self):
        """Check if the button has been pressed."""
        button_position = self.object_utils.get_geometry_center(object_path=self.button_path)
        initial_button_height = self.cfg.task.button_position[2]
        return button_position[2] < initial_button_height - 0.01  
