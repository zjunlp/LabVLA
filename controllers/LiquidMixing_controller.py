import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.pour_controller import PourController
from controllers.atomic_actions.pressZ_controller import PressZController
from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from robots.franka.rmpflow_controller import RMPFlowController
class TaskPhase(Enum):
    """Task phase enumeration"""
    PICKING1 = "picking1"        # Door opening stage
    PICKING2 = "picking2"        # Picking beaker stage
    PICKING3 = "picking3"      # Picking conical bottle stage
    POURING1 = "pouring1"
    POURING2 = "pouring2"
    POURING3 = "pouring3"
    PLACEING1 = "placing1"
    PLACEING2 = "placing2"
    PLACEING3 = "placing3"
    PRESS = "press"
    FINISHED = "finished"      # Task completed
    

class LiquidMixingController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.every_controller_index = 0
        self.controller_index = 0
        self.initial_beaker_position1 = None
        self.initial_beaker_position2 = None
        self.initial_beaker_position3 = None
        
    def _init_collect_mode(self, cfg, robot):
        """Initialize data collection mode"""
        super()._init_collect_mode(cfg, robot)
        
        self.current_phase = TaskPhase.PICKING1
        
        # Create RMP controller
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.pick_controller1 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 1, 0.05, 0.01, 1]
        )
        
        self.pour_controller1 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02
        )
        
        self.place_controller1 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        
        self.pick_controller2 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )
        
        self.pour_controller2 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02
        )
        
        self.place_controller2 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        
        self.pick_controller3 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )
        
        self.pour_controller3 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02
        )
        
        self.place_controller3 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        
        self.press_controller = PressZController(
            name="press_controller",
            cspace_controller=RMPFlowController(
                name="press_controller",
                robot_articulation=robot
            ),
            events_dt=[0.004, 0.1, 0.01],
        )
        
        self.active_controller = self.pick_controller1
        
    def _set_initial_active_controller(self):
        """Set the initial active controller based on the current phase"""
        controller_map = {
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.PICKING3: self.pick_controller3,
            TaskPhase.POURING1: self.pour_controller1,
            TaskPhase.POURING2: self.pour_controller2,
            TaskPhase.POURING3: self.pour_controller3,
            TaskPhase.PLACEING1: self.place_controller1,
            TaskPhase.PLACEING2: self.place_controller2,
            TaskPhase.PLACEING3: self.place_controller3,
            TaskPhase.PRESS: self.press_controller,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
        else:
            self.active_controller = self.pick_controller1
        
    def _init_infer_mode(self, cfg, robot):
        """Initialize inference mode"""
        super()._init_infer_mode(cfg, robot)
        
    def reset(self):
        """Reset controller state"""
        super().reset()
        self.initial_beaker_position1 = None
        self.initial_beaker_position2 = None
        self.initial_beaker_position3 = None
        self.every_controller_index = 0
        self.current_phase = TaskPhase.PICKING1
        if self.mode == "collect":
            self.phase_start_frame = 0
            self.pick_controller1.reset()
            self.pour_controller1.reset()
            self.place_controller1.reset()
            self.pick_controller2.reset()
            self.pour_controller2.reset()
            self.place_controller2.reset()
            self.pick_controller3.reset()
            self.pour_controller3.reset()
            self.place_controller3.reset()
            self.press_controller.reset()
            self.controller_index = 0
            self.active_controller = self.pick_controller1
        else:
            self.inference_engine.reset()
            
    def _check_phase_success(self, state: Dict[str, Any]) -> bool:
        """Check if the current phase is successfully completed
        
        Args:
            state: Current state dictionary
            
        Returns:
            bool: Whether the current phase is successful
        """
        if self.current_phase == TaskPhase.OPENING:
            end_effector_pos = state.get('gripper_position', np.array([0, 0, 0]))
            door_pos = state.get('door_position', np.array([0, 0, 0]))
            distance_to_door = np.linalg.norm(end_effector_pos[:2] - door_pos[:2])
            return distance_to_door < self.DOOR_OPEN_THRESHOLD
            
        elif self.current_phase == TaskPhase.PICKING:
            # Check if the beaker is picked up
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            if self.initial_beaker_position is not None:
                height_diff = beaker_pos[2] - self.initial_beaker_position[2]
                return height_diff > self.LIFT_HEIGHT_THRESHOLD
            return False
            
        elif self.current_phase == TaskPhase.TRANSPORTING:
            # Check if the beaker is successfully placed on the target platform
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            target_pos = state.get('target_position', np.array([0, 0, 0]))
            distance_to_target = np.linalg.norm(beaker_pos[:2] - target_pos[:2])
            height_close = abs(beaker_pos[2] - target_pos[2]) < 0.1
            return distance_to_target < self.TRANSPORT_SUCCESS_THRESHOLD and height_close
            
        elif self.current_phase == TaskPhase.STIRRING:
            # Check if the stirring is completed (based on the number of stirring steps and the position of the glass rod)
            self.stir_step_count += 1
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            stir_tool_pos = state.get('stir_tool_position', np.array([0, 0, 0]))
            
            # Check if the glass rod is near the beaker
            distance_to_beaker = np.linalg.norm(stir_tool_pos[:2] - beaker_pos[:2])
            in_beaker = distance_to_beaker < 0.05 and stir_tool_pos[2] < beaker_pos[2] + 0.1
            
            return self.stir_step_count > self.STIR_SUCCESS_STEPS and in_beaker
            
        return False
        
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current phase
        
        Returns:
            Optional[str]: Language instruction for the current phase
        """
        phase_instructions = {
            TaskPhase.PICKING1: "Pick up the beaker from the table", 
            TaskPhase.PICKING2: "Pick up the conical bottle",
            TaskPhase.PICKING3: "Pick up the beaker from the table",
            TaskPhase.POURING1: "Pour the contents into the beaker",
            TaskPhase.POURING2: "Pour the contents into the beaker",
            TaskPhase.POURING3: "Pour the contents into the beaker",
            TaskPhase.PLACEING1: "Place the beaker on the table",
            TaskPhase.PLACEING2: "Place the conical bottle on the table",
            TaskPhase.PLACEING3: "Place the beaker on the table",
            TaskPhase.PRESS: "Press the beaker"
        }
        return phase_instructions.get(self.current_phase, "Complete the laboratory task")
        
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control
        
        Args:
            state: Current state dictionary
            
        Returns:
            Tuple: (action, done, success)
        """
        if self.initial_beaker_position1 is None:
            self.initial_beaker_position1 = self.object_utils.get_geometry_center(object_path="/World/beaker_05")
        if self.initial_beaker_position2 is None:
            self.initial_beaker_position2 = self.object_utils.get_geometry_center(object_path="/World/beaker_04")
        if self.initial_beaker_position3 is None:
            self.initial_beaker_position3 = self.object_utils.get_geometry_center(object_path="/World/beaker_03")
        
        self.every_controller_index += 1
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in data collection mode"""
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        # If the current controller is not completed, continue to execute
        if not self.active_controller.is_done():
            action = self._get_phase_action(state)
            
            # Cache data for training
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
                
            return action, False, False
        else:
            next_phase = self._get_next_phase()
            if next_phase:
                self.current_phase = next_phase
                self._switch_active_controller()
                return None, False, False
            else:
                # All phases completed
                print("All phases completed, task successful!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = TaskPhase.FINISHED
                return None, True, True
            
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in inference mode"""
        state['language_instruction'] = ""
        # Use the inference engine to get the action
        action = self.inference_engine.step_inference(state)
        
        # Check if the task is successful (simplified version, actually may need more complex logic)
        return action, False, self.is_success()
        
    def _get_phase_action(self, state: Dict[str, Any]):
        """Get the corresponding action based on the current phase"""
        if self.current_phase == TaskPhase.PICKING1:
            return self.pick_controller1.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_05"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_05"),
                object_name="beaker_05",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0
            )   
        elif self.current_phase == TaskPhase.PICKING2:
            return self.pick_controller2.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_04"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_04"),
                object_name="beaker_04",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                pre_offset_x=0.07,
                pre_offset_z=0.05,
                after_offset_z=0
            )
        elif self.current_phase == TaskPhase.PICKING3:
            return self.pick_controller3.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_03"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_03"),
                object_name="beaker_03",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0
            )
        elif self.current_phase == TaskPhase.POURING1:
            return self.pour_controller1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_05"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
                source_name="beaker_05",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
        elif self.current_phase == TaskPhase.POURING2:
            return self.pour_controller2.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_04"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
                source_name="beaker_04",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
        elif self.current_phase == TaskPhase.POURING3:
            return self.pour_controller3.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_03"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
                source_name="beaker_03",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
        elif self.current_phase == TaskPhase.PLACEING1:
            return self.place_controller1.forward(
                place_position=self.initial_beaker_position1,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_place_z=0.3,
                place_offset_z=0.02
            )
        elif self.current_phase == TaskPhase.PLACEING2:
            return self.place_controller2.forward(
                place_position=self.initial_beaker_position2,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_place_z=0.3,
                place_offset_z=0.02
            )
        elif self.current_phase == TaskPhase.PLACEING3:
            return self.place_controller3.forward(
                place_position=self.initial_beaker_position3,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                pre_place_z=0.3,
                gripper_position=state['gripper_position'],
                place_offset_z=0.02
            )
        elif self.current_phase == TaskPhase.PRESS:
            return self.press_controller.forward(
                target_position=self.object_utils.get_object_xform_position(object_path="/World/heat_device/button"),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 40])).as_quat(),
                gripper_position=state['gripper_position']
            )
        return None
        
    def _get_next_phase(self) -> Optional[TaskPhase]:
        phase_sequence = [
            TaskPhase.PICKING1,
            TaskPhase.POURING1,
            TaskPhase.PLACEING1,
            TaskPhase.PICKING2, 
            TaskPhase.POURING2,
            TaskPhase.PLACEING2,
            TaskPhase.PICKING3,
            TaskPhase.POURING3,
            TaskPhase.PLACEING3,
            TaskPhase.PRESS
        ]
        
        self.controller_index += 1
        if self.controller_index >= len(phase_sequence):
            self.controller_index = 0
            return None
            
        # Update the current phase and return
        self.current_phase = phase_sequence[self.controller_index]
        return self.current_phase
        
    def _switch_active_controller(self):
        """Switch the active controller based on the current phase"""
        self.every_controller_index = 0
        controller_map = {
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.PICKING3: self.pick_controller3,
            TaskPhase.POURING1: self.pour_controller1,
            TaskPhase.POURING2: self.pour_controller2,
            TaskPhase.POURING3: self.pour_controller3,
            TaskPhase.PLACEING1: self.place_controller1,
            TaskPhase.PLACEING2: self.place_controller2,
            TaskPhase.PLACEING3: self.place_controller3,
            TaskPhase.PRESS: self.press_controller,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
            self.active_controller.reset()
            
    def is_success(self) -> bool:
        return False