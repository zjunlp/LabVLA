import numpy as np
import random
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.pour_controller import PourController
from controllers.base_controller import BaseController
from controllers.atomic_actions.open_controller import OpenController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from robots.franka.rmpflow_controller import RMPFlowController
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

class TaskPhase(Enum):
    OPENING = "opening"
    PICKING1 = "picking1"
    TRANSPORTING = "transporting"
    PICKING2 = "picking2"
    POURING = "pouring"
    TRANSPORTING2 = "transporting2"
    FINISHED = "finished"

class OpenTransportPourController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.every_controller_index = 0
        
    def _generate_random_sequence(self):
        task_groups = [self.task_group_a, self.task_group_b, self.task_group_c]
        
        random.shuffle(task_groups)
        
        self.randomized_sequence = []
        for group in task_groups:
            self.randomized_sequence.extend(group)
            
        print(f"Generated random task sequence: {[phase.value for phase in self.randomized_sequence]}")
        
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008],
            furniture_type="door",
            door_open_direction="clockwise"
        )
        
        self.pick_controller1 = PickController(
            name="pick_controller", 
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 1, 0.05, 0.01, 1]
        )
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper
        )
        
        self.pick_controller2 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.02]
        )
        
        self.place_controller2 = PlaceController(
            name="place_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper
        )
        
        self.task_group_a = [TaskPhase.OPENING]
        self.task_group_b = [TaskPhase.PICKING1, TaskPhase.TRANSPORTING]
        self.task_group_c = [TaskPhase.PICKING2, TaskPhase.POURING, TaskPhase.TRANSPORTING2]
        
        self.randomized_sequence = []
        self._generate_random_sequence()
        self.current_phase = self.randomized_sequence[0]
        
        self._set_initial_active_controller()
        
    def _set_initial_active_controller(self):
        controller_map = {
            TaskPhase.OPENING: self.open_controller,
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.TRANSPORTING: self.place_controller,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.POURING: self.pour_controller,
            TaskPhase.TRANSPORTING2: self.place_controller2,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
        else:
            self.active_controller = self.open_controller
        
    def _init_infer_mode(self, cfg, robot):
        """Initialize inference mode"""
        super()._init_infer_mode(cfg, robot)
        
    def reset(self):
        super().reset()
        
        self.initial_beaker_position = None
        self.initial_door_position = None
        self.door_opened = False
        self.beaker_picked = False
        self.beaker_transported = False
        self.stir_step_count = 0
        self.every_controller_index = 0
        if self.mode == "collect":
            self._generate_random_sequence()
            self.current_phase = self.randomized_sequence[0]
            self.phase_start_frame = 0
            self.open_controller.reset()
            self.pick_controller1.reset()
            self.place_controller.reset()
            self.pick_controller2.reset()
            self.pour_controller.reset()
            self.place_controller2.reset()
            self._set_initial_active_controller()
        else:
            self.inference_engine.reset()
            
    def _check_phase_success(self, state: Dict[str, Any]) -> bool:
        if self.current_phase == TaskPhase.OPENING:
            end_effector_pos = state.get('gripper_position', np.array([0, 0, 0]))
            door_pos = state.get('door_position', np.array([0, 0, 0]))
            distance_to_door = np.linalg.norm(end_effector_pos[:2] - door_pos[:2])
            return distance_to_door < self.DOOR_OPEN_THRESHOLD
            
        elif self.current_phase == TaskPhase.PICKING:
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            if self.initial_beaker_position is not None:
                height_diff = beaker_pos[2] - self.initial_beaker_position[2]
                return height_diff > self.LIFT_HEIGHT_THRESHOLD
            return False
            
        elif self.current_phase == TaskPhase.TRANSPORTING:
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            target_pos = state.get('target_position', np.array([0, 0, 0]))
            distance_to_target = np.linalg.norm(beaker_pos[:2] - target_pos[:2])
            height_close = abs(beaker_pos[2] - target_pos[2]) < 0.1
            return distance_to_target < self.TRANSPORT_SUCCESS_THRESHOLD and height_close
            
        elif self.current_phase == TaskPhase.STIRRING:
            self.stir_step_count += 1
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            stir_tool_pos = state.get('stir_tool_position', np.array([0, 0, 0]))
            
            distance_to_beaker = np.linalg.norm(stir_tool_pos[:2] - beaker_pos[:2])
            in_beaker = distance_to_beaker < 0.05 and stir_tool_pos[2] < beaker_pos[2] + 0.1
            
            return self.stir_step_count > self.STIR_SUCCESS_STEPS and in_beaker
            
        return False
        
    def get_language_instruction(self) -> Optional[str]:
        phase_instructions = {
            TaskPhase.OPENING: "Open the door of the device",
            TaskPhase.PICKING1: "Pick up the beaker from the table", 
            TaskPhase.TRANSPORTING: "Transport the beaker to the target platform",
            TaskPhase.PICKING2: "Pick up the conical bottle",
            TaskPhase.POURING: "Pour the contents into the beaker",
            TaskPhase.TRANSPORTING2: "Transport the conical bottle to the target platform"
        }
        return phase_instructions.get(self.current_phase, "Complete the laboratory task")
        
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self.every_controller_index += 1
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = self._get_phase_action(state)
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
                print(f"{self.current_phase.value} phase successful! Switching to {next_phase.value} phase...")
                self.current_phase = next_phase
                self._switch_active_controller()
                return None, False, False
            else:
                print("All phases completed, task successful!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = TaskPhase.FINISHED
                return None, True, True
            
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in inference mode"""
        state['language_instruction'] = ""
        # Use inference engine to get action
        action = self.inference_engine.step_inference(state)
        
        return action, False, self.is_success()
        
    def _get_phase_action(self, state: Dict[str, Any]):
        """Get corresponding action based on current phase"""
        if self.current_phase == TaskPhase.OPENING:
            return self.open_controller.forward(
                handle_position=self.object_utils.get_geometry_center(object_path="/World/MuffleFurnace/handle"),
                revolute_joint_position=self.object_utils.get_revolute_joint_positions(
                    joint_path="/World/MuffleFurnace/RevoluteJoint"
                ),
                current_joint_positions=state['joint_positions'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False),
                angle=30,    
            )
            
        elif self.current_phase == TaskPhase.PICKING1:
            return self.pick_controller1.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker2"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker2"),
                object_name="beaker2",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 60])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0.05
            )   
        elif self.current_phase == TaskPhase.PICKING2:
            return self.pick_controller2.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/conical_bottle02"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/conical_bottle02"),
                object_name="conical_bottle02",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                pre_offset_x=0.07,
                pre_offset_z=0.05,
                after_offset_z=0.05
            )
        elif self.current_phase == TaskPhase.TRANSPORTING:
            return self.place_controller.forward(
                place_position=self.object_utils.get_geometry_center(object_path="/World/target_plat"),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 60])).as_quat(),
                gripper_position=state['gripper_position']
            )
        elif self.current_phase == TaskPhase.POURING:
            return self.pour_controller.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/conical_bottle02"),
                target_position=self.object_utils.get_geometry_center(object_path="/World/beaker1"),
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
                source_name="conical_bottle02",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
            )
        elif self.current_phase == TaskPhase.TRANSPORTING2:
            return self.place_controller2.forward(
                place_position=self.object_utils.get_geometry_center(object_path="/World/target_plat2"),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state['gripper_position'],
                place_offset_z=0.115
            )
        return None
        
    def _get_next_phase(self) -> Optional[TaskPhase]:
        """Get next phase (based on random sequence)"""
        try:
            current_idx = self.randomized_sequence.index(self.current_phase)
            if current_idx < len(self.randomized_sequence) - 1:
                return self.randomized_sequence[current_idx + 1]
        except ValueError:
            pass
        return None
        
    def _switch_active_controller(self):
        """Switch active controller based on current phase"""
        self.every_controller_index = 0
        controller_map = {
            TaskPhase.OPENING: self.open_controller,
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.TRANSPORTING: self.place_controller,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.POURING: self.pour_controller,
            TaskPhase.TRANSPORTING2: self.place_controller2,
            TaskPhase.FINISHED: self.open_controller,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
            self.active_controller.reset()
            
    def is_success(self) -> bool:
        return False