import re
from typing import Optional
from controllers.atomic_actions.close_controller import CloseController
from robots.franka.rmpflow_controller import RMPFlowController
import numpy as np

from controllers.atomic_actions.open_controller import OpenController
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from .inference_engines.inference_engine_factory import InferenceEngineFactory

class OpenCloseTaskController(BaseController):
    """Controller for managing the task of opening and closing a drawer in collect or infer mode.

    Args:
        cfg: Configuration object containing mode and other parameters.
        robot: Robot articulation instance.
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_handle_position = None
        self.current_phase = "open"
        self.open_success = False
            
    def _init_collect_mode(self, cfg, robot):
        """Initializes the controller for data collection mode.

        Args:
            cfg: Configuration object for collect mode.
            robot: Robot articulation instance.
        """
        super()._init_collect_mode(cfg, robot)
        
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            furniture_type=self.cfg.task.get("operate_type"),
        )
        
        self.close_controller = CloseController(
            name="close_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            furniture_type=self.cfg.task.get("operate_type"),
            door_open_direction="clockwise"
        )
        
    def _init_infer_mode(self, cfg, robot):
        """
        Initializes components for inference mode.
        Creates inference engine and trajectory controller.

        Args:
            cfg: Configuration object containing model paths and settings
            robot: Robot instance to control
        """
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )

    def reset(self):
        """Resets the controller to its initial state."""
        super().reset()
        self.initial_handle_position = None
        self.current_phase = "open"
        self.open_success = False
        if self.mode == "collect":
            self.open_controller.reset()
            self.close_controller.reset()
        else:
            self.inference_engine.reset()

    def step(self, state):
        """Executes one step of the task based on the current state.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        self.state = state
        if self.initial_handle_position is None:
            self.initial_handle_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Executes a step in collect mode using the open and close controllers.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        if self.current_phase == "open":
            if not self.open_controller.is_done():
                if self.cfg.task.get("operate_type") == "door":
                    action = self.open_controller.forward(
                        handle_position=state['object_position'],
                        current_joint_positions=state['joint_positions'],
                        revolute_joint_position=state['revolute_joint_position'],
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False),
                    )
                else:
                    action = self.open_controller.forward(
                        handle_position=state['object_position'],
                        current_joint_positions=state['joint_positions'],
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=euler_angles_to_quats([90, 90, 0], degrees=True, extrinsic=False),
                    )
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
                
                if self._check_open_success(state):
                    self.check_success_counter += 1
                else:
                    self.check_success_counter = 0
                    
                return action, False, False

            self.open_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
            if self.open_success:
                print("Open phase success! Starting close phase...")
                self.current_phase = "close"
                self.close_controller.reset()
                self.check_success_counter = 0
                self.initial_handle_position = state['object_position']
                return None, False, False
            else:
                print("Open phase failed!")
                self.data_collector.clear_cache()
                self._last_success = False
                self.reset_needed = True
                return None, True, False

        elif self.current_phase == "close":
            if not self.close_controller.is_done():
                if self.cfg.task.get("operate_type") == "door":
                    action = self.close_controller.forward(
                        handle_position=state['object_position'],
                        current_joint_positions=state['joint_positions'],
                        revolute_joint_position=state['revolute_joint_position'],
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False),
                    )
                else:
                    action = self.close_controller.forward(
                        handle_position=state['object_position'],
                        current_joint_positions=state['joint_positions'],
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=euler_angles_to_quats([90, 90, 0], degrees=True, extrinsic=False),
                        push_distance=0.15
                    )
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
                
                if self._check_close_success(state):
                    self.check_success_counter += 1
                else:
                    self.check_success_counter = 0
                    
                return action, False, False

            close_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
            if close_success:
                print("Close phase success! Task completed!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
            else:
                print("Close phase failed!")
                self.data_collector.clear_cache()
                self._last_success = False
                
            self.reset_needed = True
            return None, True, close_success

    def _step_infer(self, state):
        """Executes a step in infer mode using the trained policy.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Open the door of the object"
        
        action = self.inference_engine.step_inference(state)
        
        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            print("Task success!")
            self._last_success = True
            self.reset_needed = True
            return None, True, True
            
        return action, False, False
        
    def _check_open_success(self, state):
        """Checks if the opening phase has been successfully completed.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the opening is successful, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        return (
            np.linalg.norm(np.array(current_pos) - self.initial_handle_position) > 0.13 and
            np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
        )
    
    def _check_close_success(self, state):
        """Checks if the closing phase has been successfully completed.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the closing is successful, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        operate_type = self.cfg.task.get("operate_type", "door")
        
        if operate_type == "drawer":
            return (
                np.linalg.norm(np.array(current_pos) - self.initial_handle_position) > 0.13 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
            )
        else:  # door
            return (
                np.array(current_pos)[0] - self.initial_handle_position[0] > 0.08 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.08
            )

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        operate_type = self.cfg.task.get("operate_type", "door")
        
        if self.current_phase == "open":
            self._language_instruction = f"Open the {operate_type} of the {object_name}"
        else:  # close phase
            self._language_instruction = f"Close the {operate_type} of the {object_name}"
        
        return self._language_instruction
