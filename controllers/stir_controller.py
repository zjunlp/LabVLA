from typing import Optional
import numpy as np

from isaacsim.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController
class StirTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            position_threshold=0.005,
            events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
        )
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=self.rmp_controller,
        )

        self.gripper_control.release_object()
        self._last_pick_joint_data = None

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt = [0.004, 0.002, 0.005, 1, 0.05, 0.004, 1]
        )
        self.use_stir_model = False
        self.frame_count = 0
        
    def reset(self):
        super().reset()
        self.gripper_control.release_object()
        self.pick_controller.reset()
        if self.mode == "collect":
            self.stir_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
        self.use_stir_model = False
        self.frame_count = 0
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                after_offset_z= 0.15,
                gripper_distances=0.005
            )
            
            self.gripper_control.update_grasped_object_position()

            return action, False, False
            
        elif not self.stir_controller.is_done():
            target_position = self.object_utils.get_object_xform_position(
                object_path=state['target_beaker']
            )
            if target_position is None:
                target_position = state['target_position']
            
            action = self.stir_controller.forward(
                center_position=target_position,
                current_joint_positions=state['joint_positions'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )

            self.gripper_control.update_grasped_object_position()
                
            return action, False, False
            
        else:
            self.reset_needed = True
            final_object_position = state['glass_rod_position']
            target_position = state['target_position']
            self.gripper_control.release_object()
            if (final_object_position is not None and 
                final_object_position[2] > 0.85 and
                np.linalg.norm(final_object_position[0:2] - target_position[0:2]) < 0.04):
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                return None, True, True
            else:
                self.data_collector.clear_cache()
                self._last_success = False
                return None, True, False

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Processes observations and generates control actions using learned policy.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                after_offset_z=0.15,
            )
            
            final_object_position = state['glass_rod_position']
            if final_object_position is not None and final_object_position[2] > 0.82:
                self.use_stir_model = True

            self.gripper_control.update_grasped_object_position()

            return action, False, False
        
        state['language_instruction'] = self.get_language_instruction()
        
        if self.use_stir_model:
            action = self.inference_engine.step_inference(state)
            self.gripper_control.update_grasped_object_position()

            return action, False, self._check_success()
        
        return ArticulationAction(), False, False
    
    def _check_success(self):
        object_pos = self.state['glass_rod_position']
        target_position = self.state['target_position']
        if object_pos[2] > 0.85 and np.linalg.norm(object_pos[0:2] - target_position[0:2]) < 0.04:
            self.check_success_counter += 1
            if self.check_success_counter > 240:
                self._last_success = True
                return True
        return False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        # Default instruction for shake tasks
        self._language_instruction = "Use the glass rod to stir the liquid."
        return self._language_instruction
