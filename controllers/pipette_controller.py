# ============================================================================
# pipette_controller.py: pipette picking controller.
# Implement pipette picking control.
# Level 1 atomic task.
# ============================================================================

import re
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController


class PipettePickController(BaseController):
    """Pick a pipette from its rack using a grasp strategy for a narrow, elongated object."""
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        # Create a pipette-specific pick controller.
        # Adjust phase increments for the pipette's elongated shape.
        self.pick_controller = PickController(
            name="pipette_pick",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.006]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        return self.state['object_position'][2] > self.initial_position[2] + 0.08
        
    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.pick_controller.is_done():
            # Use a pipette-specific grasp angle.
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_offset_x=0.03,
                pre_offset_z=0.05,
                gripper_distances=0.015  # The pipette is narrow.
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True

        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False
        
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
            
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Pick up the pipette from the pipette stand"
        return self._language_instruction


class TestTubePickController(BaseController):
    """Pick a fragile test tube from its rack with gentle handling."""
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="testtube_pick",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.015, 0.05, 0.003, 0.008]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        return self.state['object_position'][2] > self.initial_position[2] + 0.1
        
    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_offset_z=0.08,  # Approach from above.
                gripper_distances=0.012  # The test tube is narrow.
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True

        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False
        
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
            
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Pick up the test tube from the tube rack"
        return self._language_instruction
