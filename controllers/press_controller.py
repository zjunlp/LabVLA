from typing import Optional
import numpy as np

from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.press_controller import PressController

class PressTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot, use_default_config=False)
        
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt = [0.005, 0.1, 0.005]  # Default phase durations
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.press_controller.reset()
        else:
            self.inference_engine.reset()
    
    def step(self, state):
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        return final_object_position is not None and final_object_position[0] > 0.405

    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.press_controller.is_done():
            action = self.press_controller.forward(
                target_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
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
        else:
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
        if not self._language_instruction:
            return "Press the different color button"
        return self._language_instruction
