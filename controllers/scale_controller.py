# ============================================================================
# scale_controller.py: weighing controller.
# Place an object on the electronic scale for weighing.
# Level 2 combined task.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController


class ScaleMeasureController(BaseController):
    """Level 2 weighing controller. Pick up an object and place it gently on the scale for an accurate reading."""
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._current_step = 1
        self.frame_count = 0
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
    
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)

        # 1. Pick up the object.
        self.pick_item = PickController(
            name="pick_item",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 2. Place it gently on the scale.
        self.place_item = PlaceController(
            name="place_on_scale",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.004, 0.015, 0.05, 0.006, 0.02]  # Slower placement speed.
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_item.reset()
            self.place_item.reset()
        else:
            self.inference_engine.reset()
        
        self._current_step = 1
        self.frame_count = 0
    
    def step(self, state):
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state):
        action = None
        done = False
        success = False
        
        if 'camera_data' in state:
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1],
                language_instruction=self.get_language_instruction()
            )

        if self._current_step == 1:
            # Pick up the object.
            action = self.pick_item.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state.get('object_size', [0.05, 0.05, 0.08]),
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                pre_offset_x=0.05,
                after_offset_z=0.2
            )
            if self.pick_item.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Place it on the scale.
            scale_pos = state['scale_position']
            place_pos = scale_pos + np.array([0, 0, 0.05])  # Slightly above the scale surface.
            
            action = self.place_item.forward(
                place_position=place_pos,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_item.is_done():
                success = self._check_success(state)
                if success:
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                else:
                    self.data_collector.clear_cache()
                    self._last_success = False
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def _check_success(self, state):
        # Check whether the object is on the scale.
        obj_pos = state['object_position']
        scale_pos = state['scale_position']
        # Check the horizontal distance tolerance.
        horizontal_dist = np.linalg.norm(obj_pos[:2] - scale_pos[:2])
        # Check that the height is consistent with the scale surface.
        height_ok = abs(obj_pos[2] - scale_pos[2]) < 0.1
        return horizontal_dist < 0.05 and height_ok
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Pick up the beaker and place it gently on the electronic scale for measurement"
        return self._language_instruction
