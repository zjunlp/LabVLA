# ============================================================================
# rotavap_controller.py: rotary evaporator setup controller.
# Mount a round-bottom flask on the rotary evaporator.
# Level 3 task of moderate complexity.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController


class RotavapSetupController(BaseController):
    """Level 3 rotary evaporator setup controller. Precisely align the round-bottom flask with the mounting position."""
    
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

        # 1. Pick up the round-bottom flask.
        self.pick_flask = PickController(
            name="pick_round_flask",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 2. Mount it on the rotary evaporator.
        self.mount_flask = PlaceController(
            name="mount_on_rotavap",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.005, 0.015, 0.05, 0.008, 0.02]
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_flask.reset()
            self.mount_flask.reset()
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
            # Grasp the round-bottom flask by its neck.
            action = self.pick_flask.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.08, 0.08, 0.12],
                object_name="round_bottom_flask",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_z=0.08,
                gripper_distances=0.02  # The flask neck is narrow.
            )
            if self.pick_flask.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Move to the rotary evaporator mounting position.
            # Align at the required angle.
            mount_pos = state['holder_position']
            
            action = self.mount_flask.forward(
                place_position=mount_pos,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([-45, 90, 30])).as_quat(),  # Tilt angle.
                gripper_position=state['gripper_position']
            )
            if self.mount_flask.is_done():
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
        # Check whether the flask is near the mounting position.
        flask_pos = state['object_position']
        holder_pos = state['holder_position']
        distance = np.linalg.norm(flask_pos - holder_pos)
        return distance < 0.08
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Pick up the round-bottom flask and mount it on the rotary evaporator holder at the correct angle"
        return self._language_instruction
