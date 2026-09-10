# ============================================================================
# incubator_controller.py: incubator loading controller.
# Implement Petri dish loading into the incubator.
# Level 3 task of moderate complexity.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.open_controller import OpenController


class IncubatorLoadController(BaseController):
    """Level 3 incubator loading controller: open the door, pick up the dish, place it inside, and close the door."""
    
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

        # 1. Open the incubator door.
        self.open_door = OpenController(
            name="open_incubator_door",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.01, 0.02]
        )

        # 2. Pick up the Petri dish.
        self.pick_dish = PickController(
            name="pick_petri_dish",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 3. Place it inside the incubator.
        self.place_dish = PlaceController(
            name="place_in_incubator",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 4. Close the incubator door (reuse OpenController).
        self.close_door = OpenController(
            name="close_incubator_door",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.01, 0.02]
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.open_door.reset()
            self.pick_dish.reset()
            self.place_dish.reset()
            self.close_door.reset()
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
            # Open the incubator door.
            action = self.open_door.forward(
                target_position=state['incubator_position'] + np.array([0.1, 0, 0.1]),
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
            )
            if self.open_door.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Pick up the Petri dish.
            action = self.pick_dish.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.06, 0.06, 0.015],
                object_name="petri_dish",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                pre_offset_z=0.05
            )
            if self.pick_dish.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            # Place it on the incubator shelf.
            action = self.place_dish.forward(
                place_position=state['shelf_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_dish.is_done():
                self._current_step = 4

        elif self._current_step == 4:
            # Close the incubator door.
            action = self.close_door.forward(
                target_position=state['incubator_position'] + np.array([-0.1, 0, 0.1]),
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
            )
            if self.close_door.is_done():
                success = True
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Open the incubator door, pick up the petri dish, place it on the shelf inside the incubator, and close the door"
        return self._language_instruction
