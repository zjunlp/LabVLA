# ============================================================================
# microscope_controller.py: microscope sample controller.
# Implement sample preparation and microscope placement.
# Level 4 long-sequence task.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController


class MicroscopeSampleController(BaseController):
    """Level 4 microscope controller: prepare a slide, add the sample, apply the coverslip, and place the slide in the microscope."""
    
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

        # 1. Pick up the slide.
        self.pick_slide = PickController(
            name="pick_slide",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.015, 0.05, 0.003, 0.006]
        )

        # 2. Use the dropper.
        self.pick_dropper = PickController(
            name="pick_dropper",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.008]
        )

        # 3. Put down the dropper.
        self.place_dropper = PlaceController(
            name="place_dropper",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 4. Pick up the coverslip.
        self.pick_coverslip = PickController(
            name="pick_coverslip",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.01, 0.05, 0.003, 0.005]
        )

        # 5. Apply the coverslip.
        self.place_coverslip = PlaceController(
            name="place_coverslip",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.002, 0.004, 0.01, 0.05, 0.005, 0.015]
        )

        # 6. Pick up the slide for microscope placement.
        self.pick_prepared_slide = PickController(
            name="pick_prepared_slide",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.015, 0.05, 0.003, 0.006]
        )

        # 7. Place it on the microscope stage.
        self.place_in_microscope = PlaceController(
            name="place_in_microscope",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.005, 0.015, 0.05, 0.006, 0.018]
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_slide.reset()
            self.pick_dropper.reset()
            self.place_dropper.reset()
            self.pick_coverslip.reset()
            self.place_coverslip.reset()
            self.pick_prepared_slide.reset()
            self.place_in_microscope.reset()
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
            # Add the sample with the dropper.
            action = self.pick_dropper.forward(
                picking_position=state['dropper_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.015, 0.015, 0.08],
                object_name="dropper",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_distances=0.012
            )
            if self.pick_dropper.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Move the dropper above the slide and put it down.
            action = self.place_dropper.forward(
                place_position=state['slide_position'] + np.array([0, 0, 0.02]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_dropper.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            # Pick up the coverslip.
            action = self.pick_coverslip.forward(
                picking_position=state['coverslip_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.02, 0.02, 0.001],
                object_name="coverslip",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_distances=0.008
            )
            if self.pick_coverslip.is_done():
                self._current_step = 4

        elif self._current_step == 4:
            # Apply the coverslip.
            action = self.place_coverslip.forward(
                place_position=state['slide_position'] + np.array([0, 0, 0.001]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_coverslip.is_done():
                self._current_step = 5

        elif self._current_step == 5:
            # Pick up the prepared slide.
            action = self.pick_prepared_slide.forward(
                picking_position=state['slide_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.075, 0.025, 0.001],
                object_name="glass_slide",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_distances=0.01
            )
            if self.pick_prepared_slide.is_done():
                self._current_step = 6

        elif self._current_step == 6:
            # Place it on the microscope stage.
            action = self.place_in_microscope.forward(
                place_position=state['microscope_position'] + np.array([0, 0, 0.05]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_in_microscope.is_done():
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
        self._language_instruction = "Prepare a microscope slide by adding sample with a dropper, covering with a coverslip, and placing it on the microscope stage"
        return self._language_instruction
