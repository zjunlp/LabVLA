# ============================================================================
# fumehood_controller.py: fume hood task controller.
# Implement operations inside the fume hood.
# Level 4 long-sequence task.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.open_controller import OpenController


class FumehoodOperationController(BaseController):
    """Level 4 fume hood controller: open the sash, place the flask inside, retrieve it, and close the sash."""
    
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

        # 1. Open the fume hood sash.
        self.open_sash = OpenController(
            name="open_fumehood_sash",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.015, 0.02]
        )

        # 2. Pick up the flask.
        self.pick_flask = PickController(
            name="pick_flask",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 3. Place it inside the fume hood.
        self.place_flask = PlaceController(
            name="place_in_fumehood",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 4. Retrieve the flask.
        self.retrieve_flask = PickController(
            name="retrieve_flask",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 5. Place the flask down.
        self.place_flask_out = PlaceController(
            name="place_flask_out",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 6. Close the fume hood sash.
        self.close_sash = OpenController(
            name="close_fumehood_sash",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.015, 0.02]
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.open_sash.reset()
            self.pick_flask.reset()
            self.place_flask.reset()
            self.retrieve_flask.reset()
            self.place_flask_out.reset()
            self.close_sash.reset()
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
            # Open the fume hood sash.
            action = self.open_sash.forward(
                target_position=state['fumehood_position'] + np.array([0, 0, 0.5]),
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
            )
            if self.open_sash.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Pick up the flask.
            action = self.pick_flask.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.06, 0.06, 0.12],
                object_name="erlenmeyer_flask",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat()
            )
            if self.pick_flask.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            # Place it on the fume hood work surface.
            action = self.place_flask.forward(
                place_position=state['surface_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_flask.is_done():
                self._current_step = 4

        elif self._current_step == 4:
            # Retrieve the flask.
            action = self.retrieve_flask.forward(
                picking_position=state['surface_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.06, 0.06, 0.12],
                object_name="erlenmeyer_flask",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat()
            )
            if self.retrieve_flask.is_done():
                self._current_step = 5

        elif self._current_step == 5:
            # Return the flask to its original position.
            action = self.place_flask_out.forward(
                place_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_flask_out.is_done():
                self._current_step = 6

        elif self._current_step == 6:
            # Close the fume hood sash.
            action = self.close_sash.forward(
                target_position=state['fumehood_position'] + np.array([0, 0, 0.1]),
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
            )
            if self.close_sash.is_done():
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
        self._language_instruction = "Open the fume hood sash, place the flask inside for operation, then retrieve it and close the sash"
        return self._language_instruction
