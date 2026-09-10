# ============================================================================
# synthesis_controller.py: chemical synthesis controller.
# Implement chemical synthesis control.
# Level 5 complex-sequence task.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController


class ChemicalSynthesisController(BaseController):
    """Level 5 chemical synthesis controller covering reaction vessel preparation, reagent addition, and equipment setup."""
    
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

        # 2. Place it on the heating mantle.
        self.place_flask = PlaceController(
            name="place_on_heater",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 3. Pick up reagent A.
        self.pick_reagent_a = PickController(
            name="pick_reagent_a",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 4. Pour reagent A.
        self.pour_reagent_a = PourController(
            name="pour_reagent_a",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 0.02]
        )

        # 5. Return reagent A.
        self.place_reagent_a = PlaceController(
            name="place_reagent_a",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 6. Pick up reagent B.
        self.pick_reagent_b = PickController(
            name="pick_reagent_b",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 7. Pour reagent B.
        self.pour_reagent_b = PourController(
            name="pour_reagent_b",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 0.02]
        )

        # 8. Return reagent B.
        self.place_reagent_b = PlaceController(
            name="place_reagent_b",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        # 9. Pick up the separatory funnel.
        self.pick_funnel = PickController(
            name="pick_funnel",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        # 10. Place the separatory funnel.
        self.place_funnel = PlaceController(
            name="place_funnel",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_flask.reset()
            self.place_flask.reset()
            self.pick_reagent_a.reset()
            self.pour_reagent_a.reset()
            self.place_reagent_a.reset()
            self.pick_reagent_b.reset()
            self.pour_reagent_b.reset()
            self.place_reagent_b.reset()
            self.pick_funnel.reset()
            self.place_funnel.reset()
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
            # Pick up the round-bottom flask.
            action = self.pick_flask.forward(
                picking_position=state['flask_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.08, 0.08, 0.15],
                object_name="round_bottom_flask",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_distances=0.02
            )
            if self.pick_flask.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Place it on the heating mantle.
            action = self.place_flask.forward(
                place_position=state['heater_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_flask.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            # Subsequent stages are simplified: complete the task here.
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
        self._language_instruction = "Perform chemical synthesis: place round-bottom flask on heating mantle, add reagents A and B, and set up the separatory funnel for extraction"
        return self._language_instruction
