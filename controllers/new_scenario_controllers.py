# ============================================================================
# new_scenario_controllers.py: controllers for additional scenarios.
# Define controllers for five additional scenarios.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.open_controller import OpenController


class SpectrophotometerController(BaseController):
    """Spectrophotometer controller (additional scenario 1)."""
    
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

        self.pick_cuvette = PickController(
            name="pick_cuvette",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.006]
        )

        self.place_cuvette = PlaceController(
            name="place_in_spectrophotometer",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.005, 0.015, 0.05, 0.006, 0.018]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_cuvette.reset()
            self.place_cuvette.reset()
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
            action = self.pick_cuvette.forward(
                picking_position=state['cuvette_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.012, 0.012, 0.045],
                object_name="cuvette",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_distances=0.01
            )
            if self.pick_cuvette.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            action = self.place_cuvette.forward(
                place_position=state['spectrophotometer_position'] + np.array([0, 0, 0.05]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_cuvette.is_done():
                success = True
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        return "Pick up the cuvette and place it in the spectrophotometer sample chamber"


class PCRSetupController(BaseController):
    """PCR preparation controller (additional scenario 2)."""
    
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

        self.pick_tube = PickController(
            name="pick_pcr_tube",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.006]
        )

        self.place_tube = PlaceController(
            name="place_in_pcr",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.005, 0.015, 0.05, 0.006, 0.018]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_tube.reset()
            self.place_tube.reset()
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
            action = self.pick_tube.forward(
                picking_position=state['pipette_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.01, 0.01, 0.02],
                object_name="pcr_tube",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_distances=0.008
            )
            if self.pick_tube.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            action = self.place_tube.forward(
                place_position=state['pcr_machine_position'] + np.array([0, 0, 0.05]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_tube.is_done():
                success = True
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        return "Prepare PCR tubes with primers and template, then place them in the PCR machine"


class GelElectrophoresisController(BaseController):
    """Gel electrophoresis controller (additional scenario 3)."""
    
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

        self.pick_pipette = PickController(
            name="pick_micropipette",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.008]
        )

        self.place_pipette = PlaceController(
            name="place_pipette",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_pipette.reset()
            self.place_pipette.reset()
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
            action = self.pick_pipette.forward(
                picking_position=state['pipette_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.02, 0.02, 0.15],
                object_name="micropipette",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_distances=0.015
            )
            if self.pick_pipette.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            action = self.place_pipette.forward(
                place_position=state['gel_position'] + np.array([0, 0, 0.02]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_pipette.is_done():
                success = True
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        return "Use the micropipette to load samples into the gel wells for electrophoresis"


class pHMeasurementController(BaseController):
    """pH measurement controller (additional scenario 4)."""
    
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

        self.pick_beaker = PickController(
            name="pick_sample_beaker",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        self.place_beaker = PlaceController(
            name="place_under_electrode",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_beaker.reset()
            self.place_beaker.reset()
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
            action = self.pick_beaker.forward(
                picking_position=state['beaker_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.05, 0.05, 0.08],
                object_name="sample_beaker",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat()
            )
            if self.pick_beaker.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            action = self.place_beaker.forward(
                place_position=state['ph_meter_position'] + np.array([0, 0, -0.1]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker.is_done():
                success = True
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        return "Place the sample beaker under the pH electrode for measurement"


class AutoclaveSterilizationController(BaseController):
    """Autoclave sterilization controller (additional scenario 5)."""
    
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

        self.open_door = OpenController(
            name="open_autoclave_door",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.015, 0.02]
        )

        self.pick_tray = PickController(
            name="pick_sterilization_tray",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )

        self.place_tray = PlaceController(
            name="place_in_autoclave",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
        )

        self.close_door = OpenController(
            name="close_autoclave_door",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.015, 0.02]
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.open_door.reset()
            self.pick_tray.reset()
            self.place_tray.reset()
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
            action = self.open_door.forward(
                target_position=state['autoclave_position'] + np.array([0.2, 0, 0]),
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
            )
            if self.open_door.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            action = self.pick_tray.forward(
                picking_position=state['tray_position'],
                current_joint_positions=state['joint_positions'],
                object_size=[0.2, 0.15, 0.05],
                object_name="sterilization_tray",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat()
            )
            if self.pick_tray.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            action = self.place_tray.forward(
                place_position=state['autoclave_position'] + np.array([0, 0, 0.05]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_tray.is_done():
                self._current_step = 4

        elif self._current_step == 4:
            action = self.close_door.forward(
                target_position=state['autoclave_position'] + np.array([-0.1, 0, 0]),
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
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        return "Open the autoclave door, load the sterilization tray with items, and close the door for sterilization"
