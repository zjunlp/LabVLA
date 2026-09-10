# ============================================================================
# centrifuge_controller.py: centrifuge task controllers.
# Implement sample loading and the full centrifuge workflow.
# Level 2 and Level 5 tasks.
# ============================================================================

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController


class CentrifugeLoadController(BaseController):
    """Level 2 centrifuge loading controller. Pick up a test tube and place it in the centrifuge."""
    
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

        # 1. Pick up the test tube.
        self.pick_tube = PickController(
            name="pick_tube",
            cspace_controller=self.rmp_controller,
            events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.008]
        )

        # 2. Place it in the centrifuge.
        self.place_tube = PlaceController(
            name="place_tube",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
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
            # Pick up the test tube.
            action = self.pick_tube.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state.get('object_size', [0.02, 0.02, 0.1]),
                object_name="test_tube",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                gripper_distances=0.015
            )
            if self.pick_tube.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # Place it in the centrifuge.
            action = self.place_tube.forward(
                place_position=state['centrifuge_position'] + np.array([0, 0, 0.1]),
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_tube.is_done():
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
        # Check whether the tube is near the centrifuge position.
        tube_pos = state['object_position']
        centrifuge_pos = state['centrifuge_position']
        distance = np.linalg.norm(tube_pos[:2] - centrifuge_pos[:2])
        return distance < 0.1
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Pick up the test tube from the rack and place it into the centrifuge"
        return self._language_instruction


class FullCentrifugeController(BaseController):
    """Level 5 full centrifuge controller: load sample tubes, balance the load, close the lid, and start the centrifuge."""
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._current_step = 1
        self._current_tube = 0
        self.frame_count = 0
        self.num_tubes = 4
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
    
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)

        # Create pick and place controllers for multiple sample tubes.
        self.pick_controllers = []
        self.place_controllers = []
        
        for i in range(self.num_tubes):
            pick_ctrl = PickController(
                name=f"pick_tube_{i}",
                cspace_controller=self.rmp_controller,
                events_dt=[0.003, 0.002, 0.008, 0.02, 0.05, 0.003, 0.008]
            )
            place_ctrl = PlaceController(
                name=f"place_tube_{i}",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                events_dt=[0.003, 0.006, 0.02, 0.05, 0.008, 0.02]
            )
            self.pick_controllers.append(pick_ctrl)
            self.place_controllers.append(place_ctrl)

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            for pc in self.pick_controllers:
                pc.reset()
            for pc in self.place_controllers:
                pc.reset()
        else:
            self.inference_engine.reset()
        
        self._current_step = 1
        self._current_tube = 0
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

        # Alternate pick and place operations.
        total_steps = self.num_tubes * 2  # pick + place for each tube
        
        if self._current_step <= total_steps:
            tube_idx = (self._current_step - 1) // 2
            is_pick = (self._current_step - 1) % 2 == 0
            
            if is_pick:
                # Pick operation.
                action = self.pick_controllers[tube_idx].forward(
                    picking_position=state['sample_positions'][tube_idx],
                    current_joint_positions=state['joint_positions'],
                    object_size=[0.02, 0.02, 0.1],
                    object_name=f"sample_tube_{tube_idx}",
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_distances=0.015
                )
                if self.pick_controllers[tube_idx].is_done():
                    self._current_step += 1
            else:
                # Place operation.
                # Compute symmetric placement positions in the centrifuge.
                offset_angle = (tube_idx * 90) * np.pi / 180
                place_offset = np.array([0.03 * np.cos(offset_angle), 0.03 * np.sin(offset_angle), 0.1])
                place_pos = state['centrifuge_position'] + place_offset
                
                action = self.place_controllers[tube_idx].forward(
                    place_position=place_pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_position=state['gripper_position']
                )
                if self.place_controllers[tube_idx].is_done():
                    self._current_step += 1
        else:
            # All operations completed.
            success = True
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
            done = True
            self.reset_needed = True

        return action, done, success
    
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()
    
    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        self._language_instruction = "Load multiple sample tubes into the centrifuge with proper balance, close the lid, and start the centrifugation process"
        return self._language_instruction
