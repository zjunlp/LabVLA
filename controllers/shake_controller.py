import queue
import numpy as np
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .atomic_actions.shake_controller import ShakeController
from .base_controller import BaseController

class ShakeTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
        )
        
        self._shake_positions = []
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=60)
        self._hold_step = 0
        self._shake_success = False
        self._initial_position = None
        self._task_started = False
            
    def _init_collect_mode(self, cfg, robot):
        """Initialize data collection mode"""
        super()._init_collect_mode(cfg, robot)
        
        self.shake_controller = ShakeController(
            name="shake_controller",
            cspace_controller=self.rmp_controller,
        )

    def reset(self):
        super().reset()
        self.pick_controller.reset()
        self._shake_positions = []
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=60)
        self._hold_step = 0
        self._shake_success = False
        self._initial_position = None
        self._task_started = False

        if self.mode == "collect":
            self.shake_controller.reset()
            self.data_collector.clear_cache()
        else:
            self.inference_engine.reset()
        
    def step(self, state):
        self.state = state

        if self._initial_position is None:
            self._initial_position = state['object_position']
        
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_2",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05
            )
            
            return action, False, False
            
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state):
        if not self.shake_controller.is_done():
            action = self.shake_controller.forward(
                current_joint_positions=self.robot.get_joint_positions(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            return action, False, self.is_success()
        elif self.is_success():
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
            self.reset_needed = True
            return None, True, True
        else:
            self.data_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False

    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)

        return action, self._last_success, self.is_success()
        
    def is_success(self):
        if self._initial_position is None:
            return False
        
        height_diff = self.state['object_position'][2] - self._initial_position[2]
        if height_diff < 0.05:
            return False
        
        if self._shake_count < 5:
            if self.state['object_position'] is not None:
                xy = self.state['object_position'][:2]
                self._shake_positions.append(xy)
                if len(self._shake_positions) > 1:
                    start_xy = np.array(self._shake_positions[0])
                    end_xy = np.array(self._shake_positions[-1])
                    dist = np.linalg.norm(end_xy - start_xy)
                    if dist >= 0.05:  # 5cm
                        self._shake_count += 1
                        self._shake_positions = []
        elif not self._shake_success:
            self._hold_positions.put(self.state['object_position'][:2])
            self._hold_step += 1
            if self._hold_step >= 60:
                arr = np.array(list(self._hold_positions.queue))
                max_xy = arr.max(axis=0)
                min_xy = arr.min(axis=0)
                delta = max_xy - min_xy

                if np.all(delta <= 0.01):  # 1cm
                    self._shake_success = True
                    return True
                else:
                    self._hold_positions.get()
                    self._hold_step -= 1
            return False
        else:
            return True

    def get_language_instruction(self) -> str:
        self._language_instruction = "Pick up the container and shake it"
        return self._language_instruction
