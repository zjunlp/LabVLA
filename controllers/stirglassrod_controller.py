from typing import Optional
import numpy as np
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController

class StirGlassrodTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.obj_added = False
        
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, stir controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            position_threshold=0.005,
            events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
        )
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=RMPFlowController(
                name="stir_controller",
                robot_articulation=robot
            ),
        )
        
    def reset(self):
        super().reset()
        self.obj_added = False
        self.gripper_control.release_object()
        if self.mode == "collect":
            self.pick_controller.reset()
            self.stir_controller.reset()
        else:
            self.inference_engine.reset()
        
    def step(self, state):
        self.state = state
        
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state):
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05,
                after_offset_z=0.2
            )

            # Cache demonstration data
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            self.gripper_control.update_grasped_object_position()
            return action, False, False
        
        elif not self.stir_controller.is_done():            
            target_position = state['target_position']
            action = self.stir_controller.forward(
                center_position=target_position,
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
                gripper_position=state['gripper_position'],
            )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            self.gripper_control.update_grasped_object_position()
            return action, False, False
            
        else:
            self.reset_needed = True
            final_object_position = state['glass_rod_position']
            target_position = state['target_position']
            self.gripper_control.release_object()
            if final_object_position[2] > 0.85 and np.linalg.norm(final_object_position[0:2] - target_position[0:2]) < 0.04:
                # Task successful - save collected data
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                return None, True, True
            else:
                # Task failed - discard collected data
                self.data_collector.clear_cache()
                self._last_success = False
                return None, True, False

    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        if action is not None:
            if len(action.joint_positions) == 9:
                # Ensure gripper positions are set
                if action.joint_positions[7] == None and action.joint_positions[8] == None:
                    action.joint_positions[7] = 0.015
                    action.joint_positions[8] = 0.015

                # Add object to gripper when gripper is closed
                if action.joint_positions[7] < np.float32(0.01) and action.joint_positions[8] < np.float32(0.01) and not self.obj_added:
                    self.gripper_control.add_object_to_gripper("/World/glass_rod", "/World/Franka/panda_hand/tool_center")
                    print("glassrod is added to franka gripper center!")
                    self.obj_added = True
                    
        self.gripper_control.update_grasped_object_position()
        return action, False, self._check_success()
    
    def _check_success(self):
        object_pos = self.state['glass_rod_position']
        target_position = self.state['target_position']
        if object_pos[2] > 0.85 and np.linalg.norm(object_pos[0:2] - target_position[0:2]) < 0.04:
            self.check_success_counter += 1
            if self.check_success_counter > 240:
                self._last_success = True
                return True
        return False
    
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        # Default instruction for shake tasks
        self._language_instruction = "Use the glass rod to stir the liquid."
        return self._language_instruction