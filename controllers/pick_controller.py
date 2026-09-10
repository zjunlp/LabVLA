# ============================================================================
# pick_controller.py: picking task controller.
# Implement control logic for picking tasks.
# Support demonstration collection and policy inference.
# ============================================================================

import re  # Regular expressions.
from typing import Optional  # Type annotations.
import numpy as np  # NumPy.
from robots.franka.rmpflow_controller import RMPFlowController  # RMPFlow controller.
from scipy.spatial.transform import Rotation as R  # Rotation utilities.

from .base_controller import BaseController  # Base controller.
from .atomic_actions.pick_controller import PickController  # Atomic pick controller.
from .robot_controllers.trajectory_controller import FrankaTrajectoryController  # Trajectory controller.
from .inference_engines.inference_engine_factory import InferenceEngineFactory  # Inference engine factory.


class PickTaskController(BaseController):
    """Picking controller supporting collect and infer modes. Collection uses scripted actions and saves successful camera, joint, and action demonstrations to HDF5. Inference executes a trained policy such as ACT or Diffusion Policy and measures success. REQUIRED_SUCCESS_STEPS defines the consecutive success threshold; initial_position records the object's starting position for lift checks."""
    
    def __init__(self, cfg, robot):
        """Initialize the picking controller with the task configuration and Franka Panda robot."""
        super().__init__(cfg, robot)  # Initialize the base class.
        self.initial_position = None  # Initialize the object's starting position to None.
            
    def _init_collect_mode(self, cfg, robot):
        """Initialize collection mode: create the data collector through the base class and build the scripted pick controller from cfg and robot."""
        super()._init_collect_mode(cfg, robot)  # Initialize the collector through the base class.
        
        # Create the pick controller.
        # events_dt defines the approach, descent, grasp, and lift phase increments.
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,  # Use the RMPFlow controller.
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]  # Time increments for each phase.
        )

    def reset(self):
        """Reset all controller state at the start of a new episode."""
        super().reset()  # Reset the base class.
        if self.mode == "collect":
            self.pick_controller.reset()  # Reset the pick controller.
        else:
            self.inference_engine.reset()  # Reset the inference engine.
        self.initial_position = None  # Clear the recorded initial position.
    
    def step(self, state):
        """Execute the step function for the selected mode. The state contains joint_positions, camera_data, object_position, object_size, and other observations. Return (action, done, is_success)."""
        # Record the initial object position for the lift check.
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state  # Store the current state.
        
        # Dispatch to the selected mode's step function.
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        """Return whether the object has been lifted more than 0.1 m above its initial height."""
        return self.state['object_position'][2] > self.initial_position[2] + 0.1

    def _init_infer_mode(self, cfg, robot):
        """Initialize inference mode by creating a trajectory controller and a local or remote inference engine for cfg and robot."""
        # Create the trajectory controller for predicted actions.
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        
        # Create the inference engine through the factory.
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
        
    def _step_collect(self, state):
        """Execute a scripted collection step: check success, compute the pick action, cache the current frame, and save successful demonstrations. Return (action, done, is_success)."""
        # Check whether the pick succeeded.
        if self._check_success():
            self.check_success_counter += 1  # Increment the success counter.
        else:
            self.check_success_counter = 0  # Reset the counter.
        
        # Continue while the pick action is unfinished.
        if not self.pick_controller.is_done():
            # Compute the pick action.
            action = self.pick_controller.forward(
                picking_position=state['object_position'],  # Target object position.
                current_joint_positions=state['joint_positions'],  # Current joint positions.
                object_size=state['object_size'],  # Object size.
                object_name=state['object_name'],  # Object name.
                gripper_control=self.gripper_control,  # Gripper controller.
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),  # End-effector orientation.
                gripper_position=state['gripper_position'],  # Gripper position.
                pre_offset_x=0.05,  # Pre-grasp offset along X.
                after_offset_z=0.25  # Post-lift offset along Z.
            )
            
            # Cache the current frame.
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],  # Camera images.
                    joint_angles=state['joint_positions'][:-1],  # Joint angles excluding the gripper.
                    language_instruction=self.get_language_instruction()  # Language instruction.
                )
            
            return action, False, False  # Return the action with done=False and is_success=False.
        
        # The pick action finished; check success.
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            # Save data on success.
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True  # Completed successfully.

        # Clear cached data on failure.
        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False  # Completed unsuccessfully.
        
    def _step_infer(self, state):
        """Execute a policy inference step: add the language instruction, predict an action, and check success. Return (action, done, is_success)."""
        # Add the language instruction to the state.
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
            
        # Obtain an action from the inference engine.
        action = self.inference_engine.step_inference(state)
        
        # Check success.
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        # Check the required number of consecutive successful steps.
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True  # Completed successfully.
            
        return action, False, False  # Continue execution.

    def get_language_instruction(self) -> Optional[str]:
        """Generate a language instruction from the object name, stripping digits and replacing underscores with spaces; for example, "Pick up the conical bottle from the table"."""
        # Strip digits and replace underscores with spaces in the object name.
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ', ' ').lower()
        # Generate the language instruction.
        self._language_instruction = f"Pick up the {object_name} from the table"
        return self._language_instruction
