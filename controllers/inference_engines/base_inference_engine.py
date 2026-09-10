from abc import ABC, abstractmethod
import torch
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple


class BaseInferenceEngine(ABC):
    """
    Abstract inference engine base class, defining common inference processes and interfaces.

    All specific inference engines must inherit this class and implement the necessary abstract methods.
    """
    
    def __init__(self, cfg, trajectory_controller):
        """
        Initialize the inference engine base class
        
        Args:
            cfg: Configuration object
            trajectory_controller: Trajectory controller
        """
        self.cfg = cfg
        self.trajectory_controller = trajectory_controller
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Observation related configuration
        self.obs_names = cfg.infer.obs_names
        self.camera_to_obs = {k: v for k, v in self.obs_names.items()}
        self.n_obs_steps = self._get_n_obs_steps()
        
        # Initialize observation history
        self.obs_history_dict = {
            obs_key: deque(maxlen=self.n_obs_steps) 
            for obs_key in self.obs_names.values()
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        # Initialize language instruction history
        self.language_instruction = ""
        
        # ============================================================================
        # Gripper adjustment settings used during inference only.
        # Record raw VLA outputs and adjusted actions for diagnostic comparison.
        # ============================================================================
        self.GRIPPER_TIGHTEN_THRESHOLD = 0.035  # Values below this threshold indicate a closing grasp.
        # Read gripper_tighten_amount from the config; default to 0.01.
        # For example, PourLiquid uses 0.00 to disable extra tightening while pouring.
        self.GRIPPER_TIGHTEN_AMOUNT = getattr(cfg.infer, 'gripper_tighten_amount', 0.01)
        self.last_raw_actions = None            # Preserve raw, unmodified VLA outputs for comparison.
        self.last_modified_actions = None       # Preserve adjusted actions for comparison.
        
        # Initialize inference engine
        self._init_inference_engine()
    
    @abstractmethod
    def _get_n_obs_steps(self) -> int:
        """Get the number of observation steps"""
        pass
    
    @abstractmethod
    def _init_inference_engine(self):
        """Initialize the specific inference engine"""
        pass
    
    @abstractmethod
    def _predict_action(self, obs_dict: Dict[str, torch.Tensor], language_instruction: str = "") -> np.ndarray:
        """
        Use specific inference methods to predict actions
        
        Args:
            obs_dict: Observation data dictionary
            language_instruction: Language instruction string, if not empty
            
        Returns:
            Predicted action array
        """
        pass
    
    def reset(self):
        """Reset the inference engine state"""
        self.obs_history_dict = {
            obs_key: deque(maxlen=self.n_obs_steps) 
            for obs_key in self.obs_names.values()
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        self.language_instruction = ""
        self.last_raw_actions = None       # Clear recorded raw VLA outputs.
        self.last_modified_actions = None  # Clear recorded adjusted actions.
        self.trajectory_controller.reset()
    
    def update_observations(self, state: Dict[str, Any]):
        """
        Update observation history
        
        Args:
            state: Current state dictionary
        """
        for cam_name, image in state['camera_data'].items():
            if cam_name in self.camera_to_obs:
                obs_key = self.camera_to_obs[cam_name]
                self.obs_history_dict[obs_key].append(image)
        
        self.obs_history_pose.append(state['joint_positions'][:-1])
        
        if 'language_instruction' in state:
            self.language_instruction = state['language_instruction']
        else:
            self.language_instruction = ""
    
    def _check_histories_complete(self) -> bool:
        """Check if the observation history is complete"""
        return (
            len(self.obs_history_pose) == self.n_obs_steps and
            all(len(hist) == self.n_obs_steps for hist in self.obs_history_dict.values())
        )
    
    def _prepare_observation_dict(self) -> Dict[str, torch.Tensor]:
        """
        Prepare observation data dictionary
        
        Returns:
            Processed observation data dictionary
        """
        obs_dict = {
            obs_key: torch.from_numpy(np.stack(list(hist))).float().to(self.device) / 255.0
            for obs_key, hist in self.obs_history_dict.items()
        }
        obs_dict['agent_pose'] = torch.from_numpy(
            np.stack(list(self.obs_history_pose))
        ).float().to(self.device)
        
        for key in obs_dict.keys():
            if obs_dict[key].shape[0] != 1:
                obs_dict[key] = obs_dict[key].unsqueeze(0)
        
        return obs_dict
    
    def step_inference(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Execute one step of inference
        
        Args:
            state: Current state dictionary
            
        Returns:
            Action array, if waiting return None
        """
        self.update_observations(state)
        
        if self.trajectory_controller.is_trajectory_complete() and self._check_histories_complete():
            obs_dict = self._prepare_observation_dict()
            
            joint_positions = self._predict_action(obs_dict, self.language_instruction)
            actions_chunk = joint_positions[:40, :]  # Take the first 40 action steps (7 joints and 1 gripper channel).
            
            # ============================================================================
            # Adjust only the VLA output's gripper channel (index 7).
            # Tighten a closing grasp to reduce object slipping.
            # Smaller values in the eighth channel close the gripper more tightly.
            # Subtract a small amount when the value is below the closing threshold.
            # Only controller inputs change; the original model output is preserved.
            # ============================================================================
            self.last_raw_actions = actions_chunk.copy()  # Copy the raw VLA output for comparison.
            modified_chunk = actions_chunk.copy()          # Modify a copy without changing the original array.
            gripper_values = modified_chunk[:, 7]          # Extract the gripper channel (column 8, index 7).
            # Tighten each step whose gripper value is below the closing threshold.
            mask = gripper_values < self.GRIPPER_TIGHTEN_THRESHOLD  # Mask steps that need tightening.
            gripper_values[mask] -= self.GRIPPER_TIGHTEN_AMOUNT     # Subtract the tightening amount at masked steps.
            gripper_values[gripper_values < 0.0] = 0.0              # Clamp the gripper value to zero (fully closed).
            modified_chunk[:, 7] = gripper_values                   # Write adjusted gripper values into the action sequence.
            self.last_modified_actions = modified_chunk.copy()       # Store adjusted actions for comparison.
            
            self.trajectory_controller.generate_trajectory(modified_chunk)  # Pass adjusted actions to the trajectory controller.
        
        return self.trajectory_controller.get_next_action()
    
    def close(self):
        """Close the inference engine, release resources"""
        pass 