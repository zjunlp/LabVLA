# ============================================================================
# base_controller.py: base controller module.
# Define shared interfaces and behavior for task controllers.
# Handle robot control, data collection, inference, and success checks.
# ============================================================================

from abc import ABC, abstractmethod  # Abstract base class utilities.
from typing import Dict, Any, Tuple, Optional  # Type annotations.
import torch  # PyTorch.
from omegaconf import OmegaConf  # Configuration management.
from controllers.inference_engines.inference_engine_factory import InferenceEngineFactory  # Inference engine factory.
from controllers.robot_controllers.grapper_manager import Gripper  # Gripper manager.
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController  # Trajectory controller.
from factories.collector_factory import create_collector  # Data collector factory.
from lab_utils.object_utils import ObjectUtils  # Object utilities.
from robots.franka.rmpflow_controller import RMPFlowController as FrankaRMPFlowController  # RMPFlow motion controller.


class BaseController(ABC):
    """Abstract task controller with shared motion, gripper, data collection, inference, and episode accounting. In collect mode, scripted actions generate demonstrations; in infer mode, a trained policy predicts actions."""
    
    def __init__(self, cfg, robot, use_default_config=True):
        """Initialize with the task configuration, robot, and optional default RMPFlow configuration."""
        self.cfg = cfg  # Store the configuration.
        self.robot = robot  # Store the robot reference.
        self.object_utils = ObjectUtils.get_instance()  # Get the object utility singleton.
        self.reset_needed = False  # Reset flag.
        self._last_success = False  # Whether the previous episode succeeded.
        self._episode_num = 0  # Episode count.
        self.success_count = 0  # Success count.
        self._language_instruction = ""  # Language instruction.
        self.gripper_control = Gripper()  # Create the gripper controller.
        self.REQUIRED_SUCCESS_STEPS = 60  # Consecutive successful steps required to declare success.
        self.check_success_counter = 0  # Success-check counter.
        self.rmp_controller = None  # RMPFlow controller reference.
        self._last_failure_reason = ""  # Most recent failure reason.
        
        # Create the RMPFlow motion controller.
        # RMPFlow provides potential-field motion planning in Isaac Sim.
        self.rmp_controller = FrankaRMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot,
            use_default_config=use_default_config
        )

        # Select the compute device (GPU or CPU).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Register the eval resolver for configuration expressions.
        OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
        
        # Initialize the selected run mode.
        if hasattr(cfg, "mode"):
            self.mode = cfg.mode  # "collect" or "infer".
            if self.mode == "collect":
                self._init_collect_mode(cfg, robot)  # Initialize collection mode.
            elif self.mode == "infer":
                self._init_infer_mode(cfg, robot)  # Initialize inference mode.
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Expected 'collect' or 'infer'.")
    
    @property
    def language_instruction(self) -> Optional[str]:
        """Return the current language instruction, or None."""
        return self._language_instruction
    
    @language_instruction.setter
    def language_instruction(self, instruction: Optional[str]):
        """Set the task language instruction, or None."""
        self._language_instruction = instruction
    
    def get_language_instruction(self) -> Optional[str]:
        """Return the current language instruction. Subclasses may override this for dynamic instructions."""
        return self._language_instruction
    
    @abstractmethod
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one control step. Subclasses must implement this method. The state contains sensor and robot observations; return an (action, done, is_success) tuple."""
        pass
    
    def _init_collect_mode(self, cfg, robot=None):
        """Initialize demonstration collection using cfg and an optional robot instance."""
        # Create the data collector through the factory.
        self.data_collector = create_collector(
            cfg.collector.type,  # Collector type.
            camera_configs=cfg.cameras,  # Camera configuration.
            save_dir=cfg.multi_run.run_dir,  # Output directory.
            max_episodes=cfg.max_episodes,  # Maximum episode count.
            compression=cfg.collector.compression  # Compression method.
        )
    
    def _init_infer_mode(self, cfg, robot=None): 
        """Initialize the trajectory controller and policy inference engine using cfg and an optional robot instance."""
        # Create the trajectory controller to execute predicted action sequences.
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot,
            use_interpolation=False
        )
        
        # Create the local or remote inference engine.
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
    
    def episode_num(self) -> int:
        """Return the current episode index."""
        if self.mode == "collect":
            return self.data_collector.episode_count  # Use the collector's counter in collection mode.
        return self._episode_num  # Use the internal counter in inference mode.
    
    def print_failure_reason(self) -> None:
        """Print the most recent failure reason, if available."""
        if self._last_failure_reason:
            print(f"Failure Reason: {self._last_failure_reason}")
    
    def reset(self) -> None:
        """Reset controller state between episodes and update episode statistics."""
        # Update the success count.
        if self._last_success:
            self.success_count += 1
        self._episode_num += 1
        
        # Print success-rate statistics.
        print(f"Episode Stats: Success Rate = {self.success_count}/{self._episode_num} ({(self.success_count/self._episode_num)*100:.2f}%)")
        
        # Reset state.
        self.check_success_counter = 0
        self.reset_needed = False
        self._last_success = False
        self._last_failure_reason = ""
        
        # Clear cached data in collection mode.
        if self.mode == "collect":
            self.data_collector.clear_cache()

        
    def close(self) -> None:
        """Close the controller and save collected data when collection finishes."""
        if self.mode == "collect" and hasattr(self, 'data_collector'):
            self.data_collector.close()  # Close the data collector.
        
    def need_reset(self) -> bool:
        """Return whether the controller needs a reset."""
        return self.reset_needed

    def is_success(self):
        """Return whether the task succeeded; subclasses may override this check."""
        pass
