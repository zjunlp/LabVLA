# ============================================================================
# base_task.py: base task module.
# Define shared task interfaces and behavior.
# Provide scene state, camera data, object placement, and material switching.
# ============================================================================

from abc import ABC, abstractmethod  # Abstract base class utilities.
from typing import Dict, Any  # Type annotations.
import numpy as np  # NumPy.
from isaacsim.sensors.camera import Camera  # Isaac Sim Camera.
from lab_utils.object_utils import ObjectUtils  # Object utilities.
from isaacsim.core.utils.semantics import add_update_semantics  # Semantic segmentation utilities.
from lab_utils.camera_utils import process_camera_image  # Camera image processing.
from isaacsim.core.utils.prims import set_prim_visibility  # Object visibility helper.
from pxr import UsdShade  # USD material binding.


class BaseTask(ABC):
    """Abstract simulation task with camera setup, object placement, material switching, reset/step/completion handling, and object/material index cycling. cfg, world, stage, and robot hold the run context; cameras stores camera instances; reset_needed and frame_idx track execution; object_utils accesses scene objects."""
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize a task with cfg (camera, object, and material settings), the Isaac Sim world, the USD stage, and the robot."""
        self.cfg = cfg  # Store the configuration.
        self.world = world  # Store the simulation world.
        self.stage = stage  # Store the USD stage.
        self.robot = robot  # Store the robot.
        self.reset_needed = False  # Initialize the reset flag to False.
        self.frame_idx = 0  # Initialize the frame index to zero.
        self.object_utils = ObjectUtils.get_instance()  # Get the object utility singleton.
        
        # Set up cameras, objects, and materials in order.
        self.setup_cameras()  # Set up cameras.
        self.setup_objects()  # Set up objects.
        self.setup_materials()  # Set up materials.
        
        # Initialize object and material cycle indices.
        self.current_material_idx = 0  # Current material index.
        if len(self.obj_configs) != 0:  # Object configurations are present.
            # Compute episodes per object.
            self.episodes_per_obj = int(cfg.max_episodes / len(self.obj_configs))
        else:
            self.episodes_per_obj = 0
        self.current_obj_idx = 0  # Current object index.
        self.current_obj_episodes = 0  # Episodes collected for the current object.
        
    def reset(self) -> None:
        """Reset the simulation world and task state before a new episode."""
        self.world.reset()  # Reset the simulation world.
        self.reset_needed = False  # Clear the reset flag.
        self.frame_idx = 0  # Reset the frame index.
                
        # Apply the current material when configured.
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
            
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """Advance the task by one simulation step. Subclasses must implement this method and return the current state dictionary."""
        pass

    def get_task_info(self) -> Dict[str, Any]:
        """Return task metadata containing the frame index and reset flag."""
        return {
            "frame_idx": self.frame_idx,  # Current frame index.
            "reset_needed": self.reset_needed  # Whether a reset is required.
        }
        
    def need_reset(self) -> bool:
        """Return whether the task needs a reset."""
        return self.reset_needed

    def on_task_complete(self, success: bool) -> None:
        """Handle episode completion by updating object and material cycle indices. success indicates whether the task completed successfully."""
        self.update_object_and_material_indices(success)  # Update the indices.
        self.reset_needed = True  # Mark a reset as required.
        
    def setup_cameras(self) -> None:
        """Create and initialize all configured cameras, including pose, focal length, clipping range, and RGB/depth/point-cloud/segmentation outputs."""
        self.cameras = []  # Initialize the camera list.
        
        for cam_cfg in self.cfg.cameras:  # Iterate over camera configurations.
            # Check whether the camera Prim already exists.
            if self.stage.GetPrimAtPath(cam_cfg.prim_path).IsValid():
                # Create a camera from the existing Prim.
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    name=cam_cfg.name,
                    frequency=60,  # Sample at 60 Hz.
                    resolution=tuple(cam_cfg.resolution)  # Image resolution.
                )
            else:
                # Create a new camera Prim.
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    translation=np.array(cam_cfg.translation),  # Camera position.
                    name=cam_cfg.name,
                    frequency=60,
                    resolution=tuple(cam_cfg.resolution)
                )
                # Set camera orientation.
                camera.set_local_pose(orientation=np.array(cam_cfg.orientation), camera_axes="usd")
                camera.set_focal_length(cam_cfg.focal_length)  # Set focal length.
            
            # Set clipping range.
            if hasattr(cam_cfg, 'clipping_range'):
                camera.set_clipping_range(near_distance=cam_cfg.clipping_range[0], far_distance=cam_cfg.clipping_range[1])
            else:
                camera.set_clipping_range(near_distance=0.1, far_distance=10.0)  # Default clipping range.
            self.cameras.append(camera)  # Append the camera to the list.
        
        # Reset the world to initialize cameras.
        self.world.reset()
        
        # Initialize cameras and configure outputs.
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            camera.initialize()  # Initialize the camera.
            # Parse image_type, including combinations such as "rgb+depth".
            image_types = cam_cfg.image_type.split('+') if '+' in cam_cfg.image_type else [cam_cfg.image_type]
            
            for image_type in image_types:  # Configure the camera for each modality.
                if image_type == "depth":
                    camera.add_distance_to_image_plane_to_frame()  # Add depth output.
                elif image_type == "pointcloud":
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()  # Add point-cloud output.
                elif image_type == "segmentation":
                    camera.add_instance_segmentation_to_frame()  # Add instance-segmentation output.
                    # Configure semantic label mappings.
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
                elif image_type == "semantic_pointcloud":
                    camera.add_instance_segmentation_to_frame()
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
    
    def setup_objects(self) -> None:
        """Parse object paths and position ranges from cfg into obj_configs."""
        self.obj_configs = []  # Initialize the object configuration list.
        
        if hasattr(self.cfg, 'task') and hasattr(self.cfg.task, 'obj_paths'):
            for obj in self.cfg.task.obj_paths:  # Iterate over object configurations.
                if isinstance(obj, str):  # Handle a path-only entry.
                    # Use the default position range.
                    self.obj_configs.append({
                        'path': obj,
                        'position_range': {
                            'x': [0.24, 0.30],
                            'y': [-0.05, 0.05],
                            'z': [0.85, 0.85]
                        }
                    })
                else:  # Handle a complete object configuration.
                    self.obj_configs.append(obj)
                
    def setup_materials(self) -> None:
        """Parse material paths and select training or test materials according to collect/infer mode for out-of-distribution evaluation."""
        self.material_config = None  # Material configuration.
        self.available_materials = []  # Available material list.
        
        # Check whether the run is inference mode.
        is_infer_mode = hasattr(self.cfg, "mode") and self.cfg.mode == "infer"
        has_infer = hasattr(self.cfg, "infer")
        is_ood = False  # Whether out-of-distribution test materials are enabled.
        
        if has_infer and hasattr(self.cfg.infer, "is_test_material"):
            is_ood = bool(self.cfg.infer.is_test_material)
            
        # Check whether material configuration exists.
        has_material_paths = hasattr(self.cfg, "task") and hasattr(self.cfg.task, "material_paths") and self.cfg.task.material_paths
        
        if has_material_paths:
            self.material_config = self.cfg.task.material_paths[0]  # Get the first material configuration.
            # Select the material list based on the mode.
            if is_infer_mode and is_ood and hasattr(self.material_config, "test_materials"):
                self.available_materials = getattr(self.material_config, "test_materials", [])
            elif hasattr(self.material_config, "materials"):
                self.available_materials = getattr(self.material_config, "materials", [])
    
    def get_camera_data(self):
        """Return camera_data for recording and display_data for visualization. camera_data maps camera_name_type to arrays; combined image types return a modality dictionary."""
        camera_data = {}  # Recording data dictionary.
        display_data = {}  # Display data dictionary.
        
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            # Process camera images.
            record, display = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                if isinstance(record, dict):  # A combined modality returns a dictionary.
                    for k, v in record.items():
                        camera_data[f"{cam_cfg.name}_{k}"] = v
                else:  # A single modality is assigned directly.
                    camera_data[f"{cam_cfg.name}_{cam_cfg.image_type}"] = record
            if display is not None:
                display_data[cam_cfg.name] = display
                
        return camera_data, display_data
    
    def apply_material_to_object(self, target_path: str, material_idx: int = None) -> None:
        """Bind material_idx (or the current material index when None) to target_path."""
        if not self.material_config or not self.available_materials:
            return
            
        if material_idx is None:
            material_idx = self.current_material_idx
            
        # Get the target object's Prim.
        target_prim = self.stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            material_path = self.available_materials[material_idx]
            mtl_prim = self.stage.GetPrimAtPath(material_path)
            if mtl_prim.IsValid():
                # Create and bind the USD material.
                cube_mat_shade = UsdShade.Material(mtl_prim)
                UsdShade.MaterialBindingAPI(target_prim).Bind(
                    cube_mat_shade, 
                    UsdShade.Tokens.strongerThanDescendants
                )
    
    def randomize_object_position(self, obj_path: str, position_range: Dict[str, list]) -> np.ndarray:
        """Generate a random position inside position_range for obj_path, whose x/y/z entries each contain [min, max]."""
        # Sample a random position inside the configured range.
        position = np.array([
            np.random.uniform(position_range['x'][0], position_range['x'][1]),
            np.random.uniform(position_range['y'][0], position_range['y'][1]),
            np.random.uniform(position_range['z'][0], position_range['z'][1])
        ])
        # Set the object position.
        self.object_utils.set_object_position(object_path=obj_path, position=position)
        return position
    
    def place_objects_with_visibility_management(self, current_obj_idx: int, far_distance: float = 10.0) -> str:
        """Place the current object at a random position, move all other objects far away, and toggle visibility. Return the current object's path."""
        for i, obj_config in enumerate(self.obj_configs):
            obj_path = obj_config['path']
            position_range = obj_config['position_range']
            prim = self.stage.GetPrimAtPath(obj_path)
            
            if prim.IsValid():
                if i == current_obj_idx:  # Current object.
                    # Place it randomly within its configured range.
                    self.randomize_object_position(obj_path, position_range)
                    set_prim_visibility(prim, True)  # Make it visible.
                else:  # Non-current object.
                    # Move it far away with an angular distribution.
                    angle = 2 * np.pi * i / len(self.obj_configs)
                    far_position = np.array([
                        far_distance * np.cos(angle),
                        far_distance * np.sin(angle),
                        0.1
                    ])
                    self.object_utils.set_object_position(object_path=obj_path, position=far_position)
                    set_prim_visibility(prim, False)  # Hide it.
        
        return self.obj_configs[current_obj_idx]['path']
    
    def get_basic_state_info(self, joint_positions: np.ndarray = None, object_path: str = None,
                            target_path: str = None, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Collect common state: joint_positions, camera data, object/target paths, gripper position, completion status, and any additional_info."""
        # Get joint positions.
        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None
        
        # Get camera data.
        camera_data, display_data = self.get_camera_data()
        
        # Build the base state dictionary.
        state = {
            'joint_positions': joint_positions,  # Joint positions.
            'camera_data': camera_data,  # Camera image data.
            'camera_display': display_data,  # Display image data.
            'done': self.reset_needed,  # Completion flag.
            'gripper_position': self.robot.get_gripper_position(),  # Gripper position.
        }
        
        # Add the manipulated object information.
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
                'object_name': object_path.split("/")[-1]
            })
        
        # Add target object information.
        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1]
            })
        
        # Add additional information.
        if additional_info:
            state.update(additional_info)
            
        return state
    
    def check_frame_limits(self, max_steps: int = None) -> bool:
        """Check whether the current frame exceeds max_steps, using the configured value when max_steps is None. Return whether execution may continue."""
        # Skip the first five frames while the simulator stabilizes.
        if self.frame_idx < 5:
            return False
            
        if max_steps is None:
            max_steps = self.cfg.task.max_steps
            
        # Mark the task complete after max_steps.
        if self.frame_idx > max_steps:
            self.on_task_complete(True)
            self.reset_needed = True
            
        return True
        
    def update_object_and_material_indices(self, success: bool) -> None:
        """Update object and material cycle indices after a successful task. success indicates whether the task succeeded."""
        if success:
            self.current_obj_episodes += 1
            # Check whether to advance to the next object.
            if self.current_obj_episodes >= self.episodes_per_obj and len(self.obj_configs) > 0:
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.obj_configs)
                self.current_obj_episodes = 0
            # Advance to the next material.
            if self.available_materials:
                self.current_material_idx = (self.current_material_idx + 1) % len(self.available_materials)
