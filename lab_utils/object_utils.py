# ============================================================================
# LabUtopia object utilities.
# Read and modify object position, size, and rotation in USD scenes.
# Use a singleton so one ObjectUtils instance is shared globally.
# Supported uses:
# - Find grasp positions for beakers, tubes, and other laboratory objects.
# - Read and set world-space object positions.
# - Compute geometric centers and bounding box sizes.
# - Get object rotation quaternions.
# - Get joint positions in world coordinates.
# ============================================================================

from pxr import Usd, UsdGeom, Gf, UsdPhysics  # Pixar USD modules.
# Usd provides Stage and Prim access.
# UsdGeom provides Xformable and bounding box operations.
# Gf provides vector and matrix types such as Vec3d and Matrix4f.
# UsdPhysics provides joint and physics attributes.

from isaacsim.core.utils.stage import get_stage_units  # Scene unit scale helper.
import numpy as np  # NumPy array and numerical operations.
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats  # Euler-to-quaternion conversion.


class ObjectUtils:
    """Singleton utilities for laboratory object positions, dimensions, and rotations in a USD scene. Objects can be addressed by name or full path. get_pick_position applies object-specific height offsets; get_geometry_center returns a world-space center; get_object_size computes the bounding box size; set_object_position updates position; get_transform_quat reads orientation."""
    
    _instance = None  # Store the singleton instance; initially None.

    @classmethod
    def get_instance(cls, stage: Usd.Stage = None, default_path: str = "/World") -> "ObjectUtils":
        """Return the existing singleton or create it with the supplied Usd.Stage. stage is required on the first call; default_path defaults to '/World'. Raise ValueError if the initial call omits stage."""
        if cls._instance is None:  # Check whether an instance already exists.
            if stage is None:  # Require stage when creating the first instance.
                raise ValueError("Stage must be provided for first instance")  # Raise an exception.
            cls._instance = cls(stage, default_path)  # Create the instance.
        return cls._instance  # Return the instance.

    def __new__(cls, *args, **kwargs):
        """Create the singleton only once and return the same ObjectUtils instance on subsequent construction attempts."""
        if cls._instance is None:  # No instance exists yet.
            cls._instance = super(ObjectUtils, cls).__new__(cls)  # Allocate the instance through the base class.
        return cls._instance  # Return the instance.

    def __init__(self, stage: Usd.Stage, default_path: str = "/World"):
        """Initialize the USD stage, default object path, and object-specific grasp height offsets once. The _initialized flag prevents repeated initialization. Offsets account for different grasp heights of tubes, dishes, and other laboratory objects."""
        if not hasattr(self, '_initialized'):  # Avoid repeated initialization.
            self._stage = stage  # Store the USD stage reference.
            self._default_path = default_path  # Store the default path prefix.
            
            # Object-specific grasp height offsets in meters.
            # Offsets are additional heights above the object's geometric center.
            self._pick_height_offsets = {
                "rod": 0.04,            # Tube rack or stirring rod: +4 cm.
                "tube": 0.01,           # Test tube: +1 cm.
                "beaker": 0.0,          # Beaker: no offset, grasp at its center.
                "erlenmeyer flask": 0.018,  # Conical flask: +1.8 cm.
                "cylinder": 0.0,        # Graduated cylinder: no offset.
                "petri dish": 0.005,    # Petri dish: +0.5 cm because it is thin.
                "pipette": 0.008,       # Pipette: +0.8 cm.
                "microscope slide": 0.002  # Slide: +0.2 cm because it is very thin.
            }
            self._initialized = True  # Mark initialization complete.

    def _get_object_path(self, object_name: str = None, object_path: str = None) -> str:
        """Resolve a full USD path from object_path or object_name. Use an explicit path as supplied; otherwise append the name to the default path. Raise ValueError when neither argument is supplied."""
        if object_path:  # An explicit full path was supplied.
            return object_path  # Return it directly.
        if object_name:  # An object name was supplied.
            return f"{self._default_path}/{object_name}"  # Construct the full path.
        raise ValueError("Either object_name or object_path must be provided")  # Neither required identifier was supplied.

    def get_pick_position(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Return the object's grasp position [x, y, z] by adding an object-specific height offset to its geometric center. Accept object_name or object_path, and return None if the object does not exist."""
        # Get the object's geometric center first.
        position = self.get_geometry_center(object_name, object_path)
        if position is None:  # The object does not exist.
            return None

        # Determine the object name for offset lookup.
        name = object_name or object_path.split('/')[-1]
        
        # Find a matching object category in the offset table.
        for key, offset in self._pick_height_offsets.items():
            if key in name.lower():  # Match the object category case-insensitively.
                # Apply the height offset after dividing by the scene unit scale.
                position[2] += offset / get_stage_units()
                return position  # Return the adjusted position.

        # Use a default 2 cm offset when no object category matches.
        position[2] += 0.02 / get_stage_units()
        return position

    def get_object_size(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Compute the world-space axis-aligned bounding box size [width, depth, height], accounting for the object's transform. Accept object_name or object_path; return None for a missing object."""
        path = self._get_object_path(object_name, object_path)  # Resolve the full path.
        prim = self._stage.GetPrimAtPath(path)  # Get the Prim from the stage.
        if not prim.IsValid():  # Check Prim validity.
            return None

        # Create a bounding box cache for world-space bounds.
        # Include geometry with the default purpose.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        
        # Compute the object's world-space bounds.
        bbox = bbox_cache.ComputeWorldBound(prim)
        
        # Get the minimum and maximum bounding box corners.
        min_point = bbox.GetRange().GetMin()  # Minimum corner.
        max_point = bbox.GetRange().GetMax()  # Maximum corner.
        
        # Append a homogeneous coordinate of 1 for matrix multiplication.
        world_min_point = np.array([min_point[0], min_point[1], min_point[2], 1.0])
        world_max_point = np.array([max_point[0], max_point[1], max_point[2], 1.0])
        
        # Convert the transform matrix to a NumPy array.
        transform_matrix = bbox.GetMatrix()
        transform_matrix_np = np.array(transform_matrix)
        
        # Transform the corners and subtract to obtain the world-space size.
        world_size = world_max_point @ transform_matrix_np - world_min_point @ transform_matrix_np
        
        return world_size[:3]  # Return the first three components, dropping the homogeneous coordinate.

    def get_object_xform_position(self, object_path: str) -> np.ndarray:
        """Return the world-space origin [x, y, z] from an object's Xform at object_path. This is not necessarily its geometric center. Return None if the object is missing."""
        prim = self._stage.GetPrimAtPath(object_path)  # Get the Prim.
        if not prim.IsValid():  # Validate the Prim.
            print(f"Object at path {object_path} not found.")  # Print an error message.
            return None

        # Wrap the Prim as an Xformable.
        xformable = UsdGeom.Xformable(prim)
        
        # Compute the local-to-world transform.
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Extract its translation component.
        position = transform.ExtractTranslation()
        
        return np.array(position)  # Return a NumPy array.
    
    def set_object_position(self, object_path: str, position: np.ndarray, local_position: np.ndarray = None, position_offset: np.ndarray = None) -> None:
        """Set the position of object_path, either directly from position [x, y, z] or as local_position plus position_offset. For example, use set_object_position(path, [1.0, 2.0, 0.5]) or set_object_position(path, None, local_pos, offset)."""
        prim = self._stage.GetPrimAtPath(object_path)  # Get the Prim.
        if not prim.IsValid():  # Validate the Prim.
            print(f"Object at path {object_path} not found.")  # Print an error.
            return

        xformable = UsdGeom.Xformable(prim)  # Wrap it as an Xformable.
        xform_ops = xformable.GetOrderedXformOps()  # Get the ordered transform operations.
        
        # Compute the final position.
        if local_position is not None and position_offset is not None:
            # Use a local position plus an offset.
            new_position = Gf.Vec3d(*(local_position + position_offset))
        else:
            # Use the supplied position directly.
            new_position = Gf.Vec3d(*position)

        # Set the position.
        if xform_ops:  # Existing transform operations are available.
            xform_ops[0].Set(new_position)  # Update the first operation, usually translation.
        else:  # No transform operation exists.
            xformable.AddTranslateOp().Set(new_position)  # Add a translation operation.
            
    def get_geometry_center(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Return the world-space geometric center [x, y, z] for object_name or object_path, or None if missing. Compute the bounding box center and transform it to world coordinates; this provides a useful grasp reference."""
        path = self._get_object_path(object_name, object_path)  # Resolve the full path.
        prim = self._stage.GetPrimAtPath(path)  # Get the Prim.
        if not prim.IsValid():  # Validate the Prim.
            return None

        # Create the bounding box cache.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        
        # Compute world-space bounds.
        bbox = bbox_cache.ComputeWorldBound(prim)
        range_3d = bbox.GetRange()  # Get the 3D range.
        
        # Average the minimum and maximum corners to obtain the local center.
        center = (np.array(range_3d.GetMin()) + np.array(range_3d.GetMax())) / 2.0
        
        # Append the homogeneous coordinate w=1.
        center_hom = np.append(center, 1.0)
        
        # Get the transform matrix.
        transform = np.array(bbox.GetMatrix())
        
        # Transform the center to world coordinates.
        world_center = center_hom @ transform
        
        return world_center[:3]  # Return the xyz coordinates.

    
    def get_transform_quat(self, object_path: str, w_first: bool = False) -> np.ndarray:
        """Read the rotation quaternion for object_path, accepting USD xformOp:orient or xformOp:rotateXYZ. w_first=True returns [w, x, y, z]; otherwise return [x, y, z, w]. Return None for a missing object."""
        prim = self._stage.GetPrimAtPath(object_path)  # Get the Prim.
        if not prim.IsValid():  # Validate the Prim.
            print(f"Object at path {object_path} not found.")  # Print an error.
            return None

        # Look for a quaternion orientation attribute.
        rotation = prim.GetAttribute("xformOp:orient").Get()
        
        if rotation is None:
            # Fall back to Euler angles if no quaternion orientation is present.
            rotation = prim.GetAttribute("xformOp:rotateXYZ").Get()
            # Convert Euler angles to a quaternion; degrees=True specifies degree units.
            rotation = euler_angles_to_quats(rotation, degrees=True)
            # euler_angles_to_quats returns [w, x, y, z].
            return np.array([rotation[0], rotation[1], rotation[2], rotation[3]]) if w_first else np.array([rotation[1], rotation[2], rotation[3], rotation[0]])
        
        # Extract values from the USD quaternion.
        # GetImaginary() returns [x, y, z]; GetReal() returns w.
        quat = np.array([rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2], rotation.GetReal()])
        
        # Make w positive for a consistent sign convention.
        # Swap the order when abs(x) exceeds both 0.5 and abs(w).
        if abs(quat[0]) > 0.5 and abs(quat[0]) > abs(quat[3]):
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        
        # Return the ordering selected by w_first.
        return np.array([quat[3], quat[0], quat[1], quat[2]]) if w_first else quat
        
    def get_revolute_joint_positions(self, joint_path: str) -> np.ndarray:
        """Return the world-space pivot [x, y, z] for a revolute joint at joint_path. Retrieve its body1, combine the joint's local position and rotation with the body's world transform, and extract the world position."""
        # Get the joint Prim.
        joint_prim = self._stage.GetPrimAtPath(joint_path)
        
        # Validate the Prim.
        if not joint_prim.IsValid():
            return np.array([0.0, 0.0, 0.0])

        # Access joint attributes through UsdPhysics.Joint.
        joint_api = UsdPhysics.Joint(joint_prim)
        
        # Get the connected body1 target.
        body1 = joint_api.GetBody1Rel().GetTargets()
        if not body1:  # No connected body was found.
            print("No body1 found!")  # Print an error.
            exit()  # Exit the program.

        # Get the body1 Prim.
        body1_prim = self._stage.GetPrimAtPath(body1[0])

        # Compute the world transform of body1 through Xformable.
        body1_xform = UsdGeom.Xformable(body1_prim)
        body1_world_transform = Gf.Matrix4f(body1_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        # Get the joint's local position and rotation relative to body1.
        local_pos1 = joint_api.GetLocalPos1Attr().Get()   # Local position.
        local_rot1 = joint_api.GetLocalRot1Attr().Get()   # Local rotation.

        # Build the joint's local transform.
        rotation_matrix = Gf.Matrix3f(local_rot1)  # Build a rotation matrix from the quaternion.
        local_transform = Gf.Matrix4f()  # Create a 4x4 transform matrix.
        local_transform.SetTranslateOnly(local_pos1)  # Set translation.
        local_transform.SetRotateOnly(rotation_matrix)  # Set rotation.
        
        # Compute the world transform as the local transform multiplied by body1's world transform.
        joint_position = local_transform * body1_world_transform
        
        # Extract and return the position.
        return np.array(joint_position.ExtractTranslation())
