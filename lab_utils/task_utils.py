import numpy as np
from scipy.spatial.transform import Rotation as R

class TaskUtils:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TaskUtils()
        return cls._instance
    
    def get_pickz_offset(self, item_name, object_size=None):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.
            object_size (np.ndarray, optional): Size of the object.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.03,
            "conical_bottle03": 0.06,
            "conical_bottle04": 0.08,
            "beaker2": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
        }

        for key in offsets:
            if key in item_name.lower():
                return offsets[key]

        return object_size[2] * 2 / 5 if object_size is not None else 0.02

    def get_pour_threshold(self, item_name, source_size):
        if item_name is None:
            return source_size[2] / 2
        
        offset = {
            "conical_bottle02": 0.03,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker2": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "beaker": 0.02,
        }
                
        for key in offset:
            if key in item_name.lower():
                return source_size[2] / 2 - offset[key]
        
        return source_size[2] / 2
    
    def check_rotation_angle(self, initial_quat, current_quat, threshold_degrees=45):
        """Check if rotation angle between two quaternions exceeds threshold.

        Args:
            initial_quat (np.ndarray): Initial quaternion [x,y,z,w].
            current_quat (np.ndarray): Current quaternion [x,y,z,w].
            threshold_degrees (float): Threshold angle in degrees.

        Returns:
            bool: True if rotation angle exceeds threshold.
        """
        if np.dot(initial_quat, current_quat) < 0:
            current_quat = -np.array(current_quat)
        r1 = R.from_quat(initial_quat)
        r2 = R.from_quat(current_quat)
        relative_rotation = r1.inv() * r2
        rotvec = relative_rotation.as_rotvec()
        angle_rad = np.linalg.norm(rotvec)
        angle_deg = np.degrees(angle_rad)
        # print(f"Angle: {angle_deg} degrees")
        return angle_deg > threshold_degrees
