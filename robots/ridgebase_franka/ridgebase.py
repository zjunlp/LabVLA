from typing import Optional
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage


class Ridgebase(Robot):
    """
    Ridgebase mobile platform robot.
    
    This class encapsulates the basic functionality of the Ridgebase mobile platform robot, including initialization, position retrieval, etc.
    
    Attributes:
        prim_path (str): The path of the robot in the USD scene
        name (str): The name of the robot
        usd_path (str): The USD file path
        position (np.ndarray): The initial position of the robot
        orientation (np.ndarray): The initial orientation of the robot
    """

    def __init__(
        self,
        prim_path: str = "/World/Ridgebase",
        name: str = "ridgebase",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the Ridgebase robot.
        
        Args:
            prim_path: The path of the robot in the USD scene
            name: The name of the robot
            usd_path: The USD file path, if None then use the default path
            position: The initial position [x, y, z]
            orientation: The initial orientation (quaternion)
        """
        prim = get_prim_at_path(prim_path)
        self.prim_path_str = prim_path
        
        # If the prim does not exist, load the USD file
        if not prim.IsValid():
            if usd_path is None:
                usd_path = "assets/robots/ridgeback_franka.usd"
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        
        # Call the parent class initialization
        super().__init__(
            prim_path=prim_path, 
            name=name, 
            position=position, 
            orientation=orientation, 
            articulation_controller=None
        )

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the physical properties of the robot.
        
        Args:
            physics_sim_view: The physics simulation view
        """
        super().initialize(physics_sim_view)
        return

    def post_reset(self) -> None:
        """Post-reset operations"""
        super().post_reset()
        return
    
    def get_base_pose(self) -> tuple:
        """
        Get the position and orientation of the robot's base.
        
        Returns:
            tuple: (position, orientation) The position and orientation of the robot's base
        """
        return self.get_world_pose()
    
    def get_gripper_position(self) -> np.ndarray:
        """
        Get the gripper position.
        
        Returns:
            np.ndarray: The gripper position
        """
        from lab_utils.object_utils import ObjectUtils
        return ObjectUtils.get_instance().get_object_xform_position(
            object_path=self.prim_path_str + "/panda_hand/tool_center"
        )

