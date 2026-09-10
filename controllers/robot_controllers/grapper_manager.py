from isaacsim.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom, Usd, Sdf

class Gripper:
    def __init__(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None

    def reset():
        #TODO reset pick object
        return
        
    def add_object_to_gripper(self, object_path, gripper_frame_path):
        
        self.grasped_object_path = object_path
        self.gripper_frame_path = gripper_frame_path
                
        transform_prim = get_prim_at_path("/World/glass_rod")
        if not transform_prim.IsValid():
            raise ValueError(f"Object at path is not valid.")   

        self.inverse_transform_matrix = UsdGeom.Xformable(transform_prim).ComputeLocalToWorldTransform(0).GetInverse()

    def update_grasped_object_position(self):
        if not self.grasped_object_path or not self.gripper_frame_path:
            return

        
        target_frame_prim = get_prim_at_path(self.gripper_frame_path)
        if not target_frame_prim.IsValid():
            raise ValueError(f"Gripper frame at path {self.gripper_frame_path} is not valid.")

        
        target_world_position = UsdGeom.Xformable(target_frame_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()

        
        local_position = self.inverse_transform_matrix.TransformAffine(target_world_position)
        
        object_prim = get_prim_at_path(self.grasped_object_path)
        if not object_prim.IsValid():
            raise ValueError(f"Object at path {self.grasped_object_path} is not valid.")

        if self.position_offest is None:
            self.position_offest = UsdGeom.Xformable(object_prim).GetOrderedXformOps()[0].Get() - local_position 
        
        
        xformable = UsdGeom.Xformable(object_prim)
        xform_ops = xformable.GetOrderedXformOps()
        if xform_ops:
            translate_op = xform_ops[0]
            translate_op.Set(local_position+self.position_offest)
        else:
            xformable.AddTranslateOp().Set(local_position+self.position_offest)

    def release_object(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None