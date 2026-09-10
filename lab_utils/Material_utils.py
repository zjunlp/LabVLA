import os
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdShade

class ObjectMaterial:
    COLOR_MAP = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "yellow": (1.0, 1.0, 0.0),
        "gray": (0.5, 0.5, 0.5)
    }

    def __init__(self, stage, color, obj_path, texture=None):
        """
        Initialize ObjectMaterial class, creating a new OmniPBR material for the object
        Args:
            stage: USD stage object
            color: Color as (r, g, b) tuple (RGB values 0-1) or string (lookup from COLOR_MAP)
            obj_path: Target object's prim path
            texture: Optional, texture file path (defaults to None)
        """
        self.stage = stage
        self.obj_path = obj_path
        self.texture_path = texture
        self.assets_root_path = get_assets_root_path()
        self.mat_path = f"{self.obj_path}/Looks/ObjectMaterial"
        
        self.color = self._process_color(color)
        
        self.obj = self.stage.GetPrimAtPath(self.obj_path)
        if not self.obj:
            raise ValueError(f"Object at path {self.obj_path} does not exist. Please ensure the prim is created beforehand.")
        
        self.material, self.shader = self._create_omnipbr_material()
        self._apply_material()

    def _process_color(self, color):
        """
        Process color input and return RGB tuple
        Args:
            color: String or tuple
        Returns:
            (r, g, b) tuple
        """
        if isinstance(color, (tuple, list)) and len(color) == 3:
            r, g, b = color
            if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in (r, g, b)):
                raise ValueError("Color tuple values must be numbers between 0 and 1")
            return (float(r), float(g), float(b))
        elif isinstance(color, str):
            color = color.lower()
            if color not in self.COLOR_MAP:
                raise ValueError(f"Unknown color '{color}'. Available colors: {list(self.COLOR_MAP.keys())}")
            return self.COLOR_MAP[color]
        else:
            raise ValueError("Color must be a (r, g, b) tuple/list or a string from COLOR_MAP")

    def _create_omnipbr_material(self):
        """
        Create and return OmniPBR material and shader objects
        Returns:
            tuple: (material, shader)
        """
        MDL = "OmniPBR.mdl"
        mtl_name, _ = os.path.splitext(MDL)
        
        omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=MDL, mtl_name=mtl_name, mtl_path=self.mat_path)
        material_prim = self.stage.GetPrimAtPath(self.mat_path)
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, get_prim=True))

        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
        if self.texture_path:
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)

        material = UsdShade.Material(material_prim)
        return material, shader

    def _apply_material(self):
        """Bind material to target object and set color and texture (if provided)"""
        UsdShade.MaterialBindingAPI(self.obj).Bind(self.material, UsdShade.Tokens.strongerThanDescendants)
        
        self.shader.GetInput("diffuse_color_constant").Set(self.color)
        
        if self.texture_path:
            self.shader.GetInput("diffuse_texture").Set(self.texture_path)
            print(f"Material applied to {self.obj_path} with color {self.color} and texture {self.texture_path}")
        else:
            print(f"Material applied to {self.obj_path} with color {self.color} (no texture)")

    def update_color(self, new_color):
        """
        Update material color
        Args:
            new_color: New color value as tuple or string
        """
        self.color = self._process_color(new_color)
        self.shader.GetInput("diffuse_color_constant").Set(self.color)
        print(f"Updated color to {self.color} for {self.obj_path}")

    def update_texture(self, new_texture_path=None):
        """
        Update material texture
        Args:
            new_texture_path: New texture path or None to remove texture
        """
        self.texture_path = new_texture_path
        if self.texture_path:
            if not self.shader.GetInput("diffuse_texture"):
                self.shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
            self.shader.GetInput("diffuse_texture").Set(self.texture_path)
            print(f"Updated texture to {self.texture_path} for {self.obj_path}")
        else:
            if self.shader.GetInput("diffuse_texture"):
                self.shader.GetInput("diffuse_texture").Set(None)
            print(f"Removed texture for {self.obj_path}")

class MaterialUtils:
    def __init__(self, stage, obj_path, material_index=1):
        """
        Initialize MaterialUtils class
        Args:
            stage: USD stage
            obj_path: Object path to bind material to
            material_index: Default material number (int, 1-7, corresponds to Material_1_x to Material_7_x), defaults to 1
        """
        self.stage = stage
        self.obj_path = obj_path
        if not isinstance(material_index, int) or material_index < 1 or material_index > 7:
            raise ValueError("material_index must be an integer between 1 and 7")
        self.material_index = material_index
        self.material_paths = self.get_table_material_paths()

    def get_table_material_paths(self):
        """
        Get material paths named Material_1_x to Material_7_x under /World/Looks/
        Returns:
            dict: Material paths with keys 1-7
        """
        looks_path = "/World/Looks"
        material_paths = {i: None for i in range(1, 8)}
        
        looks_prim = self.stage.GetPrimAtPath(looks_path)
        if looks_prim.IsValid():
            for prim in looks_prim.GetChildren():
                if prim.IsA(UsdShade.Material):
                    prim_name = prim.GetName()
                    for i in range(1, 8):
                        if prim_name.startswith(f"Material_{i}_"):
                            material_paths[i] = prim.GetPath().pathString
                            break
        
        return material_paths

    def bind_material_to_object(self, material_index=None):
        """
        Bind material with specified index to the object path
        Args:
            material_index: Material index to bind (int, 1-7), defaults to None (uses self.material_index)
        """
        target_index = material_index if material_index is not None else self.material_index
        if not isinstance(target_index, int) or target_index < 1 or target_index > 7:
            raise ValueError("material_index must be an integer between 1 and 7")
        
        material_path = self.material_paths.get(target_index)
        if not material_path:
            print(f"No material found for Material_{target_index}_* under /World/Looks/")
            return
        
        obj_prim = self.stage.GetPrimAtPath(self.obj_path)
        if obj_prim.IsValid():
            material = UsdShade.Material.Get(self.stage, material_path)
            if material:
                UsdShade.MaterialBindingAPI(obj_prim).Bind(material)
                print(f"Successfully bound material {material_path} to {self.obj_path}")
            else:
                print(f"Material at {material_path} is not valid")
        else:
            print(f"Object at {self.obj_path} is not valid")

def bind_material_to_object(stage, obj_path, material_path):
    """
    Bind specified material to specified object
    Args:
        stage: USD stage
        obj_path: Object path to bind material to
        material_path: Material path to bind
    """
    obj_prim = stage.GetPrimAtPath(obj_path)
    if obj_prim.IsValid():
        material = UsdShade.Material.Get(stage, material_path)
        if material:
            UsdShade.MaterialBindingAPI(obj_prim).Bind(material)
            print(f"Successfully bound material {material_path} to {obj_path}")
        else:
            print(f"Material at {material_path} is not valid")
    else:
        print(f"Object at {obj_path} is not valid")

if __name__ == "__main__":
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

    cube_path = "/World/Cube"
    omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_path=cube_path, prim_type="Cube")
    cube = stage.GetPrimAtPath(cube_path)
    UsdGeom.Xformable(cube).AddTranslateOp().Set((0.0, 0.0, 1.0))
    UsdGeom.Xformable(cube).AddScaleOp().Set((1.0, 1.0, 1.0))

    cube_material = ObjectMaterial(stage, "red", obj_path=cube_path)

    cube_material = ObjectMaterial(stage, (0.1, 0.5, 0.3), obj_path=cube_path)

    texture_path = get_assets_root_path() + "/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg"
    cube_material.update_texture(texture_path)

    cube_material.update_color("blue")

    cube_material.update_color((0.8, 0.2, 0.2))

    cube_material.update_texture(None)