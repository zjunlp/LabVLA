# ============================================================================
# new_scenario_tasks.py: additional scenario task classes.
# Define tasks for five additional scenarios.
# ============================================================================

import numpy as np
from .base_task import BaseTask


class SpectrophotometerTask(BaseTask):
    """Spectrophotometer operation task: prepare a cuvette, add a sample, place it in the spectrophotometer, and read the result."""
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.spectrophotometer_path = cfg.spectrophotometer_path
        self.sample_chamber_path = cfg.sample_chamber_path
        self.cuvette_path = cfg.cuvette_path
        self.pipette_path = cfg.pipette_path
        
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        for obj_cfg in self.cfg.task.obj_paths:
            obj_path = obj_cfg['path']
            pos_range = obj_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=obj_path, position=position)
        
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=1800):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        spectro_pos = self.object_utils.get_geometry_center(object_path=self.spectrophotometer_path)
        cuvette_pos = self.object_utils.get_geometry_center(object_path=self.cuvette_path)
        pipette_pos = self.object_utils.get_geometry_center(object_path=self.pipette_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'spectrophotometer_position': spectro_pos,
            'cuvette_position': cuvette_pos,
            'pipette_position': pipette_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }


class PCRSetupTask(BaseTask):
    """PCR preparation task: prepare PCR tubes, add primers/template and master mix, and place them in the PCR machine."""
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.pcr_machine_path = cfg.pcr_machine_path
        self.pcr_machine_lid_path = cfg.pcr_machine_lid_path
        self.micropipette_path = cfg.micropipette_path
        self.pcr_tubes = cfg.pcr_tubes
        
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        for obj_cfg in self.cfg.task.obj_paths:
            obj_path = obj_cfg['path']
            pos_range = obj_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=obj_path, position=position)
        
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2500):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        pcr_machine_pos = self.object_utils.get_geometry_center(object_path=self.pcr_machine_path)
        pipette_pos = self.object_utils.get_geometry_center(object_path=self.micropipette_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'pcr_machine_position': pcr_machine_pos,
            'pipette_position': pipette_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }


class GelElectrophoresisTask(BaseTask):
    """Gel electrophoresis task: prepare the gel, load samples and DNA ladder, connect power, and run electrophoresis."""
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.electrophoresis_path = cfg.electrophoresis_path
        self.electrophoresis_lid_path = cfg.electrophoresis_lid_path
        self.gel_tray_path = cfg.gel_tray_path
        self.micropipette_path = cfg.micropipette_path
        self.sample_tubes = cfg.sample_tubes
        
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        for obj_cfg in self.cfg.task.obj_paths:
            obj_path = obj_cfg['path']
            pos_range = obj_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=obj_path, position=position)
        
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2200):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        electro_pos = self.object_utils.get_geometry_center(object_path=self.electrophoresis_path)
        gel_pos = self.object_utils.get_geometry_center(object_path=self.gel_tray_path)
        pipette_pos = self.object_utils.get_geometry_center(object_path=self.micropipette_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'electrophoresis_position': electro_pos,
            'gel_position': gel_pos,
            'pipette_position': pipette_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }


class pHMeasurementTask(BaseTask):
    """pH measurement task: calibrate the meter, rinse the electrode, measure the sample, and rinse it again."""
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.ph_meter_path = cfg.ph_meter_path
        self.ph_electrode_path = cfg.ph_electrode_path
        self.sample_beaker_path = cfg.sample_beaker_path
        self.wash_bottle_path = cfg.wash_bottle_path
        
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        for obj_cfg in self.cfg.task.obj_paths:
            obj_path = obj_cfg['path']
            pos_range = obj_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=obj_path, position=position)
        
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=1600):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        ph_meter_pos = self.object_utils.get_geometry_center(object_path=self.ph_meter_path)
        beaker_pos = self.object_utils.get_geometry_center(object_path=self.sample_beaker_path)
        wash_pos = self.object_utils.get_geometry_center(object_path=self.wash_bottle_path)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'ph_meter_position': ph_meter_pos,
            'beaker_position': beaker_pos,
            'wash_bottle_position': wash_pos,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }


class AutoclaveSterilizationTask(BaseTask):
    """Autoclave task: prepare items, load the tray and autoclave, close the door, set parameters, sterilize, and unload."""
    
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.autoclave_path = cfg.autoclave_path
        self.autoclave_door_path = cfg.autoclave_door_path
        self.sterilization_tray_path = cfg.sterilization_tray_path
        self.items_to_sterilize = cfg.items_to_sterilize
        
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        for obj_cfg in self.cfg.task.obj_paths:
            obj_path = obj_cfg['path']
            pos_range = obj_cfg['position_range']
            position = np.array([
                np.random.uniform(pos_range['x'][0], pos_range['x'][1]),
                np.random.uniform(pos_range['y'][0], pos_range['y'][1]),
                pos_range['z'][0]
            ])
            self.object_utils.set_object_position(object_path=obj_path, position=position)
        
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2800):
            return None
            
        joint_positions = self.robot.get_joint_positions()
        camera_data, camera_display = self.get_camera_data()
        
        autoclave_pos = self.object_utils.get_geometry_center(object_path=self.autoclave_path)
        tray_pos = self.object_utils.get_geometry_center(object_path=self.sterilization_tray_path)
        
        item_positions = []
        for item in self.items_to_sterilize:
            pos = self.object_utils.get_geometry_center(object_path=item['path'])
            item_positions.append(pos)
        
        return {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': camera_display,
            'autoclave_position': autoclave_pos,
            'tray_position': tray_pos,
            'item_positions': item_positions,
            'gripper_position': self.robot.get_gripper_position(),
            'done': self.reset_needed,
        }
