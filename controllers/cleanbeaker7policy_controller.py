from scipy.spatial.transform import Rotation as R
import numpy as np
import os

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .atomic_actions.shake_controller import ShakeController
from .base_controller import BaseController
from factories.collector_factory import create_collector

class CleanBeaker7PolicyTaskController(BaseController):
    """Controller for CleanBeaker task sequence.
    
    Manages the sequence of 7 phases:
    1. Pick beaker2
    2. Pour beaker2 to beaker1
    3. Place beaker2 to plat2
    4. Pick beaker1
    5. Shake beaker1
    6. Pour beaker1 to target_beaker
    7. Place beaker1 to plat1
    Uses separate collectors for each phase.
    """
    
    def __init__(self, cfg, robot):
        """Initialize CleanBeaker task controller.
        
        Args:
            cfg: Configuration containing task parameters
            robot: Robot instance to control
            object_utils: Utility class for object manipulation
        """
        super().__init__(cfg, robot)
        
        # Create separate collectors for each phase
        self.collectors = {
            1: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "1_pick_beaker2_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            ),
            2: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "2_pour_beaker2_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            ),
            3: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "3_place_beaker2_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            ),
            4: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "4_pick_beaker1_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            ),
            5: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "5_shake_beaker1_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            ),
            6: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "6_pour_beaker1_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            ),
            7: create_collector(
                cfg.collector.type,
                camera_configs=cfg.cameras,
                save_dir=os.path.join(cfg.multi_run.run_dir, "7_place_beaker1_data"),
                max_episodes=cfg.max_episodes,
                compression=cfg.collector.compression
            )
        }
        
        self.pick_beaker2 = PickController(
            name="pick_beaker2",
            cspace_controller=self.rmp_controller,
        )
        
        self.pour_beaker2 = PourController(
            name="pour_beaker2",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.5]
        )
        
        
        self.place_beaker2 = PlaceController(
            name="place_beaker2",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        )

        self.pick_beaker1 = PickController(
            name="pick_beaker1",
            cspace_controller=self.rmp_controller,
        )
        
        self.shake_beaker1 = ShakeController(
            name="shake_beaker1",
            cspace_controller=self.rmp_controller,
        )
        
        self.pour_beaker1 = PourController(
            name="pour_beaker1",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.5]
        )
        self.place_beaker1 = PlaceController(
            name="place_beaker1",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        
        self._current_step = 1  
        self._last_joint_data = {i: None for i in range(1, 8)}  
        self._current_collector = self.collectors[1]  
        self.frame_count = 0  
        
    def reset(self):
        """Reset controller state."""
        super().reset()
        self.pick_beaker2.reset()
        self.pour_beaker2.reset()
        self.place_beaker2.reset()
        self.pick_beaker1.reset()
        self.shake_beaker1.reset()
        self.pour_beaker1.reset()
        self.place_beaker1.reset()
        self._current_step = 1
        self._last_joint_data = {i: None for i in range(1, 8)}
        self._current_collector = self.collectors[1]  
        
        self.frame_count = 0

    def step(self, state):
        """Execute one step of the CleanBeaker sequence.
        
        Args:
            state: Current state dict containing sensor data and robot state
            
        Returns:
            Tuple[ArticulationAction, bool, bool]: Action, done flag, success flag
        """
        action = None
        done = False
        success = False
        self.current_beaker_pos = None
        self.target_pos = None
        
        
        self._current_collector = self.collectors[self._current_step]
        if 'camera_data' in state:
            self._current_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1],
                language_instruction=self.get_language_instruction()
            )

        self.frame_count += 1
        
        if self._current_step == 1:
            # 1. Pick beaker2
            action = self.pick_beaker2.forward(
                picking_position=state['beaker_2_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_2",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat()
            )
            
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
            if self.current_beaker_pos[2] > 0.90:
                self._last_joint_data[1] = state['joint_positions'][:-1]
                self._current_step = 2
                self._current_collector = self.collectors[2]
                self.frame_count = 0
            
            if self.frame_count >2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self._current_step == 2:
            # 2. Pour beaker2 to beaker1
            action = self.pour_beaker2.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=state['object_size'],
                target_position=state['beaker_1_position'],
                source_name="beaker_2",
                gripper_position=state['gripper_position'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1
            )
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
            self.target_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            if (self.current_beaker_pos[2] <= 1.12 and
                abs(self.current_beaker_pos[0] - self.target_pos[0]) < 0.05 and
                abs(self.current_beaker_pos[1] - self.target_pos[1]) < 0.05 and
                self.frame_count > 800): # 800
                self._last_joint_data[2] = state['joint_positions'][:-1]
                self._current_step = 3
                self._current_collector = self.collectors[3]
                self.frame_count = 0
            
            if self.frame_count >2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self._current_step == 3:
            # 3. Place beaker2 to plat2
            action = self.place_beaker2.forward(
                place_position=state['plat_2_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 40])).as_quat(),
                gripper_position=state['gripper_position']
            )
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
            self.target_pos = self.object_utils.get_geometry_center(object_path=self.cfg.plat_2)
            if (self.current_beaker_pos[2] <= 0.76 and 
                abs(self.current_beaker_pos[0] - self.target_pos[0]) < 0.2 and
                abs(self.current_beaker_pos[1] - self.target_pos[1]) < 0.2 and
                self.frame_count > 400):
                self._last_joint_data[3] = state['joint_positions'][:-1]
                self._current_step = 4
                self._current_collector = self.collectors[4]
                self.frame_count = 0
            
            if self.frame_count >2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self._current_step == 4:
            # 4. Pick beaker1
            action = self.pick_beaker1.forward(
                picking_position=state['beaker_1_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker1",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            if self.current_beaker_pos[2] > 0.82:
                self._last_joint_data[4] = state['joint_positions'][:-1]
                self._current_step = 5
                self._current_collector = self.collectors[5]
                self.frame_count = 0
            
            if self.frame_count >2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self._current_step == 5:
            # 5. Shake beaker1
            action = self.shake_beaker1.forward(
                current_joint_positions=self.robot.get_joint_positions(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            if self.current_beaker_pos[2] >= 0.88 and self.frame_count > 620:
                self._last_joint_data[5] = state['joint_positions'][:-1]
                self._current_step = 6
                self._current_collector = self.collectors[6]
                self.frame_count = 0
            
            if self.frame_count >2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self._current_step == 6:
            # 6. Pour beaker1 to target_beaker
            action = self.pour_beaker1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path=self.cfg.beaker_1),
                target_position=state['target_position'],
                source_name="beaker_1",
                gripper_position=state['gripper_position'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1
            )
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            self.target_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.target_sub_beaker)
            if (self.current_beaker_pos[2] <= 1.0 and
                abs(self.current_beaker_pos[0] - self.target_pos[0]) < 0.05 and
                abs(self.current_beaker_pos[1] - self.target_pos[1]) < 0.05 and
                self.frame_count > 750):
                self._last_joint_data[6] = state['joint_positions'][:-1]
                self._current_step = 7
                self._current_collector = self.collectors[7]
                self.frame_count = 0
            
            if self.frame_count >2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self._current_step == 7:
            # 7. Place beaker1 to plat1
            action = self.place_beaker1.forward(
                place_position=state['plat_1_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker1.is_done():
                # Check final state
                beaker1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
                beaker2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
                
                success = (
                    beaker1_pos is not None and
                    abs(beaker1_pos[0] - state['plat_1_position'][0]) < 0.2 and
                    abs(beaker1_pos[1] - state['plat_1_position'][1]) < 0.2 and
                    beaker1_pos[2] <= 0.78 and
                    beaker2_pos is not None and
                    abs(beaker2_pos[0] - state['plat_2_position'][0]) < 0.2 and
                    abs(beaker2_pos[1] - state['plat_2_position'][1]) < 0.2 and
                    beaker2_pos[2] <= 0.78
                )
                if success:
                    # Task succeeded
                    for step in range(1, 7):
                        if self._last_joint_data[step] is not None:
                            self.collectors[step].write_cached_data(self._last_joint_data[step])
                    self.collectors[7].write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                    self.reset_needed = True
                else:
                    # Task failed
                    for collector in self.collectors.values():
                        collector.clear_cache()
                    self._last_success = False
                    self.reset_needed = True
                done = True
                action = None
            
        return action, done, success
  
    def close(self):
        """Close data collectors."""
        for collector in self.collectors.values():
            collector.close()

    def episode_num(self):
        """Return the number of completed episodes."""
        return min(collector.episode_count for collector in self.collectors.values())