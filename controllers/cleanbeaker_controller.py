import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .atomic_actions.shake_controller import ShakeController

class CleanBeakerTaskController(BaseController):
    """
    Controller for clean beaker tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous cleaning

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        _current_step (int): Current step in the task sequence
        frame_count (int): Frame counter for episode management
    """
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._current_step = 1
        self.frame_count = 0
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
    
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up atomic action controllers and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)

        # 1. Pick beaker2
        self.pick_beaker2 = PickController(
            name="pick_beaker2",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 1, 0.05, 0.004, 1]
        )

        # 2. Pour beaker2 to beaker1
        self.pour_beaker2 = PourController(
            name="pour_beaker2",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 1]
        )

        # 3. Place beaker2 to plat2
        self.place_beaker2 = PlaceController(
            name="place_beaker2",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 1, 0.05, 0.01, 1]
        )

        # 4. Pick beaker1
        self.pick_beaker1 = PickController(
            name="pick_beaker1",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 1, 0.05, 0.004, 1]
        )

        # 5. Shake beaker1
        self.shake_beaker1 = ShakeController(
            name="shake_beaker1",
            cspace_controller=self.rmp_controller
        )

        # 6. Pour beaker1 to target_beaker
        self.pour_beaker1 = PourController(
            name="pour_beaker1",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 1]
        )

        # 7. Place beaker1 to plat1
        self.place_beaker1 = PlaceController(
            name="place_beaker1",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 1, 0.05, 0.01, 1]
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_beaker2.reset()
            self.pour_beaker2.reset()
            self.place_beaker2.reset()
            self.pick_beaker1.reset()
            self.shake_beaker1.reset()
            self.pour_beaker1.reset()
            self.place_beaker1.reset()
        else:
            self.inference_engine.reset()
        
        self._current_step = 1
        self.frame_count = 0
    
    def step(self, state):
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        action = None
        done = False
        success = False
        
        if 'camera_data' in state:
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1],
                language_instruction=self.get_language_instruction()
            )

        if self._current_step == 1:
            # 1. Pick beaker2
            action = self.pick_beaker2.forward(
                picking_position=state['beaker_2_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['target_size'],
                object_name="beaker_l",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05,
                gripper_distances=0.027
            )
            if self.pick_beaker2.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # 2. Pour beaker2 to beaker1
            action = self.pour_beaker2.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=state['target_size'],
                target_position=state['beaker_1_position'],
                gripper_position=state['gripper_position'],
                source_name="beaker",
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker2.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            # 3. Place beaker2 to plat2
            action = self.place_beaker2.forward(
                place_position=state['plat_2_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 40])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker2.is_done():
                self._current_step = 4

        elif self._current_step == 4:
            # 4. Pick beaker1
            action = self.pick_beaker1.forward(
                picking_position=state['beaker_1_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['target_size'],
                object_name="beaker_l",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_distances=0.027
            )
            if self.pick_beaker1.is_done():
                self._current_step = 5

        elif self._current_step == 5:
            # 5. Shake beaker1
            action = self.shake_beaker1.forward(
                current_joint_positions=self.robot.get_joint_positions(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            if self.shake_beaker1.is_done():
                self._current_step = 6

        elif self._current_step == 6:
            # 6. Pour beaker1 to target_beaker
            action = self.pour_beaker1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=state['target_size'],
                source_name="beaker",
                target_position=state['target_position'],
                gripper_position=state['gripper_position'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker1.is_done():
                self._current_step = 7

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
                success = self._check_success()
                if success:
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                else:
                    self.data_collector.clear_cache()
                    self._last_success = False
                done = True
                self.reset_needed = True
                action = None

        return action, done, success
    
    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses policy to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Pick up the object from the table"
        
        action = self.inference_engine.step_inference(state)
        
        return action, False, self.is_success()
    
    def _check_success(self):
        beaker1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_1+"/mesh")
        beaker2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_2+"/mesh")

        plat1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.plat_1)
        plat2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.plat_2)

        if beaker1_pos is None or beaker2_pos is None or plat1_pos is None or plat2_pos is None:
            return False

        success = (
            abs(beaker1_pos[0] - plat1_pos[0]) < 0.04 and
            abs(beaker1_pos[1] - plat1_pos[1]) < 0.04 and
            beaker1_pos[2] <= 0.78 and
            abs(beaker2_pos[0] - plat2_pos[0]) < 0.04 and
            abs(beaker2_pos[1] - plat2_pos[1]) < 0.04 and
            beaker2_pos[2] <= 0.78
        )
        
        return success
    
    def is_success(self):
        Maxframe = 5000
        self.frame_count += 1
        
        if self.frame_count > Maxframe:
            self.reset_needed = True
            return True
        
        if self.frame_count > 5000:
            beaker1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_1+"/mesh")
            beaker2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_2+"/mesh")
            plat1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.plat_1)
            plat2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.plat_2)

            print("===  ===")
            print(f"beaker1_pos: {beaker1_pos}")
            print(f"plat1_pos: {plat1_pos}")
            print(f"beaker2_pos: {beaker2_pos}")
            print(f"plat2_pos: {plat2_pos}")
            print("\n===  ===")

            cond1 = beaker1_pos is not None
            print(f" 1: beaker1_pos is not None -> {cond1}")
            
            diff_x1 = abs(beaker1_pos[0] - plat1_pos[0])
            cond2 = diff_x1 < 0.04
            print(f" 2: abs(beaker1_pos[0] - plat1_pos[0]) = {diff_x1:.6f} < 0.04 -> {cond2}")

            diff_y1 = abs(beaker1_pos[1] - plat1_pos[1])
            cond3 = diff_y1 < 0.04
            print(f" 3: abs(beaker1_pos[1] - plat1_pos[1]) = {diff_y1:.6f} < 0.04 -> {cond3}")

            z1 = beaker1_pos[2]
            cond4 = z1 <= 0.78
            print(f" 4: beaker1_pos[2] = {z1:.6f} <= 0.78 -> {cond4}")

            cond5 = beaker2_pos is not None
            print(f" 5: beaker2_pos is not None -> {cond5}")
            
            diff_x2 = abs(beaker2_pos[0] - plat2_pos[0])
            cond6 = diff_x2 < 0.04
            print(f" 6: abs(beaker2_pos[0] - plat2_pos[0]) = {diff_x2:.6f} < 0.04 -> {cond6}")
            
            diff_y2 = abs(beaker2_pos[1] - plat2_pos[1])
            cond7 = diff_y2 < 0.04
            print(f" 7: abs(beaker2_pos[1] - plat2_pos[1]) = {diff_y2:.6f} < 0.04 -> {cond7}")

            z2 = beaker2_pos[2]
            cond8 = z2 <= 0.78
            print(f" 8: beaker2_pos[2] = {z2:.6f} <= 0.78 -> {cond8}")

            success = cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8
            print("\n===  ===")
            print(f"success = {success}")
            if success:
                print("(1-8) True")
            else:
                print("(1-8) False")
                if not cond1: print("-  1: beaker1_pos is not None")
                if not cond3: print("-  3: abs(beaker1_pos[1] - plat1_pos[1]) < 0.04")
                if not cond4: print("-  4: beaker1_pos[2] <= 0.78")
                if not cond5: print("-  5: beaker2_pos is not None")
                if not cond6: print("-  6: abs(beaker2_pos[0] - plat2_pos[0]) < 0.04")
                if not cond7: print("-  7: abs(beaker2_pos[1] - plat2_pos[1]) < 0.04")
                if not cond8: print("-  8: beaker2_pos[2] <= 0.78")

        else:
            success = False

        if success:
            self.reset_needed = True
            self.print_success = True
            return True
        return False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        self._language_instruction = "First, pick up the second beaker from the table. Next, pour its contents into the first beaker. Then, place the now-empty second beaker on the second platform. After that, pick up the first beaker from the table and shake it to mix the contents thoroughly. Once mixed, pour the contents from the first beaker into the target beaker. Finally, place the empty first beaker on the first platform."
        return self._language_instruction
