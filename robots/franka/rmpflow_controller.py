# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    """RMPFlow motion controller

    Args:
        name (str): Controller name
        robot_articulation (SingleArticulation): Robot articulation object
        physics_dt (float, optional): Physics time step. Defaults to 1.0/60.0.
        use_default_config (bool, optional): Whether to use default config files. Defaults to True.
    """

    def __init__(
        self, 
        name: str, 
        robot_articulation: SingleArticulation, 
        physics_dt: float = 1.0 / 60.0,
        use_default_config: bool = True
    ) -> None:
        # Choose between default config or custom config based on parameter
        if use_default_config:
            # Use system default RMPflow configuration
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        else:
            # Use custom configuration file paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rmpflow_dir = os.path.join(current_dir, "rmpflow")
            
            self.rmp_flow_config = {
                'end_effector_frame_name': 'right_gripper',
                'maximum_substep_size': 0.00334,
                'ignore_robot_state_updates': False,
                'robot_description_path': os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
                'urdf_path': os.path.join(current_dir, "lula_franka_gen.urdf"),
                'rmpflow_config_path': os.path.join(rmpflow_dir, "franka_rmpflow_common.yaml")
            }
        
        print(self.rmp_flow_config)
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
