# ============================================================================
# LabUtopia simulation entry point.
# Start Isaac Sim and run demonstration collection or policy inference.
# Usage: python main.py --config-name <config-name>
# Example: python main.py --config-name level3_pick
# ============================================================================

import os  # Operating system and path utilities.
import argparse  # Command-line argument parsing.
from isaacsim import SimulationApp  # SimulationApp starts the Isaac Sim runtime.



def parse_args():
    """Parse simulator CLI options and return an argparse.Namespace. Options select the numpy (CPU) or gpu backend, headless mode, video suppression, the config name without its .yaml suffix, and the config directory."""
    # Create the argument parser and program description.
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    # Select the backend; default to numpy (CPU).
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    # Enable headless mode when the flag is present.
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    # Add the video suppression flag.
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    # Add the task configuration name option.
    parser.add_argument('--config-name', type=str, default='level3_Heat_Liquid',
                       help='Configuration file name (without .yaml extension)')
    # Add the configuration directory option, defaulting to config/.
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Configuration directory path (default: config)')
    # Optionally print actions at runtime; disabled by default.
    parser.add_argument('--print-action', action='store_true',
                       help='Enable real-time action printing (default: disabled)')
    # Optionally save per-episode action trajectories; disabled by default.
    parser.add_argument('--save-action-trajectory', action='store_true',
                       help='Save action trajectory for each episode as JSON (default: disabled)')
    return parser.parse_args()  # Parse and return the arguments.

# Parse CLI options before importing simulator-dependent modules.
# SimulationApp must be created before importing omni modules.
args = parse_args()

# Configure the simulation application from CLI options.
# headless=True disables the GUI for server-side collection.
simulation_config = {"headless": args.headless}
# Create the Isaac Sim application.
# Create SimulationApp before importing omni modules.
simulation_app = SimulationApp(simulation_config)

# ============================================================================
# These imports must follow SimulationApp creation.
# They require an initialized Omniverse runtime.
# ============================================================================

import hydra  # Hydra loads and manages YAML configurations.
from omegaconf import OmegaConf  # OmegaConf provides configuration manipulation and serialization.
import cv2  # OpenCV handles image processing and video output.
import numpy as np  # NumPy provides array and numerical operations.
import json  # JSON stores episode outcomes and accuracy statistics.
import time  # Time utilities record inference timing.

import omni  # Omniverse core.
from isaacsim.core.api import World  # World represents the simulation world.
from isaacsim.core.utils.stage import add_reference_to_stage  # Scene reference helper.
import omni.usd  # USD scene utilities.
from isaacsim.core.utils import extensions  # Extension management.

# Enable the Omniverse extensions required for physics simulation.
extensions.enable_extension("omni.physx.bundle")  # Enable the PhysX physics extension bundle.
extensions.enable_extension("omni.usdphysics.ui")  # Enable the USD physics UI extension.

# Project factories and utilities.
from factories.robot_factory import create_robot  # Robot factory.
from lab_utils.object_utils import ObjectUtils  # Object utilities.
from factories.task_factory import create_task  # Task factory.
from factories.controller_factory import create_controller  # Controller factory.


def main():
    """Load the Hydra configuration, initialize physics and the scene, create the robot/task/controller, and run the simulation loop. collect mode records scripted demonstrations; infer mode executes policy predictions and evaluates success."""
    # ========== Load configuration ==========
    # Initialize Hydra with the config directory and task name.
    hydra.initialize(config_path=args.config_dir, job_name=args.config_name, version_base="1.1")
    # Load the configuration without its .yaml suffix.
    cfg = hydra.compose(config_name=args.config_name)
    # ============================================================================
    # Run is_test_material=True and False sequentially for dual-phase inference.
    # In infer mode, an is_test_material setting enables True followed by False.
    # Write results to is_test_material_True/ and is_test_material_False/.
    # ============================================================================
    enable_dual_material_run = (
        cfg.mode == "infer" 
        and hasattr(cfg, "infer") 
        and hasattr(cfg.infer, "is_test_material")
    )
    
    if enable_dual_material_run:
        # Create one parent output directory for the two material phases.
        parent_run_dir = cfg.multi_run.run_dir  # Parent run directory, for example outputs/infer/<date>/<run>.
        material_phases = [True, False]  # Run test materials first, then training materials.
        current_phase_idx = 0  # Current material phase index.
        # Set the first phase's output directory.
        phase_label = f"is_test_material_{material_phases[current_phase_idx]}"
        phase_run_dir = os.path.join(parent_run_dir, phase_label)
        os.makedirs(phase_run_dir, exist_ok=True)
        os.makedirs(parent_run_dir, exist_ok=True)
        # Update the output directory and is_test_material in cfg.
        cfg_mutable = OmegaConf.to_container(cfg, resolve=False)
        cfg_mutable['multi_run'] = {'run_dir': phase_run_dir}
        cfg_mutable['infer']['is_test_material'] = material_phases[current_phase_idx]
        cfg = OmegaConf.create(cfg_mutable)
        OmegaConf.save(cfg, phase_run_dir + "/config.yaml")
        print(f"[Info] 双轮推理模式已启用: 第1轮 is_test_material={material_phases[current_phase_idx]}")
        print(f"[Info] 父输出目录: {parent_run_dir}")
        print(f"[Info] 当前轮输出目录: {phase_run_dir}")
    else:
        # Single-phase execution.
        parent_run_dir = None
        material_phases = None
        current_phase_idx = None
        # Create the directory for data, video, and configuration outputs.
        os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
        # Save the run configuration for reproducibility.
        OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")

    # ========== Initialize simulation world ==========
    # Select the compute backend from CLI options.
    if args.backend == 'gpu':
        # Use GPU physics; Isaac Sim rendering requires a supported GPU, not A100/A800.
        world = World(stage_units_in_meters=1, device="cpu")
        physx_interface = omni.physx.get_physx_interface()  # Get the PhysX interface.
        physx_interface.overwrite_gpu_setting(1)  # Enable GPU physics.
    else:
        # Use the default NumPy/CPU backend.
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    
    # ========== Video settings ==========
    # Configure video recording and display.
    if args.no_video:
        save_video = False  # Disable video saving.
        show_video = False  # Disable video display.
    else:
        save_video = False   # Video saving remains disabled by default in this checkout.
        show_video = False  # Disable live video display by default to avoid overhead.

    # ========== Create robot ==========
    # Instantiate the robot through the factory.
    # Read robot type and position from the configuration.
    robot = create_robot(
        cfg.robot.type,  # Robot type, such as "franka" for Franka Panda.
        position=np.array(cfg.robot.position)  # Robot base position in world coordinates.
    )
    
    # ========== Load scene ==========
    # Get the current USD stage.
    stage = omni.usd.get_context().get_stage()
    # Reference the configured USD scene under /World.
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    # ========== Initialize utilities ==========
    # Initialize the object utility singleton for scene positions and dimensions.
    ObjectUtils.get_instance(stage)
    
    # ========== Create task ==========
    # Instantiate the task through the factory.
    # Tasks provide scene state, camera data, object placement, and task status.
    task = create_task(
        cfg.task_type,  # Task type, such as "pick", "place", or "pour".
        cfg=cfg,        # Complete configuration.
        world=world,    # Simulation world.
        stage=stage,    # USD stage.
        robot=robot,    # Robot instance.
    )
    
    # ========== Create controller ==========
    # Instantiate the task controller through the factory.
    # Controllers handle motion, collection or inference, and success checks.
    task_controller = create_controller(
        cfg.controller_type,  # Controller type, usually matching the task type.
        cfg=cfg,              # Complete configuration.
        robot=robot,          # Robot instance.
    )
    
    # Initialize the video writer to None.
    video_writer = None
    # ============================================================================
    # Initialize episode result tracking for inference accuracy.
    # Save per-episode success/failure outcomes to JSON when inference ends.
    # ============================================================================
    episode_results = []  # Episode outcomes, for example [{"episode": 0, "success": True}, ...].
    current_episode_success = None  # None indicates that the current episode has not finished.
    current_video_path = None  # Temporary video path before appending the success/failure suffix.
    # ============================================================================
    # Initialize the current episode's action trajectory.
    # Enable trajectory recording with --save-action-trajectory.
    # Save each episode to a separate JSON file with a success/failure suffix.
    # ============================================================================
    current_episode_actions = []  # Actions recorded at each step of the current episode.
    # Reset task state before the first episode.
    task.reset()
    
    # ========== Main simulation loop ==========
    while simulation_app.is_running():  # Continue while the simulation application is running.
        # Step the simulation and render the frame.
        world.step(render=True)
        
        # Detect simulator pause events.
        if world.is_stopped():
            task_controller.reset_needed = True  # Mark the controller for reset.
            
        # Execute task logic only while the world is playing.
        if world.is_playing():
            # ========== Episode reset check ==========
            # A task or controller reset request marks an episode boundary.
            if task_controller.need_reset() or task.need_reset():
                # Release the current episode's video writer.
                if video_writer is not None:
                    video_writer.release()  # Release video writer resources.
                    video_writer = None  # Clear the video writer reference.

                # ============================================================================
                # Save the current episode's action trajectory to JSON.
                # Filename: episode_{N}_{success/fail}.json.
                # ============================================================================
                if args.save_action_trajectory and len(current_episode_actions) > 0:
                    ep_success_traj = task_controller._last_success  # Get the episode outcome.
                    ep_num_traj = task_controller._episode_num  # Get the episode index.
                    suffix_traj = "success" if ep_success_traj else "fail"  # Select the success/failure suffix.
                    # Create the action_episodes directory.
                    action_episodes_dir = os.path.join(cfg.multi_run.run_dir, "action_episodes")
                    os.makedirs(action_episodes_dir, exist_ok=True)
                    # Build the episode_{N}_{success/fail}.json path.
                    traj_filename = f"episode_{ep_num_traj}_{suffix_traj}.json"
                    traj_save_path = os.path.join(action_episodes_dir, traj_filename)
                    # Build the trajectory record.
                    trajectory_data = {
                        "episode": ep_num_traj,
                        "success": bool(ep_success_traj),
                        "total_steps": len(current_episode_actions),
                        "steps": current_episode_actions
                    }
                    # Write indented JSON for readability.
                    with open(traj_save_path, "w", encoding="utf-8") as traj_f:
                        json.dump(trajectory_data, traj_f, ensure_ascii=False, indent=2)
                    print(f"[Info] Episode {ep_num_traj} 动作轨迹已保存到: {traj_save_path}")
                    current_episode_actions = []  # Clear the current episode's trajectory list.

                # ============================================================================
                # Record inference outcomes and rename the episode video.
                # ============================================================================
                if cfg.mode == "infer":  # Record outcomes only in inference mode.
                    ep_success = task_controller._last_success  # Get the episode outcome.
                    ep_num = task_controller._episode_num  # Get the episode index.
                    episode_results.append({  # Append this episode's result.
                        "episode": ep_num,  # Episode index.
                        "success": bool(ep_success)  # Convert the success flag to a Python bool.
                    })
                    # Append a success/failure suffix to the video filename.
                    if current_video_path is not None and os.path.exists(current_video_path):  # A video exists for the current episode.
                        suffix = "success" if ep_success else "fail"  # Select the outcome suffix.
                        dir_name = os.path.dirname(current_video_path)  # Get the video directory.
                        base_name = os.path.splitext(os.path.basename(current_video_path))[0]  # Get the filename without its extension.
                        new_video_path = os.path.join(dir_name, f"{base_name}_{suffix}.mp4")  # Construct the path with the outcome suffix.
                        os.rename(current_video_path, new_video_path)  # Rename the video.
                    current_video_path = None  # Clear the current video path.

                # Reset the controller and print success statistics.
                task_controller.reset()
                
                # Check the configured episode limit.
                if task_controller.episode_num() >= cfg.max_episodes:
                    # ============================================================================
                    # Save inference accuracy statistics to JSON.
                    # Store episode outcomes and final accuracy beside the video directory.
                    # ============================================================================
                    if cfg.mode == "infer" and episode_results:  # Save only in inference mode when results are available.
                        total_episodes = len(episode_results)  # Total episode count.
                        success_episodes = sum(1 for r in episode_results if r["success"])  # Successful episode count.
                        final_accuracy = success_episodes / total_episodes if total_episodes > 0 else 0.0  # Compute final accuracy.
                        result_summary = {  # Build the result summary.
                            "task_name": cfg.name,  # Run name, for example pi0_level3_TransportBeaker_10000.
                            "total_episodes": total_episodes,  # Total episode count.
                            "success_episodes": success_episodes,  # Successful episode count.
                            "final_accuracy": round(final_accuracy, 4),  # Final accuracy rounded to four decimal places.
                            "final_accuracy_percent": f"{final_accuracy * 100:.2f}%",  # Accuracy formatted as a percentage.
                            "episode_details": episode_results  # Per-episode results.
                        }
                        json_save_path = os.path.join(cfg.multi_run.run_dir, "inference_results.json")  # Result JSON path.
                        with open(json_save_path, "w", encoding="utf-8") as jf:  # Write the JSON file.
                            json.dump(result_summary, jf, ensure_ascii=False, indent=2)  # Format the JSON output.
                        print(f"[Info] 推理结果已保存到: {json_save_path}")  # Print the output path.
                        print(f"[Info] 最终准确率: {result_summary['final_accuracy_percent']} ({success_episodes}/{total_episodes})")  # Print final accuracy.
                    
                    # ============================================================================
                    # Check whether another material phase is required.
                    # ============================================================================
                    if enable_dual_material_run and current_phase_idx < len(material_phases) - 1:
                        # Switch is_test_material and continue with the next phase.
                        current_phase_idx += 1
                        next_material_value = material_phases[current_phase_idx]
                        phase_label = f"is_test_material_{next_material_value}"
                        phase_run_dir = os.path.join(parent_run_dir, phase_label)
                        os.makedirs(phase_run_dir, exist_ok=True)
                        
                        # Update is_test_material and the output directory in cfg.
                        cfg_mutable = OmegaConf.to_container(cfg, resolve=False)
                        cfg_mutable['multi_run'] = {'run_dir': phase_run_dir}
                        cfg_mutable['infer']['is_test_material'] = next_material_value
                        cfg = OmegaConf.create(cfg_mutable)
                        OmegaConf.save(cfg, phase_run_dir + "/config.yaml")
                        
                        print(f"\n{'='*60}")
                        print(f"[Info] 第{current_phase_idx}轮完成，开始第{current_phase_idx+1}轮")
                        print(f"[Info] 切换 is_test_material = {next_material_value}")
                        print(f"[Info] 输出目录: {phase_run_dir}")
                        print(f"{'='*60}\n")
                        
                        # Reconfigure materials for the new is_test_material value.
                        task.cfg = cfg
                        task.setup_materials()
                        
                        # Close the current controller.
                        task_controller.close()
                        
                        # Recreate the controller with the updated cfg.
                        task_controller = create_controller(
                            cfg.controller_type,
                            cfg=cfg,
                            robot=robot,
                        )
                        
                        # Reset phase state variables.
                        episode_results = []
                        current_episode_success = None
                        current_video_path = None
                        current_episode_actions = []
                        video_writer = None
                        
                        # Reset task state.
                        task.reset()
                        continue  # Continue the main loop for the second phase.
                    
                    # All phases are complete; exit.
                    task_controller.close()  # Close the controller.
                    simulation_app.close()   # Close the simulation application.
                    cv2.destroyAllWindows()  # Close all OpenCV windows.
                    break  # Exit the main loop.
                    
                # Reset task state for the next episode.
                task.reset()
                
                continue  # Skip the remainder of this iteration.
                
            # ========== Task step ==========
            # Step the task and retrieve its current state.
            # State includes joint positions, camera images, object positions, and gripper position.
            state = task.step()
            if state is None:
                continue  # Skip invalid states, such as the first few simulator frames.
            
            # ========== Controller step ==========
            # Generate an action from the current state.
            # Collect mode executes scripted actions and records demonstrations.
            # Infer mode predicts actions with the policy.
            action, done, is_success = task_controller.step(state)
            
            # ========== Optional action trajectory recording ==========
            # Append per-step actions to current_episode_actions.
            if args.save_action_trajectory and action is not None:
                action_joint_pos = action.joint_positions
                if action_joint_pos is not None:
                    action_arr = np.array([x if x is not None else float('nan') for x in action_joint_pos])
                    # Ignore all-NaN actions, such as waiting stages in collection mode.
                    if not np.all(np.isnan(action_arr)):
                        step_data = {
                            "step": task.frame_idx,
                            "joint_positions": action_arr.tolist()
                        }
                        # Record observed joint positions when available.
                        if 'joint_positions' in state:
                            step_data["current_joint_positions"] = state['joint_positions'].tolist()
                        # Record observed gripper position when available.
                        if 'gripper_position' in state and state['gripper_position'] is not None:
                            gripper_pos = state['gripper_position']
                            step_data["gripper_position"] = gripper_pos.tolist() if hasattr(gripper_pos, 'tolist') else list(gripper_pos)
                        current_episode_actions.append(step_data)
            
            # ========== Optional live action logging (--print-action) ==========
            # Collect mode prints nine controller outputs: seven joints and two gripper fingers.
            # Infer mode also prints the raw and gripper-adjusted eight-dimensional VLA outputs.
            if args.print_action and action is not None:
                action_joint_pos = action.joint_positions
                if action_joint_pos is not None:
                    action_arr = np.array([x if x is not None else float('nan') for x in action_joint_pos])
                    # Ignore all-NaN actions, such as waiting stages in collection mode.
                    if not np.all(np.isnan(action_arr)):
                        print(f"[{cfg.mode}] Step {task.frame_idx} | Action ({len(action_arr)}-dim): {np.array2string(action_arr, precision=4, suppress_small=True)}")
                        # ============================================================================
                        # In inference mode, compare raw VLA output with gripper-adjusted actions.
                        # This comparison helps diagnose grasp behavior before and after tightening.
                        # Print raw -> adjusted gripper values for the currently executed step.
                        # ============================================================================
                        if cfg.mode == "infer" and hasattr(task_controller, 'inference_engine'):  # Only print when inference mode has an inference engine.
                            ie = task_controller.inference_engine  # Get the inference engine.
                            if ie.last_raw_actions is not None and ie.last_modified_actions is not None:  # Check that recorded action arrays are available.
                                # Find the action index corresponding to trajectory progress.
                                tc = ie.trajectory_controller  # Get the trajectory controller.
                                act_idx = max(0, tc._action_sequence_index - 1)  # The action index has already advanced; subtract one.
                                # Map action_sequence_index back to the original 40-step action chunk.
                                if hasattr(tc, 'gripper_indices') and len(tc.gripper_indices) > 0 and act_idx < len(tc.gripper_indices):
                                    raw_idx = tc.gripper_indices[act_idx]  # Use gripper_indices to recover the original chunk index.
                                else:
                                    raw_idx = min(act_idx, len(ie.last_raw_actions) - 1)  # Otherwise use the index directly, clipped to its valid range.
                                raw_gripper = ie.last_raw_actions[raw_idx, 7]       # Raw VLA gripper value.
                                mod_gripper = ie.last_modified_actions[raw_idx, 7]   # Adjusted gripper value.
                                delta = mod_gripper - raw_gripper                    # Compute the adjustment difference.
                                # Print raw -> adjusted gripper values and their difference.
                                print(f"  [Gripper] VLA原始: {raw_gripper:.4f} -> 增量后: {mod_gripper:.4f} (delta: {delta:.4f})")
            
            # Apply valid actions to robot joints.
            if action is not None:
                robot.get_articulation_controller().apply_action(action)
                
            # The current episode is complete.
            if done:
                task.on_task_complete(is_success)  # Notify the task of the episode outcome.
                continue  # Reset on the next iteration.
            
            # ========== Video recording ==========
            if save_video or show_video:
                camera_images = []
                # Gather preview images from all cameras.
                for _, image_data in state['camera_display'].items():
                    # Convert channel-first CHW images to HWC.
                    # Convert RGB to OpenCV's BGR color order.
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)
                
                if camera_images:
                    # Concatenate camera images horizontally.
                    combined_img = np.hstack(camera_images)
                    
                    # Add camera labels to the combined image.
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                        
                    # Display the live video window.
                    if show_video:
                        cv2.imshow('Camera Views', combined_img)
                        cv2.waitKey(1)
                        
                    # Save video frames to a file.
                    if save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")  # Video output directory.
                        os.makedirs(output_dir, exist_ok=True)  # Create the directory if needed.
                        # Initially use a temporary video name without an outcome suffix.
                        # Rename it to episode_X_success.mp4 or episode_X_fail.mp4 after completion.
                        output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")  # Temporary video path.
                        
                        # Initialize one video writer per episode.
                        if video_writer is None:
                            height, width = combined_img.shape[:2]  # Get the image dimensions.
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the mp4v codec.
                            video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))  # Create the video writer.
                            current_video_path = output_path  # Remember the path for renaming after the episode.
                            
                        video_writer.write(combined_img)  # Write the current frame.


import traceback  # Traceback utilities print detailed exception stacks.
import os  # Import os for os._exit.

if __name__ == "__main__":
    """Program entry point.

    The guarded shutdown prints uncaught exceptions and uses os._exit(0) to
    avoid segmentation faults during Isaac Sim cleanup.
    """
    try:
        main()  # Run the main function.
    except Exception as e:
        # Catch exceptions and print diagnostic details.
        print("\n" + "="*60)
        print("🛑 CAUGHT EXCEPTION (程序异常退出):")
        print("="*60)
        traceback.print_exc()  # Print the full exception traceback.
        print("="*60 + "\n")
    finally:
        # Terminate the process immediately.
        # os._exit(0) skips Isaac Sim cleanup instead of using sys.exit().
        # Forced exit avoids segmentation faults during Isaac Sim cleanup.
        print("[Info] Forcing exit with os._exit(0) to prevent segfault...")
        os._exit(0)
