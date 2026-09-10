"""
Script to convert LabSim dataset to LeRobot format.

LabSim dataset uses HDF5 format containing camera images, robot joint angles and action data.
This script converts it to LeRobot standard format.

Usage:
python scripts/convert_labsim_data_to_lerobot.py --data_dir /path/to/your/labsim/dataset --num_processes 4

To push to Hugging Face Hub:
python scripts/convert_labsim_data_to_lerobot.py --data_dir /path/to/your/labsim/dataset --push_to_hub --num_processes 8

Note: This script requires LeRobot installation:
pip install lerobot
"""

import os
import h5py
import numpy as np
import tyro
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import queue
import time

# Try to import LeRobot modules
try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed, please run: pip install lerobot")


def get_image_shape_from_h5(h5_file: h5py.File, camera_name: str) -> tuple:
    """Get image shape from HDF5 file"""
    # Get camera data shape from first episode group
    for key in h5_file.keys():
        episode_group = h5_file[key]
        if isinstance(episode_group, h5py.Group) and camera_name in episode_group:
            dataset = episode_group[camera_name]
            if hasattr(dataset, 'shape'):
                if len(dataset.shape) == 4:  # [T, H, W, C]
                    return dataset.shape[1:]  # Return [H, W, C]
                elif len(dataset.shape) == 3:  # [H, W, C]
                    return dataset.shape
            break
    return (256, 256, 3)  # Default shape


def get_state_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get state data shape from HDF5 file"""
    # Get state data shape from first episode group
    for key in h5_file.keys():
        episode_group = h5_file[key]
        if isinstance(episode_group, h5py.Group) and "agent_pose" in episode_group:
            dataset = episode_group["agent_pose"]
            if hasattr(dataset, 'shape'):
                if len(dataset.shape) == 2:  # [T, num_joints]
                    return (dataset.shape[1],)  # Return [num_joints]
                elif len(dataset.shape) == 1:  # [num_joints]
                    return (dataset.shape[0],)
            break
    return (7,)  # Default joint count


def get_action_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get action data shape from HDF5 file"""
    # Get action data shape from first episode group
    for key in h5_file.keys():
        episode_group = h5_file[key]
        if isinstance(episode_group, h5py.Group) and "actions" in episode_group:
            dataset = episode_group["actions"]
            if hasattr(dataset, 'shape'):
                if len(dataset.shape) == 2:  # [T, num_joints]
                    return (dataset.shape[1],)  # Return [num_joints]
                elif len(dataset.shape) == 1:  # [num_joints]
                    return (dataset.shape[0],)
            break
    return (7,)  # Default joint count


def detect_camera_names(h5_file: h5py.File) -> list:
    """Detect camera names in HDF5 file"""
    camera_names = []
    
    # First find an episode group to detect camera names
    for key in h5_file.keys():
        episode_group = h5_file[key]
        if isinstance(episode_group, h5py.Group):
            # Look for camera data in episode group
            for data_key in episode_group.keys():
                if data_key not in ["agent_pose", "actions"]:
                    # Check if it's image data (by shape)
                    dataset = episode_group[data_key]
                    if hasattr(dataset, 'shape') and len(dataset.shape) >= 3:
                        # Assume 3D or 4D data is image
                        if data_key not in camera_names:
                            camera_names.append(data_key)
            break  # Only need to check first episode group
    
    return camera_names


def get_source_fps_from_h5(h5_file: h5py.File) -> int:
    """Detect source fps from HDF5 file metadata or estimate from data"""
    # Try to get fps from metadata first
    if 'fps' in h5_file.attrs:
        return int(h5_file.attrs['fps'])
    
    # Estimate fps from episode duration and frame count
    for key in h5_file.keys():
        episode_group = h5_file[key]
        if isinstance(episode_group, h5py.Group):
            # Look for duration metadata
            if 'duration' in episode_group.attrs:
                duration = episode_group.attrs['duration']
                # Find frame count from any camera data
                for data_key in episode_group.keys():
                    if data_key not in ["agent_pose", "actions"]:
                        dataset = episode_group[data_key]
                        if hasattr(dataset, 'shape') and len(dataset.shape) >= 3:
                            frame_count = dataset.shape[0] if len(dataset.shape) == 4 else 1
                            if duration > 0:
                                estimated_fps = frame_count / duration
                                return int(estimated_fps)
            break
    
    return 60  # Default assumption


def check_language_instructions_availability(h5_file: h5py.File) -> tuple:
    """Check if language instructions are available in the dataset"""
    episodes_with_instructions = 0
    total_episodes = 0
    
    for key in h5_file.keys():
        episode_group = h5_file[key]
        if isinstance(episode_group, h5py.Group):
            total_episodes += 1
            if "language_instruction" in episode_group:
                episodes_with_instructions += 1
    
    return episodes_with_instructions, total_episodes


def calculate_frame_indices(source_fps: int, target_fps: int, total_frames: int) -> list:
    """Calculate which frames to keep when converting fps"""
    if source_fps <= target_fps:
        # If source fps is lower or equal, keep all frames
        return list(range(total_frames))
    
    # Calculate frame interval
    interval = source_fps / target_fps
    
    # Generate frame indices to keep
    frame_indices = []
    current_frame = 0
    
    while current_frame < total_frames:
        frame_indices.append(int(current_frame))
        current_frame += interval
    
    return frame_indices


def process_episode(args):
    """Process a single episode - designed to be run in parallel"""
    episode_file, episode_name, camera_names, source_fps, target_fps, image_shape, state_shape, action_shape = args
    
    try:
        frame_data_list = []
        
        with h5py.File(episode_file, 'r') as h5_file:
            episode_group = h5_file[episode_name]
            
            # Get time steps
            if camera_names and camera_names[0] in episode_group:
                time_steps = episode_group[camera_names[0]].shape[0]
            elif "agent_pose" in episode_group:
                time_steps = episode_group["agent_pose"].shape[0]
            else:
                return None, f"No valid data found in episode {episode_name}"
            
            # Get language instruction if available
            language_instruction = None
            if "language_instruction" in episode_group:
                try:
                    # Read the HDF5 dataset content
                    instruction_data = episode_group["language_instruction"]
                    if hasattr(instruction_data, 'asstr'):
                        # For string datasets
                        language_instruction = instruction_data.asstr()[()]
                    elif hasattr(instruction_data, 'decode'):
                        # For bytes datasets
                        language_instruction = instruction_data.decode('utf-8')
                    else:
                        # For other types, try to get the actual value
                        language_instruction = instruction_data[()]
                        if isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                        elif isinstance(language_instruction, np.ndarray):
                            language_instruction = str(language_instruction.item())
                        else:
                            language_instruction = str(language_instruction)
                    print(f"Found language instruction for episode {episode_name}: {language_instruction}")
                except Exception as e:
                    print(f"Warning: Could not decode language instruction for episode {episode_name}: {e}")
            
            # Calculate which frames to keep based on fps conversion
            frame_indices = calculate_frame_indices(source_fps, target_fps, time_steps)
            
            # Process each frame
            for t in frame_indices:
                frame_data = {}
                
                # Add camera data
                for camera_name in camera_names:
                    if camera_name in episode_group:
                        camera_data = episode_group[camera_name]
                        if len(camera_data.shape) == 4:  # [T, H, W, C]
                            frame_data[camera_name] = camera_data[t]
                        else:  # [H, W, C]
                            frame_data[camera_name] = camera_data
                
                # Add state data
                if "agent_pose" in episode_group:
                    pose_data = episode_group["agent_pose"]
                    if len(pose_data.shape) == 2:  # [T, num_joints]
                        frame_data["state"] = pose_data[t]
                    else:  # [num_joints]
                        frame_data["state"] = pose_data
                
                # Add action data
                if "actions" in episode_group:
                    action_data = episode_group["actions"]
                    if len(action_data.shape) == 2:  # [T, num_joints]
                        frame_data["actions"] = action_data[t]
                    else:  # [num_joints]
                        frame_data["actions"] = action_data
                
                # Set task description - use language instruction if available, otherwise use episode name
                if language_instruction:
                    frame_data["task"] = language_instruction
                else:
                    frame_data["task"] = f"LabSim episode {episode_name}"
                
                frame_data_list.append(frame_data)
        
        return frame_data_list, None
    
    except Exception as e:
        return None, f"Error processing episode {episode_name}: {str(e)}"


def main(data_dir: str, repo_name: str, *, push_to_hub: bool = False, fps: int = 60, robot_type: str = "franka", num_processes: int = 4):
    """Main conversion function
    
    Args:
        data_dir: Path to the LabSim dataset directory
        push_to_hub: Whether to push to Hugging Face Hub
        fps: Target fps for conversion
        robot_type: Type of robot (default: franka)
        num_processes: Number of processes for parallel processing (default: 4)
    """
    if not LEROBOT_AVAILABLE:
        print("Error: LeRobot not installed, cannot continue conversion")
        return
    
    # Validate num_processes
    if num_processes < 1:
        num_processes = 1
    max_processes = mp.cpu_count()
    if num_processes > max_processes:
        print(f"Warning: Requested {num_processes} processes, but only {max_processes} CPUs available. Using {max_processes} processes.")
        num_processes = max_processes

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        return

    # Find episode_data.hdf5 file
    episode_file = data_path / "episode_data.hdf5"
    if not episode_file.exists():
        print(f"Error: episode_data.hdf5 file not found at: {episode_file}")
        return

    # Clean output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"Reading dataset: {episode_file}")
    
    # First read HDF5 file to determine data structure
    with h5py.File(episode_file, 'r') as h5_file:
        # Detect camera names
        camera_names = detect_camera_names(h5_file)
        print(f"Detected cameras: {camera_names}")
        
        # Check language instructions availability
        episodes_with_instructions, total_episodes = check_language_instructions_availability(h5_file)
        print(f"Language instructions: {episodes_with_instructions}/{total_episodes} episodes have language instructions")
        
        # Get data shapes
        if camera_names:
            image_shape = get_image_shape_from_h5(h5_file, camera_names[0])
        else:
            image_shape = (256, 256, 3)
        
        state_shape = get_state_shape_from_h5(h5_file)
        action_shape = get_action_shape_from_h5(h5_file)
        
        # Detect source fps
        source_fps = get_source_fps_from_h5(h5_file)
        print(f"Source fps: {source_fps}, Target fps: {fps}")
        
        print(f"Image shape: {image_shape}")
        print(f"State shape: {state_shape}")
        print(f"Action shape: {action_shape}")

    # Create LeRobot dataset
    features = {}
    
    # Add camera features
    for camera_name in camera_names:
        features[camera_name] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    
    # Add state and action features
    features["state"] = {
        "dtype": "float32",
        "shape": state_shape,
        "names": ["state"],
    }
    
    features["actions"] = {
        "dtype": "float32",
        "shape": action_shape,
        "names": ["actions"],
    }

    print("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=8,
    )

    # Read and convert data using multiprocessing
    print(f"Converting data using {num_processes} processes...")
    
    # Get list of episodes for parallel processing
    episode_names = []
    with h5py.File(episode_file, 'r') as h5_file:
        for episode_name in h5_file.keys():
            episode_group = h5_file[episode_name]
            if isinstance(episode_group, h5py.Group):
                episode_names.append(episode_name)
    
    print(f"Found {len(episode_names)} episodes to process")
    
    # Prepare arguments for parallel processing
    process_args = [
        (episode_file, episode_name, camera_names, source_fps, fps, image_shape, state_shape, action_shape)
        for episode_name in episode_names
    ]
    
    # Use multiprocessing to process episodes in parallel
    episode_count = 0
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_episode = {executor.submit(process_episode, args): args[1] for args in process_args}
        
        # Process results as they complete
        for future in future_to_episode:
            episode_name = future_to_episode[future]
            try:
                frame_data_list, error = future.result()
                if error:
                    print(f"Error: {error}")
                    continue
                
                print(f"Processing episode: {episode_name} ({len(frame_data_list)} frames)")
                
                # Add frames to dataset sequentially (thread-safe)
                for frame_data in frame_data_list:
                    # dataset.add_frame(frame_data)
                    task_description = frame_data.pop("task")
                    # Pass task as a separate argument.
                    dataset.add_frame(frame_data, task=task_description)
                
                # Save episode
                dataset.save_episode()
                episode_count += 1
                
            except Exception as e:
                print(f"Exception processing episode {episode_name}: {str(e)}")
                continue

    print(f"Successfully converted {episode_count} episodes")

    # Optional: Push to Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["labsim", robot_type, "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset already pushed to Hugging Face Hub")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_name}")
    else:
        print(f"Dataset saved locally to: {output_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensure proper multiprocessing on all platforms
    tyro.cli(main) 