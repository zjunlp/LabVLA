# ============================================================================
# data_collector.py: core data collection module.
# Collect and save camera images, joint states, and robot actions.
# Storage format: HDF5 (.h5).
# ============================================================================

import os  # Operating system utilities.
import numpy as np  # NumPy.
import cv2  # OpenCV image processing.
from datetime import datetime  # Date and time utilities.
import json  # JSON handling.
import h5py  # HDF5 handling.
from concurrent.futures import ProcessPoolExecutor, Future  # Process pool executor.
from typing import List, Optional  # Type annotations.
from glob import glob  # File pattern matching.


def _write_episode_data(episode_path: str, episode_name: str, 
                       camera_data: dict, agent_pose_data: np.ndarray, 
                       actions_data: np.ndarray, language_instruction: Optional[str] = None, compression=None):
    """Write one episode to a separate HDF5 file in a worker process. episode_path and episode_name identify the output; camera_data maps names to image arrays [T, C, H, W], agent_pose_data has shape [T, num_joints], and actions_data has shape [T, action_dim]. language_instruction is optional. compression=None disables compression; "gzip" enables gzip."""
    
    with h5py.File(episode_path, 'w') as h5_file:  # Open the HDF5 file for writing.
        print(f"Writing episode {episode_name} to {episode_path}")  # Log the write operation.
        
        # Store camera arrays with dynamic chunks and optional compression.
        for camera_name, image_data in camera_data.items():  # Iterate over camera arrays.
            # Limit each chunk to 64 frames along the time axis.
            chunk_size = (min(64, image_data.shape[0]),) + image_data.shape[1:]
            kwargs = {  # Prepare dataset creation arguments.
                'data': image_data,  # Image array.
                'dtype': 'uint8',  # Unsigned 8-bit integer dtype.
                'chunks': chunk_size  # Chunk shape.
            }
            if compression == "gzip":  # Enable gzip compression if requested.
                kwargs.update({
                    'compression': "gzip",  # Compression method.
                    'compression_opts': 5  # Compression level (1-9).
                })
            h5_file.create_dataset(camera_name, **kwargs)  # Create the camera dataset.
        
        # Store joint positions without compression: the array is small and frequently accessed.
        h5_file.create_dataset(
            "agent_pose",  # Dataset name.
            data=agent_pose_data,  # Joint angle array.
            dtype='float32',  # 32-bit floating-point dtype.
            chunks=True  # Enable chunked storage.
        )
        # Store the action array.
        h5_file.create_dataset(
            "actions",  # Dataset name.
            data=actions_data,  # Action array.
            dtype='float32',  # 32-bit floating-point dtype.
            chunks=True  # Enable chunked storage.
        )
        
        # Store the language instruction when supplied.
        if language_instruction is not None:
            h5_file.create_dataset(
                "language_instruction",  # Dataset name.
                data=language_instruction,  # Language instruction string.
                dtype=h5py.special_dtype(vlen=str)  # Variable-length string dtype.
            )
        
        print(f"Finished writing episode {episode_name}")  # Log completion.


class DataCollector:
    """Collect multi-camera RGB images, robot joint angles (agent_pose), actions, and an optional language_instruction. Save each episode as a separate .h5 file using asynchronous worker processes."""
    
    def __init__(self, camera_configs: List[dict], save_dir="output", 
                 max_episodes=10, max_workers=4, compression=None):
        """Initialize the collector. camera_configs contains name and image_type entries; save_dir is the output root; max_episodes limits collection; max_workers limits parallel writers; compression is None or "gzip"."""
        self.save_dir = save_dir  # Output root.
        self.max_episodes = max_episodes  # Maximum episode count.
        self.compression = compression  # Compression method.
        self.session_dir = os.path.join(save_dir, "dataset")  # Dataset directory.
        self.mate_dir = os.path.join(self.session_dir, "meta")  # Metadata directory.
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")  # Episode index file path.
        self.episode_count = 0  # Current episode count.
        self.camera_configs = camera_configs  # Camera configuration.
        self.task_instructions = None  # Task instruction.
        
        # Create required directories.
        os.makedirs(self.session_dir, exist_ok=True)  # Create the dataset directory.
        os.makedirs(self.mate_dir, exist_ok=True)  # Create the metadata directory.
        
        # Initialize per-camera temporary buffers.
        # Support combined image types such as "rgb+depth".
        self.temp_cameras = {}
        for config in camera_configs:  # Iterate over camera configurations.
            if '+' in config['image_type']:  # Handle a combined image type.
                types = config['image_type'].split('+')  # Split the image types.
                for t in types:  # Create a buffer for each type.
                    self.temp_cameras[f"{config['name']}_{t}"] = []
            else:  # Handle a single image type.
                self.temp_cameras[f"{config['name']}_{config['image_type']}"] = []
        
        self.temp_agent_pose = []  # Temporary joint angle buffer.
        self.temp_actions = []  # Temporary action buffer.
        self.temp_language_instruction = None  # Temporary language instruction.
        
        # Initialize the worker pool and pending task list.
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)  # Create the process pool.
        self.pending_futures: List[Future] = []  # Pending asynchronous write tasks.
        
    def cache_step(self, camera_images: dict, joint_angles: np.ndarray, language_instruction: Optional[str] = None):
        """Cache each simulation frame until the episode ends. camera_images maps camera_name_type to an image array; joint_angles contains current joint angles, and language_instruction is optional."""
        # Set the task instruction on first use.
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = language_instruction
        # Cache camera images.
        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(image)
        # Cache joint angles.
        self.temp_agent_pose.append(joint_angles)
        # Update the language instruction.
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
        
    def write_cached_data(self, final_joint_positions):
        """Convert the completed episode's buffers to NumPy arrays and submit an asynchronous write. final_joint_positions supplies the action target for the final frame."""
        # Check the maximum episode count.
        if self.episode_count >= self.max_episodes:
            self.close()
            return
            
        # Use the next frame's joint angles as the current frame's action target.
        self.temp_actions = self.temp_agent_pose[1:] + [final_joint_positions]
        
        # Convert buffered lists to NumPy arrays.
        camera_data = {
            name: np.array(images) 
            for name, images in self.temp_cameras.items()
        }
        agent_pose_data = np.array(self.temp_agent_pose)  # Joint angle array.
        actions_data = np.array(self.temp_actions)  # Action array.

        # Construct the episode file path.
        episode_name = f"episode_{self.episode_count:04d}"  # Format the name, for example episode_0001.
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")  # Full path.
        
        # Submit the write task to the process pool.
        future = self.process_pool.submit(
            _write_episode_data,
            episode_path,
            episode_name,
            camera_data,
            agent_pose_data,
            actions_data,
            self.temp_language_instruction,
            self.compression
        )
        self.pending_futures.append(future)  # Track the pending task.

        # Append episode metadata to the JSONL index.
        info = {
            "episode_index": self.episode_count,  # Episode index.
            "tasks": [self.task_instructions] if self.task_instructions else [],  # Task list.
            "length": len(self.temp_agent_pose)  # Episode length in frames.
        }
        
        with open(self.episode_file_path, "a", encoding="utf-8") as f:  # Open in append mode.
            f.write(json.dumps(info, ensure_ascii=False) + "\n")  # Write one JSON line.
        
        # Clear the buffers.
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        
        # Increment the episode count.
        self.episode_count += 1

    def clear_cache(self):
        """Clear temporary episode data without writing it, for failed or discarded episodes."""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.task_instructions = None
        
    def close(self, merge=True):
        """Wait for pending writes and shut down the worker pool. If merge is True, merge all episode files into one HDF5 file."""
        # Wait for pending write tasks.
        for future in self.pending_futures:
            future.result()
        
        # Shut down the process pool.
        self.process_pool.shutdown(wait=True)
        
        # Optionally merge all episode files.
        if merge:
            merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
            episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*.h5")))
            
            if not episode_files:
                print("No episodes to merge")
                return
                
            with h5py.File(merged_path, 'w') as merged_file:
                # Copy each episode file into the merged file.
                for episode_path in episode_files:
                    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                    with h5py.File(episode_path, 'r') as episode_file:
                        # Create an episode group in the merged file.
                        episode_group = merged_file.create_group(episode_name)
                        
                        # Copy datasets while preserving compression settings.
                        for key in episode_file.keys():
                            episode_file.copy(key, episode_group)
                    
                    # Delete the original episode file after merging it.
                    os.remove(episode_path)
            # Rename the merged file.
            os.rename(merged_path, os.path.join(self.session_dir, "episode_data.hdf5"))
            print(f"Successfully merged {len(episode_files)} episodes into {merged_path}")
