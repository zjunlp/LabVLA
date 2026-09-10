import h5py
import numpy as np
import os
from glob import glob
from datetime import datetime

class PickDataCollector:
    def __init__(self, save_dir="outputs/nav", max_episodes=10):
        """Initialize the pick task data collector
        
        Args:
            save_dir (str): Root directory for saving data
            max_episodes (int): Maximum number of episodes to save
        """
        self.save_dir = save_dir
        # Add timestamp to session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(save_dir, f"nav_dataset_{timestamp}")
        self.max_episodes = max_episodes
        self.current_episode = 0
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize temporary storage
        self.cache = []
        
    def cache_step(self, joint_positions, robot_position, object_position, **kwargs):
        """Cache one step of data including camera images
        
        Args:
            joint_positions: Robot joint angles
            robot_position: Robot position
            object_position: Object position
            kwargs: Additional data such as camera images
        """
        step_data = {
            'joint_positions': joint_positions,
            'robot_position': robot_position,
            'object_position': object_position
        }
        # Add camera data
        step_data.update(kwargs)
        self.cache.append(step_data)
        
    def save_episode(self, episode_id, success=True):
        """Save the episode data to an HDF5 file
        
        Args:
            episode_id: Episode identifier
            success: Whether the episode was successful
        Returns:
            bool: Whether the save was successful
        """
        if not self.cache or self.current_episode >= self.max_episodes:
            return False
            
        episode_path = os.path.join(self.session_dir, f"episode_{self.current_episode:04d}.h5")
        
        with h5py.File(episode_path, 'w') as f:
            # Store episode metadata
            f.attrs['episode_id'] = episode_id
            f.attrs['success'] = success
            
            # Get the first step to determine data structure
            first_step = self.cache[0]
            
            # Create datasets for each key in the data
            for key, value in first_step.items():
                shape = (len(self.cache),) + np.array(value).shape
                dtype = 'float32' if key not in ['cam_01_rgb', 'cam_02_rgb', 'hand_cam_rgb'] else 'uint8'
                f.create_dataset(key, shape, dtype=dtype)
            
            # Fill datasets
            for i, step_data in enumerate(self.cache):
                for key, value in step_data.items():
                    f[key][i] = value
                    
        self.current_episode += 1
        self.cache = []
        return True
        
    def clear_cache(self):
        """Clear the cached data"""
        self.cache = []
        
    def is_collection_complete(self):
        """Check if we've collected maximum number of episodes"""
        return self.current_episode >= self.max_episodes

    def close(self):
        """Close the data collector and merge all episode files"""
        # Merge all episode files into a single HDF5 file
        merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
        episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*.h5")))
        
        if not episode_files:
            print("No episodes to merge")
            return
            
        with h5py.File(merged_path, 'w') as merged_file:
            # Copy each episode file into the merged file
            for episode_path in episode_files:
                episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                with h5py.File(episode_path, 'r') as episode_file:
                    # Create episode group in merged file
                    episode_group = merged_file.create_group(episode_name)
                    
                    # Copy all datasets with their original settings
                    for key in episode_file.keys():
                        episode_file.copy(key, episode_group)
                    
                    # Copy success attribute
                    if 'success' in episode_file.attrs:
                        episode_group.attrs['success'] = episode_file.attrs['success']
                
                # Remove individual episode file after merging
                os.remove(episode_path)
                
        # Rename merged file to final name
        final_path = os.path.join(self.session_dir, "data.hdf5")
        os.rename(merged_path, final_path)
        print(f"Successfully merged {len(episode_files)} episodes into {final_path}")
