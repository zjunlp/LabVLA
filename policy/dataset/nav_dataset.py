import copy
import os
import numpy as np
import h5py
import torch
import cv2  # Add this import at the top with other imports
from typing import Dict, Optional
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer
from policy.model.common.normalizer import SingleFieldLinearNormalizer
from policy.common.normalize_util import get_image_range_normalizer
from torch.nn.utils.rnn import pad_sequence

class NavDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta,
                 dataset_path: str,
                 seed: int = 42,
                 horizon: int = None,
                 n_obs_steps: int = None,
                 val_ratio: float = 0.00,
                 max_train_episodes: Optional[int] = None):
        
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        
        self.h5_file = h5py.File(dataset_path, 'r')
        self.episode_ids = list(self.h5_file.keys())
        self.camera_names = ['cam_01', 'cam_02']
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        self.sequences = []
        for episode_id in self.episode_ids:
            episode = self.h5_file[episode_id]
            n_steps = episode['joint_positions'].shape[0]
            for start_idx in range(n_steps-self.horizon-1):
                self.sequences.append((episode_id, start_idx))

    def __len__(self):
        return len(self.sequences)

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode_id in self.episode_ids:
            episode = self.h5_file[episode_id]
            actions = torch.from_numpy(episode['joint_positions'][:].astype(np.float32))
            all_actions.append(actions)
        return torch.cat(all_actions, dim=0)
    
    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        
        # Joint positions normalizer
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy())
        
        # Robot pose normalizer
        all_poses = []
        all_position = []
        for episode_id in self.episode_ids:
            episode = self.h5_file[episode_id]
            poses = episode['joint_positions'][:].astype(np.float32)
            positions = episode['robot_position'][:].astype(np.float32)
            all_position.append(positions)
            all_poses.append(poses)
        all_poses = np.concatenate(all_poses, axis=0)
        normalizer['joint_positions'] = SingleFieldLinearNormalizer.create_fit(all_poses)
        
        # Image normalizers
        normalizer['cam_01'] = get_image_range_normalizer()
        normalizer['cam_02'] = get_image_range_normalizer()
        normalizer['cam_03'] = get_image_range_normalizer()
        normalizer['robot_position'] = SingleFieldLinearNormalizer.create_fit(np.concatenate(all_position, axis=0))
        
        return normalizer
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        return val_set
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_id, start_idx = self.sequences[index]
        episode = self.h5_file[episode_id]
        
        # Get observations
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps

        cam1_obs = np.transpose(episode['cam_01_rgb'][obs_start_idx:obs_end_idx], (0,3,1,2))
        cam2_obs = np.transpose(episode['cam_02_rgb'][obs_start_idx:obs_end_idx], (0,3,1,2))
        cam3_obs = np.transpose(episode['hand_cam_rgb'][obs_start_idx:obs_end_idx], (0,3,1,2))
        robot_position = episode['robot_position'][obs_start_idx:obs_end_idx]
        joint_positions = episode['joint_positions'][obs_start_idx:obs_end_idx]
        
        action_end_idx = obs_start_idx + self.horizon
        action_start_idx = obs_start_idx
        action = episode['joint_positions'][action_start_idx:action_end_idx]

        # Convert to tensors
        cam1_obs = torch.from_numpy(cam1_obs).float() / 255.0
        cam2_obs = torch.from_numpy(cam2_obs).float() / 255.0
        cam3_obs = torch.from_numpy(cam3_obs).float() / 255.0
        robot_position = torch.from_numpy(robot_position).float()
        joint_positions = torch.from_numpy(joint_positions).float()
        action = torch.from_numpy(action).float()
        
        return {
            'obs': {
                'cam_01': cam1_obs,
                'cam_02': cam2_obs,
                'cam_03': cam3_obs,
                'joint_positions': joint_positions,
                'robot_position': robot_position,
            },
            'action': action,
        }

    @staticmethod
    def collate_fn(batch):
        cam1_images = torch.stack([item['obs']['cam_01'] for item in batch])
        cam2_images = torch.stack([item['obs']['cam_02'] for item in batch])
        cam3_images = torch.stack([item['obs']['cam_03'] for item in batch])
        joint_positions = torch.stack([item['obs']['joint_positions'] for item in batch])
        robot_positions = torch.stack([item['obs']['robot_position'] for item in batch])
        action = torch.stack([item['action'] for item in batch])
    
        return {
            'obs': {
                'cam_01': cam1_images,
                'cam_02': cam2_images,
                'cam_03': cam3_images,
                'joint_positions': joint_positions,
                
                'robot_position': robot_positions,
            },
            'action': action,
        }

    def save_episode_to_video(self, episode_id: str, output_dir: str):
        """
        Save camera views of an episode as MP4 videos.
        
        Args:
            episode_id: ID of the episode to save
            output_dir: Directory to save the video files
        """
        episode = self.h5_file[episode_id]
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all frames for each camera
        cam1_frames = episode['cam_01_rgb'][:]
        cam2_frames = episode['cam_02_rgb'][:]
        cam3_frames = episode['hand_cam_rgb'][:]
        
        # Video parameters
        fps = 60
        
        # Define video writers for each camera
        writers = {
            'cam_01': cv2.VideoWriter(
                os.path.join(output_dir, f'{episode_id}_cam1.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (cam1_frames.shape[2], cam1_frames.shape[1])
            ),
            'cam_02': cv2.VideoWriter(
                os.path.join(output_dir, f'{episode_id}_cam2.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (cam2_frames.shape[2], cam2_frames.shape[1])
            ),
            'cam_03': cv2.VideoWriter(
                os.path.join(output_dir, f'{episode_id}_cam3.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (cam3_frames.shape[2], cam3_frames.shape[1])
            )
        }
        
        # Write frames
        for i in range(len(cam1_frames)):
            # Convert from RGB to BGR for OpenCV
            writers['cam_01'].write(cv2.cvtColor(cam1_frames[i], cv2.COLOR_RGB2BGR))
            writers['cam_02'].write(cv2.cvtColor(cam2_frames[i], cv2.COLOR_RGB2BGR))
            writers['cam_03'].write(cv2.cvtColor(cam3_frames[i], cv2.COLOR_RGB2BGR))
        
        # Release all writers
        for writer in writers.values():
            writer.release()

def main():
    
    dataset_path = 'outputs/nav/nav_dataset_20250523_141324/data.hdf5'
    shape_meta = {
        'cam_01': (3, 640, 640),
        'cam_02': (3, 640, 640),
        'cam_03': (3, 640, 640),
        'agent_pose': (12,)
    }
    horizon = 8
    n_obs_steps = 3
    batch_size = 2
    
    dataset = NavDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        val_ratio=0.1,
        seed=42
    )

    # Save videos for first episode
    output_dir = 'outputs/nav/episode_videos'
    first_episode_id = dataset.episode_ids[0]
    dataset.save_episode_to_video(first_episode_id, output_dir)
    print(f"Videos saved to {output_dir}")

    val_dataset = dataset.get_validation_dataset()

    print(f": {len(dataset)}")
    print(f": {len(val_dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=NavDataset.collate_fn,
        shuffle=True
    )

    for batch in train_loader:
        print("\n:")
        print(f"Camera 1 shape: {batch['obs']['cam_01'].shape}")  # [B,T,3,H,W]
        print(f"Camera 2 shape: {batch['obs']['cam_02'].shape}")  # [B,T,3,H,W]
        print(f"Camera 3 shape: {batch['obs']['cam_03'].shape}")  # [B,T,3,H,W]
        print(f"Agent pose shape: {batch['obs']['joint_positions'].shape}")  # [B,T,7]
        print(f"Actions shape: {batch['action'].shape}")  # [B,T,dim]
        print("\n:")
        print("Agent pose (first frame):")
        print(batch['obs']['joint_positions'][0, 0])
        print("\nAction (first frame):")
        print(batch['action'][0, 0])
        break

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    main()
