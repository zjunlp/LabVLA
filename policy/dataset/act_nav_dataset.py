import copy
import numpy as np
import h5py
import torch
from typing import Dict, Optional
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer
from policy.model.common.normalizer import SingleFieldLinearNormalizer
from policy.common.normalize_util import get_image_range_normalizer
from torch.nn.utils.rnn import pad_sequence

class ACTNavDataset(BaseImageDataset):
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
        self.camera_names = ['cam_01', 'cam_02', 'cam_03']
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        self.sequences = []
            
        for episode_id in self.episode_ids:
            episode = self.h5_file[episode_id]
            n_steps = episode['joint_positions'].shape[0]
            for start_idx in range(n_steps):
                self.sequences.append((episode_id, start_idx, n_steps))

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        
        
        all_actions = []
        all_poses = []
        all_positions = []
        for episode_id in self.episode_ids:
            episode = self.h5_file[episode_id]
            poses = episode['joint_positions'][:].astype(np.float32)
            positions = episode['robot_position'][:].astype(np.float32)
            all_actions.append(poses)
            all_positions.append(positions)
            all_poses.append(poses)

        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(all_actions, axis=0))
        normalizer['agent_pose'] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(all_poses, axis=0))
        normalizer['robot_position'] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(all_positions, axis=0))
        
        
        for cam in self.camera_names:
            normalizer[cam] = get_image_range_normalizer()
        
        return normalizer

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_id, start_idx, episode_len = self.sequences[index]
        episode = self.h5_file[episode_id]
        
        
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        
        
        obs_dict = {}
        for cam in self.camera_names:
            if cam == 'cam_03':
                cam_data = episode['hand_cam_rgb'][obs_start_idx]
            else:
                cam_data = episode[f'{cam}_rgb'][obs_start_idx]
            obs_dict[cam] = torch.from_numpy(
                np.transpose(cam_data, (2,0,1))).float() / 255.0

        
        joint_positions = episode['joint_positions'][obs_start_idx]
        robot_position = episode['robot_position'][obs_start_idx]
        
        
        action = episode['joint_positions'][obs_start_idx:]
        padded_action = np.zeros((episode_len, action.shape[-1]), dtype=np.float32)
        padded_action[:action.shape[0]] = action
        
        
        is_pad = np.zeros(episode_len)
        is_pad[action.shape[0]:] = 1
        
        return {
            'obs': {
                'cam_01': obs_dict['cam_01'],
                'cam_02': obs_dict['cam_02'],
                'cam_03': obs_dict['cam_03'],
                'agent_pose': torch.from_numpy(joint_positions).float(),
                'robot_position': torch.from_numpy(robot_position).float(),
            },
            'action': torch.from_numpy(padded_action).float(),
            'is_pad': torch.from_numpy(is_pad).bool()
        }

    @staticmethod
    def collate_fn(batch):
        
        cam1_images = [item['obs']['cam_01'] for item in batch]
        cam2_images = [item['obs']['cam_02'] for item in batch]
        cam3_images = [item['obs']['cam_03'] for item in batch]
        joint_positions = [item['obs']['agent_pose'] for item in batch]
        robot_positions = [item['obs']['robot_position'] for item in batch]
        actions = [item['action'] for item in batch]
        is_pads = [item['is_pad'] for item in batch]
        
        
        return {
            'obs': {
                'cam_01': torch.stack(cam1_images),
                'cam_02': torch.stack(cam2_images),
                'cam_03': torch.stack(cam3_images),
                'agent_pose': torch.stack(joint_positions),
                'robot_position': torch.stack(robot_positions),
            },
            'action': pad_sequence(actions, batch_first=True),
            'is_pad': pad_sequence(is_pads, batch_first=True)
        }

    def __len__(self):
        return len(self.sequences)

def main():
    
    dataset_path = 'outputs/nav/nav_dataset_20250523_011359/data.hdf5'
    shape_meta = {
        'cam_01': (3, 84, 84),
        'cam_02': (3, 84, 84),
        'cam_03': (3, 84, 84),
    }
    
    dataset = ACTNavDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=60,
        n_obs_steps=1,
        val_ratio=0.1
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=ACTNavDataset.collate_fn)
    
    for batch in loader:
        print("Batch shapes:")
        print(f"Camera 1: {batch['obs']['cam_01'].shape}")
        print(f"Actions: {batch['action'].shape}")
        print(f"Padding mask: {batch['is_pad'].shape}")
        break

if __name__ == "__main__":
    main()
