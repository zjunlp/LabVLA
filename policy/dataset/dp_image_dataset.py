import os
import numpy as np
import h5py
import torch
import copy
from typing import Dict
import glob
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer
from policy.model.common.normalizer import SingleFieldLinearNormalizer
from policy.common.normalize_util import get_image_range_normalizer

class DPImageDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta,
                 dataset_path: str,
                 horizon: int = None,
                 pad_before: int = None,
                 pad_after: int = None,
                 n_obs_steps: int = None,
                 n_latency_steps: int = None,
                 use_cache: bool = True,
                 seed: int = 42,
                 val_ratio: float = 0.00,
                 delta_action: bool = False,
                 in_memory: bool = True):
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.shape_meta = shape_meta
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.use_cache = use_cache
        self.seed = seed
        self.val_ratio = val_ratio
        self.delta_action = delta_action
        self.episode_map = []
        self.in_memory = in_memory
        h5_path = os.path.join(dataset_path, "episode_data.hdf5")
        self.h5_file = h5py.File(h5_path, 'r')
        self.episode_ids = list(self.h5_file.keys())
        self.camera_names = ['camera_1', 'camera_2']
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        self.memory_data = {} if self.in_memory else None
        
        h5_files = glob.glob(os.path.join(dataset_path, "*.h5"))
        if not h5_files:
            h5_path = os.path.join(dataset_path, "episode_data.hdf5")
            self.h5_file = h5py.File(h5_path, 'r')
            for episode_name in self.h5_file.keys():
                n_frames = self.h5_file[episode_name]['actions'].shape[0]
                self.episode_map.append((
                    episode_name,
                    n_frames
                ))
                if self.in_memory:
                    episode = self.h5_file[episode_name]
                    self.memory_data[episode_name] = {
                        'camera_1_rgb': episode['camera_1_rgb'][:],
                        'camera_2_rgb': episode['camera_2_rgb'][:],
                        'agent_pose': episode['agent_pose'][:],
                        'actions': episode['actions'][:]
                    }
        else:
            self.h5_file = {}
            for i in h5_files:
                h5_file = h5py.File(i, 'r')
                episode_name = os.path.splitext(os.path.basename(i))[0]
                self.h5_file[os.path.basename(i)] = h5_file
                n_frames = h5_file['actions'].shape[0]
                self.episode_map.append((
                    episode_name,
                    n_frames
                ))
                if self.in_memory:
                    self.memory_data[episode_name] = {
                        'camera_1_rgb': h5_file['camera_1_rgb'][:],
                        'camera_2_rgb': h5_file['camera_2_rgb'][:],
                        'agent_pose': h5_file['agent_pose'][:],
                        'actions': h5_file['actions'][:]
                    }
        
        
        self.sequences = []
        for episode_name, n_frames in self.episode_map:
            total_steps = n_frames
            if self.horizon is not None and self.n_obs_steps is not None:
                total_steps = n_frames - (self.horizon + self.n_obs_steps) + 1
            for start_idx in range(total_steps):
                self.sequences.append((episode_name, start_idx))

    def __len__(self) -> int:
        return len(self.sequences)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        if self.in_memory:
            val_set.memory_data = self.memory_data
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy())
        all_poses = []
        for episode_name, _ in self.episode_map:
            if self.in_memory:
                poses = self.memory_data[episode_name]['agent_pose'].astype(np.float32)
            else:
                episode = self.h5_file[episode_name]
                poses = episode['agent_pose'][:].astype(np.float32)
            all_poses.append(poses)
        all_poses = np.concatenate(all_poses, axis=0)
        normalizer['agent_pose'] = SingleFieldLinearNormalizer.create_fit(all_poses)
        normalizer['camera_1_rgb'] = get_image_range_normalizer()
        normalizer['camera_2_rgb'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode_name, _ in self.episode_map:
            if self.in_memory:
                actions = torch.from_numpy(self.memory_data[episode_name]['actions'].astype(np.float32))
            else:
                episode = self.h5_file[episode_name]
                actions = torch.from_numpy(episode['actions'][:].astype(np.float32))
            all_actions.append(actions)
        return torch.cat(all_actions, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_name, start_idx = self.sequences[idx]
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        action_start_idx = obs_start_idx
        action_end_idx = action_start_idx + self.horizon

        if self.in_memory:
            episode = self.memory_data[episode_name]
            cam1_obs = episode['camera_1_rgb'][obs_start_idx:obs_end_idx]
            cam2_obs = episode['camera_2_rgb'][obs_start_idx:obs_end_idx]
            robot_eef_obs = episode['agent_pose'][obs_start_idx:obs_end_idx]
            actions = episode['agent_pose'][action_start_idx:action_end_idx]
        else:
            episode = self.h5_file[episode_name]
            cam1_obs = episode['camera_1_rgb'][obs_start_idx:obs_end_idx]
            cam2_obs = episode['camera_2_rgb'][obs_start_idx:obs_end_idx]
            robot_eef_obs = episode['agent_pose'][obs_start_idx:obs_end_idx]
            actions = episode['agent_pose'][action_start_idx:action_end_idx]

        if cam1_obs.shape[1] == 1:
            cam1_obs = np.repeat(cam1_obs, 3, axis=1)
        if cam2_obs.shape[1] == 1:
            cam2_obs = np.repeat(cam2_obs, 3, axis=1)

        cam1_obs = torch.from_numpy(cam1_obs).float() / 255.0
        cam2_obs = torch.from_numpy(cam2_obs).float() / 255.0
        robot_eef_obs = torch.from_numpy(robot_eef_obs).float()
        actions = torch.from_numpy(actions).float()

        return {
            'obs': {
                'camera_1_rgb': cam1_obs,
                'camera_2_rgb': cam2_obs,
                'agent_pose': robot_eef_obs,
            },
            'action': actions,
        }

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None and isinstance(self.h5_file, h5py.File):
            self.h5_file.close()
            
    @staticmethod
    def collate_fn(batch):

        cam1_images = torch.stack([item['obs']['camera_1_rgb'] for item in batch])
        cam2_images = torch.stack([item['obs']['camera_2_rgb'] for item in batch])
        robot_eef_pose = torch.stack([item['obs']['agent_pose'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        
        return {
            'obs': {
                'camera_1_rgb': cam1_images,        # [B,T,3,H,W]
                'camera_2_rgb': cam2_images,        # [B,T,3,H,W]
                'agent_pose': robot_eef_pose, # [B,T,7]
            },
            'action': actions,                  # [B,T,7]
        }

import torch
from torch.utils.data import DataLoader
def main():
    
    dataset_path = ''  
    shape_meta = {'camera_1': (3, 480, 480), 'camera_2': (3, 480, 480), 'agent_pose': (8,)}  
    horizon = 8
    n_obs_steps = 3
    n_latency_steps = 0
    batch_size = 1

    dataset = DPImageDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_latency_steps=n_latency_steps,
        use_cache=True,
        seed=42,
        val_ratio=0.1,
        in_memory=True
    )

    val_dataset = dataset.get_validation_dataset()

    print(f": {len(dataset)}")
    print(f": {len(val_dataset)}")
    
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=DPImageDataset.collate_fn, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=DPImageDataset.collate_fn, shuffle=False)

    for batch in train_loader:
        print(":")
        print("Camera 1:", batch['obs']['camera_1_rgb'].shape)  # [B, T, 3, H, W]
        print("Camera 2:", batch['obs']['camera_2_rgb'].shape)  # [B, T, 3, H, W]
        print("Agent Pose:", batch['obs']['agent_pose'].shape)  # [B, T, 2]
        print("Actions:", batch['action'].shape)  # [B, T, 2]
        print("Agent Pose:", batch['obs']['agent_pose'])
        print(batch['action'])
        break

if __name__ == "__main__":
    main()