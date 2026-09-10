#!/usr/bin/env python3
"""
Dataset Merger Script - Merge scattered episode files into a complete h5 file

Features:
1. Reorder episode sequence numbers
2. Ignore error files (corrupted or incomplete files)
3. Merge all valid episodes into one h5 file
4. Image compression support to reduce file size

Usage:
    # Basic usage
    python scripts/merge_dataset.py /path/to/dataset/directory
    
    # Specify output file
    python scripts/merge_dataset.py /path/to/dataset/directory -o /path/to/output.h5
    
    # Enable image compression
    python scripts/merge_dataset.py /path/to/dataset/directory --compress-images
    
    # Verbose output
    python scripts/merge_dataset.py /path/to/dataset/directory -v

Example:
    python scripts/merge_dataset.py /home/ubuntu/Documents/LabSim/outputs/collect/2025.07.07/22.42.05_Level3_PourLiquid/dataset --compress-images
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetMerger:
    def __init__(self, dataset_dir: str, output_file: str = None, compress_images: bool = False):
        """
        Initialize the dataset merger
        
        Args:
            dataset_dir: Directory path containing episode files
            output_file: Output filename, auto-generated if None
            compress_images: Whether to compress image data using HDF5 gzip
        """
        self.dataset_dir = Path(dataset_dir)
        if output_file is None:
            output_file = self.dataset_dir / "merged_dataset.h5"
        self.output_file = Path(output_file)
        
        # Image compression settings
        self.compress_images = compress_images
        
        # Statistics
        self.total_files = 0
        self.valid_files = 0
        self.error_files = 0
        self.merged_episodes = 0
        
    def should_compress_dataset(self, key: str, data: np.ndarray) -> bool:
        """
        Determine if a dataset should be compressed with gzip
        
        Args:
            key: Dataset key name
            data: Dataset data
            
        Returns:
            True if dataset should be compressed
        """
        if not self.compress_images:
            return False
            
        # Don't compress pose/action data
        if key in ['agent_pose', 'actions']:
            return False
        # Compress image data (typically 4D arrays with 3 channels)
        if len(data.shape) == 4 and (data.shape[-1] == 3 or data.shape[1] == 3):
            return True
            
        return False
        
    def validate_episode_file(self, file_path: Path) -> Tuple[bool, str, Dict]:
        """
        Validate if episode file is valid
        
        Args:
            file_path: Episode file path
            
        Returns:
            (is_valid, error_message, file_info)
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # Check if required keys exist
                required_keys = ['agent_pose', 'actions']
                missing_keys = [key for key in required_keys if key not in f.keys()]
                if missing_keys:
                    return False, f"Missing required keys: {missing_keys}", {}
                
                # Check data dimensions
                agent_pose = f['agent_pose'][:]
                actions = f['actions'][:]
                
                if len(agent_pose.shape) != 2 or len(actions.shape) != 2:
                    return False, "Incorrect data dimensions", {}
                
                if agent_pose.shape[0] < 10:  # At least 10 timesteps
                    return False, f"Too few timesteps: {agent_pose.shape[0]}", {}
                
                if agent_pose.shape[1] != actions.shape[1]:
                    return False, "agent_pose and actions dimensions don't match", {}
                
                # Check for NaN or infinite values
                if np.any(np.isnan(agent_pose)) or np.any(np.isnan(actions)):
                    return False, "Data contains NaN values", {}
                
                if np.any(np.isinf(agent_pose)) or np.any(np.isinf(actions)):
                    return False, "Data contains infinite values", {}
                
                # Collect file information
                file_info = {
                    'agent_pose_shape': agent_pose.shape,
                    'actions_shape': actions.shape,
                    'camera_keys': [key for key in f.keys() if key not in ['agent_pose', 'actions']],
                    'file_size': file_path.stat().st_size
                }
                
                return True, "", file_info
                
        except Exception as e:
            return False, f"File read error: {str(e)}", {}
    
    def scan_episode_files(self) -> List[Tuple[Path, Dict]]:
        """
        Scan and validate all episode files
        
        Returns:
            List of valid files, each element is (file_path, file_info)
        """
        logger.info(f"Scanning directory: {self.dataset_dir}")
        
        # Find all episode files
        episode_files = list(self.dataset_dir.glob("episode_*.h5"))
        self.total_files = len(episode_files)
        
        if self.total_files == 0:
            logger.warning(f"No episode files found in {self.dataset_dir}")
            return []
        
        logger.info(f"Found {self.total_files} episode files")
        
        valid_files = []
        
        for file_path in tqdm(episode_files, desc="Validating files"):
            is_valid, error_msg, file_info = self.validate_episode_file(file_path)
            
            if is_valid:
                valid_files.append((file_path, file_info))
                self.valid_files += 1
            else:
                logger.warning(f"Skipping invalid file {file_path.name}: {error_msg}")
                self.error_files += 1
        
        logger.info(f"Validation complete: {self.valid_files} valid files, {self.error_files} error files")
        return valid_files
    
    def merge_episodes(self, valid_files: List[Tuple[Path, Dict]]):
        """
        Merge all valid episode files
        
        Args:
            valid_files: List of valid files
        """
        if not valid_files:
            logger.error("No valid files to merge")
            return
        
        logger.info(f"Starting to merge {len(valid_files)} episodes to {self.output_file}")
        
        # Sort by filename to ensure consistent order
        valid_files.sort(key=lambda x: x[0].name)
        
        with h5py.File(self.output_file, 'w') as merged_file:
            for new_episode_idx, (file_path, file_info) in enumerate(tqdm(valid_files, desc="Merging episodes")):
                try:
                    with h5py.File(file_path, 'r') as source_file:
                        # Create new episode group
                        episode_name = f"episode_{new_episode_idx:04d}"
                        episode_group = merged_file.create_group(episode_name)
                        
                        # Copy all datasets
                        for key in source_file.keys():
                            data = source_file[key][:]
                            
                            # Create dataset with gzip compression if appropriate
                            if self.should_compress_dataset(key, data):
                                logger.info(f"Applying gzip compression to {episode_name}/{key}")
                                episode_group.create_dataset(
                                    key, 
                                    data=data, 
                                    compression="gzip", 
                                    compression_opts=4,
                                    chunks=True
                                )
                            else:
                                # No compression for pose/action data
                                episode_group.create_dataset(
                                    key, 
                                    data=data, 
                                    chunks=True
                                )
                        
                        self.merged_episodes += 1
                        
                except Exception as e:
                    logger.error(f"Error merging file {file_path.name}: {str(e)}")
                    continue
        
        logger.info(f"Merge complete! Successfully merged {self.merged_episodes} episodes")
        logger.info(f"Output file: {self.output_file}")
        
        # Show file size information
        output_size = self.output_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Output file size: {output_size:.2f} MB")
    
    def create_summary_report(self, valid_files: List[Tuple[Path, Dict]]):
        """
        Create merge summary report
        
        Args:
            valid_files: List of valid files
        """
        report_file = self.output_file.parent / "merge_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Dataset Merge Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Source directory: {self.dataset_dir}\n")
            f.write(f"Output file: {self.output_file}\n")
            f.write(f"Merge time: {np.datetime64('now')}\n")
            f.write(f"Image compression: {self.compress_images}\n")
            f.write("\n")
            
            f.write("Statistics:\n")
            f.write(f"  Total files: {self.total_files}\n")
            f.write(f"  Valid files: {self.valid_files}\n")
            f.write(f"  Error files: {self.error_files}\n")
            f.write(f"  Successfully merged: {self.merged_episodes}\n\n")
            
            if valid_files:
                f.write("Valid file details:\n")
                for new_idx, (file_path, file_info) in enumerate(valid_files):
                    f.write(f"  episode_{new_idx:04d} <- {file_path.name}\n")
                    f.write(f"    Timesteps: {file_info['agent_pose_shape'][0]}\n")
                    f.write(f"    Joints: {file_info['agent_pose_shape'][1]}\n")
                    f.write(f"    Cameras: {len(file_info['camera_keys'])}\n")
                    f.write(f"    File size: {file_info['file_size'] / (1024*1024):.2f} MB\n\n")
        
        logger.info(f"Merge report saved to: {report_file}")
    
    def run(self):
        """
        Execute complete merge process
        """
        logger.info("Starting dataset merge process")
        
        # 1. Scan and validate files
        valid_files = self.scan_episode_files()
        
        if not valid_files:
            logger.error("No valid episode files found, exiting")
            return False
        
        # 2. Merge episodes
        self.merge_episodes(valid_files)
        
        # 3. Create report
        self.create_summary_report(valid_files)
        
        logger.info("Dataset merge process completed!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Merge episode dataset files")
    parser.add_argument("dataset_dir", help="Directory path containing episode files")
    parser.add_argument("-o", "--output", help="Output filename (default: merged_dataset.h5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--compress-images", action="store_true", 
                       help="Enable image compression using HDF5 gzip to reduce file size")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input directory exists
    if not os.path.exists(args.dataset_dir):
        logger.error(f"Directory does not exist: {args.dataset_dir}")
        sys.exit(1)
    
    # Create merger and execute
    merger = DatasetMerger(
        args.dataset_dir, 
        args.output, 
        args.compress_images
    )
    success = merger.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 