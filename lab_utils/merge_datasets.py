import os
import h5py
from glob import glob
from datetime import datetime

def merge_nav_datasets(output_dir="outputs/nav", pattern="episode_*.h5"):
    """
     episode_XXXX.h5 
    
    Args:
        output_dir: 
        pattern: 
    """
    
    episode_files = []
    for root, _, _ in os.walk(output_dir):
        episode_files.extend(glob(os.path.join(root, pattern)))
    episode_files = sorted(episode_files)  
    
    if not episode_files:
        print("!")
        return
        
    print(f" {len(episode_files)} ")
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_dir = os.path.join(output_dir, f"merged_episodes_{timestamp}")
    os.makedirs(merged_dir, exist_ok=True)
    
    
    merged_path = os.path.join(merged_dir, "data.hdf5")
    
    with h5py.File(merged_path, 'w') as merged_file:
        
        for idx, episode_path in enumerate(episode_files):
            print(f"\n: {os.path.basename(episode_path)}")
            
            
            episode_name = f"episode_{idx:04d}"
            
            with h5py.File(episode_path, 'r') as episode_file:
                
                episode_group = merged_file.create_group(episode_name)
                
                
                for key in episode_file.keys():
                    episode_file.copy(key, episode_group)
                
                
                for attr_name, attr_value in episode_file.attrs.items():
                    episode_group.attrs[attr_name] = attr_value
                    
                
                episode_group.attrs['original_file'] = episode_path
                
                print(f"- : {episode_name}")
    
    print(f"- : {len(episode_files)}")
    print(f"- save: {merged_path}")
    
    
    # for file in episode_files:
    #     os.remove(file)

if __name__ == "__main__":
    merge_nav_datasets()
