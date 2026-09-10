import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from typing import Dict
import pandas as pd

from .base_inference_engine import BaseInferenceEngine
from policy.model.common.normalizer import LinearNormalizer


class LocalModelInferenceEngine(BaseInferenceEngine):
    """
    Local model inference engine
    
    Use locally loaded PyTorch model for inference
    """
    
    def _get_n_obs_steps(self) -> int:
        """Get observation steps"""
        # Load configuration to get n_obs_steps
        config = OmegaConf.load(self.cfg.infer.policy_config_path)
        return config.n_obs_steps
    
    def _init_inference_engine(self):
        """Initialize local model inference engine"""
        # Load model checkpoint
        self.checkpoint = torch.load(self.cfg.infer.policy_model_path, map_location=self.device)
        
        # Load configuration
        self.config = OmegaConf.load(self.cfg.infer.policy_config_path)
        
        # Create and load policy model
        self.policy = hydra.utils.instantiate(self.config.policy)
        
        # Process state_dict for multi-GPU training
        state_dict = self.checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
            if "ema" in k:
                del new_state_dict[k]
                
        self.policy.load_state_dict(new_state_dict)
        self.policy.eval()
        self.policy.to(self.device)
        
        # Load or create normalizer
        if hasattr(self.cfg.infer, 'normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(self.cfg.infer.normalizer_path, map_location=self.device))
        else:
            dataset = hydra.utils.instantiate(self.config.task.dataset)
            normalizer = dataset.get_normalizer()
        
        normalizer.to(self.device)
        self.policy.set_normalizer(normalizer)
        
        print(f"✓ Local model inference engine initialized, device: {self.device}")
    
    def _predict_action(self, obs_dict: Dict[str, torch.Tensor], language_instruction: str = "") -> np.ndarray:
        """
        Use local model for action prediction
        
        Args:
            obs_dict: Observation data dictionary
            
        Returns:
            Predicted action array
        """
        with torch.no_grad():
            prediction = self.policy.predict_action(obs_dict)
            joint_positions = prediction['action'][0].cpu().numpy()
        
        return joint_positions
    
    def close(self):
        """Close local model inference engine"""
        if hasattr(self, 'policy'):
            del self.policy
        if hasattr(self, 'checkpoint'):
            del self.checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("✓ Local model inference engine closed") 