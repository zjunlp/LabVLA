



import torch
import numpy as np
from typing import Dict

from .base_inference_engine import BaseInferenceEngine

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ModuleNotFoundError:
    print("OpenPI client not found. Please follow the instruction to install openpi-client'")


class RemoteInferenceEngine(BaseInferenceEngine):
    """
    Remote inference engine using OpenPI client
    
    Connects to OpenPI server for remote inference using WebSocket communication
    """
    
    def _get_n_obs_steps(self) -> int:
        """Get observation steps from configuration"""
        return self.cfg.infer.n_obs_steps
    
    def _init_inference_engine(self):
        """Initialize OpenPI client connection"""
        # Get server connection parameters
        self.host = getattr(self.cfg.infer, 'host', '0.0.0.0')
        self.port = getattr(self.cfg.infer, 'port', None)
        self.api_key = getattr(self.cfg.infer, 'api_key', None)
        
        # Initialize OpenPI WebSocket client
        try:
            self.client = WebsocketClientPolicy(
                host=self.host,
                port=self.port,
                api_key=self.api_key
            )
            
            # Get server metadata
            self.server_metadata = self.client.get_server_metadata()
            
            print(f"✓ OpenPI client initialized successfully")
            print(f"  - Host: {self.host}")
            print(f"  - Port: {self.port}")
            print(f"  - Server metadata: {self.server_metadata}")
            
        except Exception as e:
            print(f"❌ Failed to initialize OpenPI client: {e}")
            raise
    
    def _prepare_observation(self, obs_dict: Dict[str, torch.Tensor]) -> Dict:
        """
        Prepare observation data for OpenPI client
        
        Args:
            obs_dict: Dictionary containing observation tensors
            
        Returns:
            Dictionary formatted for OpenPI inference
        """
        observation = {}
        n_obs_steps = self._get_n_obs_steps()
        
        # Process each observation in the dictionary
        # Note on tensor shapes from _prepare_observation_dict:
        #   - When n_obs_steps == 1: shape is (time=1, C, H, W) for images, (time=1, dim) for state
        #     (no batch dim added because shape[0] == 1)
        #   - When n_obs_steps > 1: shape is (batch=1, time, C, H, W) for images
        #     (batch dim added via unsqueeze because shape[0] != 1)
        for obs_key, obs_tensor in obs_dict.items():
            arr = obs_tensor.cpu().numpy()
            if obs_key == 'agent_pose':
                # For n_obs_steps==1: arr shape is (1, dim), take last time step
                # For n_obs_steps>1: arr shape is (1, T, dim), take last time step
                if n_obs_steps == 1:
                    observation['state'] = arr[-1]  # (dim,)
                else:
                    observation['state'] = arr[0, -1]  # (dim,)
            else:
                if n_obs_steps == 1:
                    # arr shape: (time=1, C, H, W) — no batch dim
                    # Take the last (only) time step
                    latest_image = arr[-1]  # (C, H, W)
                    # Convert from (C, H, W) float [0,1] to (H, W, C) uint8
                    if np.issubdtype(latest_image.dtype, np.floating):
                        latest_image = (np.clip(latest_image, 0, 1) * 255).astype(np.uint8)
                    # Transpose from (C, H, W) to (H, W, C) if needed
                    if latest_image.ndim == 3 and latest_image.shape[0] in (1, 3, 4) and latest_image.shape[2] not in (1, 3, 4):
                        latest_image = np.transpose(latest_image, (1, 2, 0))
                    # Handle edge cases
                    if latest_image.ndim == 3:
                        if latest_image.shape[2] > 4:
                            latest_image = latest_image[:, :, :3]
                        elif latest_image.shape[2] == 1:
                            latest_image = np.repeat(latest_image, 3, axis=2)
                    elif latest_image.ndim == 2:
                        latest_image = np.repeat(latest_image[:, :, np.newaxis], 3, axis=2)
                    observation[obs_key] = latest_image
                else:
                    # arr shape: (batch=1, time, C, H, W)
                    # Take the last time step from batch 0
                    latest_image = arr[0, -1]  # (C, H, W)
                    if np.issubdtype(latest_image.dtype, np.floating):
                        latest_image = (np.clip(latest_image, 0, 1) * 255).astype(np.uint8)
                    if latest_image.ndim == 3 and latest_image.shape[0] in (1, 3, 4) and latest_image.shape[2] not in (1, 3, 4):
                        latest_image = np.transpose(latest_image, (1, 2, 0))
                    if latest_image.ndim == 3:
                        if latest_image.shape[2] > 4:
                            latest_image = latest_image[:, :, :3]
                        elif latest_image.shape[2] == 1:
                            latest_image = np.repeat(latest_image, 3, axis=2)
                    elif latest_image.ndim == 2:
                        latest_image = np.repeat(latest_image[:, :, np.newaxis], 3, axis=2)
                    observation[obs_key] = latest_image
        
        return observation
    
    def _predict_action(self, obs_dict: Dict[str, torch.Tensor], language_instruction: str = "") -> np.ndarray:
        """
        Predict action using OpenPI client
        
        Args:
            obs_dict: Dictionary containing observation tensors
            
        Returns:
            Predicted action array
        """
        try:
            # Prepare observation data
            observation = self._prepare_observation(obs_dict)
            observation['language_instruction'] = language_instruction
            observation['prompt'] = language_instruction
            # Call OpenPI inference
            result = self.client.infer(observation)
            
            # Extract action from result
            if 'action' in result:
                action = np.array(result['action'])
            elif 'actions' in result:
                action = np.array(result['actions'])
            else:
                # If no action key found, check for other possible keys
                action_keys = [k for k in result.keys() if 'action' in k.lower()]
                if action_keys:
                    action = np.array(result[action_keys[0]])
                else:
                    raise ValueError(f"No action found in server response. Available keys: {list(result.keys())}")
            return action
            
        except Exception as e:
            print(f"❌ OpenPI inference failed: {e}")
            # Return zero action as fallback
            return np.zeros((8, 8))  # Default action shape
    
    def close(self):
        """Close OpenPI client connection"""
        try:
            if hasattr(self, 'client'):
                self.client.reset()
            print("✓ OpenPI client closed successfully")
        except Exception as e:
            print(f"⚠ Error closing OpenPI client: {e}")
