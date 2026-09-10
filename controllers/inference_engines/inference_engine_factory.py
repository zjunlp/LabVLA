from .local_model_inference_engine import LocalModelInferenceEngine
from .remote_inference_engine import RemoteInferenceEngine
from .base_inference_engine import BaseInferenceEngine


class InferenceEngineFactory:
    """Inference engine factory class"""
    
    @staticmethod
    def create_inference_engine(cfg, trajectory_controller) -> BaseInferenceEngine:
        """
        Create inference engine based on configuration
        
        Args:
            cfg: Configuration object
            trajectory_controller: Trajectory controller
            
        Returns:
            Corresponding inference engine instance
        """
        inference_type = getattr(cfg.infer, 'type', 'local')
        
        if inference_type == 'local':
            return LocalModelInferenceEngine(cfg, trajectory_controller)
        elif inference_type == 'remote':
            return RemoteInferenceEngine(cfg, trajectory_controller)
        else:
            raise ValueError(f"Unsupported inference engine type: {inference_type}")
    
    @staticmethod
    def get_supported_types():
        """Get supported inference engine types"""
        return ['local', 'remote'] 