if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from policy.workspace.lightning_workspace import LightningWorkspace, create_lightning_trainer
from policy.model.diffusion.ema_model import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspaceLightning(LightningWorkspace):
    """
    Diffusion training workspace based on PyTorch Lightning
    Supports multi-GPU data parallel training
    """
    
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        
        super().__init__(cfg, output_dir=output_dir)

        # Configure EMA
        self.ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Save first batch for sampling
        if self.cfg.training.freeze_encoder:
            self.model.obs_encoder.eval()
            self.model.obs_encoder.requires_grad_(False)
            
        if self.train_sampling_batch is None:
            self.train_sampling_batch = batch
        
        # Compute loss
        raw_loss = self.model.compute_loss(batch)
        loss = raw_loss / self.cfg.training.gradient_accumulate_every
        
        # Log loss
        self.log('train_loss', raw_loss, prog_bar=True)
        
        # Update EMA
        if self.cfg.training.use_ema and self.ema is not None:
            self.ema.step(self.model)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Use EMA model for validation
        policy = self.ema_model if self.cfg.training.use_ema else self.model
        
        loss = policy.compute_loss(batch)
        
        # Log validation loss
        self.log('val_loss', loss, sync_dist=True)
        
        return loss

    def on_train_epoch_end(self):
        """Callback at the end of training epoch"""
        # Run sampling evaluation
        if (self.current_epoch % self.cfg.training.sample_every) == 0:
            self._run_sampling_evaluation()

    def _run_sampling_evaluation(self):
        """Run sampling evaluation"""
        if self.train_sampling_batch is None:
            return
            
        # Use EMA model for sampling
        policy = self.ema_model if self.cfg.training.use_ema else self.model
        policy.eval()
        
        with torch.no_grad():
            batch = self.train_sampling_batch
            obs_dict = batch['obs']
            gt_action = batch['action']
            
            result = policy.predict_action(obs_dict)
            pred_action = result['action_pred']
            
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
            self.log('train_action_mse_error', mse.item(), sync_dist=True)
        
        policy.train()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, 
            params=self.model.parameters()
        )
        
        # Configure learning rate scheduler
        from policy.model.common.lr_scheduler import get_scheduler
        
        # Calculate training steps
        train_dataloader = self.train_dataloader()
        num_training_steps = (
            len(train_dataloader) * self.cfg.training.num_epochs
        ) // self.cfg.training.gradient_accumulate_every
        
        lr_scheduler = get_scheduler(
            self.cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=self.global_step-1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1
            }
        }

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        """Save checkpoint"""
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        
        path.parent.mkdir(parents=False, exist_ok=True)
        
        # Prepare data to save
        payload = {
            'cfg': self.cfg,
            'state_dicts': {},
            'pickles': {}
        }
        
        # Save model state
        payload['state_dicts']['model'] = self.model.state_dict()
        if self.ema_model is not None:
            payload['state_dicts']['ema_model'] = self.ema_model.state_dict()
        
        # Save optimizer state
        payload['state_dicts']['optimizer'] = self.optimizers().state_dict()
        
        # Save training state
        payload['pickles']['global_step'] = self.global_step
        payload['pickles']['epoch'] = self.epoch
        
        # Save to file
        import dill
        torch.save(payload, path, pickle_module=dill)
        return str(path.absolute())

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        """Load checkpoint data"""
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if key == 'model':
                    self.model.load_state_dict(value, **kwargs)
                elif key == 'ema_model' and self.ema_model is not None:
                    self.ema_model.load_state_dict(value, **kwargs)
                elif key == 'optimizer':
                    self.optimizers().load_state_dict(value)
        
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = payload['pickles'][key]

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        """Load checkpoint"""
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        
        import dill
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    def run(self):
        trainer = create_lightning_trainer(self.cfg, self)
        trainer.fit(self)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name="train_diffusion_unet_image_workspace")
def main(cfg):
        # Create workspace
    workspace = TrainDiffusionUnetImageWorkspaceLightning(cfg)
    
    # Resume training
    if cfg.training.resume:
        latest_ckpt_path = workspace.get_checkpoint_path()
        if latest_ckpt_path.is_file():
            print(f"Resuming from checkpoint {latest_ckpt_path}")
            workspace.load_checkpoint(path=latest_ckpt_path)
    
    # Create trainer
    trainer = create_lightning_trainer(cfg, workspace)
    
    # Start training
    trainer.fit(workspace)

if __name__ == "__main__":
    main() 