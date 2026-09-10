import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from typing import Optional, Dict, Any
import numpy as np
import random
from torch.utils.data import DataLoader
import tqdm
from policy.common.json_logger import JsonLogger


class LightningWorkspace(pl.LightningModule):
    """
    Base class for distributed training based on PyTorch Lightning
    Supports multi-GPU data parallel training
    """
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # Set random seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Configure model
        self.model = hydra.utils.instantiate(cfg.policy)
        
        # Configure EMA model
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        
        # Training state
        self.epoch = 0
        
        # Save training batch for sampling
        self.train_sampling_batch = None
    
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            # Use hydra.run.dir in the configuration file first
            if hasattr(self.cfg, 'hydra') and hasattr(self.cfg.hydra, 'run') and hasattr(self.cfg.hydra.run, 'dir'):
                output_dir = self.cfg.hydra.run.dir
            else:
                # Fall back to Hydra's runtime output directory
                from hydra.core.hydra_config import HydraConfig
                output_dir = HydraConfig.get().runtime.output_dir
        return pathlib.Path(output_dir)
    
    def setup(self, stage: str):
        """Lightning setup method, called before training starts"""
        if stage == "fit":
            # Display output directory information
            print(f"Using output directory: {self.output_dir}")
            print(f"Current working directory: {pathlib.Path.cwd()}")
            
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure dataset
            dataset = hydra.utils.instantiate(self.cfg.task.dataset)
            self.train_dataset = dataset
            self.val_dataset = dataset.get_validation_dataset()
            
            # Save normalizer
            normalizer = dataset.get_normalizer()
            if self.local_rank == 0:
                path = pathlib.Path(self.output_dir).joinpath('checkpoints', 'normalize.ckpt')
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(normalizer.state_dict(), path)
                print(f"Normalizer parameters saved to {path}")
            
            # Set normalizer
            self.model.set_normalizer(normalizer)
            if self.cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)
    
    def train_dataloader(self):
        """Training data loader"""
        if hasattr(self, 'train_dataset'):
            return DataLoader(
                self.train_dataset, 
                collate_fn=self.train_dataset.collate_fn,
                **self.cfg.dataloader
            )
        else:
            dataset = hydra.utils.instantiate(self.cfg.task.dataset)
            self.train_dataset = dataset
            
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if self.cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)
            
            return DataLoader(
                self.train_dataset, 
                collate_fn=self.train_dataset.collate_fn,
                **self.cfg.dataloader
            )
    
    def val_dataloader(self):
        """Validation data loader"""
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                collate_fn=self.val_dataset.collate_fn,
                **self.cfg.val_dataloader
            )
        else:
            # If the dataset is not set, set it first
            dataset = hydra.utils.instantiate(self.cfg.task.dataset)
            self.val_dataset = dataset.get_validation_dataset()
            
            return DataLoader(
                self.val_dataset,
                collate_fn=self.val_dataset.collate_fn,
                **self.cfg.val_dataloader
            )
    
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
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Save the first batch for sampling
        if self.train_sampling_batch is None:
            self.train_sampling_batch = batch
        
        # Calculate loss
        loss_dict = self.model.compute_loss(batch)
        loss = loss_dict['loss'] / self.cfg.training.gradient_accumulate_every
        
        # Record loss
        self.log('train_loss', loss_dict['loss'], prog_bar=True)
        self.log('train_l1_loss', loss_dict['l1'])
        self.log('train_kl_loss', loss_dict['kl'])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss_dict = self.model.compute_loss(batch)
        
        # Record validation loss
        self.log('val_loss', loss_dict['loss'])
        self.log('val_l1_loss', loss_dict['l1'])
        self.log('val_kl_loss', loss_dict['kl'])
        
        return loss_dict
    
    def on_train_epoch_end(self):
        """Callback when training epoch ends"""
        # Run sampling evaluation
        if (self.current_epoch % self.cfg.training.sample_every) == 0:
            self._run_sampling_evaluation()
    
    def _run_sampling_evaluation(self):
        """Run sampling evaluation"""
        if self.train_sampling_batch is None:
            return
            
        self.model.eval()
        with torch.no_grad():
            batch = self.train_sampling_batch
            obs_dict = batch['obs']
            gt_action = batch['action']
            
            a_hat = self.model.predict_action(obs_dict)
            pred_action = a_hat['action']
            
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
            self.log('train_action_mse_error', mse.item())
        self.model.train()
    
    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        """Save checkpoint"""
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
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
        torch.save(payload, path, pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
    
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
        
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload


def create_lightning_trainer(cfg: OmegaConf, workspace: LightningWorkspace):
    """Create Lightning trainer"""
    
    # Configure callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback1 = ModelCheckpoint(
        dirpath=str(workspace.output_dir / 'checkpoints'),
        filename=cfg.checkpoint.format_str,
        monitor=cfg.checkpoint.monitor_key,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.k,
        save_last=cfg.checkpoint.save_last_ckpt,
        every_n_epochs=cfg.checkpoint.every_n_epochs
    )

    callbacks.append(checkpoint_callback1)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Configure logger
    loggers = []
    if cfg.logging.mode != 'disabled':
        wandb_logger = WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            tags=cfg.logging.tags,
            log_model=False
        )
        loggers.append(wandb_logger)
    
    # Configure distributed strategy
    strategy = None
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.gradient_accumulate_every,
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.val_every,
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        precision='16-mixed' if cfg.training.get('use_amp', False) else '32',
    )
    
    return trainer 