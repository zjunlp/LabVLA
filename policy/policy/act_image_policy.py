from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy.model.common.normalizer import LinearNormalizer
from policy.policy.base_image_policy import BaseImagePolicy
from policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from policy.model.diffusion.mask_generator import LowdimMaskGenerator
from policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from policy.common.pytorch_util import dict_apply
from policy.model.act import build_ACT_model

from argparse import Namespace
class ACTImagePolicy(BaseImagePolicy):
    def __init__(self,
            kl_weight: float,
            hidden_dim: int,
            num_queries: int,
            num_heads: int,
            num_layers: int,
            dropout: float,
            dim_feedforward: int,
            enc_layers: int,
            pre_norm: bool,
            position_embedding: str,
            lr_backbone: float,
            masks: bool,
            backbone: str,
            dilation: bool = False,
            camera_names: list = ["camera_0", "camera_1"],
            nheads: int = 8,
            dec_layers: int = 6,
            robot_state_dim: int = 8,
            action_dim: int = 8,
            **kwargs):
        super().__init__()
        
        args = Namespace(
            kl_weight=kl_weight,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            pre_norm=pre_norm,
            position_embedding=position_embedding,
            lr_backbone=lr_backbone,
            masks=masks,
            backbone=backbone,
            dilation=dilation,
            camera_names=camera_names,
            nheads=nheads,
            dec_layers=dec_layers,
            robot_state_dim=robot_state_dim,
            action_dim=action_dim
        )
        
        model = build_ACT_model(args)
        self.model = model # CVAE decoder
        self.optimizer = None
        self.kl_weight = kl_weight
        self.normalizer = LinearNormalizer()
        print(f'KL Weight {self.kl_weight}')

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld


    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        actions = self.normalizer['action'].normalize(batch['action'])
        is_pad = batch['is_pad']
        
        qpos = nobs['agent_pose']
        
        image = torch.stack([nobs[cam_name] for cam_name in self.model.camera_names], dim=1)
        env_state = None

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        qpos = nobs['agent_pose']
        
        image = torch.stack([nobs[cam_name] for cam_name in self.model.camera_names], dim=1)
        env_state = None
        a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
        a_hat_unnormalized = self.normalizer['action'].unnormalize(a_hat)
        return {"action": a_hat_unnormalized}

    def configure_optimizers(self):
        return self.optimizer
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
