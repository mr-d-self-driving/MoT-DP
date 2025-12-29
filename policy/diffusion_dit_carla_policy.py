import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable, Union
from collections import defaultdict
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
from model.interfuser_bev_encoder import InterfuserBEVEncoder
from model.interfuser_bev_encoder import load_lidar_submodules
import os



def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

class DiffusionDiTCarlaPolicy(nn.Module):
    def __init__(self, config: Dict, action_stats: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        
        # config
        self.cfg = config
        policy_cfg = config['policy']

        obs_as_global_cond = policy_cfg.get('obs_as_global_cond', True)
        self.obs_as_global_cond = obs_as_global_cond
        shape_meta = config['shape_meta']
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        
        # Action normalization settings
        self.enable_action_normalization = config.get('enable_action_normalization', False)
        self.action_stats = action_stats
        if self.enable_action_normalization and self.action_stats is not None:
            print(f"✓ Action normalization enabled with stats:")
            print(f"  Action min: {self.action_stats['min']}")
            print(f"  Action max: {self.action_stats['max']}")
        else:
            print("⚠ Action normalization disabled")
            self.action_stats = None
        

        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))
        
        # Load BEV encoder configuration from config file
        bev_encoder_cfg = config.get('bev_encoder', {})
        obs_encoder = InterfuserBEVEncoder(
            perception_backbone=None,
            state_dim=bev_encoder_cfg.get('state_dim', 13),  # 修改为13维以支持拼接后的ego_status
            feature_dim=bev_encoder_cfg.get('feature_dim', 256),
            use_group_norm=bev_encoder_cfg.get('use_group_norm', True),
            freeze_backbone=bev_encoder_cfg.get('freeze_backbone', False),
            bev_input_size=tuple(bev_encoder_cfg.get('bev_input_size', [448, 448]))
        )
        
        # Load pretrained weights from config
        pretrained_path = bev_encoder_cfg.get('pretrained_path', None)
        if pretrained_path is not None and os.path.exists(pretrained_path):
            load_lidar_submodules(obs_encoder, pretrained_path, strict=False, logger=None)
            print(f"✓ BEV encoder loaded from: {pretrained_path}")
        else:
            print(f"⚠ BEV encoder pretrained_path not found or not specified: {pretrained_path}")
            print("  Continuing with random initialization...")
        
        self.obs_encoder = obs_encoder

        vlm_feature_dim = 2560  # 隐藏层维度
        self.feature_encoder = nn.Linear(vlm_feature_dim, 1536)

        obs_feature_dim = 256  

        # Get status_dim from bev_encoder config
        status_dim = bev_encoder_cfg.get('state_dim', 15)
        status_dim_anchor_goal =  bev_encoder_cfg.get('status_dim_anchor_goal', 14)
        # Get ego_status_seq_len from policy config (defaults to n_obs_steps)
        ego_status_seq_len = policy_cfg.get('ego_status_seq_len', self.n_obs_steps)

        model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 16),
            n_obs_steps=self.n_obs_steps,  
            cond_dim=256,   
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=policy_cfg.get('n_emb', 512),
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            obs_as_cond=obs_as_global_cond,
            n_cond_layers=policy_cfg.get('n_cond_layers', 4),
            status_dim_anchor_goal=status_dim_anchor_goal,  # 传入 ego_status 维度
            ego_status_seq_len=ego_status_seq_len  # 传入 ego_status 序列长度
        )

        self.model = model
        
        # ========== Truncated Diffusion Configuration (DiffusionDriveV2 style) ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        self.trunc_timesteps = diffusion_cfg.get('trunc_timesteps', 8)  # Truncated timestep for anchor during inference
        self.train_trunc_timesteps = diffusion_cfg.get('train_trunc_timesteps', 50)  # Max timestep during training (DiffusionDrive uses 50)
        self.num_diffusion_steps = diffusion_cfg.get('num_diffusion_steps', 2)  # Number of denoising steps
        self.diffusion_eta = diffusion_cfg.get('eta', 1.0)  # 1.0 for stochastic multiplicative noise
        
        # Normalization parameters (DiffusionDrive v1 style: linear mapping to [-1, 1])
        # x: 2*(x + x_offset)/x_range - 1
        # y: 2*(y + y_offset)/y_range - 1
        self.norm_x_offset = diffusion_cfg.get('norm_x_offset', 2.0)  # x range: [-2, 78]
        self.norm_x_range = diffusion_cfg.get('norm_x_range', 80.0)
        self.norm_y_offset = diffusion_cfg.get('norm_y_offset', 20.0)  # y range: [-20, 36]
        self.norm_y_range = diffusion_cfg.get('norm_y_range', 56.0)
        
        # DDIMScheduler for variance computation (DiffusionDriveV2 style)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",  # Predict clean sample directly
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_global_cond) else obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,  
            fix_obs_steps=True,
            action_visible=False
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)
    
    # ========== Normalization Functions ==========
    def norm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory coordinates to [-1, 1] range.
        Following DiffusionDrive v1: 2*(x + offset)/range - 1
        
        For our data (x: [-0.066, 74.045], y: [-17.526, 32.736]):
        - x: 2*(x + 1)/76 - 1, maps [-1, 75] to [-1, 1]
        - y: 2*(y + 18)/52 - 1, maps [-18, 34] to [-1, 1]
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        
        # Linear mapping to [-1, 1]
        odo_info_fut_x = 2 * (odo_info_fut_x + self.norm_x_offset) / self.norm_x_range - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + self.norm_y_offset) / self.norm_y_range - 1
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def denorm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Denormalize trajectory from [-1, 1] back to original scale.
        Following DiffusionDrive v1: (x + 1)/2 * range - offset
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        
        # Inverse linear mapping from [-1, 1]
        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * self.norm_x_range - self.norm_x_offset
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * self.norm_y_range - self.norm_y_offset
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    

    def add_multiplicative_noise_scheduled(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, int],
        eta: float = 1.0,
        std_min: float = 0.04
    ) -> torch.Tensor:
        """
        Add multiplicative noise with scheduler-based variance (DiffusionDriveV2 style).
        The noise level is determined by the diffusion scheduler's variance at the given timestep.
        
        DiffusionDriveV2 formula:
            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
        
        When eta > 0:
            - std_dev_t_mul = clip(std_dev_t, min=0.04) for multiplicative noise
            - std_dev_t_add = 0.0 (no additive noise)
        
        Multiplicative noise is applied separately to x (horizon) and y (vert) directions,
        then combined: sample * noise_mul
        
        Args:
            sample: (B, T, 2) normalized trajectory
            timestep: current diffusion timestep (scalar or tensor)
            eta: scaling factor for variance (0.0 = deterministic, 1.0 = full stochasticity)
            std_min: minimum standard deviation to prevent zero noise (V2 uses 0.04)
            
        Returns:
            Noisy sample with timestep-scheduled multiplicative noise applied
        """
        device = sample.device
        dtype = sample.dtype
        bs = sample.shape[0]
        T = sample.shape[1]  # trajectory length (num_points)
        
        # Get timestep as integer
        if torch.is_tensor(timestep):
            t = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t = timestep
        t = int(t)
        
        # Compute variance from scheduler (DDIM style)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        prev_t = t - self.num_train_timesteps // max(self.num_diffusion_steps, 1)
        prev_t = max(prev_t, 0)
        
        alpha_prod_t = self.diffusion_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.diffusion_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.diffusion_scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Variance formula from DDIM
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = max(variance.item(), 1e-10)
        
        # std_dev_t with eta scaling
        std_dev_t = eta * (variance ** 0.5)
        
        # DiffusionDriveV2 style: std_dev_t_mul = clip(std_dev_t, min=0.04)
        std_dev_t_mul = max(std_dev_t, std_min)
        
        # Generate multiplicative noise for horizon (x) and vert (y) separately
        # DiffusionDriveV2: variance_noise_horizon/vert shape is (B, G, 1, 1), then repeat
        # Our shape: (B, 1, 1) for horizon and vert, then cat to (B, 1, 2), then repeat to (B, T, 2)
        
        # variance_noise_horizon = randn * std_dev_t_mul + 1.0  (for x direction)
        variance_noise_horizon = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        # variance_noise_vert = randn * std_dev_t_mul + 1.0  (for y direction)
        variance_noise_vert = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        
        # Concatenate horizon and vert: (B, 1, 1) + (B, 1, 1) -> (B, 1, 2)
        variance_noise_mul = torch.cat([variance_noise_horizon, variance_noise_vert], dim=-1)
        
        # Repeat across trajectory length: (B, 1, 2) -> (B, T, 2)
        variance_noise_mul = variance_noise_mul.expand(-1, T, -1)
        
        # Apply multiplicative noise: sample * variance_noise_mul
        # This matches DiffusionDriveV2: prev_sample = prev_sample_mean * variance_noise_mul
        # (when std_dev_t_add = 0, the additive term is zero)
        noisy_sample = sample * variance_noise_mul
        
        return noisy_sample

    def add_multiplicative_noise_scheduled_batch(
        self, 
        sample: torch.Tensor, 
        timesteps: torch.Tensor,
        eta: float = 1.0,
        std_min: float = 0.04
    ) -> torch.Tensor:
        """
        Add multiplicative noise with per-sample timesteps (batch version).
        Each sample in the batch gets noise corresponding to its own timestep.
        
        This is the correct implementation for training where each sample should have
        noise added according to its own sampled timestep.
        
        Args:
            sample: (B, T, 2) normalized trajectory
            timesteps: (B,) tensor of timesteps, one per sample
            eta: scaling factor for variance (0.0 = deterministic, 1.0 = full stochasticity)
            std_min: minimum standard deviation to prevent zero noise (V2 uses 0.04)
            
        Returns:
            Noisy sample with per-sample timestep-scheduled multiplicative noise applied
        """
        device = sample.device
        dtype = sample.dtype
        bs = sample.shape[0]
        T = sample.shape[1]  # trajectory length (num_points)
        
        # Compute variance for each sample based on its timestep
        # Pre-compute alpha_cumprod values on CPU then move to device
        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod
        
        # Get prev_t for each sample
        step_ratio = self.num_train_timesteps // max(self.num_diffusion_steps, 1)
        prev_timesteps = (timesteps - step_ratio).clamp(min=0)
        
        # Gather alpha_prod values for each sample
        alpha_prod_t = alphas_cumprod[timesteps.cpu()].to(device=device, dtype=dtype)  # (B,)
        alpha_prod_t_prev = alphas_cumprod[prev_timesteps.cpu()].to(device=device, dtype=dtype)  # (B,)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Variance formula from DDIM: (B,)
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = variance.clamp(min=1e-10)
        
        # std_dev_t with eta scaling: (B,)
        std_dev_t = eta * (variance ** 0.5)
        
        # DiffusionDriveV2 style: std_dev_t_mul = clip(std_dev_t, min=0.04)
        std_dev_t_mul = std_dev_t.clamp(min=std_min)  # (B,)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        std_dev_t_mul = std_dev_t_mul.view(bs, 1, 1)
        
        # Generate multiplicative noise for horizon (x) and vert (y) separately
        # variance_noise_horizon = randn * std_dev_t_mul + 1.0  (for x direction)
        variance_noise_horizon = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        # variance_noise_vert = randn * std_dev_t_mul + 1.0  (for y direction)
        variance_noise_vert = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        
        # Concatenate horizon and vert: (B, 1, 1) + (B, 1, 1) -> (B, 1, 2)
        variance_noise_mul = torch.cat([variance_noise_horizon, variance_noise_vert], dim=-1)
        
        # Repeat across trajectory length: (B, 1, 2) -> (B, T, 2)
        variance_noise_mul = variance_noise_mul.expand(-1, T, -1)
        
        # Apply multiplicative noise: sample * variance_noise_mul
        noisy_sample = sample * variance_noise_mul
        
        return noisy_sample

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Normalize action from original range to [-1, 1]
        Args:
            action: tensor of shape (..., action_dim)
        Returns:
            normalized action in range [-1, 1]
        """
        if not self.enable_action_normalization or self.action_stats is None:
            return action
        
        device = action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        
        # Normalize to [0, 1]
        normalized = (action - action_min) / (action_max - action_min + 1e-8)
        # Normalize to [-1, 1]
        normalized = normalized * 2 - 1
        return normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize action from [-1, 1] back to original range
        Args:
            normalized_action: tensor of shape (..., action_dim) in range [-1, 1]
        Returns:
            action in original range
        """
        if not self.enable_action_normalization or self.action_stats is None:
            return normalized_action
        
        device = normalized_action.device
        action_min = self.action_stats['min'].to(device)
        action_max = self.action_stats['max'].to(device)
        
        # Unnormalize from [-1, 1] to [0, 1]
        unnormalized = (normalized_action + 1) / 2
        # Unnormalize to original range
        unnormalized = unnormalized * (action_max - action_min) + action_min
        return unnormalized

    def extract_tcp_features(self, obs_dict, return_attention=False):
        """
        使用InterfuserBEVEncoder提取特征
        支持两种模式：
        1. 使用预处理好的BEV特征（推荐，快速）
        2. 使用原始lidar_bev图像（兼容模式，慢）
        
        Args:
            obs_dict: 观测字典，应包含：
                - 'lidar_token': (B, seq_len, 512) 预处理的空间特征，或
                - 'lidar_token_global': (B, 1, 512) 预处理的全局特征，或
                - 'lidar_bev': (B, 3, 448, 448) 原始BEV图像（兼容模式）
                - 'ego_status' (B, 13): [accel(3), rot_rate(3), vel(3), steer(1), command(3)]
                  已经拼接好的13维状态向量，直接作为state输入
            return_attention: 是否返回attention map
            
        Returns:
            如果return_attention=False: j_ctrl特征 (B, 256)
            如果return_attention=True: (j_ctrl特征, attention_map) 
        """
        try:
            device = next(self.parameters()).device
            # Get the dtype of the model parameters
            model_dtype = next(self.parameters()).dtype
            state = obs_dict['ego_status'].to(device=device, dtype=model_dtype)  # Use model dtype instead of float32
            use_precomputed = 'lidar_token' in obs_dict and 'lidar_token_global' in obs_dict
            if use_precomputed:
                lidar_token = obs_dict['lidar_token'].to(device=device, dtype=model_dtype)
                lidar_token_global = obs_dict['lidar_token_global'].to(device=device, dtype=model_dtype)
                
                if return_attention:
                    j_ctrl, attention_map = self.obs_encoder(
                        state=state,
                        lidar_token=lidar_token,
                        lidar_token_global=lidar_token_global,
                        normalize=True,
                        return_attention=True
                    )
                else:
                    j_ctrl = self.obs_encoder(
                        state=state,
                        lidar_token=lidar_token,
                        lidar_token_global=lidar_token_global,
                        normalize=True,
                        return_attention=False
                    )
                    attention_map = None
            else:
                if 'lidar_bev' not in obs_dict:
                    raise KeyError("Neither pre-computed features (lidar_token, lidar_token_global) nor raw BEV images (lidar_bev) found in obs_dict")
                
                lidar_bev_img = obs_dict['lidar_bev'].to(device=device, dtype=model_dtype)  # Use model dtype instead of float32
                if return_attention:
                    j_ctrl, attention_map = self.obs_encoder(
                        image=lidar_bev_img,
                        state=state,
                        normalize=True,
                        return_attention=True
                    )
                else:
                    j_ctrl = self.obs_encoder(
                        image=lidar_bev_img,
                        state=state,
                        normalize=True,
                        return_attention=False
                    )
                    attention_map = None
            
            if return_attention:
                return j_ctrl, attention_map
            else:
                return j_ctrl
                
        except KeyError as e:
            raise KeyError(f"Missing required field in obs_dict for TCP feature extraction: {e}")
        except Exception as e:
            raise RuntimeError(f"Error in TCP feature extraction: {e}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward method for DDP compatibility.
        DDP only synchronizes gradients when forward() is called, not for other methods.
        This method simply calls compute_loss() to enable proper gradient synchronization
        in distributed training.
        """
        return self.compute_loss(batch)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch: {
            # BEV特征（二选一）
            'lidar_token': (B, obs_horizon, seq_len, 512) - 预处理的空间特征（推荐）
            'lidar_token_global': (B, obs_horizon, 1, 512) - 预处理的全局特征（推荐）
            'lidar_bev': (B, obs_horizon, 3, 448, 448) - 原始LiDAR BEV图像（兼容模式）
            
            'agent_pos': (B, horizon, 2) - 未来轨迹点
            'ego_status': (B, obs_horizon, 12) - 车辆状态 [speed(1), theata(1), command(6) target_point(2), waypoints_hist(2)]
            'anchor': (B, horizon, 2) - anchor轨迹点（用于truncated diffusion）
        }
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype  # Get model dtype
        nobs = {}
        required_fields = ['lidar_token', 'lidar_token_global', 'lidar_bev', 'ego_status', 'agent_pos', 'reasoning_query_tokens', 'gen_vit_tokens', 'anchor']
        
        for field in required_fields:
            if field in batch:
                if field in ['lidar_bev', 'lidar_token', 'lidar_token_global']:
                    nobs[field] = batch[field].to(device=device, dtype=model_dtype)  # Use model dtype
                else:
                    nobs[field] = batch[field].to(device)

        raw_agent_pos = batch['agent_pos'].to(device)

        # (B, horizon, 2)
        To = self.n_obs_steps
        nactions = raw_agent_pos
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        cond = None
        
        # Get ground truth trajectory
        trajectory = nactions.to(dtype=model_dtype)  # (B, horizon, 2)
        
        # Get anchor trajectory for truncated diffusion
        anchor = batch.get('anchor', None)
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)  # (B, horizon, 2)
        
        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
        feature_dim = batch_features.shape[-1]
        cond = batch_features.reshape(batch_size, To, feature_dim)  # Already in model_dtype

        # Prepare VL tokens
        gen_vit_tokens = batch['gen_vit_tokens']
        reasoning_query_tokens = batch['reasoning_query_tokens']
        gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=model_dtype)
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)
        reasoning_query_tokens = reasoning_query_tokens.to(device=device, dtype=model_dtype)
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)
        
        # Get ego_status
        ego_status = nobs['ego_status'].to(dtype=model_dtype)  # (B, obs_horizon, state_dim)
        
        # Get ground truth anchor point for prediction (mid-point between 2nd and 3rd waypoints)
        # anchor_goal_gt: (B, 2) - ground truth mid-point between index 1 and 2 (shorter horizon than last waypoint)
        anchor_goal_gt = (trajectory[:, 1, :] + trajectory[:, 2, :]) / 2.0  # Mid-point between 2nd and 3rd waypoints
        
        # Add noise to anchor_goal before putting into ego_status to prevent input-output identity
        # Use small Gaussian noise to maintain information while preventing overfitting
        anchor_goal_noise_std = self.cfg.get('anchor_goal_noise_std', 0.5)  # Default 0.5m standard deviation
        anchor_goal_noisy = anchor_goal_gt + torch.randn_like(anchor_goal_gt) * anchor_goal_noise_std
        
        # Extend ego_status with noisy anchor_goal repeated across history
        # Keep original target_point unchanged (they have different distributions: ~50m vs ~0-15m)
        # ego_status: (B, obs_horizon, state_dim) -> (B, obs_horizon, state_dim+2)
        ego_status = ego_status.clone()
        # Repeat the current noisy anchor goal across all obs steps (stable training)
        anchor_goal_expanded = anchor_goal_noisy.unsqueeze(1).expand(-1, ego_status.shape[1], -1)  # (B, obs_horizon, 2)
        ego_status = torch.cat([ego_status, anchor_goal_expanded], dim=-1)  # (B, obs_horizon, state_dim+2)


        # ========== Compute Loss (Truncated Diffusion DiffusionDriveV2 style) ==========
        loss_dict = self._compute_truncated_diffusion_loss(
            trajectory=trajectory,
            anchor=anchor,
            anchor_goal_gt=anchor_goal_gt,  # Pass ground truth anchor for loss computation
            cond=cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status,
            device=device,
            model_dtype=model_dtype
        )
        
        # Return the complete loss dictionary for logging
        # Training code will extract total_loss for backward pass
        return loss_dict
    
    def _compute_truncated_diffusion_loss(
        self,
        trajectory: torch.Tensor,
        anchor: torch.Tensor,
        anchor_goal_gt: torch.Tensor,  # Ground truth anchor (B, 2) - the 5th waypoint
        cond: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss using truncated diffusion (DiffusionDriveV2 style).
        Instead of starting from pure noise, we start from anchor with multiplicative noise.
        The model predicts the clean sample directly.
        
        Additionally, predicts the anchor goal (mid-point between 2nd and 3rd waypoints) from encoder hidden states.
        The noisy anchor goal is fed as input to prevent input-output identity, while clean anchor goal is used as loss target.
        
        Training: Add scheduler-based multiplicative noise (noise level depends on timestep)
        
        Returns:
            dict with keys:
                - 'total_loss': weighted sum of diffusion loss and anchor loss
                - 'diffusion_loss': trajectory prediction loss
                - 'anchor_loss': anchor point (5th waypoint) prediction loss
        """
        batch_size = trajectory.shape[0]
        
        # 1. Normalize trajectories using DiffusionDriveV2 style normalization
        trajectory_norm = self.norm_odo(trajectory)  # (B, T, 2)
        anchor_norm = self.norm_odo(anchor)  # (B, T, 2)
        
        # 2. Sample random timesteps within truncated range (like DiffusionDrive training)
        timesteps = torch.randint(
            0, self.train_trunc_timesteps,  # Training uses larger range [0, 50)
            (batch_size,), device=device
        ).long()
        
        # 3. Add scheduler-based multiplicative noise to anchor (DiffusionDriveV2 style)
        # FIXED: Use per-sample timestep for noise generation to match the timesteps passed to model
        # This ensures consistency between the noise level and the timestep condition
        noisy_anchor = self.add_multiplicative_noise_scheduled_batch(
            anchor_norm,
            timesteps=timesteps,  # Use per-sample timesteps
            eta=1.0,
            std_min=0.04
        )
        
        # 4. Clamp to valid range
        noisy_anchor = torch.clamp(noisy_anchor, min=-1, max=1)
        
        # 5. Denormalize for model input (model expects denormalized coordinates)
        noisy_trajectory_denorm = self.denorm_odo(noisy_anchor)
        
        # 6. Predict clean sample and anchor points
        pred, pred_anchor = self.model(
            noisy_trajectory_denorm,
            timesteps,  # Use the sampled timesteps
            cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status,
            return_anchor=True  # Request anchor prediction
        )
        
        # 7. Compute diffusion loss - predict clean sample (not noise)
        # Target is the ground truth trajectory (denormalized)
        # DiffusionDrive uses L1 loss
        target = trajectory
        
        diffusion_loss = F.l1_loss(pred, target, reduction='none')
        
        if diffusion_loss.shape[-1] > 2:
            diffusion_loss = diffusion_loss[..., :2]
        
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()
        
        # 8. Compute anchor loss - predict the mid-point between 2nd and 3rd waypoints from encoder
        # pred_anchor: (B, 2), anchor_goal_gt: (B, 2) - mid-point between 2nd and 3rd waypoints (clean GT, no noise)
        anchor_loss = F.l1_loss(pred_anchor, anchor_goal_gt, reduction='mean')
        
        # 9. Combine losses with weighting
        # You can adjust the weight for anchor_loss
        anchor_loss_weight = self.cfg.get('anchor_loss_weight', 0.5)  # Default weight is 0.5
        total_loss = diffusion_loss + anchor_loss_weight * anchor_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'anchor_loss': anchor_loss
        }

    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None, gen_vit_tokens=None, 
            reasoning_query_tokens=None, ego_status=None,
            anchor=None,  # Anchor trajectory for truncated diffusion
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        Generate trajectory samples using truncated diffusion (DiffusionDriveV2 style).
        """
        model = self.model
        device = condition_data.device
        bs = condition_data.shape[0]
        model_dtype = condition_data.dtype
        
        # Truncated Diffusion with Anchor (DiffusionDriveV2 style)
        trajectory = self._truncated_diffusion_sample(
            anchor=anchor,
            cond=cond,
            ego_status=ego_status,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            device=device,
            model_dtype=model_dtype,
            generator=generator
        )

        return trajectory
    
    def _truncated_diffusion_sample(
        self,
        anchor: torch.Tensor,
        cond: torch.Tensor,
        ego_status: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        generator=None
    ) -> torch.Tensor:
        """
        Truncated diffusion sampling (DiffusionDriveV2 style with multiplicative noise).
        Start from anchor with multiplicative noise, denoise for few steps.
        
        Key insight:
        - The model predicts the CLEAN trajectory directly (not noise, not residual)
        - Uses multiplicative noise with scheduler-based variance (timestep-dependent)
        - Final output is the model's direct prediction
        
        Note: ego_status should already have the predicted anchor in target_point position (8:10)
              from the predict_action method before calling this function.
        """
        bs = anchor.shape[0]
        
        # Set up scheduler
        self.diffusion_scheduler.set_timesteps(self.num_train_timesteps, device)
        
        # Compute rollout timesteps
        step_ratio = 20 / self.num_diffusion_steps
        roll_timesteps = (np.arange(0, self.num_diffusion_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        
        # 1. Normalize anchor
        diffusion_output = self.norm_odo(anchor)  # (B, T, 2)
        
        # 2. Add initial multiplicative noise using truncated timestep (scheduler-based)
        # Use trunc_timesteps to determine initial noise level
        diffusion_output = self.add_multiplicative_noise_scheduled(
            diffusion_output, 
            timestep=self.trunc_timesteps,
            eta=1.0,
            std_min=0.04
        )
        
        # 3. Denoising loop
        pred = None  # Will hold the final model prediction
        for i, k in enumerate(roll_timesteps):
            # Clamp and denormalize
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)  # (B, T, 2)
            
            # Get timestep
            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(bs)
            
            # Predict clean sample - model directly outputs the trajectory
            pred = self.model(
                sample=noisy_traj_points.to(dtype=model_dtype),
                timestep=timesteps,
                cond=cond,
                gen_vit_tokens=gen_vit_tokens,
                reasoning_query_tokens=reasoning_query_tokens,
                ego_status=ego_status
            )
            
            # For next iteration, use the normalized prediction as input
            # Apply scheduler-based multiplicative noise for refinement
            x_start = self.norm_odo(pred)  # (B, T, 2)
            
            # Add noise for next iteration based on the next timestep
            if i < len(roll_timesteps) - 1:
                next_k = roll_timesteps[i + 1]
                # Apply multiplicative noise based on scheduler variance at next timestep
                diffusion_output = self.add_multiplicative_noise_scheduled(
                    x_start,
                    timestep=next_k,
                    eta=1.0,
                    std_min=0.02
                )
            else:
                diffusion_output = x_start
        
        # 4. Return the model's direct prediction (NOT denorm_odo(diffusion_output))
        return pred

    def predict_anchor_goal(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PART 1: Encode observations and predict anchor goal.
        
        Returns:
            (predicted_anchor_goal, cond, gen_vit_tokens, reasoning_query_tokens, memory_pooled_for_decoder, ego_status)
            - predicted_anchor_goal: (B, 2) - raw predicted anchor
            - cond: (B, To, feature_dim) - encoded condition features
            - gen_vit_tokens: (B, seq_len, n_emb) - projected VIT tokens
            - reasoning_query_tokens: (B, seq_len, n_emb) - projected reasoning tokens
            - memory_pooled_for_decoder: (B, n_emb) - encoder memory (for potential future use)
            - ego_status: (B, status_dim) - base ego status without anchor
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # Extract and encode BEV features
        batch_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))  # (B*To, ...)
        batch_features = self.extract_tcp_features(batch_nobs)  # (B*To, feature_dim)
        feature_dim = batch_features.shape[-1]
        cond = batch_features.reshape(B, To, feature_dim)  # (B, To, feature_dim)
        cond = cond.to(dtype=model_dtype)

        # Process VIT and reasoning tokens
        gen_vit_tokens = nobs['gen_vit_tokens'].to(device=device, dtype=model_dtype)
        gen_vit_tokens = self.feature_encoder(gen_vit_tokens)
        
        reasoning_query_tokens = nobs['reasoning_query_tokens'].to(device=device, dtype=model_dtype)
        reasoning_query_tokens = self.feature_encoder(reasoning_query_tokens)

        # Get ego_status (base status without anchor)
        ego_status = nobs['ego_status'].to(dtype=model_dtype)  # (B, status_dim) or (B, To, status_dim)

        # Encode conditions to get fused anchor features (fusion is now inside encoder_block)
        memory, vl_features, reasoning_features, anchor_features, vl_padding_mask, reasoning_padding_mask = self.model.encode_conditions(
            cond=cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            reasoning_use_first_token=True,  # Use only first text token for anchor prediction
        )

        # Predict anchor goal from fused features (B, n_emb) -> (B, 2)
        predicted_anchor_goal = self.model.anchor_prediction_head(anchor_features)  # (B, 2)

        return predicted_anchor_goal, cond, gen_vit_tokens, reasoning_query_tokens, anchor_features, ego_status

    def predict_action_from_anchor(
        self, 
        ego_status_with_anchor: torch.Tensor,
        cond: torch.Tensor,
        gen_vit_tokens: torch.Tensor,
        reasoning_query_tokens: torch.Tensor,
        anchor: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        PART 2: Decoder - sample trajectory using anchor-augmented ego_status.
        
        Args:
            ego_status_with_anchor: (B, [obs_horizon,] status_dim+2) - ego_status with appended anchor_goal
            cond: (B, To, feature_dim) - encoded condition features
            gen_vit_tokens: (B, seq_len, n_emb) - projected VIT tokens
            reasoning_query_tokens: (B, seq_len, n_emb) - projected reasoning tokens
            anchor: (B, T, 2) - anchor trajectory (optional, for truncated diffusion)
            
        Returns:
            {
                'action': numpy array of shape (B, T, 2)
                'action_pred': torch tensor of shape (B, T, 2)
            }
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        B = ego_status_with_anchor.shape[0]
        T = self.horizon
        Da = self.action_dim

        # Prepare condition data and mask for sampling
        cond_data = torch.zeros((B, T, Da), device=device, dtype=model_dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Ensure inputs have correct dtype
        ego_status_with_anchor = ego_status_with_anchor.to(dtype=model_dtype)
        cond = cond.to(dtype=model_dtype)
        gen_vit_tokens = gen_vit_tokens.to(dtype=model_dtype)
        reasoning_query_tokens = reasoning_query_tokens.to(dtype=model_dtype)

        # Handle anchor
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)
            if anchor.dim() == 2:
                anchor = anchor.unsqueeze(0)
            if anchor.shape[0] != B:
                anchor = anchor.expand(B, -1, -1)
        
        # Conditional sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            ego_status=ego_status_with_anchor,
            anchor=anchor
        )
        
        naction_pred = nsample[...,:Da]
        action_pred = naction_pred.detach().float().cpu().numpy()
        
        result = {
            'action': action_pred,
            'action_pred': action_pred,
        }
        return result

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Combined predict_action for backward compatibility.
        Returns both action and predicted_anchor_goal for agent to process.
        
        NOTE: In inference with KF smoothing:
        1. predict_anchor_goal() predicts raw anchor
        2. Agent applies KF smoothing and concatenates smoothed anchor to ego_status
        3. predict_action_from_anchor() does the decoding
        
        This method combines them for direct inference without agent-side KF.
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # PART 1: Predict anchor goal
        predicted_anchor_goal, cond, gen_vit_tokens, reasoning_query_tokens, memory_pooled, ego_status = \
            self.predict_anchor_goal(obs_dict)

        # Extend ego_status with predicted anchor (this will be replaced with smoothed anchor in agent)
        if ego_status.dim() == 3:  # (B, obs_horizon, status_dim)
            ego_status = ego_status.clone()
            anchor_goal_expanded = predicted_anchor_goal.unsqueeze(1).expand(-1, ego_status.shape[1], -1)
            ego_status = torch.cat([ego_status, anchor_goal_expanded], dim=-1)
        elif ego_status.dim() == 2:  # (B, status_dim)
            ego_status = ego_status.clone()
            ego_status = torch.cat([ego_status, predicted_anchor_goal], dim=-1)

        # Get anchor for truncated diffusion (if available)
        nobs = dict_apply(obs_dict, lambda x: x.to(device))
        anchor = nobs.get('anchor', None)
        if anchor is not None:
            anchor = anchor.to(device=device, dtype=model_dtype)
            if anchor.dim() == 2:
                anchor = anchor.unsqueeze(0)
            if anchor.shape[0] != ego_status.shape[0]:
                anchor = anchor.expand(ego_status.shape[0], -1, -1)

        # PART 2: Decode trajectory
        result = self.predict_action_from_anchor(
            ego_status_with_anchor=ego_status,
            cond=cond,
            gen_vit_tokens=gen_vit_tokens,
            reasoning_query_tokens=reasoning_query_tokens,
            anchor=anchor
        )

        # Add predicted anchor goal for visualization
        anchor_goal_np = predicted_anchor_goal.detach().float().cpu().numpy()
        result['anchor_goal'] = anchor_goal_np

        return result