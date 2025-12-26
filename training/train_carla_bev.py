#!/usr/bin/env python3
import os
import sys
import torch
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import argparse
import datetime
from torch.distributed.elastic.multiprocessing.errors import record

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(project_root, "config", "pdm_server.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_action_stats(config):
    if 'action_stats' in config:
        action_stats = {
            'min': torch.tensor(config['action_stats']['min'], dtype=torch.float32),
            'max': torch.tensor(config['action_stats']['max'], dtype=torch.float32),
            'mean': torch.tensor(config['action_stats']['mean'], dtype=torch.float32),
            'std': torch.tensor(config['action_stats']['std'], dtype=torch.float32),
        }
        print("✓ Loaded action_stats from config file")
        return action_stats
    else:
        print("⚠ No action_stats found in config, action normalization will be disabled")
        return None


def safe_wandb_log(data, use_wandb=True):
    if not use_wandb:
        return
    try:
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    continue
            
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    continue
            
            if isinstance(value, (np.integer, np.floating)):
                value = value.item() 
            
            if isinstance(value, (int, float, np.integer, np.floating)):
                if np.isnan(value):
                    continue
                elif np.isinf(value):
                    value = 1e10 if value > 0 else -1e10
            
            if isinstance(value, np.generic):
                value = value.item()
            
            cleaned_data[key] = value
        
        if not cleaned_data:
            return
        sys.stdout.flush()
        sys.stderr.flush()
    
        wandb.log(cleaned_data)
        
        sys.stdout.flush()
        sys.stderr.flush()
        
    except Exception as e:
        import traceback
        traceback.print_exc()


def safe_wandb_finish(use_wandb=True):
    if not use_wandb:
        return
    wandb.finish(quiet=True, exit_code=0)




def compute_driving_metrics(predicted_trajectories, target_trajectories, fut_obstacles=None):
    """
    计算驾驶性能指标
    
    Args:
        predicted_trajectories: (B, T, 2) 预测轨迹
        target_trajectories: (B, T, 2) 真实轨迹
        fut_obstacles: List of B lists, each containing T dicts with obstacle info
                      Each dict has 'gt_boxes' (N, 7), 'gt_names' (N,), 'gt_velocity' (N, 2)
                      Optional - if None, collision metrics will not be computed
    
    Returns:
        metrics: 包含L2误差和碰撞率的字典
        
    """
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    
    B, T, _ = predicted_trajectories.shape
    
    l2_errors = np.linalg.norm(
        predicted_trajectories - target_trajectories, axis=-1
    )
    
    metrics = {}
    
    # === L2 误差指标 ===

    if T >= 2:  
        metrics['L2_1s'] = np.mean(l2_errors[:, 1])
    
    if T >= 4:  
        metrics['L2_2s'] = np.mean(l2_errors[:, 3])
    
    if T >= 6: 
        metrics['L2_3s'] = np.mean(l2_errors[:, 5])
    
    # L2_avg: 只计算1s, 2s, 3s时间步的平均
    l2_avg_values = []
    if T >= 2:
        l2_avg_values.append(l2_errors[:, 1])
    if T >= 4:
        l2_avg_values.append(l2_errors[:, 3])
    if T >= 6:
        l2_avg_values.append(l2_errors[:, 5])
    
    if len(l2_avg_values) > 0:
        metrics['L2_avg'] = np.mean(np.concatenate(l2_avg_values))
    else:
        metrics['L2_avg'] = 0.0
    
    safe_metrics = {}
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"Warning: Metric '{key}' is NaN, replacing with 0.0")
            safe_metrics[key] = 0.0
        elif np.isinf(value):
            print(f"Warning: Metric '{key}' is Inf, replacing with large value")
            safe_metrics[key] = 1e10 if value > 0 else -1e10
        else:
            safe_metrics[key] = value
    
    return safe_metrics

def validate_model(policy, val_loader, device, rank=0, world_size=1):
    """
    Validation function for distributed training
    Only rank 0 will compute and log metrics
    """
    policy.eval()
    
    # Get the actual model (unwrap DDP if needed)
    model_for_inference = policy.module if world_size > 1 else policy
    
    val_metrics = defaultdict(list)
    
    # Only rank 0 performs validation
    if rank == 0:
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                loss = model_for_inference.compute_loss(batch)
                
                # Extract loss components if available
                if isinstance(loss, dict):
                    total_loss = loss.get('total_loss', loss.get('loss', 0))
                    diffusion_loss = loss.get('diffusion_loss', 0)
                    anchor_loss = loss.get('anchor_loss', 0)
                    val_metrics['loss'].append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)
                    val_metrics['diffusion_loss'].append(diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss)
                    val_metrics['anchor_loss'].append(anchor_loss.item() if isinstance(anchor_loss, torch.Tensor) else anchor_loss)
                else:
                    val_metrics['loss'].append(loss.item())
                    val_metrics['diffusion_loss'].append(loss.item())
                    val_metrics['anchor_loss'].append(0.0)
                
                obs_dict = {
                    'lidar_token': batch['lidar_token'][:, :model_for_inference.n_obs_steps],
                    'lidar_token_global': batch['lidar_token_global'][:, :model_for_inference.n_obs_steps],
                    'ego_status': batch['ego_status'][:, :model_for_inference.n_obs_steps],  
                    'gen_vit_tokens': batch['gen_vit_tokens'],
                    'reasoning_query_tokens': batch['reasoning_query_tokens'],
                    'anchor': batch['anchor']  # Pass anchor for truncated diffusion
                }
                target_actions = batch['agent_pos']  
                
                try:
                    result = model_for_inference.predict_action(obs_dict)
                    predicted_actions = torch.from_numpy(result['action']).to(device)
                    
                    if target_actions.dim() == 3:  # (B, T, 2)
                        target_actions = target_actions[:, :predicted_actions.shape[1]]
                    elif target_actions.dim() == 2:  # (B, 2) 
                        target_actions = target_actions.unsqueeze(1)  # (B, 1, 2)
                    
                    fut_obstacles = batch.get('fut_obstacles', None)

                    driving_metrics = compute_driving_metrics(
                        predicted_actions, 
                        target_actions, 
                        fut_obstacles=fut_obstacles 
                    )
                    for key, value in driving_metrics.items():
                        val_metrics[key].append(value)
                    
                    # Get current loss for progress bar display
                    current_loss = val_metrics['loss'][-1] if val_metrics['loss'] else 0.0
                    pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})
                except Exception as e:
                    print(f"Warning: Error in action prediction during validation: {e}")
                    continue
        
            pbar.close() 
    
        # Compute averaged metrics
        averaged_metrics = {}
        for key, values in val_metrics.items():
            if values:  
                mean_value = np.mean(values)
                if np.isnan(mean_value):
                    print(f"Warning: computed NaN for metric 'val_{key}'")
                    averaged_metrics[f'val_{key}'] = 0.0  
                elif np.isinf(mean_value):
                    print(f"Warning: computed Inf for metric 'val_{key}'")
                    averaged_metrics[f'val_{key}'] = 1e10 if mean_value > 0 else -1e10 
                else:
                    averaged_metrics[f'val_{key}'] = mean_value
        
        return averaged_metrics
    else:
        return {}

@record  # Records error and tracebacks in case of failure
def train_pdm_policy(config_path):
    """
    Multi-GPU distributed training for PDM policy
    
    Args:
        config_path: 配置文件路径
    """
    torch.cuda.empty_cache()
    
    print("Initializing pdm driving policy training...")
    config = load_config(config_path=config_path)

    # Initialize distributed training
    rank = int(os.environ.get('RANK', 0))  # Rank across all processes
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Rank on Node
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # Number of processes
    
    # Single GPU fallback
    if world_size == 1:
        print("Running in single GPU mode")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        print(f'RANK, LOCAL_RANK and WORLD_SIZE: {rank}/{local_rank}/{world_size}')
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=15)
        )
        
        torch.cuda.set_device(device)
    
    # Enable performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    print(f'Rank: {rank}, Device: {device}, World size: {world_size}')
    
    # Only rank 0 should initialize wandb
    wandb_mode = os.environ.get('WANDB_MODE', 'online') 
    use_wandb = config.get('logging', {}).get('use_wandb', True) and (rank == 0)
    
    if use_wandb:
        try:
            # 支持从config动态读取wandb账号信息并自动登录（谨慎：不要把真实api key提交到仓库）
            logging_cfg = config.get('logging', {})
            wandb_entity = logging_cfg.get('wandb_entity')
            wandb_api_key = logging_cfg.get('wandb_api_key')

            if wandb_api_key:
                # 优先通过环境变量注入，然后尝试login，这样在CI中也能工作
                os.environ['WANDB_API_KEY'] = str(wandb_api_key)
                try:
                    wandb.login(key=str(wandb_api_key))
                    print("✓ WandB login succeeded using provided api key")
                except Exception as e:
                    print(f"⚠ WandB login failed: {e}")

            init_kwargs = dict(
                project=logging_cfg.get('wandb_project', "carla-diffusion-policy"),
                name=logging_cfg.get('run_name', "carla_dit_full_validation"),
                mode=wandb_mode,
                resume='allow',
                config={
                    "learning_rate": config.get('optimizer', {}).get('lr', 5e-5),
                    "epochs": config.get('training', {}).get('num_epochs', 50),
                    "batch_size": config.get('dataloader', {}).get('batch_size', 16),
                    "obs_horizon": config.get('obs_horizon', 2),
                    "action_horizon": config.get('action_horizon', 4),
                    "pred_horizon": config.get('pred_horizon', 8),
                    "dataset_path": config.get('training', {}).get('dataset_path', ""),
                    "max_files": None,
                    "train_split": 0.8,
                    "weight_decay": config.get('optimizer', {}).get('weight_decay', 1e-5),
                    "num_workers": config.get('dataloader', {}).get('num_workers', 4)
                }
            )

            # 如果config里指定了entity（账号/组织），把它传给wandb.init
            if wandb_entity:
                init_kwargs['entity'] = wandb_entity

            wandb.init(**init_kwargs)
            print(f"✓ WandB initialized in {wandb_mode} mode")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            use_wandb = False

    # dataset
    dataset_path_root = config.get('training', {}).get('dataset_path')
    train_dataset_path = os.path.join(dataset_path_root, 'train')
    val_dataset_path = os.path.join(dataset_path_root, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    train_dataset = CARLAImageDataset(dataset_path=train_dataset_path, image_data_root=image_data_root)
    val_dataset = CARLAImageDataset(dataset_path=val_dataset_path, image_data_root=image_data_root)

    if rank == 0:
        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    


    action_stats = {
    'min': torch.tensor([-0.06638534367084503, -17.525903701782227]),
    'max': torch.tensor([74.04539489746094, 32.73622512817383]),
    'mean': torch.tensor([12.758530616760254, 0.354688435792923]),
    'std': torch.tensor([6.723825454711914, 2.5529885292053223]),
    } 
    
    batch_size = config.get('dataloader', {}).get('batch_size', 32)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    persistent_workers = config.get('dataloader', {}).get('persistent_workers', True)
    prefetch_factor = config.get('dataloader', {}).get('prefetch_factor', 2)
    
    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True
        )
        # For validation, only rank 0 needs the full dataset
        # Other ranks don't participate in validation
        sampler_val = None
    else:
        sampler_train = None
        sampler_val = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=sampler_train,
        shuffle=(sampler_train is None),  # Only shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True
    )
    
    # Validation loader: only create meaningful loader for rank 0
    # Other ranks get an empty loader since they don't validate
    if world_size > 1 and rank != 0:
        # Create empty validation loader for non-rank 0 processes
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False,
                num_replicas=world_size,
                rank=rank,
                drop_last=True
            ),
            shuffle=False,
            num_workers=0,  # No workers needed for empty validation
            pin_memory=False,
            drop_last=True
        )
    else:
        # Rank 0 or single GPU: use full validation dataset
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler_val,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=True
        )
    
    if rank == 0:
        print("Initializing policy model...")
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    # Wrap model with DistributedDataParallel for multi-GPU training
    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Required: some parameters in obs_encoder may not be used in all forward passes
        )
        if rank == 0:
            print(f"✓ Model wrapped with DistributedDataParallel (find_unused_parameters=True)")
            print(f"Policy action steps (n_action_steps): {policy.module.n_action_steps}")
    else:
        if rank == 0:
            print(f"Policy action steps (n_action_steps): {policy.n_action_steps}")
    
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    
    # Linear learning rate scaling for distributed training
    # With N GPUs and same batch_size per GPU, effective batch_size = N * batch_size
    # Scale learning rate linearly: lr_scaled = lr * world_size
    scale_lr = config.get('optimizer', {}).get('scale_lr', True)  # Default to True
    if scale_lr and world_size > 1:
        lr_scaled = lr * world_size
        if rank == 0:
            print(f"✓ Learning rate scaled for {world_size} GPUs: {lr} -> {lr_scaled}")
        lr = lr_scaled
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup for multi-GPU training stability
    # Warmup prevents large gradient updates in early training when model parameters are random
    warmup_epochs = int(config.get('training', {}).get('warmup_epochs', 5))
    lr_final = float(config.get('training', {}).get('lr_final', 1e-7))
    use_lr_scheduler = config.get('training', {}).get('use_lr_scheduler', True)
    
    if use_lr_scheduler:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        # Calculate total training steps
        total_epochs = int(config.get('training', {}).get('num_epochs', 50))
        
        # Warmup scheduler: linearly increase LR from lr/10 to lr over warmup_epochs
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        # Cosine annealing scheduler: decay LR from lr to lr_final
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs - warmup_epochs, 
            eta_min=lr_final
        )
        
        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_epochs]
        )
        
        if rank == 0:
            print(f"✓ Learning rate scheduler: {warmup_epochs} epochs warmup + cosine annealing to {lr_final}")
    else:
        scheduler = None
        if rank == 0:
            print("✓ No learning rate scheduler used")

    # 设置 checkpoint 目录
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"✓ Checkpoint directory: {checkpoint_dir}")
    
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    best_l2_avg = float('inf')  # Use average L2 error as best metric
    val_loss = None  # 初始化验证损失
    val_metrics = {}  # 初始化验证指标  
    
    for epoch in range(num_epochs):
        # Update the seed depending on the epoch for distributed sampler
        if world_size > 1:
            sampler_train.set_epoch(epoch)
        
        policy.train()
        train_losses = []
        
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        else:
            pbar = train_loader
            
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # IMPORTANT: zero_grad BEFORE forward pass, not after
            # This is the standard PyTorch training pattern
            optimizer.zero_grad()
            
            # Call policy(batch) which invokes forward() method
            # DDP only synchronizes gradients when forward() is called
            # This ensures proper gradient synchronization across all GPUs
            loss = policy(batch)
            
            # Extract loss components if available (for logging)
            if isinstance(loss, dict):
                total_loss = loss.get('total_loss', loss.get('loss', 0))
                diffusion_loss = loss.get('diffusion_loss', 0)
                anchor_loss = loss.get('anchor_loss', 0)
                loss = total_loss  # Use total_loss for backward pass
            else:
                # Backward compatibility: if loss is a scalar
                diffusion_loss = loss
                anchor_loss = 0
            
            # Check for NaN/Inf loss to prevent gradient explosion
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, skipping this batch")
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            # Gradient clipping for training stability (CRITICAL for multi-GPU training)
            # This prevents gradient explosion which can cause val_loss to spike
            max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)
            if world_size > 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.module.parameters(), max_norm=max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            
            # Skip optimizer step if gradients are invalid
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if rank == 0:
                    print(f"Warning: NaN/Inf gradient norm detected at batch {batch_idx}, skipping optimizer step")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            train_losses.append(loss.item())
            
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{np.mean(train_losses):.4f}',
                    'grad_norm': f'{grad_norm.item():.4f}' if isinstance(grad_norm, torch.Tensor) else f'{grad_norm:.4f}'
                })
            
            if batch_idx % 10 == 0 and rank == 0:
                step = epoch * len(train_loader) + batch_idx
                log_dict = {
                    "train/loss_step": loss.item(),
                    "train/epoch":  epoch ,
                    "train/step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/batch_idx": batch_idx,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                }
                # Add loss components if available
                if isinstance(diffusion_loss, torch.Tensor):
                    log_dict["train/diffusion_loss"] = diffusion_loss.item()
                if isinstance(anchor_loss, torch.Tensor) and anchor_loss != 0:
                    log_dict["train/anchor_loss"] = anchor_loss.item()
                safe_wandb_log(log_dict, use_wandb)
        
        if rank == 0:
            pbar.close() 
        
        avg_train_loss = np.mean(train_losses)
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_train_loss:.4f}")
        
        # Update learning rate scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Learning rate: {current_lr:.2e}")
        
        # Get model state dict (handle DDP wrapper)
        model_to_save = policy.module if world_size > 1 else policy
        
        if rank == 0:
            torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'config': config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': avg_train_loss,
                        'val_metrics': val_metrics
                        }, os.path.join(checkpoint_dir, "carla_policy.pt"))
        
            safe_wandb_log({
                "train/loss_epoch": avg_train_loss,
                "train/epoch": epoch,
                "train/samples_processed": (epoch + 1) * len(train_dataset)
            }, use_wandb)

        validation_freq = config.get('training', {}).get('validation_freq', 1)
        if (epoch + 1) % validation_freq == 0:
            if rank == 0:
                print(f"Validating (Epoch {epoch+1}/{num_epochs})...")
                sys.stdout.flush()
            try:
                val_metrics = validate_model(policy, val_loader, device, rank=rank, world_size=world_size)
                if rank == 0:
                    print(f"✓ Validation completed")
                    sys.stdout.flush()
            except Exception as e:
                if rank == 0:
                    print(f"✗ Error during validation: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                continue

            if rank == 0:
                sys.stdout.flush()
                    
                log_dict = {
                        "epoch": epoch,
                        "train/loss": avg_train_loss,
                    }
            
                if 'val_loss' in val_metrics:
                    log_dict["val/loss"] = val_metrics['val_loss']
                
                # Add validation loss components
                if 'val_diffusion_loss' in val_metrics:
                    log_dict["val/diffusion_loss"] = val_metrics['val_diffusion_loss']
                if 'val_anchor_loss' in val_metrics:
                    log_dict["val/anchor_loss"] = val_metrics['val_anchor_loss']
            
                for key, value in val_metrics.items():
                    if key in ['val_loss', 'val_diffusion_loss', 'val_anchor_loss']:
                        continue  # Already logged above
                    elif key.startswith('val_'):
                        # Remove 'val_' prefix
                        metric_name = key.replace('val_', '')
                        if 'L2' in metric_name:
                            log_dict[f"val/driving_metrics/{metric_name}"] = value
                        elif 'collision' in metric_name:
                            log_dict[f"val/collision/{metric_name}"] = value
                        else:
                            log_dict[f"val/{metric_name}"] = value
                    else:
                        # Shouldn't happen, but handle it just in case
                        if 'L2' in key:
                            log_dict[f"val/driving_metrics/{key}"] = value
                        else:
                            log_dict[f"val/{key}"] = value
            
                print("Logging to wandb...")
                sys.stdout.flush()
                safe_wandb_log(log_dict, use_wandb)
                sys.stdout.flush()

            
                print(f"Validation metrics: (total {len(val_metrics)} metrics)")
                if len(val_metrics) == 0:
                    print("  Warning: No validation metrics were computed!")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
        
                val_loss = val_metrics.get('val_loss', float('inf'))
                l2_avg = val_metrics.get('val_L2_avg', float('inf'))
                
                # Save best model based on L2_avg (average L2 error across all timesteps)
                if l2_avg < best_l2_avg:
                    best_l2_avg = l2_avg
                    torch.save({
                            'model_state_dict': model_to_save.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'train_loss': avg_train_loss,
                            'val_metrics': val_metrics
                            }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
                    print(f"✓ New best model saved with L2_avg: {l2_avg:.4f} (val_loss: {val_loss:.4f})")
                   
                    safe_wandb_log({
                            "best_model/epoch": epoch,
                            "best_model/L2_avg": l2_avg,
                            "best_model/val_loss": val_loss,
                            "best_model/train_loss": avg_train_loss
                        }, use_wandb)
    
    if rank == 0:
        print("Training completed!")
        print(f"Best L2_avg: {best_l2_avg:.4f}")
        safe_wandb_log({
            "training/completed": 0.0,
            "training/total_epochs": num_epochs,
            "training/best_l2_avg": best_l2_avg,
            "training/final_train_loss": avg_train_loss
        }, use_wandb)
        
        safe_wandb_finish(use_wandb)
        print("✓ Training session finished")
    
    # Clean up distributed process group
    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pdm Driving Policy with Diffusion DiT - Multi-GPU Distributed Training")
    parser.add_argument('--config_path', type=str, default="/root/z_projects/code/MoT-DP-1/config/pdm_mini_server.yaml", 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    train_pdm_policy(config_path=args.config_path)