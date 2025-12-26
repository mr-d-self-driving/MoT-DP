#!/bin/bash

# Multi-GPU Distributed Training Script for CARLA BEV Policy
# Usage: bash train_carla_bev_multi_gpu.sh [num_gpus] [config_path]
# pkill -9 -f "train_carla_bev"

# Select GPUs 1-7 (excluding GPU 0)
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=1

# Default values
NUM_GPUS=${1:-1}  # Default to 1 GPU
CONFIG_PATH=${2:-"/root/z_projects/code/MoT-DP-1/config/pdm_mini_server.yaml"}

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# Print configuration
echo "========================================="
echo "Multi-GPU Distributed Training"
echo "========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config Path: $CONFIG_PATH"
echo "========================================="

# Launch distributed training with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --max_restarts=0 \
    --rdzv_id=123456789 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    training/train_carla_bev.py \
    --config_path "$CONFIG_PATH"