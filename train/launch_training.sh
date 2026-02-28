#!/bin/bash
# launch_training_fixed.sh

# CRITICAL: Disable ALL DeepSpeed JIT compilation
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_UTILS=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_AIO=0
export DS_BUILD_FUSED_LAMB=0
export DS_SKIP_CUDA_CHECK=1

# Disable all quantization/FP8 flags
unset TRANSFORMER_ENGINE_FP8
unset NVTE_FP8_AMAX_HISTORY_LEN
unset NVTE_FP8_DP_BINS

# Standard distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=6000
export CUDA_VISIBLE_DEVICES=0,1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable BF16 and TF32
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# Performance tuning
export NCCL_ALGO=Tree
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=13

echo "=========================================="
echo "Minnow-Math-1.5B Training - No Fused Ops"
echo "=========================================="
echo "Precision: BF16 + TF32"
echo "DeepSpeed Fused Ops: DISABLED"
echo "Using PyTorch AdamW instead"
echo "=========================================="

# Launch with fixed config
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config ./config.json \
    --deepspeed ./ds_config_fixed.json