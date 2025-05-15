#!/usr/bin/env bash

# 指定可见 GPU 列表
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# # 可选：打印当前可见 GPU（方便调试）
# echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
# nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# 执行微调脚本
python scripts/load_dataset.py \
  --dataset-path ./demo_data/tumai_pickneedle_Dataset \
  --embodiment_tag new_embodiment \
  --video_backend decord