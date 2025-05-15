#!/usr/bin/env bash

# 指定可见 GPU 列表
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# # 可选：打印当前可见 GPU（方便调试）
# echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
# nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# 执行微调脚本（ok）
# python scripts/gr00t_finetune.py \
#   --dataset-path ./demo_data/robot_sim.PickNPlace \
#   --batch_size 4 \
#   --num-gpus 1

python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/tumai_pickneedle_Dataset \
  --batch_size 4 \
  --num-gpus 1 \
  --output-dir /tmp/tumai  \
  --max-steps 2000 \
  --data-config tumaisingle \
  --video-backend torchvision_av