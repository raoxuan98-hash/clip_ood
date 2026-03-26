#!/bin/bash
# Launch Phase 3 - Method D on GPU 1

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/raoxuan/projects/clip_ood:$PYTHONPATH

cd /home/raoxuan/projects/clip_ood

mkdir -p experiments/phase3

echo "Starting Phase 3 - Method D: LoRA-NSP Only on GPU 1..."
python src/experiments/run_continual_learning.py \
    --method lora_nsp \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 \
    --iterations 800 \
    --fd_weight 1.0 \
    --cd_weight 1.0 \
    --output_dir experiments/phase3/D_lora_nsp_only \
    > experiments/phase3/D_lora_nsp_only.log 2>&1

echo "Method D completed!"
