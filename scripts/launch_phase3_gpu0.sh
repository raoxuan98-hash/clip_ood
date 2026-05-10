#!/bin/bash
# Launch Phase 3 - Method C on GPU 0

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/raoxuan/projects/clip_ood:$PYTHONPATH

cd /home/raoxuan/projects/clip_ood

mkdir -p experiments/phase3

echo "Starting Phase 3 - Method C: LoRA Vanilla on GPU 0..."
python src/experiments/run_continual_learning.py \
    --method lora_vanilla \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 \
    --iterations 800 \
    --fd_weight 0 \
    --cd_weight 0 \
    --output_dir experiments/phase3/C_lora_vanilla \
    > experiments/phase3/C_lora_vanilla.log 2>&1

echo "Method C completed!"
