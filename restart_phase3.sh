#!/bin/bash
# Restart Phase 3 experiments after crash

cd /home/raoxuan/projects/clip_ood
export PYTHONPATH=/home/raoxuan/projects/clip_ood:$PYTHONPATH

echo "=========================================="
echo "Restarting Phase 3 Experiments"
echo "Started at: $(date)"
echo "=========================================="

# Method C: LoRA Vanilla (GPU 0)
export CUDA_VISIBLE_DEVICES=0
echo "[$(date)] Starting Method C: LoRA Vanilla on GPU 0..."
python src/experiments/run_continual_learning.py \
    --method lora_vanilla \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 \
    --iterations 800 \
    --fd_weight 0 \
    --cd_weight 0 \
    --output_dir experiments/phase3/C_lora_vanilla \
    > logs/phase3_C.log 2>&1 &
PID_C=$!

# Method D: LoRA-NSP Only (GPU 1)  
export CUDA_VISIBLE_DEVICES=1
echo "[$(date)] Starting Method D: LoRA-NSP Only on GPU 1..."
python src/experiments/run_continual_learning.py \
    --method lora_nsp \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 \
    --iterations 800 \
    --fd_weight 1.0 \
    --cd_weight 1.0 \
    --output_dir experiments/phase3/D_lora_nsp_only \
    > logs/phase3_D.log 2>&1 &
PID_D=$!

echo ""
echo "=========================================="
echo "Experiments restarted!"
echo "Method C PID: $PID_C (GPU 0)"
echo "Method D PID: $PID_D (GPU 1)"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tail -f logs/phase3_C.log"
echo "  tail -f logs/phase3_D.log"
echo "  watch -n 5 nvidia-smi"
