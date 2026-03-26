#!/bin/bash
# Launch Phase 3 - Method E on GPU 2 (waits for D to complete)

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/raoxuan/projects/clip_ood:$PYTHONPATH

cd /home/raoxuan/projects/clip_ood

mkdir -p experiments/phase3

# Wait for Method D to complete
echo "Waiting for Method D to complete..."
while [ ! -f "experiments/phase3/D_lora_nsp_only/final_model.pt" ]; do
    echo "  Waiting for D_lora_nsp_only/final_model.pt..."
    sleep 60
done

echo "Method D checkpoint found! Starting Method E..."
python main.py \
    --load_checkpoint experiments/phase3/D_lora_nsp_only/final_model.pt \
    --enable_routing \
    --output_dir experiments/phase3/E_lora_nsp_full \
    > experiments/phase3/E_lora_nsp_full.log 2>&1

echo "Method E completed!"
