#!/bin/bash
# Phase 3: 主实验微调部分 (Table 1-C/D/E)
# GPU 分配: 使用 GPU 0, 1, 2 (明确排除 GPU 3)
# ⚠️ 系统级约束: GPU 3 禁止使用 (硬件错误会导致 CUDA 崩溃)

set -e

# 安全设置
export CUDA_VISIBLE_DEVICES=0,1,2,4,5
cd /home/raoxuan/projects/clip_ood

echo "=========================================="
echo "Phase 3: Main Experiments - Fine-tuning"
echo "=========================================="
echo "GPU Allocation: 0, 1, 2 (GPU 3 excluded)"
echo "=========================================="

# 安全检查
echo "[Safety Check] Verifying CUDA environment..."
if ! bash scripts/check_cuda_safety.sh; then
    echo "ERROR: CUDA safety check failed!"
    exit 1
fi

mkdir -p experiments/phase3

# ========================================
# Method C: LoRA Vanilla (GPU 0)
# ========================================
launch_method_c() {
    echo ""
    echo "[$(date)] Starting Method C: LoRA Vanilla on GPU 0..."
    export CUDA_VISIBLE_DEVICES=0
    
    python src/experiments/run_continual_learning.py \
        --method lora_vanilla \
        --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
        --num_shots 16 \
        --iterations 800 \
        --fd_weight 0 \
        --cd_weight 0 \
        --output_dir experiments/phase3/C_lora_vanilla \
        > experiments/phase3/C_lora_vanilla.log 2>&1
    
    echo "[$(date)] Method C completed!"
}

# ========================================
# Method D: LoRA-NSP Only (GPU 1)
# ========================================
launch_method_d() {
    echo ""
    echo "[$(date)] Starting Method D: LoRA-NSP Only on GPU 1..."
    export CUDA_VISIBLE_DEVICES=1
    
    python src/experiments/run_continual_learning.py \
        --method lora_nsp \
        --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
        --num_shots 16 \
        --iterations 800 \
        --fd_weight 1.0 \
        --cd_weight 1.0 \
        --output_dir experiments/phase3/D_lora_nsp_only \
        > experiments/phase3/D_lora_nsp_only.log 2>&1
    
    echo "[$(date)] Method D completed!"
}

# ========================================
# Method E: LoRA-NSP Full (GPU 2, 等待 D)
# ========================================
launch_method_e() {
    echo ""
    echo "[$(date)] Waiting for Method D to complete..."
    
    # 等待 D 的 checkpoint
    while [ ! -f "experiments/phase3/D_lora_nsp_only/final_model.pt" ]; do
        if [ -f "experiments/phase3/D_lora_nsp_only/D_lora_nsp_only_results.json" ]; then
            echo "  D results found, checking for checkpoint..."
        fi
        sleep 60
    done
    
    echo "[$(date)] Checkpoint found! Starting Method E on GPU 2..."
    export CUDA_VISIBLE_DEVICES=2
    
    # 注意: main.py 需要使用缓存特征或加载模型
    # 这里使用 run_cached_experiment.py 的方式复用 D 的模型
    python scripts/run_cached_experiment_with_checkpoint.py \
        --checkpoint experiments/phase3/D_lora_nsp_only/final_model.pt \
        --enable_routing \
        --output_dir experiments/phase3/E_lora_nsp_full \
        > experiments/phase3/E_lora_nsp_full.log 2>&1
    
    echo "[$(date)] Method E completed!"
}

# ========================================
# 并行启动 C 和 D
# ========================================
echo ""
echo "Launching Methods C and D in parallel..."
launch_method_c &
PID_C=$!

launch_method_d &
PID_D=$!

echo "  Method C (LoRA Vanilla) PID: $PID_C"
echo "  Method D (LoRA-NSP) PID: $PID_D"

# 等待 C 和 D 完成
wait $PID_C
STATUS_C=$?
wait $PID_D
STATUS_D=$?

if [ $STATUS_C -ne 0 ]; then
    echo "WARNING: Method C exited with error (code: $STATUS_C)"
fi

if [ $STATUS_D -ne 0 ]; then
    echo "WARNING: Method D exited with error (code: $STATUS_D)"
fi

# 启动 E (依赖 D)
if [ $STATUS_D -eq 0 ]; then
    launch_method_e
else
    echo "ERROR: Method D failed, cannot run Method E"
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 3 Completed!"
echo "=========================================="
echo "Results:"
echo "  C: experiments/phase3/C_lora_vanilla/"
echo "  D: experiments/phase3/D_lora_nsp_only/"
echo "  E: experiments/phase3/E_lora_nsp_full/"
echo "=========================================="
