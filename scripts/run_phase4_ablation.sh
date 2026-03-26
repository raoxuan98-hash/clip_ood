#!/bin/bash
# Phase 4: 训练端消融实验 (Table 2)
# GPU 分配: 使用 GPU 4, 5 (明确排除 GPU 3)
# ⚠️ 系统级约束: GPU 3 禁止使用 (硬件错误会导致 CUDA 崩溃)

set -e

# 安全设置
export CUDA_VISIBLE_DEVICES=0,1,2,4,5
cd /home/raoxuan/projects/clip_ood

echo "=========================================="
echo "Phase 4: Training-side Ablation Experiments"
echo "=========================================="
echo "GPU Allocation: 4, 5 (GPU 3 excluded)"
echo "=========================================="

# 安全检查
echo "[Safety Check] Verifying CUDA environment..."
if ! bash scripts/check_cuda_safety.sh; then
    echo "ERROR: CUDA safety check failed!"
    exit 1
fi

mkdir -p experiments/phase4

# ========================================
# 8 组消融配置
# ========================================
declare -A CONFIGS=(
    ["baseline"]="lora_vanilla 0 0"
    ["nsp_only"]="lora_nsp 0 0"
    ["fd_only"]="lora_vanilla 1.0 0"
    ["cd_only"]="lora_vanilla 0 1.0"
    ["fd_cd"]="lora_vanilla 1.0 1.0"
    ["nsp_fd"]="lora_nsp 1.0 0"
    ["nsp_cd"]="lora_nsp 0 1.0"
    ["full"]="lora_nsp 1.0 1.0"
)

# ========================================
# Batch 1: 在 GPU 4 上运行 configs 1-4
# ========================================
launch_batch1() {
    echo ""
    echo "[$(date)] Launching Batch 1 on GPU 4..."
    export CUDA_VISIBLE_DEVICES=4
    
    configs=("baseline" "nsp_only" "fd_only" "cd_only")
    
    for name in "${configs[@]}"; do
        IFS=' ' read -r method fd cd <<< "${CONFIGS[$name]}"
        
        echo "  Starting: $name (method=$method, fd=$fd, cd=$cd)"
        
        python src/experiments/run_continual_learning.py \
            --method $method \
            --fd_weight $fd \
            --cd_weight $cd \
            --task_sequence aircraft caltech101 dtd eurosat flowers \
            --num_shots 16 \
            --iterations 800 \
            --output_dir experiments/phase4/${name} \
            > experiments/phase4/${name}.log 2>&1 &
    done
    
    wait
    echo "[$(date)] Batch 1 completed!"
}

# ========================================
# Batch 2: 在 GPU 5 上运行 configs 5-8
# ========================================
launch_batch2() {
    echo ""
    echo "[$(date)] Launching Batch 2 on GPU 5..."
    export CUDA_VISIBLE_DEVICES=5
    
    configs=("fd_cd" "nsp_fd" "nsp_cd" "full")
    
    for name in "${configs[@]}"; do
        IFS=' ' read -r method fd cd <<< "${CONFIGS[$name]}"
        
        echo "  Starting: $name (method=$method, fd=$fd, cd=$cd)"
        
        python src/experiments/run_continual_learning.py \
            --method $method \
            --fd_weight $fd \
            --cd_weight $cd \
            --task_sequence aircraft caltech101 dtd eurosat flowers \
            --num_shots 16 \
            --iterations 800 \
            --output_dir experiments/phase4/${name} \
            > experiments/phase4/${name}.log 2>&1 &
    done
    
    wait
    echo "[$(date)] Batch 2 completed!"
}

# ========================================
# 主执行流程
# ========================================

# 先运行 Batch 1 (GPU 4)
launch_batch1

# 再运行 Batch 2 (GPU 5)
launch_batch2

# 生成汇总
echo ""
echo "=========================================="
echo "Phase 4 Completed!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo ""
echo "| NSP | FD | CD | Experiment |"
echo "|-----|----|----|------------|"
echo "| ✗   | ✗  | ✗  | baseline   |"
echo "| ✓   | ✗  | ✗  | nsp_only   |"
echo "| ✗   | ✓  | ✗  | fd_only    |"
echo "| ✗   | ✗  | ✓  | cd_only    |"
echo "| ✗   | ✓  | ✓  | fd_cd      |"
echo "| ✓   | ✓  | ✗  | nsp_fd     |"
echo "| ✓   | ✗  | ✓  | nsp_cd     |"
echo "| ✓   | ✓  | ✓  | full       |"
echo ""
echo "All results saved to: experiments/phase4/"
echo "=========================================="
