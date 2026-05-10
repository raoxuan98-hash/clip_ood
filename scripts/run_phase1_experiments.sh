#!/bin/bash
# Phase 1: 推理端实验 (Table 3/4/5)
# 使用特征缓存加速

set -e

# 配置
CACHE_DIR="cache/pretrained_features"
EXPERIMENTS_DIR="experiments/phase1"
FIGURES_DIR="figures"
DATASETS="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

# ID和OOD数据集配置
ID_DATASETS=("caltech101" "flowers" "oxford_pets")
OOD_DATASETS=("dtd" "eurosat" "mnist")

# 创建目录
mkdir -p $CACHE_DIR
mkdir -p $EXPERIMENTS_DIR
mkdir -p $FIGURES_DIR

echo "=========================================="
echo "Phase 1: Inference-side Experiments"
echo "=========================================="

# Step 1: 提取特征（如果缓存不存在）
if [ ! -f "$CACHE_DIR/caltech101_features.pkl" ]; then
    echo ""
    echo "[Step 1] Extracting features..."
    python scripts/extract_cached_features.py \
        --datasets $DATASETS \
        --cache_dir $CACHE_DIR \
        --device cuda \
        --batch_size 32
else
    echo ""
    echo "[Step 1] Cache already exists, skipping feature extraction"
fi

# Step 2: Table 3 - OOD检测器对比
echo ""
echo "=========================================="
echo "[Table 3] OOD Detector Comparison"
echo "=========================================="

for detector in mahalanobis lda qda lr_rgda; do
    echo "Running detector: $detector"
    python scripts/run_cached_experiment.py \
        --id_datasets ${ID_DATASETS[@]} \
        --ood_datasets ${OOD_DATASETS[@]} \
        --cache_dir $CACHE_DIR \
        --ood_detector_type $detector \
        --enable_routing \
        --ood_threshold 0.993 \
        --output_dir $EXPERIMENTS_DIR/detector_${detector} \
        > $EXPERIMENTS_DIR/detector_${detector}.log 2>&1 &
done

# 等待所有检测器实验完成
echo "Waiting for detector experiments to complete..."
wait
echo "Detector experiments completed!"

# Step 3: Table 4 - 推理端消融
echo ""
echo "=========================================="
echo "[Table 4] Inference Ablation"
echo "=========================================="

# 3.1 纯零样本分类器
echo "Running: Zero-shot only"
python scripts/run_cached_experiment.py \
    --id_datasets ${ID_DATASETS[@]} \
    --ood_datasets ${OOD_DATASETS[@]} \
    --cache_dir $CACHE_DIR \
    --classifier_type zeroshot \
    --output_dir $EXPERIMENTS_DIR/ablation_zeroshot \
    > $EXPERIMENTS_DIR/ablation_zeroshot.log 2>&1

# 3.2 集成分类器 (无路由, alpha=0.8)
echo "Running: Ensemble (alpha=0.8, no routing)"
python scripts/run_cached_experiment.py \
    --id_datasets ${ID_DATASETS[@]} \
    --ood_datasets ${OOD_DATASETS[@]} \
    --cache_dir $CACHE_DIR \
    --classifier_type ensemble \
    --alpha 0.8 \
    --output_dir $EXPERIMENTS_DIR/ablation_ensemble \
    > $EXPERIMENTS_DIR/ablation_ensemble.log 2>&1

# 3.3 自适应路由 (完整推理端)
echo "Running: Adaptive Routing"
python scripts/run_cached_experiment.py \
    --id_datasets ${ID_DATASETS[@]} \
    --ood_datasets ${OOD_DATASETS[@]} \
    --cache_dir $CACHE_DIR \
    --ood_detector_type lr_rgda \
    --enable_routing \
    --ood_threshold 0.993 \
    --output_dir $EXPERIMENTS_DIR/ablation_routing \
    > $EXPERIMENTS_DIR/ablation_routing.log 2>&1

echo "Ablation experiments completed!"

# Step 4: Table 5 - Alpha敏感性分析 (21个点)
echo ""
echo "=========================================="
echo "[Table 5] Alpha Sensitivity Analysis"
echo "=========================================="

ALPHA_VALUES=(0.50 0.525 0.55 0.575 0.60 0.625 0.65 0.675 0.70 0.725 0.75 0.775 0.80 0.825 0.85 0.875 0.90 0.925 0.95 0.975 1.00)

# 并行运行alpha扫描（每批4个）
BATCH_SIZE=4
for ((i=0; i<${#ALPHA_VALUES[@]}; i+=BATCH_SIZE)); do
    echo "Running alpha batch $((i/BATCH_SIZE+1))/$(((${#ALPHA_VALUES[@]}+BATCH_SIZE-1)/BATCH_SIZE))"
    
    for ((j=i; j<i+BATCH_SIZE && j<${#ALPHA_VALUES[@]}; j++)); do
        alpha=${ALPHA_VALUES[$j]}
        alpha_str=$(echo $alpha | tr '.' '_')
        
        echo "  Starting alpha=$alpha"
        python scripts/run_cached_experiment.py \
            --id_datasets ${ID_DATASETS[@]} \
            --ood_datasets ${OOD_DATASETS[@]} \
            --cache_dir $CACHE_DIR \
            --classifier_type ensemble \
            --alpha $alpha \
            --output_dir $EXPERIMENTS_DIR/alpha_${alpha_str} \
            > $EXPERIMENTS_DIR/alpha_${alpha_str}.log 2>&1 &
    done
    
    # 等待当前批次完成
    wait
done

echo "Alpha sensitivity experiments completed!"

# Step 5: 生成可视化
echo ""
echo "=========================================="
echo "[Step 5] Generating Visualizations"
echo "=========================================="

python scripts/plot_alpha_sensitivity.py \
    --input_dir $EXPERIMENTS_DIR \
    --pattern "alpha_*" \
    --output_dir $FIGURES_DIR \
    --output_prefix alpha_sensitivity

echo ""
echo "=========================================="
echo "Phase 1 Completed!"
echo "=========================================="
echo "Results saved to: $EXPERIMENTS_DIR"
echo "Figures saved to: $FIGURES_DIR"
echo ""
echo "Summary:"
echo "  - Table 3: OOD Detector Comparison"
echo "  - Table 4: Inference Ablation"
echo "  - Table 5: Alpha Sensitivity (21 points)"
echo "=========================================="
