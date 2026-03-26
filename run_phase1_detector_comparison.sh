#!/bin/bash
# Phase 1 - Table 3: OOD检测器类型对比实验
# 对比 4 种 OOD 检测器在预训练 CLIP 上的性能

set -e  # 遇到错误立即退出

WORK_DIR="/home/raoxuan/projects/clip_ood"
OUTPUT_BASE="$WORK_DIR/experiments/phase1"

# 实验配置
ID_DATASETS="caltech101 flowers oxford_pets stanford_cars food101"
OOD_DATASETS="dtd eurosat mnist sun397"
ITERATIONS=0  # 不训练，使用预训练 CLIP
GPU_ID=0

# 定义检测器类型
DETECTORS=("mahalanobis" "lda" "qda" "lr_rgda")

echo "================================================"
echo "Phase 1 - Table 3: OOD检测器类型对比实验"
echo "================================================"
echo ""
echo "实验配置:"
echo "  ID datasets: $ID_DATASETS"
echo "  OOD datasets: $OOD_DATASETS"
echo "  Iterations: $ITERATIONS (预训练 CLIP)"
echo "  GPU: $GPU_ID"
echo ""

# 结果汇总文件
RESULTS_SUMMARY="$OUTPUT_BASE/detector_comparison_results.csv"
echo "detector,auroc,fpr_at_95_tpr,detection_error" > "$RESULTS_SUMMARY"

# 遍历所有检测器
for DETECTOR in "${DETECTORS[@]}"; do
    echo "================================================"
    echo "运行检测器: $DETECTOR"
    echo "================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE/detector_$DETECTOR"
    LOG_FILE="$OUTPUT_DIR/experiment.log"
    
    # 构建命令
    CMD="cd $WORK_DIR && python main.py \
        --id_datasets $ID_DATASETS \
        --ood_datasets $OOD_DATASETS \
        --iterations $ITERATIONS \
        --ood_detector_type $DETECTOR \
        --device cuda:$GPU_ID \
        --seed 42 \
        2>&1 | tee $LOG_FILE"
    
    echo "命令:"
    echo "$CMD"
    echo ""
    
    # 执行实验
    eval "$CMD"
    
    # 从日志中提取关键指标
    echo ""
    echo "--- $DETECTOR 实验结果 ---"
    
    # 提取 AUROC
    AUROC=$(grep -oP "AUROC: \K[0-9.]+" "$LOG_FILE" || echo "N/A")
    echo "AUROC: $AUROC"
    
    # 提取 FPR@95TPR
    FPR=$(grep -oP "FPR@95TPR: \K[0-9.]+" "$LOG_FILE" || echo "N/A")
    echo "FPR@95TPR: $FPR"
    
    # 提取 Detection Error
    DET_ERR=$(grep -oP "Detection Error: \K[0-9.]+" "$LOG_FILE" || echo "N/A")
    echo "Detection Error: $DET_ERR"
    
    # 保存到汇总文件
    echo "$DETECTOR,$AUROC,$FPR,$DET_ERR" >> "$RESULTS_SUMMARY"
    
    echo ""
    echo "✓ $DETECTOR 实验完成，结果已保存到 $OUTPUT_DIR"
    echo ""
done

echo "================================================"
echo "所有实验完成!"
echo "================================================"
echo ""
echo "结果汇总:"
cat "$RESULTS_SUMMARY"
echo ""
echo "详细结果保存在: $OUTPUT_BASE"
