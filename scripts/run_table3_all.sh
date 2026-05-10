#!/bin/bash
# Table 3: OOD检测器对比实验 - 批量运行脚本
# 运行5组交叉验证 × 4种检测器 = 20组实验

set -e

# 配置
OUTPUT_BASE="experiments/table3"
DETECTORS=("mahalanobis" "lda" "qda" "lr_rgda")

# 定义5组数据集组合
# Combo 1: 5ID + 5OOD (平衡)
COMBO1_ID=("aircraft" "caltech101" "dtd" "eurosat" "flowers")
COMBO1_OOD=("food101" "mnist" "oxford_pets" "stanford_cars" "sun397")

# Combo 2: 6ID + 4OOD (ID偏多)
COMBO2_ID=("aircraft" "caltech101" "dtd" "eurosat" "flowers" "food101")
COMBO2_OOD=("mnist" "oxford_pets" "stanford_cars" "sun397")

# Combo 3: 4ID + 6OOD (OOD偏多)
COMBO3_ID=("aircraft" "caltech101" "dtd" "eurosat")
COMBO3_OOD=("flowers" "food101" "mnist" "oxford_pets" "stanford_cars" "sun397")

# Combo 4: 3ID + 7OOD (ID < OOD)
COMBO4_ID=("aircraft" "caltech101" "dtd")
COMBO4_OOD=("eurosat" "flowers" "food101" "mnist" "oxford_pets" "stanford_cars" "sun397")

# Combo 5: 7ID + 3OOD (ID很多)
COMBO5_ID=("aircraft" "caltech101" "dtd" "eurosat" "flowers" "food101" "mnist")
COMBO5_OOD=("oxford_pets" "stanford_cars" "sun397")

echo "=========================================="
echo "Table 3: OOD Detector Comparison"
echo "=========================================="
echo "Total experiments: 5 combos × 4 detectors = 20"
echo "=========================================="

# 创建输出目录
mkdir -p $OUTPUT_BASE

# 运行单个组合的函数
run_combo() {
    local combo_id=$1
    shift
    local id_datasets=("$@")
    shift ${#id_datasets[@]}
    local ood_datasets=("$@")
    
    echo ""
    echo "=========================================="
    echo "Running Combo $combo_id"
    echo "ID datasets (${#id_datasets[@]}): ${id_datasets[*]}"
    echo "OOD datasets (${#ood_datasets[@]}): ${ood_datasets[*]}"
    echo "=========================================="
    
    for detector in "${DETECTORS[@]}"; do
        echo "  Running detector: $detector"
        
        output_dir="$OUTPUT_BASE/combo${combo_id}_${detector}"
        
        python scripts/run_ood_detector_eval.py \
            --id_datasets "${id_datasets[@]}" \
            --ood_datasets "${ood_datasets[@]}" \
            --detector_type $detector \
            --output_dir $output_dir \
            > $output_dir/eval.log 2>&1 &
        
        # 控制并行度（每批4个）
        if [[ $(jobs -r | wc -l) -ge 4 ]]; then
            wait -n
        fi
    done
}

# 运行5组组合
run_combo 1 "${COMBO1_ID[@]}" "${COMBO1_OOD[@]}"
run_combo 2 "${COMBO2_ID[@]}" "${COMBO2_OOD[@]}"
run_combo 3 "${COMBO3_ID[@]}" "${COMBO3_OOD[@]}"
run_combo 4 "${COMBO4_ID[@]}" "${COMBO4_OOD[@]}"
run_combo 5 "${COMBO5_ID[@]}" "${COMBO5_OOD[@]}"

# 等待所有后台任务完成
echo ""
echo "Waiting for all experiments to complete..."
wait
echo "All experiments completed!"

# 生成汇总结果
echo ""
echo "=========================================="
echo "Generating summary..."
echo "=========================================="

python3 << 'PYTHON_EOF'
import json
import os
import numpy as np

detectors = ['mahalanobis', 'lda', 'qda', 'lr_rgda']
results = {}

for detector in detectors:
    aurocs = []
    fpr95s = []
    detection_errors = []
    
    for combo in range(1, 6):
        result_file = f'experiments/table3/combo{combo}_{detector}/results.json'
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
                metrics = data['metrics']
                aurocs.append(metrics['auroc'])
                fpr95s.append(metrics['fpr_at_95_tpr'])
                detection_errors.append(metrics['detection_error'])
    
    if aurocs:
        results[detector] = {
            'auroc_mean': np.mean(aurocs),
            'auroc_std': np.std(aurocs),
            'fpr95_mean': np.mean(fpr95s),
            'fpr95_std': np.std(fpr95s),
            'detection_error_mean': np.mean(detection_errors),
            'detection_error_std': np.std(detection_errors),
        }

# 打印汇总表格
print("\nTable 3: OOD Detector Comparison (5-fold Cross-validation)")
print("="*80)
print(f"{'Detector':<15} {'AUROC (↑)':<20} {'FPR@95TPR (↓)':<20} {'Detection Error (↓)':<20}")
print("="*80)

for detector in detectors:
    if detector in results:
        r = results[detector]
        auroc_str = f"{r['auroc_mean']*100:.1f}±{r['auroc_std']*100:.1f}%"
        fpr95_str = f"{r['fpr95_mean']*100:.1f}±{r['fpr95_std']*100:.1f}%"
        det_err_str = f"{r['detection_error_mean']*100:.1f}±{r['detection_error_std']*100:.1f}%"
        print(f"{detector:<15} {auroc_str:<20} {fpr95_str:<20} {det_err_str:<20}")

print("="*80)

# 保存汇总结果
with open('experiments/table3/summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSummary saved to: experiments/table3/summary.json")
PYTHON_EOF

echo ""
echo "=========================================="
echo "Table 3 Completed!"
echo "=========================================="
echo "Best detector and threshold can be found in:"
echo "  experiments/table3/summary.json"
echo "=========================================="
