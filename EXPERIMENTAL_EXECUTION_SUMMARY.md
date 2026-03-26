# 实验执行优化总结

**文档版本**: 1.0  
**更新日期**: 2026-03-18

---

## 🎯 核心优化思路

### 优化原则
1. **先做轻量级实验**: 预训练 CLIP 的实验（无需训练）
2. **使用特征缓存**: 避免重复提取特征（20-50倍加速）
3. **合理复用结果**: Table 1-A/B 直接复用 Table 4 的结果
4. **延后重量级实验**: 训练端消融（Table 2）放到最后

---

## 📊 实验执行顺序（优化后）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: 推理端实验 (Table 3/4/5)                                            │
│ • 使用预训练 CLIP                                                            │
│ • 提取特征并缓存（一次，~2小时）                                              │
│ • 完成所有 OOD 检测器、路由实验和集成分类器分析                                 │
│ • Table 5: 21个alpha值的集成分类器性能扫描                                     │
│ • 预计时间: ~2.5小时                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: 主实验 - 预训练部分 (Table 1-A/B)                                   │
│ • 直接复用 Phase 1 的结果                                                    │
│ • 创建符号链接即可                                                           │
│ • 预计时间: ~10分钟                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: 主实验 - 微调部分 (Table 1-C/D/E)                                   │
│ • C: LoRA (Vanilla) - 训练                                                  │
│ • D: LoRA-NSP (仅微调) - 训练                                               │
│ • E: LoRA-NSP Full - 复用 D 的模型                                           │
│ • 预计时间: ~8小时 (3×GPU并行)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: 训练端消融 (Table 2)                                                │
│ • 8组消融实验                                                                │
│ • 每组需要训练                                                               │
│ • 预计时间: ~12小时 (3×GPU并行)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**总时间**: ~22小时（vs 传统顺序 ~35小时）

---

## 💾 特征缓存使用指南

### Step 1: 提取特征（只做一次）
```bash
python scripts/extract_cached_features.py \
    --datasets aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --cache_dir cache/pretrained_features \
    --device cuda
```

### Step 2: 使用缓存运行实验
```bash
python main.py \
    --use_cached_features \
    --cache_dir cache/pretrained_features \
    --enable_routing \
    --output_dir experiments/phase1/routing_cached
```

**加速效果**:
- 无缓存：~10分钟/实验
- 有缓存：~30秒/实验
- **加速比**: 20-50倍

---

## 🔗 结果复用关系

| 表格 | 实验 | 复用来源 | 说明 |
|------|------|---------|------|
| Table 1-A | Zero-shot | Table 4-零样本 | 直接复用 |
| Table 1-B | Pretrain + Routing | Table 4-自适应路由 | 直接复用 |
| Table 1-E | LoRA-NSP Full | Table 1-D 的模型 | 复用微调后的模型 |

---

## 📋 各阶段详细命令

### Phase 1: 推理端实验 (~2小时)
```bash
# 1. 提取特征（一次）
python scripts/extract_cached_features.py \
    --datasets aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --cache_dir cache/pretrained_features

# 2. Table 3: OOD 检测器对比
for detector in mahalanobis lda qda lr_rgda; do
    python main.py \
        --use_cached_features \
        --cache_dir cache/pretrained_features \
        --ood_detector_type $detector \
        --enable_routing \
        --output_dir experiments/phase1/detector_${detector} &
done
wait

# 3. Table 4: 推理端消融
python main.py --use_cached_features ... --classifier_type zeroshot ...
python main.py --use_cached_features ... --classifier_type ensemble ...
python main.py --use_cached_features ... --enable_routing ...

# 4. Table 5: 集成分类器 Alpha 敏感性分析（21个点）
mkdir -p experiments/phase1/alpha_sweep
for alpha in 0.50 0.525 0.55 0.575 0.60 0.625 0.65 0.675 0.70 0.725 0.75 0.775 0.80 0.825 0.85 0.875 0.90 0.925 0.95 0.975 1.00; do
    alpha_str=$(echo $alpha | tr '.' '_')
    python main.py \
        --use_cached_features --cache_dir cache/pretrained_features \
        --classifier_type ensemble --alpha $alpha \
        --output_dir experiments/phase1/alpha_sweep/alpha_${alpha_str}
done

# 生成集成分类器性能可视化
python scripts/plot_ensemble_performance.py \
    --input_dir experiments/phase1/alpha_sweep \
    --output_dir figures/ensemble_analysis
```

### Phase 2: 复用结果 (~10分钟)
```bash
# 直接创建符号链接
mkdir -p experiments/phase2
ln -sf experiments/phase1/zeroshot experiments/phase2/A_zeroshot
ln -sf experiments/phase1/routing experiments/phase2/B_pretrain_routing
```

### Phase 3: 微调实验 (~8小时)
```bash
# C: LoRA (Vanilla)
python src/experiments/run_continual_learning.py \
    --method lora_vanilla \
    --task_sequence aircraft caltech101 ... \
    --output_dir experiments/phase3/C_lora_vanilla &

# D: LoRA-NSP (仅微调)
python src/experiments/run_continual_learning.py \
    --method lora_nsp \
    --fd_weight 1.0 --cd_weight 1.0 \
    --use_zeroshot_only \
    --output_dir experiments/phase3/D_lora_nsp_only &

wait

# E: LoRA-NSP Full（复用 D 的模型）
python main.py \
    --load_checkpoint experiments/phase3/D_lora_nsp_only/final_model.pt \
    --enable_routing \
    --output_dir experiments/phase3/E_lora_nsp_full
```

### Phase 4: 训练端消融 (~12小时)
```bash
# 8组实验并行运行
for config in "${configs[@]}"; do
    # 并行启动
    python src/experiments/run_continual_learning.py ... &
done
wait
```

---

## ⚡ 关键优势

| 优化点 | 效果 | 说明 |
|--------|------|------|
| **特征缓存** | 20-50倍加速 | 避免重复提取特征 |
| **结果复用** | 节省时间 ~3小时 | Table 1-A/B 直接复用 |
| **模型复用** | 节省 ~4小时 | E 复用 D 的模型 |
| **并行执行** | 节省 ~10小时 | 3×GPU 并行 |
| **总节省时间** | ~13小时 (37%) | 从 ~35小时降至 ~22小时 |

---

## 📝 注意事项

1. **显存管理**: Phase 3 和 Phase 4 需要较大显存，建议分批运行
2. **检查点保存**: 确保 D 的模型正确保存，以便 E 复用
3. **缓存位置**: `cache/pretrained_features` 需要足够磁盘空间 (~10GB)
4. **日志记录**: 每个实验都要记录日志，便于排查问题

---

## 🎯 预期产出

| 表格 | 内容 | 实验数量 | 预计时间 |
|------|------|---------|---------|
| Table 1 | 主实验（正交性验证） | 5组 | ~8小时 |
| Table 2 | 训练端消融 | 8组 | ~12小时 |
| Table 3 | OOD 检测器对比 | 4组 | ~1小时（含缓存） |
| Table 4 | 推理端消融 | 3组 | ~0.5小时（含缓存） |
| Table 5 | 集成分类器性能变化 | 21个alpha | ~0.5小时（含缓存） |

**Table 5 详细说明**:
- **Alpha 扫描范围**: 0.50 ~ 1.00，步长 0.025，共 21 个点
- **实验目的**: 详细分析集成分类器在不同权重下的 ID/OOD 性能变化
- **关键产出**: 
  - ID Acc 随 alpha 变化曲线（单调上升）
  - OOD Acc 随 alpha 变化曲线（单调下降）
  - 综合得分曲线（抛物线形，峰值在 α ≈ 0.8）
  - 相对单一分类器的性能提升对比
- **预期发现**: α = 0.8 时达到 ID/OOD 最佳平衡，验证超参数优化结果

---

*优化执行计划完成*
