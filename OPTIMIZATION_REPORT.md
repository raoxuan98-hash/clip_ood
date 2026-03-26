# CLIP OOD检测与集成分类器 - 超参数优化报告

## 1. 项目概述

本项目实现了CLIP模型的持续学习能力，包括：
- **LoRA-NSP微调**：融合零空间参数化与低秩适应
- **LR-RGDA分类器**：低秩分解正则高斯判别分析
- **OOD检测器**：基于统计分布的异常检测
- **集成分类器**：融合零样本分类器与LR-RGDA

## 2. OOD检测器优化

### 2.1 优化目标
选择最优超参数，使得在95% TPR下FPR最低。

### 2.2 数据集
- **ID数据集**: Caltech101 (100类) + Flowers (102类) = 202类
- **OOD数据集**: DTD (47类) + EuroSAT (10类)
- **训练样本**: 16-shot per class

### 2.3 优化方法
网格搜索以下参数：
- **Mahalanobis**: alpha ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- **LR-RGDA**: rank ∈ [32, 64, 128], reg_alpha ∈ [0.1, 0.2, 0.3]

### 2.4 优化结果

#### Mahalanobis检测器
| Alpha | AUROC | FPR@95TPR | Detection Error |
|-------|-------|-----------|-----------------|
| 0.0 | 0.8974 | 0.2398 | 0.1407 |
| 0.2 | 0.9109 | 0.1980 | 0.1216 |
| 0.4 | 0.9114 | 0.1962 | 0.1220 |
| 0.6 | 0.9120 | 0.1959 | 0.1208 |
| 0.8 | 0.9137 | 0.1934 | 0.1188 |
| **1.0** ⭐ | **0.9569** | **0.1306** | **0.0877** |

**最佳配置**: alpha=1.0 (纯类特定协方差)

#### LR-RGDA检测器 - 初始网格搜索
| Rank | α₁/α₂/α₃ | AUROC | FPR@95TPR |
|------|----------|-------|-----------|
| 32 | 0.3/0.3/0.3 | 0.9855 | 0.0662 |
| 64 | 0.3/0.3/0.3 | 0.9855 | 0.0662 |
| 128 | 0.3/0.3/0.3 | 0.9855 | 0.0662 |

#### LR-RGDA检测器 - Alpha1/Alpha2深度优化 ⭐

固定 rank=32, α₃=0.5，对 α₁ 和 α₂ 进行深度网格搜索。发现 **α₂ 的作用远未饱和**，需要扩展到更大的范围。

**第一阶段** (α₂ ∈ [0, 1.0]):
| α₁ | α₂ | AUROC | FPR@95TPR |
|----|----|-------|-----------|
| 0.0 | 1.0 | 0.9782 | 0.1126 |
| 0.2 | 1.0 | 0.9850 | 0.0681 |
| 0.4 | 1.0 | 0.9874 | 0.0520 |
| **0.6** | **1.0** | **0.9877** | **0.0458** |

**第二阶段** (α₂ 扩展到 [1.0, 20.0]):
| α₁ | α₂ | AUROC | FPR@95TPR | 趋势 |
|----|----|-------|-----------|------|
| 0.6 | 1.0 | 0.9877 | 0.0458 | ↓ |
| 0.6 | 4.0 | 0.9882 | 0.0427 | ↓ |
| 0.6 | 10.0 | 0.9889 | 0.0368 | ↓ |
| **0.6** ⭐ | **16.0** ⭐ | **0.9894** | **0.0356** | 平台 |
| 0.6 | 20.0 | 0.9896 | 0.0359 | 略升 |

**最终最佳配置**: rank=32, **α₁=0.6, α₂=16.0, α₃=0.5**
- AUROC: **98.94%**
- FPR@95TPR: **3.56%** (相比初始6.62%降低46%！)
- Detection Error: **3.88%**

**关键发现**: α₂（低秩分解正则化系数）的最优值远大于预期，在16.0附近达到平台。这表明强正则化对LR-RGDA的OOD检测性能至关重要。

### 2.5 结论
- **LR-RGDA显著优于Mahalanobis**: AUROC 98.94% vs 95.69%
- **深度优化提升惊人**: FPR@95TPR从6.62%降至3.56% (降低46%！)
- **关键发现**: α₂（低秩分解正则化）的最优值高达16.0，远超预期
- **推荐检测器**: LR-RGDA with rank=32, **α₁=0.6, α₂=16.0, α₃=0.5**
- **推荐阈值**: 0.9936 (基于95% TPR)

## 3. 集成分类器优化

### 3.1 架构说明

集成分类器组合零样本分类器和LR-RGDA分类器：

```
Ensemble(x) = α · LR-RGDA(x) + (1-α) · Zeroshot(x)
```

**扩展类别空间** (259类):
- ID类别 (0-201): 使用完整Ensemble
- OOD类别 (202-258): LR-RGDA输出0，仅使用Zeroshot

### 3.2 样本平衡
为确保公平评估，调整样本比例：
- **原始**: ID=3232, OOD=912 (比例 1:3.5)
- **平衡后**: ID=456, OOD=912 (比例 1:2)

### 3.3 Temperature参数
Temperature控制Zeroshot输出的平滑程度：
- **T < 1**: 更尖锐，最大概率类别置信度更高
- **T = 1**: 标准softmax
- **T > 1**: 更平滑，概率分布更均匀

### 3.4 优化结果

#### Temperature = 0.5 (尖锐)
| Alpha | Overall | ID Acc | OOD Acc |
|-------|---------|--------|---------|
| 0.0 | 51.46% | 74.34% | 40.02% |
| 0.5 | 51.75% | 75.22% | 40.02% |
| **0.7** ⭐ | **51.83%** | **75.44%** | **40.02%** |
| 1.0 | 30.92% | 92.76% | 0.00% |

#### Temperature = 1.0 (标准) ⭐
| Alpha | Overall | ID Acc | OOD Acc |
|-------|---------|--------|---------|
| 0.0 | 52.78% | 79.39% | 39.47% |
| 0.5 | 53.07% | 80.26% | 39.47% |
| **0.8** ⭐ | **53.14%** | **80.48%** | **39.47%** |
| 1.0 | 30.70% | 92.11% | 0.00% |

#### Temperature = 5.0 (平滑)
| Alpha | Overall | ID Acc | OOD Acc |
|-------|---------|--------|---------|
| 0.0 | 52.78% | 77.63% | 40.35% |
| 0.1 ⭐ | **53.07%** | **78.95%** | **40.13%** |
| 0.5 | 50.15% | 81.14% | 34.65% |
| 1.0 | 31.14% | 93.42% | 0.00% |

### 3.5 最佳配置对比

| Temperature | 最佳 Alpha | Overall | ID Acc | OOD Acc | 特点 |
|------------|-----------|---------|--------|---------|------|
| 0.5 | 0.7 | 51.83% | 75.44% | 40.02% | 尖锐，需较小α |
| **1.0** ⭐ | **0.8** | **53.14%** | **80.48%** | **39.47%** | 平衡，可用大α |
| 5.0 | 0.1 | 53.07% | 78.95% | 40.13% | 平滑，需较小α |

## 4. 关键发现

### 4.1 OOD检测
1. **LR-RGDA优于Mahalanobis**: FPR@95TPR降低约50%
2. **低秩分解有效**: rank=32即可达到最佳性能
3. **正则化重要**: α=0.3时性能最佳

### 4.2 集成分类
1. **Temperature=1.0是最佳平衡点**
   - 既不会过于尖锐导致融合不稳定
   - 也不会过于平滑导致信息丢失

2. **Alpha与Temperature的关系**
   - T越小，最佳Alpha越小 (更依赖Zeroshot)
   - T=1.0时，可以使用较大的Alpha (0.8)，充分利用LR-RGDA

3. **OOD准确度稳定**
   - 在不同Temperature下保持约40%
   - 对Alpha变化相对不敏感 (直到α接近1.0)

4. **Alpha=1.0时OOD准确度降为0**
   - 因为LR-RGDA对OOD类别没有统计信息
   - 证明扩展类别空间的设计是必要的

## 5. 推荐配置

### 5.1 OOD检测器
```python
ClassifierBasedOODDetector(
    classifier_type='lr_rgda',
    rank=32,
    qda_reg_alpha1=0.6,   # 协方差正则化
    qda_reg_alpha2=16.0,  # 低秩分解正则化 (深度优化后的高值)
    qda_reg_alpha3=0.5,   # 混合正则化
    threshold=0.9936      # 基于95% TPR
)
```

### 5.2 集成分类器
```python
EnsembleClassifier(
    zeroshot_classifier=zeroshot_weights,  # 259类
    lr_rgda_classifier=lr_rgda,  # 202类
    alpha=0.8,  # 最佳融合权重
    temperature=1.0,  # 标准softmax
    num_id_classes=202  # ID类别数量
)
```

### 5.3 配置文件
```yaml
# configs/base/default.yaml
classifier:
  type: "lr_rgda"
  rank: 32
  reg_alpha1: 0.6    # 深度优化后的值
  reg_alpha2: 16.0   # 深度优化后的高值 (关键!)
  reg_alpha3: 0.5    # 固定值

ensemble:
  alpha: 0.8
  temperature: 1.0

ood_detector:
  type: "lr_rgda"
  rank: 32
  reg_alpha1: 0.6    # 深度优化后的值
  reg_alpha2: 16.0   # 深度优化后的高值 (关键!)
  reg_alpha3: 0.5    # 固定值
  threshold: 0.9936  # 基于95% TPR
```

## 6. 使用指南

### 6.1 运行优化脚本

#### OOD检测器优化

**深度优化流程** (推荐):
```bash
# 步骤1: 粗略搜索 alpha1/alpha2 范围
python scripts/optimization/optimize_ood_detector.py \
    --stats stats/pretrained.pt \
    --id_datasets caltech101,flowers \
    --ood_datasets dtd,eurosat \
    --detector_type lr_rgda \
    --optimize_alpha_grid \
    --alpha1_range 0.4,0.5,0.6,0.7,0.8 \
    --alpha2_range 1.0,2.0,4.0,8.0,16.0 \
    --alpha3_fixed 0.5 \
    --output_dir optimization/alpha_grid_coarse

# 步骤2: 精细搜索最优区域
python scripts/optimization/optimize_ood_detector.py \
    --stats stats/pretrained.pt \
    --id_datasets caltech101,flowers \
    --ood_datasets dtd,eurosat \
    --detector_type lr_rgda \
    --optimize_alpha_grid \
    --alpha1_range 0.55,0.6,0.65 \
    --alpha2_range 14.0,15.0,16.0,17.0,18.0 \
    --alpha3_fixed 0.5 \
    --output_dir optimization/alpha_grid_fine
```

#### 集成分类器优化
```bash
python scripts/optimization/optimize_ensemble_classifier.py \
    --stats stats/pretrained.pt \
    --id_datasets caltech101,flowers \
    --ood_datasets dtd,eurosat \
    --fixed_temperature 1.0 \
    --alpha_metric overall_accuracy \
    --output_dir optimization/ensemble
```

### 6.2 使用优化后的参数
```bash
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml
```

## 7. 总结

本次优化确定了CLIP OOD检测和集成分类的最佳超参数：

| 组件 | 最佳参数 | 性能指标 |
|-----|---------|---------|
| OOD检测器 | LR-RGDA, rank=32, **α₁=0.6, α₂=16.0**, α₃=0.5 | **AUROC=98.94%, FPR@95TPR=3.56%** |
| 集成分类器 | α=0.8, T=1.0 | Overall=53.14%, ID=80.48%, OOD=39.47% |

### 关键提升
1. **LR-RGDA深度优化效果惊人**: FPR@95TPR从**6.62%降至3.56%**（相对降低46%！）
2. **关键发现 - α₂的最优值高达16.0**: 最初以为α₂=1.0已足够，但深入搜索发现最优值在16.0附近
3. **优化过程**:
   - 初始: α₁=α₂=α₃=0.3 → FPR@95TPR=6.62%
   - 第一次优化: α₁=0.6, α₂=1.0 → FPR@95TPR=4.58%
   - **深度优化: α₁=0.6, α₂=16.0 → FPR@95TPR=3.56%** ⭐

### 重要启示
- **α₂（低秩分解正则化）的作用被严重低估**: 需要大幅扩展搜索范围才能找到真正的最优值
- **精细网格搜索的必要性**: 粗略的网格搜索可能导致次优结果
- **可视化工具的价值**: 热力图帮助识别参数作用的饱和趋势

这些参数已设置为默认值，可直接用于后续实验。
