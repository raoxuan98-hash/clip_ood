# CLIP持续学习项目实现笔记

> 本文档记录项目的关键实现细节、设计决策和使用说明

---

## 1. 架构设计

### 1.1 核心原则

**模块化 + 职责分离**

```
src/                    # 核心算法模块
├── trainers/          # LoRA-NSP训练器
├── classifiers/       # LR-RGDA分类器
├── detectors/         # OOD检测器
└── routing/           # 自适应路由

scripts/
├── core/              # 核心脚本（每个只做一件事）
├── workflows/         # 工作流脚本（串联核心脚本）
├── ablations/         # 消融实验
└── optimization/      # 超参数优化
```

**关键设计**：基于统计分布构建，而非原始数据

### 1.2 数据流

```
原始数据 → 特征提取 → 统计分布(stats_dict) → 分类器/检测器
                                    ↓
                              支持增量学习
                                    ↓
                         合并历史 + 新任务统计
```

---

## 2. LoRA-NSP实现细节

### 2.1 核心机制

**标准LoRA**：
```
W = W_0 + B·A
A: [r, d_in]  - 低秩矩阵
B: [d_out, r] - 低秩矩阵
```

**LoRA-NSP（带零空间约束）**：
```
W = W_0 + B·A·P
P: [d_in, d_in] - 投影矩阵（由协方差计算，固定）
```

### 2.2 协方差提取与维护

**提取**：所有LoRA层的非中心化协方差
```python
Σ = X^T X / N  # X: [N, D] 层输入特征
```

**更新**：滑动平均
```python
Σ_history = α·Σ_history + (1-α)·Σ_new
α = 0.9 (默认，可配置)
```

**应用**：构建投影矩阵P
```python
# 特征分解
Σ = V·Λ·V^T

# 软投影
w_i = 1 / (1 + β·log(1 + λ_i^p))
P = V·diag(w)·V^T
```

### 2.3 增量学习流程

```
Task 1:
  1. 从预训练CLIP → LoRA微调
  2. 提取协方差 Σ_1
  3. 保存 Σ_history = Σ_1
  4. 合并: W = W + B·A·P, A重新初始化, B归零

Task 2:
  1. 从Task 1模型 → LoRA微调（加载Σ_history，自动应用零空间约束）
  2. 提取协方差 Σ_2
  3. 更新: Σ_history = 0.9·Σ_1 + 0.1·Σ_2
  4. 更新投影矩阵P
  5. 合并权重，重置A/B

Task 3+:
  重复Task 2流程...
```

**关键**：每个任务学习A和B（可学习），P固定（由协方差计算）

---

## 3. 分类器与检测器

### 3.1 基于统计分布构建

**LR-RGDA分类器**：
```python
stats_dict = {
    class_id: GaussianStatistics(mean, cov)
    for class_id in all_classes
}

classifier = LRRGDAClassifier(stats_dict, device='cuda')
```

**优势**：
- 无需原始数据，仅保存均值+协方差
- 增量学习：直接合并stats_dict
- 内存高效：几百KB vs 几百MB（特征缓存）

### 3.2 集成分类器

```python
EnsembleClassifier(
    zeroshot_classifier=zeroshot_weights,  # [D, C]
    lr_rgda_classifier=lr_rgda_classifier,  # 基于统计
    alpha=0.5,                              # 集成权重
    temperature=2.0                         # 零样本温度
)

# 输出
P_ensemble = α·P_lr_rgda + (1-α)·P_zeroshot
```

### 3.3 OOD检测器

**类型**：
- Mahalanobis（基于马氏距离）
- ClassifierBased（LDA/LR-RGDA/QDA）

**OOD分数**：
```python
score = 1 - max(Posterior Probability)
# 越高越可能是OOD
```

---

## 4. 超参数优化

### 4.1 两阶段优化策略

**集成分类器**：
1. **Stage 1**：优化temperature
   - 目标：匹配零样本分类器输出分布
   - 指标：`max_prob_mse`, `kl_divergence`, `entropy_match`
   - 方式：离散网格搜索

2. **Stage 2**：优化alpha
   - 目标：最大化ID准确率
   - 范围：[0.0, 1.0] 离散取值
   - 方式：离散网格搜索

**OOD检测器**：
- 优化检测器特定参数（alpha/rank等）
- 目标：在TPR=95%时最小化FPR

### 4.3 优化结果

已完成超参数优化，详细结果和推荐参数见：
- [OPTIMIZATION_REPORT.md](./OPTIMIZATION_REPORT.md) - 完整的优化报告
- [OPTIMIZATION_LOG.md](./OPTIMIZATION_LOG.md) - 实验记录

**关键结论**：
| 组件 | 最佳参数 | 性能 |
|-----|---------|------|
| OOD检测器 | LR-RGDA, rank=32, α=0.3 | AUROC=98.55% |
| 集成分类器 | α=0.8, T=1.0 | Overall=53.14% |

### 4.2 特征缓存

**为什么**：
- CLIP前向传播是瓶颈
- 超参数优化需要评估数十到数百次
- 同一特征被重复计算

**使用**：
```bash
# 1. 提取并缓存（一次，慢）
python optimize_xxx.py --cache_features --cache_dir cache/exp ...

# 2. 使用缓存（多次，快）
python optimize_xxx.py --use_cached_features --cache_dir cache/exp ...
```

**加速比**：60x-144x

---

## 5. 增量学习工作流

### 5.1 运行完整序列

```bash
python scripts/workflows/run_incremental.py \
  --task_sequence caltech101,flowers,oxford_pets \
  --all_datasets caltech101,flowers,oxford_pets,stanford_cars,food101 \
  --output_dir experiments/incremental_exp1 \
  --cov_momentum 0.9
```

**输出结构**：
```
experiments/incremental_exp1/
├── state.json                    # 实验状态
├── task_00_caltech101/
│   ├── model.pt                  # 微调后的CLIP
│   ├── covariance_history.pt     # 协方差历史
│   ├── stats.pt                  # 类别统计分布
│   └── results.json              # 评估结果
├── task_01_flowers/
│   └── ...
└── forgetting_curve.png          # 可视化
```

### 5.2 关键文件

- `covariance_history.pt`：所有LoRA层的协方差矩阵（用于零空间约束）
- `stats.pt`：类别均值和协方差（用于构建分类器）

---

## 6. 使用场景速查

### 场景1：验证LoRA-NSP效果（不微调）

```bash
# 仅使用零样本分类器评估
python scripts/core/train_clip.py --datasets caltech101 --output ckpt.pt
python scripts/core/evaluate.py --model ckpt.pt --mode zeroshot
```

### 场景2：单独测试分类器设计

```bash
# 不微调CLIP，直接测试分类器
python scripts/core/extract_stats.py --datasets caltech101 --output stats.pt
python scripts/core/build_classifier.py --stats stats.pt --output cls.pt
python scripts/core/evaluate.py --classifier cls.pt --mode ensemble
```

### 场景3：超参数优化

```bash
# OOD检测器
python scripts/optimization/optimize_ood_detector.py \
  --stats stats.pt --cache_features --cache_dir cache/ood \
  --id_datasets caltech101 --ood_datasets dtd \
  --target_tpr 0.95

# 集成分类器
python scripts/optimization/optimize_ensemble_classifier.py \
  --stats stats.pt --cache_features --cache_dir cache/ensemble \
  --id_datasets caltech101 --match_metric max_prob_mse
```

### 场景4：增量学习

```bash
python scripts/workflows/run_incremental.py \
  --task_sequence caltech101,flowers,oxford_pets \
  --all_datasets caltech101,flowers,oxford_pets,cars,food \
  --output_dir experiments/incremental
```

---

## 7. 关键参数说明

### 7.1 LoRA-NSP

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `lora_rank` | 低秩维度 | 4-8 |
| `cov_momentum` | 协方差滑动平均系数 | 0.9 |
| `nsp_eps` | 零空间阈值（硬投影） | 0.05 |
| `nsp_weight` | 单位矩阵混合权重 | 0.02 |

### 7.2 分类器

| 参数 | 含义 | 范围 |
|------|------|------|
| `alpha` | LR-RGDA贡献系数 | [0.0, 1.0] |
| `temperature` | 零样本温度 | 0.5-10.0 |
| `rank` | LR-RGDA低秩 | 32, 64, 128 |

### 7.3 OOD检测

| 参数 | 含义 | 说明 |
|------|------|------|
| `target_tpr` | 目标OOD检测率 | 0.95 (95%) |
| `threshold` | OOD判定阈值 | 根据target_tpr自动确定 |

---

## 8. 常见问题

### Q1: 是否需要微调CLIP才能优化超参数？

**不需要**。超参数优化可以基于预训练CLIP：
```bash
# 不传--model，默认使用预训练CLIP
python optimize_ood_detector.py --stats stats.pt ...
```

### Q2: 如何快速迭代超参数？

**使用特征缓存**：
```bash
# 第一次：提取特征（慢）
python optimize_xxx.py --cache_features --cache_dir cache/exp ...

# 后续：使用缓存（快）
python optimize_xxx.py --use_cached_features --cache_dir cache/exp ...
```

### Q3: 增量学习时如何合并历史知识？

**协方差历史**：滑动平均更新
**统计分布**：直接拼接（类别互斥）
**零样本分类器**：权重拼接

### Q4: 如何选择alpha和temperature？

**两阶段优化**：
1. 先优化temperature（匹配零样本输出）
2. 再优化alpha（最大化ID准确率）

使用脚本自动完成：
```bash
python optimize_ensemble_classifier.py \
  --match_metric max_prob_mse \
  --alpha_metric id_accuracy
```

---

## 9. 性能基准

### 时间开销（基于V100 GPU）

| 任务 | 时间（无缓存） | 时间（有缓存） | 加速比 |
|------|---------------|---------------|--------|
| 超参数优化（100组合） | 2小时 | 2分钟 | **60x** |
| 提取统计分布 | 5分钟 | - | - |
| 增量学习（3任务） | 30分钟 | - | - |

### 空间开销

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.pt` | ~500MB | 微调后的CLIP |
| `stats.pt` | ~1MB | 类别统计分布 |
| `covariance_history.pt` | ~500KB | 协方差历史 |
| `cache/id_features.pt` | ~50MB | ID特征缓存（1000样本×768维） |

---

## 10. 扩展指南

### 添加新的OOD检测器

1. 在`src/detectors/ood_detector.py`中实现类
2. 继承接口：`fit()`, `predict_score()`
3. 支持基于统计分布构建

### 添加新的超参数优化目标

1. 修改优化脚本中的评估函数
2. 添加新的metric选项
3. 更新可视化部分

---

## 11. 配置管理系统

### 11.1 设计原则

- **继承**: 基础配置 + 实验特定配置
- **验证**: 自动检查必填字段和数值范围
- **覆盖**: 命令行参数可覆盖配置
- **生成**: 自动生成实验目录结构

### 11.2 配置层次

```
base/default.yaml          # 最基础配置
experiments/my_exp.yaml    # 继承并覆盖
--override key=value      # 命令行覆盖
```

### 11.3 使用示例

```bash
# 运行预定义实验
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml

# 覆盖参数
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml \
    --override training.lr=5e-5 training.iterations=1000
```

### 11.4 实验目录自动生成

```
outputs/experiment_name_20240314_123456/
├── config.yaml              # 配置副本
├── checkpoints/             # 模型检查点
├── logs/                    # 日志
├── results/                 # 评估结果
├── cache/                   # 特征缓存
└── visualizations/          # 可视化
```

### 11.5 配置验证

自动检查：
- 必填字段（`experiment.name`, `data.id_datasets`）
- 数值范围（`lr`在(0,1)之间）
- 类型检查（`iterations`为正整数）

---

*本文档将持续更新，记录实现细节和最佳实践。*

*最后更新: 2026-03-14*
