# CLIP持续学习项目文档

> **版本**: 重构中  
> **最后更新**: 2026-03-14  
> **文档状态**: 草稿

**相关文档**:
- 📘 [技术文档 (TECHNICAL_DOCUMENTATION.md)](./TECHNICAL_DOCUMENTATION.md) - LoRA-NSP算法详解，供论文写作参考
- 📗 [脚本指南 (scripts/README.md)](./scripts/README.md) - 脚本使用说明
- 📙 [增量学习指南 (scripts/workflows/README.md)](./scripts/workflows/README.md) - 增量学习工作流说明

---

## 1. 项目概述

### 1.1 项目目标

本项目旨在解决CLIP模型在下游任务适配过程中的**灾难性遗忘问题**，通过训练端与推理端的协同优化，实现：
- **训练端**: 在保持CLIP通用能力的同时适配下游任务
- **推理端**: 平衡ID（分布内）性能与OOD（分布外）检测能力
- **持续学习**: 支持多任务增量学习，避免遗忘历史知识

### 1.2 论文内容与创新性

本研究围绕CLIP模型的持续学习展开，从训练端与推理端协同优化的角度，提出了一套系统性的解决方案。以下从四个方面阐述论文的核心内容与创新点。

#### 1.2.1 训练端：CLIP主干的抗遗忘微调

**问题背景**: CLIP模型在下游任务适配过程中面临严重的灾难性遗忘问题。标准微调会显著破坏预训练获得的通用知识，导致模型在OOD数据上的性能大幅下降。

**解决方案**: 本研究提出了**LoRA-NSP**（Low-Rank Adaptation with Null Space Parameterization）微调方法。

- **零空间参数化（NSP）**: 通过在参数更新过程中引入零空间约束，将梯度更新限制在预训练关键知识的正交方向上，有效限制对通用能力的干扰
- **低秩适应（LoRA）**: 采用低秩分解形式 $W = W_0 + B 	imes A$ 进行高效微调，仅训练少量参数即可适配下游任务
- **复合蒸馏损失**: 设计了跨模态蒸馏（Cross-modal Distillation, CD）与模态内特征蒸馏（Feature Distillation, FD）相结合的复合损失，进一步约束CLIP在参考数据集上的表征稳定性

**技术协同**: LoRA-NSP与知识蒸馏机制相互协同，共同实现CLIP模型对下游任务的适配与其通用能力的保持。LoRA-NSP从参数更新空间层面限制干扰，蒸馏从输出表征层面保持稳定性。

**实现位置**: `main.py`, `src/trainers/lora_nsp_trainer.py`

#### 1.2.2 推理端：集成分类器的构建

**问题背景**: 经LoRA-NSP微调后，CLIP的零样本分类器在下游任务上的性能获得显著提升。然而，零样本分类器仅依赖视觉-文本编码的相似度进行决策，未能充分利用下游监督数据集中蕴含的视觉细粒度信息。

**解决方案**: 本研究提出了融合**LR-RGDA**（Low-Rank Factorized Regularized Gaussian Discriminant Analysis）与零样本分类器的集成分类框架。
dansh
- **LR-RGDA分类器**: 基于下游监督数据构建的高斯判别分析分类器，充分利用训练数据的统计分布信息
- **零样本分类器**: 保持CLIP原有的视觉-文本对齐能力，维护OOD泛化性能
- **动态集成机制**: 通过贡献系数 $\alpha$ 动态调节两类分类器的权重分配
  $$P_{\text{ensemble}} = \alpha \cdot P_{\text{lr-rgda}} + (1-\alpha) \cdot P_{\text{zeroshot}}$$

**关键特性**: LR-RGDA分类器被设计为仅对分布内（ID）样本输出正置信度，而对分布外（OOD）样本输出零置信度。这一特性源于LR-RGDA作为监督分类器的本质——其决策边界完全由ID数据的统计分布决定，OOD数据落在所有类别的高斯分布之外，自然获得零后验概率。

**实现位置**: `demo_ood.ipynb`, `src/classifiers/lr_rgda_classifier.py`

#### 1.2.3 推理端：自适应路由分类器

**问题背景**: 尽管集成分类器显著增强了ID性能，但由于LR-RGDA对OOD样本输出零置信度，导致集成分类器在OOD数据上的性能相较于零样本分类器出现退化。这一性能损失与本文的研究目标相悖——期望CLIP经微调后在下游任务性能提升的同时，尽可能保持其OOD数据的零样本分类能力（此为CLIP区别于传统视觉分类器的核心优势）。

**解决方案**: 本研究提出了基于LR-RGDA的OOD检测器，并据此构建**自适应路由分类器**。

- **OOD检测**: 利用LR-RGDA分类器的特性，通过最大后验概率阈值判定样本的分布属性
  $$\text{OOD Score} = 1 - \max(P_{\text{posterior}})$$
- **自适应路由策略**:
  - **若判定为OOD样本** → 激活零样本分类器进行决策
  - **若判定为ID样本** → 激活集成分类器进行决策
- **计算效率**: 该路由机制的关键优势在于，当样本被判定为OOD时，仅需切换分类器分支而无需重新执行主干网络的前向传播，因而引入的计算开销极为有限

**实现位置**: `demo_ood.ipynb`, `src/routing/adaptive_router.py`

#### 1.2.4 创新总结与拓展性

综上所述，本研究从训练端与推理端协同切入，系统提升了CLIP持续学习的有效性：

| 创新维度 | 核心技术 | 作用机制 | 独立性 |
|:-------:|:--------|:--------|:------:|
| **训练端** | LoRA-NSP + 跨模态/模态内蒸馏 | 约束微调过程，缓解灾难性遗忘 | ✅ 可独立应用于任何ViT微调场景 |
| **推理端** | LR-RGDA集成分类器 | 融合监督学习与零样本能力 | ✅ 可独立应用于预训练CLIP模型 |
| **推理端** | 自适应路由分类器 | 动态决策，平衡ID/OOD性能 | ✅ 即使不微调CLIP也能带来增益 |

**正交性优势**:
- 训练端与推理端创新是正交的，具备独立应用于不同场景的灵活性
- **特别地**，实验表明自适应路由分类器即便在不微调CLIP的情况下，仍可为持续学习性能带来可观增益
- **广泛适用性**: LoRA-NSP方法在ViT类增量学习、大语言模型领域适应等任务中亦展现出广泛的适用潜力

**核心贡献总结**:
1. 提出了LoRA-NSP微调方法，通过零空间约束与知识蒸馏协同缓解灾难性遗忘
2. 设计了LR-RGDA集成分类器，充分利用监督数据的同时保持零样本能力
3. 构建了自适应路由机制，实现ID/OOD动态决策与性能平衡
4. 验证了各组件的正交性与独立应用价值，拓展了方法的适用范围

### 1.3 核心创新点（技术实现角度）

#### 创新点1: LoRA-NSP抗遗忘微调 (训练端)
- **方法**: 融合零空间参数化（NSP）与低秩适应（LoRA）
- **目标**: 限制参数更新对预训练关键知识的干扰
- **技术**: 跨模态蒸馏 + 模态内特征蒸馏的复合蒸馏损失
- **文件**: `src/trainers/lora_nsp_trainer.py`

#### 创新点2: LR-RGDA集成分类器 (推理端)
- **方法**: 低秩分解正则高斯判别分析 + 零样本分类器集成
- **特性**: LR-RGDA仅对ID样本输出正置信度，OOD样本输出零置信度
- **参数**: 通过贡献系数α动态调节两类分类器权重
- **文件**: `src/classifiers/lr_rgda_classifier.py`

#### 创新点3: 自适应路由分类器 (推理端)
- **方法**: 基于LR-RGDA的OOD检测器 + 动态分类器选择
- **路由策略**: 
  - OOD样本 → 零样本分类器
  - ID样本 → 集成分类器
- **优势**: 无需重新执行主干网络前向传播，计算开销极小
- **文件**: `src/routing/adaptive_router.py`

#### 创新点4: 正交性与拓展性
- 训练端与推理端创新相互独立，可灵活组合
- 自适应路由即使在不微调CLIP的情况下也能带来性能增益
- LoRA-NSP在ViT增量学习、LLM领域适应等任务中具有广泛适用性

---

## 2. 项目结构

```
clip_ood/
├── README.md                      # 项目说明（面向用户）
├── PROJECT_DOCUMENTATION.md       # 项目文档（本文档）- 整体架构和进度
├── TECHNICAL_DOCUMENTATION.md     # 技术文档 - LoRA-NSP算法详解（论文素材）
├── AGENTS.md                      # Agent配置（待完善）
├── main.py                        # 简化版入口（重构前）
├── demo_ood.ipynb                 # 完整Demo（最完整的实现参考）
├── utils_data.py                  # 数据加载工具
│
├── src/                           # 重构后的核心代码
│   ├── main.py                    # 完整版主入口
│   ├── trainers/                  # 训练器模块
│   │   └── lora_nsp_trainer.py    # LoRA-NSP训练器 ✅
│   ├── classifiers/               # 分类器模块
│   │   ├── lr_rgda_classifier.py  # LR-RGDA + 集成分类器 ✅
│   │   ├── da_classifier_builder.py
│   │   ├── gaussian_classifier.py
│   │   └── gaussian_statistics.py
│   ├── detectors/                 # OOD检测器模块
│   │   └── ood_detector.py        # 多种OOD检测器 ✅
│   ├── routing/                   # 路由模块
│   │   └── adaptive_router.py     # 自适应路由分类器 ✅
│   ├── models/                    # 模型定义
│   │   ├── clip.py                # CLIP模型封装
│   │   ├── lora_sgp.py            # LoRA-SGP实现
│   │   └── utils.py               # 模型工具函数
│   └── utils/                     # 通用工具
│       ├── feature_extractor.py   # 特征提取
│       ├── evaluation.py          # 评估指标
│       ├── reference_loader.py    # 参考数据集加载
│       └── hyperparameter_optimizer.py  # 超参数优化（待完善）
│
├── models/                        # 重构前的模型代码（待迁移/删除）
├── classifier/                    # 重构前的分类器代码（待迁移/删除）
├── scripts/                       # 脚本目录
│   └── optimize_hyperparameters.py # 超参数优化脚本（待完善）
└── configs/                       # 配置目录（待建立）
```

---

## 3. 功能模块详解

### 3.1 训练模块 (`src/trainers/`)

#### `LoRANSPTrainer` (已完成 ✅)
- **功能**: 实现LoRA-NSP微调，支持增量学习
- **核心方法**:
  - `train()`: 执行微调训练（自动应用零空间约束）
  - `extract_layer_covariances()`: 提取所有LoRA层的非中心化协方差
  - `update_covariance_history()`: 滑动平均更新协方差历史并更新投影矩阵
  - `zeroshot_classifier()`: 构建零样本分类器
- **协方差管理**:
  - 支持从checkpoint恢复 `covariance_history`
  - 滑动平均系数可配置 (`cov_momentum`)
  - 所有LoRA层的协方差独立维护
- **损失函数**:
  - 交叉熵损失 (监督学习)
  - 特征蒸馏损失 (保持预训练特征)
  - 跨模态蒸馏损失 (保持视觉-文本对齐)

**技术细节**: 详见 [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) - 包含数学公式、算法流程、实现细节和论文写作素材

**状态**: ✅ 完全实现，支持增量学习场景

### 设计原则: 统计分布优先

在重构后的代码中，**分类器和OOD检测器的构建完全基于类别统计分布**，而非原始数据或特征向量。这是为了：

1. **支持增量学习**: 在无法访问历史数据的情况下，只需保存每个类别的均值和协方差
2. **内存效率**: 避免缓存大量特征向量
3. **模块化**: 特征提取与分类器构建解耦

**核心数据结构**:
```python
# 类别统计分布字典
stats_dict = {
    class_id: GaussianStatistics(mean, cov)
    for class_id in all_classes
}
```

**使用流程**:
```python
# 1. 从数据提取统计分布（仅在训练时执行一次）
stats_dict = build_stats_dict_from_features(features, labels)

# 2. 基于统计分布构建分类器和检测器
classifier = LRRGDAClassifier(stats_dict, device='cuda')
detector = ClassifierBasedOODDetector(stats_dict, classifier_type='lr_rgda')

# 3. 增量学习时合并统计分布
merged_stats = {**old_stats_dict, **new_stats_dict}
new_classifier = LRRGDAClassifier(merged_stats)
```

---

### 3.2 分类器模块 (`src/classifiers/`)

#### `LRRGDAClassifier` (已完成 ✅)
- **功能**: 基于类别统计分布构建LR-RGDA分类器
- **设计原则**: **完全基于统计分布构建，无需原始数据或特征向量**
- **核心接口**:
  ```python
  # 从统计分布直接构建
  stats_dict = {0: GaussianStatistics(mean_0, cov_0), ...}
  classifier = LRRGDAClassifier(stats_dict, device='cuda')
  predictions = classifier.predict(features)
  ```
- **参数**:
  - `stats_dict`: 类别统计分布字典 {class_id: GaussianStatistics}
  - `rank`: 低秩分解的秩
  - `qda_reg_alpha1/2/3`: 协方差正则化权重
  - `temperature`: 温度参数

#### `EnsembleClassifier` (已完成 ✅)
- **功能**: 融合零样本分类器与LR-RGDA分类器
- **集成公式**: `P_ensemble = α * P_lr_rgda + (1-α) * P_zeroshot`
- **温度参数**: 支持对零样本分类器输出进行温度缩放

**状态**: 基础功能完成，需支持增量学习下的分类器拼接

### 3.3 OOD检测器模块 (`src/detectors/`)

#### `ClassifierBasedOODDetector` (已完成 ✅)
- **支持类型**: LDA / LR-RGDA / QDA
- **设计原则**: **完全基于统计分布构建，无需原始数据或特征向量**
- **OOD分数**: `score = 1 - max(Posterior Probability)`
- **核心接口**:
  ```python
  # 从统计分布直接构建
  detector = ClassifierBasedOODDetector(stats_dict, classifier_type='lr_rgda')
  ood_scores = detector.predict_score(features)
  ```

#### `MahalanobisOODDetector` (已完成 ✅)
- **方法**: 基于马氏距离的OOD检测
- **设计原则**: **完全基于类别均值和精度矩阵构建**
- **核心接口**:
  ```python
  # 从均值和精度矩阵构建
  detector = MahalanobisOODDetector(class_means, class_precisions)
  # 或从stats_dict构建
  detector = MahalanobisOODDetector.from_stats_dict(stats_dict, alpha=0.2)
  ```

#### 辅助函数
- `build_stats_dict_from_features()`: 从特征/标签构建统计分布字典
- `extract_stats_dict_from_model()`: 从模型和数据集提取统计分布

**状态**: 多种检测器已实现，需统一接口和评估流程

### 3.4 路由模块 (`src/routing/`)

#### `AdaptiveRouter` (已完成 ✅)
- **功能**: 根据OOD检测结果动态选择分类器
- **核心方法**:
  - `predict()`: 预测类别并返回OOD判定
  - `predict_proba()`: 预测概率分布
  - `set_threshold()`: 动态调整OOD阈值

**状态**: 基础路由逻辑完成，需优化阈值自适应策略

### 3.5 工具模块 (`src/utils/`)

| 模块 | 功能 | 状态 |
|------|------|------|
| `feature_extractor.py` | 特征提取 | ✅ 完成 |
| `evaluation.py` | 评估指标计算 | ✅ 基础完成 |
| `reference_loader.py` | 参考数据集加载 | ✅ 完成 |
| `hyperparameter_optimizer.py` | 超参数优化 | ⚠️ 待完善 |

---

## 4. 已完成的部分

### 4.1 核心算法实现
- [x] LoRA-NSP训练器 (`src/trainers/lora_nsp_trainer.py`)
  - ✅ 基础微调（LoRA + 知识蒸馏）
  - ✅ 协方差提取（所有LoRA层）
  - ✅ 协方差历史管理（滑动平均）
  - ✅ 零空间投影矩阵自动更新
  - ✅ 支持增量学习（从checkpoint恢复）
- [x] LR-RGDA分类器 (`src/classifiers/lr_rgda_classifier.py`)
  - ✅ 基于统计分布构建，无需原始数据
- [x] 集成分类器 (`EnsembleClassifier`)
- [x] 多种OOD检测器 (`src/detectors/ood_detector.py`)
  - ✅ 基于统计分布构建
  - ✅ Mahalanobis检测器支持从统计分布构建
  - ✅ 分类器基础检测器支持LDA/LR-RGDA/QDA
- [x] 自适应路由分类器 (`src/routing/adaptive_router.py`)

### 4.2 数据处理
- [x] X-TAIL数据集加载器 (`utils_data.py`)
- [x] 参考数据集(Flickr8K)支持
- [x] 数据变换管道

### 4.3 演示与验证
- [x] 完整Demo笔记本 (`demo_ood.ipynb`)
- [x] 基础主程序 (`main.py` 简化版)

---

## 5. 未完成/待完善的部分

### 5.1 Pipeline架构重构 (已完成 ✅)

#### 架构设计: 核心脚本 + 工作流脚本

**已完成**:
- [x] 设计核心脚本架构（train_clip / extract_stats / build_classifier / evaluate）
- [x] 分类器和OOD检测器基于统计分布构建
- [x] 增量学习工作流脚本 (`scripts/workflows/run_incremental.py`)
  - ✅ 连续微调（从上一个Task模型继承）
  - ✅ 协方差历史维护（滑动平均）
  - ✅ 零空间约束自动应用
  - ✅ 跨任务评估（ID+OOD）

**两种学习模式的统一**:
```bash
# 模式1 (联合微调): 单任务，多数据集
python scripts/core/train_clip.py --datasets a,b,c

# 模式2 (增量学习): 多任务，连续微调+零空间约束
python scripts/workflows/run_incremental.py \
  --task_sequence caltech101,flowers,oxford_pets \
  --all_datasets caltech101,flowers,oxford_pets,cars,food \
  --output_dir experiments/incremental_1
```

### 5.2 配置管理系统 (已完成 ✅)

#### 功能特性
- [x] **YAML配置支持** (`src/utils/config_manager.py`)
  - ✅ 配置继承（base + experiment）
  - ✅ 配置验证（必填字段、数值范围、类型检查）
  - ✅ 命令行参数覆盖（`--override key=value`）
  - ✅ 嵌套配置访问（`config.training.lr`）
  - ✅ 自动生成实验目录结构

- [x] **配置文件示例** (`configs/`)
  - ✅ 基础配置 (`base/default.yaml`)
  - ✅ 实验配置 (`experiments/*.yaml`)
  - ✅ 支持三种模式：joint/incremental/optimization

- [x] **基于配置的运行脚本** (`scripts/run_from_config.py`)
  - ✅ 一键运行完整实验流程
  - ✅ 自动创建标准目录结构
  - ✅ 支持参数覆盖

**使用示例**:
```bash
# 运行实验
python scripts/run_from_config.py --config configs/experiments/lora_nsp_caltech.yaml

# 覆盖参数
python scripts/run_from_config.py \
  --config configs/experiments/lora_nsp_caltech.yaml \
  --override training.lr=5e-5 training.iterations=1000

# 增量学习
python scripts/run_from_config.py \
  --config configs/experiments/incremental_3tasks.yaml
```

### 5.3 超参数优化模块 (已完成 ✅)

#### 已完成功能

**1. OOD检测器超参数优化** (`scripts/optimization/optimize_ood_detector.py`)
- ✅ Mahalanobis检测器 (alpha参数)
- ✅ ClassifierBased检测器 (LDA/LR-RGDA/QDA)
- ✅ 基于TPR@95%的阈值自动确定
- ✅ FPR@95TPR优化指标
- ✅ 结果可视化

**2. 集成分类器超参数优化** (`scripts/optimization/optimize_ensemble_classifier.py`)
- ✅ 两阶段优化：先temperature，后alpha
- ✅ Temperature优化：匹配零样本分类器输出分布
  - 支持max_prob_mse、kl_divergence、entropy_match
- ✅ Alpha优化：在[0,1]范围内搜索
  - 支持id_accuracy、ood_auroc、balanced指标
- ✅ 结果可视化

**使用示例**:
```bash
# 优化OOD检测器
python scripts/optimization/optimize_ood_detector.py \
  --stats stats/pretrained.pt \
  --id_datasets caltech101,flowers \
  --ood_datasets dtd,eurosat \
  --target_tpr 0.95 \
  --output_dir opt/ood

# 优化集成分类器
python scripts/optimization/optimize_ensemble_classifier.py \
  --stats stats/pretrained.pt \
  --id_datasets caltech101,flowers \
  --match_metric max_prob_mse \
  --output_dir opt/ensemble
```

**输出**: 
- 最优超参数配置
- 推荐阈值（实现95% OOD检测率）
- 可视化图表
- 可直接用于build_classifier.py的配置文件

### 5.4 消融实验框架 (中优先级)

#### 需支持的对比实验
| 实验 | 描述 | 优先级 |
|------|------|--------|
| OOD检测器对比 | Mahalanobis vs LDA vs LR-RGDA vs QDA | 高 |
| 分类器对比 | 纯零样本 vs 集成分类器 vs 自适应路由 | 高 |
| α敏感性分析 | ID/OOD准确率随α变化的曲线 | 中 |
| 微调vs不微调 | 自适应路由在不微调CLIP时的效果 | 中 |

**建议实现方式**:
- 创建 `scripts/ablation_studies/` 目录
- 每个实验一个独立脚本
- 统一的结果收集与可视化模块

### 5.5 代码清理与迁移 (低优先级)
- [ ] 清理重构前的旧代码 (`models/`, `classifier/`)
- [ ] 统一导入路径（解决`classifier` vs `classifiers`命名不一致）
- [ ] 完善类型注解和文档字符串
- [ ] 单元测试覆盖

---

## 6. 重构规划与讨论

### 6.1 重构目标

1. **链路清晰化**: 明确区分两种学习模式，统一数据处理流程
2. **模块化**: 各组件（训练器/分类器/检测器/路由器）职责清晰，接口统一
3. **可配置化**: 支持YAML/JSON配置，便于实验管理
4. **可扩展性**: 便于添加新的OOD检测器、分类器、训练策略

### 6.2 关键设计决策

#### Q1: 增量学习下的历史知识管理（已解决 ✅）
**问题**: 模式2中不能获取过去任务的数据，如何构建全局LR-RGDA分类器？

**解决方案**: **基于统计分布的合并**（已在代码中实现）

```python
# 每个任务保存: stats_dict = {class_id: GaussianStatistics(mean, cov)}

# 增量学习流程:
# 1. 学习新任务，提取新任务的统计分布
new_stats = extract_stats_dict_from_model(model, new_dataset, args)

# 2. 合并历史与新任务的统计分布（类别互斥，直接拼接）
merged_stats = {**historical_stats_dict, **new_stats}

# 3. 基于合并后的统计分布重建分类器和检测器
global_classifier = LRRGDAClassifier(merged_stats)
global_detector = ClassifierBasedOODDetector(merged_stats, classifier_type='lr_rgda')

# 4. 同时拼接零样本分类器权重
zeroshot_weights = torch.cat([historical_zeroshot, new_zeroshot], dim=1)

# 5. 重建集成分类器
ensemble = EnsembleClassifier(zeroshot_weights, global_classifier, alpha=0.5)
```

**优势**:
- 无需保存原始数据，符合增量学习约束
- 统计分布（均值+协方差）占用内存极小
- 重建分类器计算开销低

#### Q2: OOD检测器的阈值如何确定？
**方案选项**:
- A. 在验证集上搜索最优阈值（最大化ID准确率+OOD检测AUROC）
- B. 使用统计方法（如FPR@95%TPR对应的阈值）
- C. 固定经验值（如0.5）

**建议**: 先实现B方案作为默认，A方案在超参数优化脚本中实现

#### Q3: 超参数优化与消融实验的代码组织？
**建议结构**:
```
scripts/
├── optimize_hyperparameters.py    # 主超参数优化
├── ablation_studies/
│   ├── compare_ood_detectors.py   # OOD检测器对比
│   ├── compare_classifiers.py     # 分类器对比
│   ├── alpha_sensitivity.py       # α敏感性分析
│   └── visualize_results.py       # 结果可视化
└── run_full_pipeline.py           # 一键运行完整流程
```

### 6.3 架构决策: 核心脚本 vs 工作流脚本

#### 问题分析
当前`main.py`将所有功能耦合在一起，导致：
- 无法单独测试某个模块（如仅测试分类器而不微调）
- 参数过多且冲突（训练参数 vs 分类器参数 vs 评估参数）
- 不利于并行开发和超参数优化

#### 解决方案: 分层架构

```
scripts/
├── core/              # 核心脚本（职责单一，可独立运行）
│   ├── train_clip.py         # 仅微调CLIP
│   ├── extract_stats.py      # 仅提取统计分布
│   ├── build_classifier.py   # 仅从统计分布构建分类器
│   └── evaluate.py           # 仅评估（支持多种模式）
│
├── workflows/         # 工作流脚本（串联核心脚本）
│   ├── run_full_pipeline.py      # 完整流程
│   ├── run_zeroshot_only.py      # 仅零样本评估
│   └── run_classifier_only.py    # 不微调，仅测试分类器
│
├── ablations/         # 消融实验
│   ├── compare_classifiers.py
│   ├── compare_ood_detectors.py
│   └── alpha_sensitivity.py
│
└── optimization/      # 超参数优化
    └── optimize_hyperparameters.py
```

#### 设计优势

1. **功能独立性**
   - 每个核心脚本只做一件事，便于单独测试
   - 脚本间通过文件（checkpoint/stats/classifier）传递数据

2. **组合灵活性**
   - 可以只评估预训练CLIP（不微调）
   - 可以只微调CLIP（用零样本评估）
   - 可以只测试分类器（使用预计算的统计分布）

3. **并行优化**
   - 超参数优化时可以独立优化分类器参数（无需重新训练CLIP）

#### 核心脚本说明

| 脚本 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `train_clip.py` | 数据集 | `model.pt` | 微调CLIP |
| `extract_stats.py` | 模型 + 数据集 | `stats.pt` | 提取统计分布 |
| `build_classifier.py` | `stats.pt` | `classifier.pt` | 构建分类器/检测器 |
| `evaluate.py` | 模型 + 分类器 | `results.json` | 评估性能 |

#### 使用示例

```bash
# 场景1: 仅评估预训练CLIP（不微调）
python scripts/core/extract_stats.py \
    --datasets caltech101 \
    --output stats/pretrained.pt

python scripts/core/build_classifier.py \
    --stats stats/pretrained.pt \
    --output classifiers/pretrained.pt

python scripts/core/evaluate.py \
    --classifier classifiers/pretrained.pt \
    --id_datasets caltech101 \
    --mode ensemble

# 场景2: 仅验证LoRA-NSP效果（用零样本分类器）
python scripts/core/train_clip.py \
    --datasets caltech101 \
    --output checkpoints/lora_nsp.pt

python scripts/core/evaluate.py \
    --model checkpoints/lora_nsp.pt \
    --id_datasets caltech101 \
    --mode zeroshot

# 场景3: 完整流程（使用工作流脚本）
python scripts/workflows/run_full_pipeline.py \
    --id_datasets caltech101,flowers \
    --ood_datasets dtd,eurosat \
    --output_dir experiments/exp1
```

### 6.4 项目完成状态（更新）

| 优先级 | 任务 | 状态 | 完成度 |
|--------|------|------|--------|
| **P0 - 核心架构** |
| | 核心脚本架构设计 | ✅ | 100% |
| | `scripts/core/`下的4个核心脚本 | ✅ | 100% |
| | `scripts/workflows/`工作流脚本 | ✅ | 100% |
| | LoRA-NSP协方差历史管理 | ✅ | 100% |
| | 增量学习工作流 | ✅ | 100% |
| | 基于统计分布的分类器/检测器 | ✅ | 100% |
| **P1 - 优化与文档** |
| | OOD检测器超参数优化 | ✅ | 100% |
| | 集成分类器超参数优化 | ✅ | 100% |
| | 特征缓存（60x-144x加速） | ✅ | 100% |
| | 超参数优化文档 | ✅ | 100% |
| | 消融实验框架 | ⚠️ | 50% |
| **P2 - 配置与可视化** |
| | 配置管理系统（YAML+继承） | ✅ | 100% |
| | 可视化模块 | ⚠️ | 70% |
| | 主程序更新 | ⚠️ | 80% |
| **P3 - 代码质量** |
| | 代码清理 | ⚠️ | 60% |
| | 类型注解 | ⚠️ | 50% |
| | 单元测试 | ⚠️ | 20% |

**总体完成度**: ~85%

**核心功能**: ✅ 全部完成（可运行完整实验）
**优化功能**: ✅ 全部完成（超参数优化+特征缓存）
**配置系统**: ✅ 已完成（YAML+继承+验证）
**文档**: ✅ 完整（技术文档+实现笔记+使用指南）

---

## 7. 快速开始

### 7.1 环境准备
```bash
pip install -r requirements.txt
```

### 7.2 运行完整Demo
参考 `demo_ood.ipynb` 笔记本，涵盖：
- 联合微调
- OOD检测器构建
- 自适应路由评估
- 超参数调优
- 可视化

### 7.3 使用核心脚本（推荐）

#### 场景A: 仅评估预训练CLIP（不微调）
```bash
# 1. 提取统计分布
python scripts/core/extract_stats.py \
  --datasets caltech101,flowers \
  --output stats/pretrained.pt

# 2. 构建分类器
python scripts/core/build_classifier.py \
  --stats stats/pretrained.pt \
  --output classifiers/pretrained.pt

# 3. 评估
python scripts/core/evaluate.py \
  --classifier classifiers/pretrained.pt \
  --id_datasets caltech101,flowers \
  --mode ensemble
```

#### 场景B: 验证LoRA-NSP效果（零样本评估）
```bash
# 1. 微调CLIP
python scripts/core/train_clip.py \
  --datasets caltech101 \
  --output checkpoints/lora_nsp.pt

# 2. 用零样本分类器评估（验证LoRA-NSP）
python scripts/core/evaluate.py \
  --model checkpoints/lora_nsp.pt \
  --id_datasets caltech101 \
  --mode zeroshot
```

#### 场景C: 完整流程（使用工作流脚本）
```bash
python scripts/workflows/run_full_pipeline.py \
  --id_datasets caltech101,flowers,oxford_pets \
  --ood_datasets dtd,eurosat \
  --output_dir experiments/exp1
```

#### 场景D: 消融实验（分类器对比）
```bash
python scripts/ablations/compare_classifiers.py \
  --stats stats/finetuned.pt \
  --id_datasets caltech101,flowers \
  --ood_datasets dtd,eurosat \
  --alphas 0.0,0.3,0.5,0.7,1.0 \
  --output_dir ablations/classifiers
```

---

## 8. 附录

### 8.1 关键术语

| 术语 | 解释 |
|------|------|
| CLIP | Contrastive Language-Image Pre-training |
| LoRA | Low-Rank Adaptation |
| NSP | Null Space Parameterization |
| LR-RGDA | Low-Rank Factorized Regularized Gaussian Discriminant Analysis |
| ID | In-Distribution (分布内) |
| OOD | Out-of-Distribution (分布外) |
| AUROC | Area Under ROC Curve |
| `stats_dict` | 类别统计分布字典 {class_id: GaussianStatistics} |
| `GaussianStatistics` | 高斯统计容器 (mean, cov) |

### 8.2 相关文件对照

| 论文模块 | 实现文件 | 状态 |
|----------|----------|------|
| LoRA-NSP训练 | `src/trainers/lora_nsp_trainer.py` | ✅ |
| LR-RGDA分类器 | `src/classifiers/lr_rgda_classifier.py` | ✅ 已重构为统计分布接口 |
| OOD检测器 | `src/detectors/ood_detector.py` | ✅ 已重构为统计分布接口 |
| 自适应路由 | `src/routing/adaptive_router.py` | ✅ |
| 完整Demo | `demo_ood.ipynb` | ✅ |
| 主程序 | `src/main.py` | ⚠️ 需完善 |
| 超参数优化 | `scripts/optimize_hyperparameters.py` | ⚠️ 待完善 |

---

*本文档将持续更新，记录项目进展和架构决策。*


---

## 9. API参考

### 9.1 基于统计分布构建分类器和检测器

```python
from classifiers.gaussian_statistics import GaussianStatistics
from classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from detectors.ood_detector import ClassifierBasedOODDetector, MahalanobisOODDetector

# ========== 1. 准备统计分布 ==========
# 方式A: 手动构建
stats_dict = {
    0: GaussianStatistics(mean=torch.randn(768), cov=torch.eye(768)),
    1: GaussianStatistics(mean=torch.randn(768), cov=torch.eye(768)),
    # ...
}

# 方式B: 从特征提取
from detectors.ood_detector import build_stats_dict_from_features
stats_dict = build_stats_dict_from_features(features, labels)

# ========== 2. 构建LR-RGDA分类器 ==========
classifier = LRRGDAClassifier(
    stats_dict=stats_dict,
    device='cuda',
    rank=64,
    qda_reg_alpha1=0.2,
    qda_reg_alpha2=0.2,
    qda_reg_alpha3=0.2,
    temperature=1.0
)

# 预测
predictions = classifier.predict(features)
probs = classifier.predict_proba(features)

# ========== 3. 构建OOD检测器 ==========
# 方式A: 基于分类器
detector = ClassifierBasedOODDetector(
    stats_dict=stats_dict,
    classifier_type='lr_rgda',  # 或 'lda', 'qda'
    device='cuda'
)

# 方式B: 基于Mahalanobis距离
detector = MahalanobisOODDetector.from_stats_dict(
    stats_dict=stats_dict,
    alpha=0.2,  # 类特定协方差权重
    device='cuda'
)

# 计算OOD分数
ood_scores = detector.predict_score(features)  # 越高越可能是OOD

# ========== 4. 增量学习下的合并 ==========
# 假设已有历史统计分布
historical_stats = {...}  # 历史任务的统计分布
new_stats = {...}         # 新任务的统计分布

# 直接合并（类别互斥）
merged_stats = {**historical_stats, **new_stats}

# 重建全局分类器和检测器
global_classifier = LRRGDAClassifier(merged_stats)
global_detector = ClassifierBasedOODDetector(merged_stats)

# ========== 5. 构建集成分类器 ==========
# 零样本分类器权重 [D, C]
zeroshot_weights = trainer.zeroshot_classifier(class_names, templates)

ensemble = EnsembleClassifier(
    zeroshot_classifier=zeroshot_weights,
    lr_rgda_classifier=classifier,
    alpha=0.5,        # LR-RGDA权重
    temperature=1.0   # 零样本温度
)

# 预测（需要logit_scale）
probs = ensemble.predict_proba(features, logit_scale=model.logit_scale)
```

### 9.2 完整训练-评估流程示例

```python
from trainers.lora_nsp_trainer import LoRANSPTrainer
from detectors.ood_detector import extract_stats_dict_from_model

# 1. 初始化训练器
trainer = LoRANSPTrainer(args)

# 2. 微调CLIP
model = trainer.train(train_loader, class_names, reference_loader)

# 3. 提取统计分布（而非缓存特征）
stats_dict = extract_stats_dict_from_model(
    model, 
    dataset_names=['caltech101', 'flowers'],
    args=args
)

# 4. 构建分类器和检测器
classifier = LRRGDAClassifier(stats_dict, device=args.device)
detector = ClassifierBasedOODDetector(stats_dict, classifier_type='lr_rgda')

# 5. 构建自适应路由
from routing.adaptive_router import AdaptiveRouter
zeroshot_weights = trainer.zeroshot_classifier(all_class_names, templates)
ensemble = EnsembleClassifier(zeroshot_weights, classifier, alpha=0.5)

router = AdaptiveRouter(
    zeroshot_classifier=zeroshot_weights,
    ensemble_classifier=ensemble,
    ood_detector=detector,
    threshold=0.5
)

# 6. 评估
predictions, is_ood = router.predict(test_features, model.logit_scale)
```

---

*本文档将持续更新，记录项目进展和架构决策。*


---

## 10. 架构示意图

### 10.1 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                      核心模块层 (src/)                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   trainers   │  │  classifiers │  │   detectors  │          │
│  │  LoRANSP     │  │  LRRGDA      │  │  Mahalanobis │          │
│  │  Trainer     │  │  Ensemble    │  │  Classifier  │          │
│  │              │  │  Adaptive    │  │  Based       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           │                                      │
│                    ┌──────────────┐                             │
│                    │   routing    │                             │
│                    │  Adaptive    │                             │
│                    │  Router      │                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      核心脚本层 (scripts/core/)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │train_clip.py│───▶│extract_stats│───▶│build_classi-│         │
│  │             │    │    .py      │    │   fier.py   │         │
│  │  输入:数据  │    │  输入:模型  │    │  输入:stats │         │
│  │  输出:model │    │  输出:stats │    │  输出:class │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                │
│                                                ▼                │
│                                         ┌─────────────┐        │
│                                         │ evaluate.py │        │
│                                         │             │        │
│                                         │  输入:model │        │
│                                         │       +class│        │
│                                         │  输出:results│       │
│                                         └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      工作流/实验层 (scripts/)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  workflows/      │  │  ablations/      │                    │
│  │  run_full_pipe-  │  │  compare_classi- │                    │
│  │    line.py       │  │    fiers.py      │                    │
│  │                  │  │                  │                    │
│  │  串联核心脚本    │  │  消融实验        │                    │
│  │  执行完整流程    │  │  超参数优化      │                    │
│  └──────────────────┘  └──────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 数据流

```
Data ──▶ train_clip.py ──▶ model.pt ──┬──▶ extract_stats.py ──▶ stats.pt
                                      │                           │
                                      │                           ▼
                                      │                    build_classifier.py
                                      │                           │
                                      │                           ▼
                                      │                    classifier.pt
                                      │                           │
                                      └───────────────────────────┘
                                                                  │
                                                                  ▼
                                                           evaluate.py
                                                                  │
                                                                  ▼
                                                           results.json
```

### 10.3 使用场景映射

| 使用场景 | 需要的脚本 | 命令 |
|----------|-----------|------|
| 验证LoRA-NSP效果 | train_clip + evaluate | `train_clip.py` → `evaluate.py --mode zeroshot` |
| 测试分类器设计 | extract_stats + build_classifier + evaluate | `extract_stats.py` → `build_classifier.py` → `evaluate.py` |
| 完整实验 | 工作流脚本 | `run_full_pipeline.py` |
| 超参数优化 | build_classifier + evaluate | 循环调用，改变alpha/temperature |
| 消融实验 | 对比脚本 | `compare_classifiers.py` |

---

*本文档将持续更新，记录项目进展和架构决策。*
