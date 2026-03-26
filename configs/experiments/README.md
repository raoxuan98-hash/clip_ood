# 论文实验配置

本目录包含论文所需的所有实验配置文件。

## 目录结构

```
configs/experiments/
├── README.md                           # 本文件
├── zeroshot_5datasets.yaml            # Zero-shot CLIP基线
├── lora_baseline_5datasets.yaml       # LoRA基线
├── lora_nsp_5datasets.yaml            # LoRA-NSP主实验
├── incremental_5tasks_lora_nsp.yaml   # 增量学习
├── ablation_components.yaml           # 消融实验
└── ...
```

## 快速开始

### 运行单个实验

```bash
# LoRA-NSP在5个数据集上的联合微调
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_5datasets.yaml
```

### 运行所有主实验

```bash
python scripts/run_paper_experiments.py --phase main
```

### 运行消融实验

```bash
python scripts/run_paper_experiments.py --phase ablation
```

## 配置文件说明

### 主实验配置

| 配置文件 | 说明 | 预期运行时间 |
|---------|------|-------------|
| `zeroshot_5datasets.yaml` | Zero-shot CLIP基线 | 5分钟 |
| `lora_baseline_5datasets.yaml` | 标准LoRA微调 | 2-3小时 |
| `lora_nsp_5datasets.yaml` | LoRA-NSP方法 | 2-3小时 |

### 增量学习配置

| 配置文件 | 说明 | 预期运行时间 |
|---------|------|-------------|
| `incremental_5tasks_lora_nsp.yaml` | 5任务增量学习 | 10-12小时 |

### 消融实验配置

`ablation_components.yaml` 包含5个消融实验：
- `experiment_lora_only`: 纯LoRA
- `experiment_lora_nsp`: LoRA + NSP
- `experiment_lora_fd`: LoRA + 特征蒸馏
- `experiment_lora_cd`: LoRA + 跨模态蒸馏
- `experiment_lora_nsp_full`: 完整LoRA-NSP

## 超参数说明

所有实验使用以下优化后的超参数：

### OOD检测器 (LR-RGDA)
```yaml
rank: 32
reg_alpha1: 0.6   # 协方差正则化
reg_alpha2: 2.0   # 低秩分解正则化
reg_alpha3: 0.5   # 混合正则化
threshold: 0.9930
```

### 集成分类器
```yaml
alpha: 0.8        # LR-RGDA贡献系数
temperature: 1.0  # 温度参数
```

### LoRA-NSP训练
```yaml
iterations: 800
lr: 1e-4
lora_rank: 4
cov_momentum: 0.9
fd_weight: 1.0    # 特征蒸馏权重
cd_weight: 1.0    # 跨模态蒸馏权重
```

## 自定义实验

可以通过 `--override` 参数覆盖配置：

```bash
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_5datasets.yaml \
    --override training.iterations=1000 ensemble.alpha=0.9
```

## 结果输出

实验结果默认保存在 `outputs/` 目录下，可以通过配置覆盖：

```bash
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_5datasets.yaml \
    --override paths.output_dir=my_experiment_results
```

## 注意事项

1. 确保数据集已正确放置在 `paths.data_root` 指定的目录
2. 大规模实验建议分阶段运行，避免显存不足
3. 实验结果会自动保存，可随时中断和恢复
