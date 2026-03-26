# CLIP零样本分类持续学习

## 项目概述

本项目实现了CLIP模型的持续学习能力，通过训练端与推理端的协同优化，解决CLIP在下游任务适配过程中的灾难性遗忘问题，并构建自适应路由分类器以平衡ID和OOD性能。

**📚 文档导航**: 不确定从哪开始？查看 [DOC_INDEX.md](./DOC_INDEX.md) 快速找到需要的文档

主要创新点包括：

1. **训练端**：实现了LoRA-NSP微调方法，融合零空间参数化与低秩适应，结合跨模态蒸馏与模态内特征蒸馏，有效限制对预训练关键知识的干扰。
   - 详见：[TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) 第2章
   - 快速参考：[IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) 第2节

2. **推理端**：构建了集成分类器，融合零样本分类器与LR-RGDA分类器，并设计了基于LR-RGDA的OOD检测器和自适应路由分类器，根据OOD检测结果动态选择分类器。
   - 详见：[IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) 第3节

3. **增量学习**：支持两种增量学习模式：联合微调和多任务增量学习。
   - 使用指南：[scripts/workflows/README.md](./scripts/workflows/README.md)

## 目录结构

```
clip_ood/
├── src/                          # 源代码目录
│   ├── models/                   # 模型相关代码
│   ├── trainers/                 # 训练器实现
│   ├── classifiers/              # 分类器实现
│   ├── detectors/                # OOD检测器实现
│   ├── routing/                  # 自适应路由分类器实现
│   ├── utils/                    # 工具函数
│   └── main.py                   # 主程序
├── scripts/                      # 脚本目录
│   ├── core/                     # 核心脚本（单一职责）
│   ├── workflows/                # 工作流脚本
│   ├── optimization/             # 超参数优化脚本
│   └── ablations/                # 消融实验脚本
├── configs/                      # 配置文件目录
├── doc/                          # 文档目录（新增）
│   ├── DOC_INDEX.md              # 📍 文档导航索引
│   ├── PROJECT_DOCUMENTATION.md  # 项目架构和进度
│   ├── TECHNICAL_DOCUMENTATION.md# 技术细节（论文素材）
│   └── IMPLEMENTATION_NOTES.md   # 实现笔记（快速参考）
├── main.py                       # 原始主程序
├── demo_ood.ipynb                # 演示笔记本
└── README.md                     # 项目说明
```

### 快速导航

| 你想了解 | 查看文档 |
|----------|----------|
| 项目整体架构 | [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) |
| LoRA-NSP技术细节（论文） | [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) |
| 实现细节速查 | [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) |
| **超参数优化结果** | **[OPTIMIZATION_REPORT.md](./OPTIMIZATION_REPORT.md)** ⭐ |
| **配置管理系统** | **[configs/README.md](./configs/README.md)** ⭐ |
| 脚本使用方法 | [scripts/README.md](./scripts/README.md) |
| 增量学习指南 | [scripts/workflows/README.md](./scripts/workflows/README.md) |

### 快速开始（使用配置系统）

```bash
# 运行预定义实验
python scripts/run_from_config.py --config configs/experiments/lora_nsp_caltech.yaml

# 覆盖参数
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml \
    --override training.lr=5e-5 training.iterations=1000

# 增量学习
python scripts/run_from_config.py \
    --config configs/experiments/incremental_3tasks.yaml
```
| 快速找到文档 | [DOC_INDEX.md](./DOC_INDEX.md) |

## 安装说明

### 环境依赖

- Python 3.7+
- PyTorch 1.10+
- Transformers
- scikit-learn
- tqdm
- numpy

### 安装步骤

1. 克隆项目仓库：

```bash
git clone <repository-url>
cd clip_ood
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备数据集：

本项目使用X-TAIL数据集，需要将数据集放在指定目录：

```bash
# 默认数据集路径
/home/raoxuan/projects/data/X-TAIL/
```

## 使用指南

### 主程序运行

#### 联合微调模式

```bash
python src/main.py --id_datasets ["caltech101", "flowers", "oxford_pets", "stanford_cars", "food101"] --ood_datasets ["dtd", "eurosat", "mnist", "sun397"] --root /path/to/X-TAIL
```

#### 增量学习模式

```bash
python src/main.py --incremental_mode True --dataset_sequence [["caltech101"], ["flowers"], ["oxford_pets"], ["stanford_cars"], ["food101"]] --root /path/to/X-TAIL
```

### 超参数优化

```bash
python scripts/optimize_hyperparameters.py --id_datasets ["caltech101", "flowers", "oxford_pets", "stanford_cars", "food101"] --ood_datasets ["dtd", "eurosat", "mnist", "sun397"] --root /path/to/X-TAIL
```

## 核心模块

### 1. LoRANSPTrainer

实现了基于LoRA-NSP的微调方法，融合零空间参数化与低秩适应，结合跨模态蒸馏与模态内特征蒸馏。

### 2. LRRGDAClassifier

实现了低秩分解正则高斯判别分析（LR-RGDA）分类器，仅对ID样本输出正置信度，对OOD样本输出零置信度。

### 3. EnsembleClassifier

构建了融合零样本分类器与LR-RGDA分类器的集成分类框架，通过贡献系数α动态调节两类分类器的权重分配。

### 4. OOD检测器

实现了多种OOD检测方法，包括Mahalanobis距离、LDA和基于分类器的OOD检测器。

### 5. AdaptiveRouter

设计了基于OOD检测结果的自适应路由分类器，根据OOD检测结果动态选择分类器。

## 配置参数

### 数据集参数
- `--id_datasets`: ID数据集列表
- `--ood_datasets`: OOD数据集列表
- `--root`: 数据集根目录
- `--num_shots`: 少样本学习的shot数
- `--batch_size`: 批处理大小

### 训练参数
- `--seed`: 随机种子
- `--device`: 设备（cuda或cpu）
- `--iterations`: 训练迭代次数
- `--lr`: 学习率
- `--weight_decay`: 权重衰减

### LoRA参数
- `--lora_rank`: LoRA适应的秩
- `--lora_type`: LoRA适应类型（lora_sgp或lora_nsp）
- `--nsp_eps`: NSP的epsilon参数
- `--nsp_weight`: NSP的权重参数

### 分类器参数
- `--alpha`: 集成分类器中LR-RGDA的权重
- `--temperature`: 零样本分类器的温度参数

### OOD检测参数
- `--ood_detector_type`: OOD检测器类型（lda、lr_rgda或qda）
- `--ood_threshold`: OOD检测阈值

### 增量学习参数
- `--incremental_mode`: 是否使用增量学习模式
- `--dataset_sequence`: 增量学习的数据集序列

## 评估指标

- **ID准确率**：在ID数据集上的分类准确率
- **AUROC**：OOD检测的面积 Under ROC曲线
- **FPR@95TPR**：在95%真阳性率下的假阳性率
- **Detection Error**：检测错误率

## 示例

### 联合微调示例

```python
from src.main import parse_args, main

# 解析参数
args = parse_args()
args.id_datasets = ["caltech101", "flowers", "oxford_pets", "stanford_cars", "food101"]
args.ood_datasets = ["dtd", "eurosat", "mnist", "sun397"]
args.root = "/path/to/X-TAIL"
args.incremental_mode = False

# 运行主程序
main(args)
```

### 增量学习示例

```python
from src.main import parse_args, main

# 解析参数
args = parse_args()
args.dataset_sequence = [["caltech101"], ["flowers"], ["oxford_pets"], ["stanford_cars"], ["food101"]]
args.root = "/path/to/X-TAIL"
args.incremental_mode = True

# 运行主程序
main(args)
```

## 超参数优化示例

```python
from scripts.optimize_hyperparameters import parse_args
from src.utils.hyperparameter_optimizer import optimize_hyperparameters

# 解析参数
args = parse_args()
args.id_datasets = ["caltech101", "flowers", "oxford_pets", "stanford_cars", "food101"]
args.ood_datasets = ["dtd", "eurosat", "mnist", "sun397"]
args.root = "/path/to/X-TAIL"

# 执行超参数优化
best_params, best_results = optimize_hyperparameters(args)
print(f"Best parameters: {best_params}")
print(f"Best results: {best_results}")
```

## 注意事项

1. 本项目使用Flickr8K作为参考数据集进行蒸馏，需要确保该数据集可访问。

2. 增量学习模式下，每次学习一个新的数据集，并在所有数据集上进行评估。

3. 超参数优化可能需要较长时间，建议在具有足够计算资源的环境中运行。

## 许可证

本项目采用MIT许可证。

## 引用

如果您使用本项目的代码或方法，请引用相关论文。