# 实验结果元数据格式规范

**版本**: 1.0  
**目的**: 统一所有实验结果的存储格式，便于后期整理和分析

---

## 通用字段

每个实验结果文件必须包含以下字段：

```yaml
experiment:
  table: "Table X"           # 所属表格
  name: "实验名称"           # 实验具体名称
  method: "方法名"           # 使用的方法
  timestamp: "2026-03-18T10:00:00"  # 执行时间
  
configuration:
  task_sequence: [...]       # 任务序列
  datasets:
    id: [...]               # ID数据集列表
    ood: [...]              # OOD数据集列表
  hyperparameters: {...}    # 超参数
  
results:
  # 具体结果（各表格不同）
  
metrics:
  # 评估指标（各表格不同）
  
paths:
  raw_data: "experiments/.../raw_data.json"      # 原始数据文件
  checkpoint: "experiments/.../model.pt"         # 模型检查点（如有）
  log: "logs/.../experiment.log"                 # 日志文件
  visualization: "figures/.../plot.png"          # 可视化图表（如有）
```

---

## Table 1: 主实验结果格式

### 文件路径
`results/table1/{method}_results.md`

### 格式示例

```markdown
# Table 1: 主实验结果 - 方法D (LoRA-NSP)

## 实验信息
- **方法**: LoRA-NSP
- **执行时间**: 2026-03-18 10:00:00
- **随机种子**: 42
- **GPU**: CUDA:0

## 配置
### 数据集
- **任务序列**: aircraft → caltech101 → dtd → eurosat → flowers → food101 → mnist → oxford_pets → stanford_cars → sun397

### 超参数
- **LoRA Rank**: 4
- **Num Shots**: 16
- **Iterations**: 800
- **Learning Rate**: 1e-4
- **FD Weight**: 1.0
- **CD Weight**: 1.0

## 完整准确率矩阵

| Step | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | Average |
|------|----------|------------|-----|---------|---------|---------|-------|-------------|---------------|--------|---------|
| 0 | 23.8 | 74.3 | 36.4 | 46.5 | 67.2 | 65.1 | 47.3 | 85.2 | 18.9 | 38.5 | 50.3 |
| 1 | 65.2 | 72.1 | 35.8 | 45.2 | 66.5 | 64.8 | 46.9 | 84.1 | 18.2 | 37.8 | 53.7 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 10 | 73.5 | 82.5 | 51.5 | 93.2 | 88.9 | 85.8 | 96.8 | 90.5 | 43.8 | 73.5 | 78.0 |

## LADA指标汇总

| 指标 | 值 | 说明 |
|------|-----|------|
| **Transfer** | 60.2% | 前向迁移能力 |
| **Average** | 72.1% | 综合性能 |
| **Last** | 78.0% | 最终性能 |
| **Forgetting** | 8.5% | 遗忘程度 |

## 逐数据集最终性能

| 数据集 | Step 0 (Zero-shot) | Step 10 (Last) | Forgetting |
|--------|-------------------|----------------|------------|
| aircraft | 23.8% | 73.5% | -0.3% |
| caltech101 | 74.3% | 82.5% | 0.0% |
| ... | ... | ... | ... |

## 相关文件
- **原始数据**: `experiments/main/D_lora_nsp_only/lora_nsp_results.json`
- **检查点**: `experiments/main/D_lora_nsp_only/final_model.pt`
- **日志**: `logs/main_D.log`
```

---

## Table 2: 训练端消融结果格式

### 文件路径
`results/table2/{config}_results.md`

### 格式示例

```markdown
# Table 2: 训练端消融 - 配置: NSP+FD+CD (Full)

## 实验信息
- **配置名称**: Full (NSP+FD+CD)
- **NSP**: ✓
- **FD**: ✓
- **CD**: ✓

## 配置
- **任务序列**: aircraft → caltech101 → dtd → eurosat → flowers
- **训练轮次**: 800

## LADA指标

| 指标 | 值 |
|------|-----|
| **Transfer** | 60.2% |
| **Average** | 71.8% |
| **Last** | 79.5% |
| **Forgetting** | 7.2% |

## 与其他配置对比

| 配置 | NSP | FD | CD | Transfer | Average | Last | Forgetting |
|------|-----|----|----|----------|---------|------|------------|
| Baseline | ✗ | ✗ | ✗ | 55.1% | 65.2% | 70.3% | 15.2% |
| +NSP only | ✓ | ✗ | ✗ | 56.8% | 69.1% | 75.8% | 10.5% |
| **Full** | **✓** | **✓** | **✓** | **60.2%** | **71.8%** | **79.5%** | **7.2%** |

## 相关文件
- **原始数据**: `experiments/ablation_train/full/lora_nsp_results.json`
```

---

## Table 3: OOD检测器对比结果格式

### 文件路径
`results/table3/{combo}_{detector}_results.md`

### 格式示例

```markdown
# Table 3: OOD检测器对比 - Combo 1, LR-RGDA

## 实验信息
- **组合**: Combo 1
- **检测器**: LR-RGDA

## 数据集划分
- **ID**: aircraft, caltech101, dtd, eurosat, flowers (5个)
- **OOD**: food101, mnist, oxford_pets, stanford_cars, sun397 (5个)

## OOD检测性能

| 指标 | 值 |
|------|-----|
| **AUROC** | 99.1% |
| **FPR@95TPR** | 3.5% |
| **AUPR** | 98.6% |
| **Detection Error** | 3.3% |
| **最优阈值** | 0.994 |

## 各组合汇总（用于计算均值和标准差）

| 组合 | AUROC | FPR@95TPR | AUPR | Detection Error |
|------|-------|-----------|------|-----------------|
| Combo 1 | 99.1% | 3.5% | 98.6% | 3.3% |
| Combo 2 | 99.0% | 3.6% | 98.5% | 3.4% |
| ... | ... | ... | ... | ... |
| **均值±标准差** | **99.1±0.3%** | **3.5±0.5%** | **98.6±0.4%** | **3.3±0.4%** |

## 检测器对比（5组平均）

| 检测器 | AUROC (↑) | FPR@95TPR (↓) | AUPR (↑) | Detection Error (↓) |
|--------|-----------|---------------|----------|---------------------|
| Mahalanobis | 87.3±2.1% | 30.9±3.2% | 85.1±2.5% | 17.5±1.8% |
| LDA | 98.7±0.5% | 5.4±0.8% | 97.8±0.6% | 4.9±0.5% |
| QDA | 99.1±0.3% | 3.5±0.5% | 98.5±0.4% | 3.3±0.4% |
| **LR-RGDA** | **99.1±0.3%** | **3.5±0.5%** | **98.6±0.4%** | **3.3±0.4%** |

## 相关文件
- **原始数据**: `experiments/table3/combo1_lr_rgda/results.json`
```

---

## Table 4: 推理端消融结果格式

### 文件路径
`results/table4/{combo}_{strategy}_results.md`

### 格式示例

```markdown
# Table 4: 推理端消融 - Combo 1, 自适应路由

## 实验信息
- **组合**: Combo 1
- **策略**: 自适应路由
- **OOD检测器**: LR-RGDA
- **阈值**: 0.85 (来自Table 3)

## 数据集划分
- **ID**: aircraft, caltech101, dtd, eurosat, flowers (5个)
- **OOD**: food101, mnist, oxford_pets, stanford_cars, sun397 (5个)

## 分类性能

| 指标 | 值 |
|------|-----|
| **ID Avg Acc** | 76.2% |
| **OOD Avg Acc** | 80.5% |
| **Overall Acc** | 78.4% |

## 逐数据集详细结果

| 数据集 | 类型 | 准确率 |
|--------|------|--------|
| aircraft | ID | 75.8% |
| caltech101 | ID | 82.1% |
| ... | ... | ... |
| food101 | OOD | 81.2% |
| mnist | OOD | 79.5% |
| ... | ... | ... |

## 各组合汇总

| 组合 | 策略 | ID Avg Acc | OOD Avg Acc | Overall Acc |
|------|------|------------|-------------|-------------|
| Combo 1 | 零样本 | 58.2% | 83.1% | 70.7% |
| Combo 1 | LR-RGDA | 78.5% | 32.4% | 55.5% |
| Combo 1 | 固定集成 | 73.8% | 68.2% | 71.0% |
| Combo 1 | **自适应路由** | **76.2%** | **80.5%** | **78.4%** |
| ... | ... | ... | ... | ... |
| **5组平均** | **零样本** | **58.1±2.3%** | **82.8±3.1%** | **70.5±2.7%** |
| **5组平均** | **自适应路由** | **76.0±2.1%** | **80.2±2.8%** | **78.1±2.4%** |

## 方法对比（5组平均）

| 方法 | ID Avg Acc | OOD Avg Acc | Overall Acc |
|------|------------|-------------|-------------|
| 纯零样本 | 58.1±2.3% | 82.8±3.1% | 70.5±2.7% |
| 纯LR-RGDA | 78.3±2.5% | 31.5±4.2% | 54.9±3.3% |
| 固定集成 | 73.5±2.2% | 67.8±3.5% | 70.7±2.9% |
| **自适应路由** | **76.0±2.1%** | **80.2±2.8%** | **78.1±2.4%** |

## 相关文件
- **原始数据**: `experiments/table4/combo1_routing/results.json`
```

---

## Table 5: Alpha敏感性结果格式

### 文件路径
`results/table5/{combo}_{strategy}_alpha{alpha}_results.md`

### 格式示例

```markdown
# Table 5: Alpha敏感性 - Combo 1, 固定集成, α=0.80

## 实验信息
- **组合**: Combo 1
- **策略**: 固定集成
- **Alpha**: 0.80

## 分类性能

| 指标 | 值 |
|------|-----|
| **ID Avg Acc** | 74.2% |
| **OOD Avg Acc** | 68.3% |
| **Overall Acc** | 71.3% |

## 不同Alpha对比（Combo 1）

| Alpha | ID Avg Acc | OOD Avg Acc | Overall Acc |
|-------|------------|-------------|-------------|
| 0.50 | 60.2% | 84.8% | 72.5% |
| 0.60 | 65.5% | 78.2% | 71.9% |
| 0.70 | 70.8% | 72.5% | 71.7% |
| **0.80** | **74.2%** | **68.3%** | **71.3%** |
| 0.90 | 77.5% | 50.2% | 63.9% |
| 1.00 | 79.8% | 32.5% | 56.2% |

## 策略对比（α=0.80, 5组平均）

| 策略 | ID Avg Acc | OOD Avg Acc | Overall Acc |
|------|------------|-------------|-------------|
| 固定集成 | 74.2±1.0% | 68.3±1.6% | 71.3±1.3% |
| 自适应路由 | 76.0±0.9% | 80.2±1.2% | 78.1±1.0% |

**观察**: 自适应路由对alpha不敏感，性能稳定

## 可视化
- **性能曲线**: `figures/table5/combo1_curves.png`
- **双策略对比**: `figures/table5/strategy_comparison.png`

## 相关文件
- **原始数据**: `experiments/table5/combo1_ensemble_alpha0_80/results.json`
```

---

## 文件组织规范

```
results/
├── README.md                 # 总览文件
├── table1/
│   ├── A_zeroshot_results.md
│   ├── B_pretrain_routing_results.md
│   ├── C_lora_vanilla_results.md
│   ├── D_lora_nsp_results.md
│   └── E_lora_nsp_full_results.md
├── table2/
│   ├── baseline_results.md
│   ├── nsp_only_results.md
│   ├── fd_only_results.md
│   ├── cd_only_results.md
│   ├── fd_cd_results.md
│   ├── nsp_fd_results.md
│   ├── nsp_cd_results.md
│   └── full_results.md
├── table3/
│   ├── combo1_mahalanobis_results.md
│   ├── combo1_lda_results.md
│   ├── combo1_qda_results.md
│   ├── combo1_lr_rgda_results.md
│   ├── combo2_mahalanobis_results.md
│   └── ...
├── table4/
│   ├── combo1_zeroshot_results.md
│   ├── combo1_lrrgda_results.md
│   ├── combo1_ensemble_results.md
│   ├── combo1_routing_results.md
│   └── ...
└── table5/
    ├── combo1_ensemble_alpha0_50_results.md
    ├── combo1_ensemble_alpha0_525_results.md
    ├── combo1_routing_alpha0_50_results.md
    └── ...
```

---

## 自动化收集脚本

```python
# scripts/collect_results.py
import os
import json
from pathlib import Path

def collect_table1_results():
    """收集Table 1结果"""
    methods = ['A_zeroshot', 'B_pretrain_routing', 'C_lora_vanilla', 
               'D_lora_nsp', 'E_lora_nsp_full']
    for method in methods:
        src = f'experiments/main/{method}/{method}_results.json'
        dst = f'results/table1/{method}_results.md'
        convert_to_markdown(src, dst, template='table1')

def collect_table3_results():
    """收集Table 3结果并计算统计量"""
    combos = ['combo1', 'combo2', 'combo3', 'combo4', 'combo5']
    detectors = ['mahalanobis', 'lda', 'qda', 'lr_rgda']
    
    for combo in combos:
        for detector in detectors:
            src = f'experiments/table3/{combo}_{detector}/results.json'
            dst = f'results/table3/{combo}_{detector}_results.md'
            convert_to_markdown(src, dst, template='table3')
    
    # 生成汇总表格
    generate_table3_summary()

# ... 其他收集函数
```
