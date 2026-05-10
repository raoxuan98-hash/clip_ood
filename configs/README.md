# 配置管理系统

基于YAML的配置管理，支持继承、验证和命令行覆盖。

## 快速开始

### 1. 运行预定义实验

```bash
# 运行单个实验
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml

# 覆盖参数
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml \
    --override training.lr=5e-5 training.iterations=1000

# 指定输出目录
python scripts/run_from_config.py \
    --config configs/experiments/lora_nsp_caltech.yaml \
    --exp-dir ./my_experiments/exp1
```

### 2. 创建自定义配置

```bash
# 复制示例配置
cp configs/experiments/lora_nsp_caltech.yaml configs/experiments/my_exp.yaml

# 编辑配置
vim configs/experiments/my_exp.yaml

# 运行
python scripts/run_from_config.py --config configs/experiments/my_exp.yaml
```

## 配置结构

### 基础配置 (`base/default.yaml`)

包含所有默认参数，实验配置继承并覆盖：

```yaml
experiment:
  name: "default_experiment"
  description: "..."

data:
  num_shots: 16
  batch_size: 32

training:
  iterations: 800
  lr: 1e-4
  # ...
```

### 实验配置 (`experiments/*.yaml`)

继承基础配置，只覆盖需要修改的参数：

```yaml
# 继承基础配置
inherits: "default"

experiment:
  name: "my_experiment"
  description: "..."

data:
  id_datasets: ["caltech101"]
  # 其他参数使用默认值
```

## 配置继承

### 继承语法

```yaml
inherits: "default"  # 继承 configs/base/default.yaml

# 覆盖特定参数
training:
  lr: 5e-5  # 覆盖默认的1e-4
  iterations: 1000  # 覆盖默认的800
```

### 继承链

```yaml
# base/default.yaml          - 最基础配置
# base/advanced.yaml         - 继承default，添加高级参数
#   inherits: "default"
# experiments/my_exp.yaml    - 继承advanced
#   inherits: "advanced"
```

## 配置验证

系统自动验证配置：

- **必填字段**: `experiment.name`, `data.id_datasets`
- **数值范围**: `lr`必须在(0,1)之间
- **类型检查**: `iterations`必须是正整数

验证失败时会显示错误信息。

## 命令行覆盖

### 覆盖语法

```bash
python scripts/run_from_config.py \
    --config configs/exp.yaml \
    --override key1=value1 key2=value2
```

### 嵌套参数

使用点号(`.`)访问嵌套参数：

```bash
# 覆盖 training.lr
--override training.lr=5e-5

# 覆盖 data.id_datasets
--override data.id_datasets=[caltech101,flowers]
```

### 类型推断

系统自动推断值类型：

```bash
--override training.lr=0.001        # float
--override training.iterations=1000 # int
--override training.enabled=true    # bool
--override data.id_datasets=[a,b]   # list
```

## 实验目录结构

运行实验后自动生成：

```
outputs/
└── experiment_name_20240314_123456/
    ├── config.yaml              # 配置副本
    ├── checkpoints/             # 模型检查点
    │   └── model.pt
    ├── stats.pt                 # 统计分布
    ├── classifier.pt            # 分类器
    ├── logs/                    # 日志文件
    ├── results/                 # 评估结果
    │   └── evaluation.json
    ├── cache/                   # 特征缓存
    └── visualizations/          # 可视化图表
```

## 配置示例

### 示例1：单任务微调

```yaml
# configs/experiments/single_task.yaml
inherits: "default"

experiment:
  name: "single_task_caltech"
  description: "Finetune on Caltech101"

data:
  id_datasets: ["caltech101"]
  ood_datasets: ["dtd", "eurosat"]

training:
  iterations: 1000
  lr: 1e-4
```

### 示例2：增量学习

```yaml
# configs/experiments/incremental.yaml
inherits: "default"

mode: "incremental"

experiment:
  name: "incremental_3tasks"

data:
  task_sequence:
    - ["caltech101"]
    - ["flowers"]
    - ["oxford_pets"]
  all_datasets: ["caltech101", "flowers", "oxford_pets", "stanford_cars", "food101"]

training:
  cov_momentum: 0.9
```

### 示例3：超参数优化

```yaml
# configs/experiments/optimization.yaml
inherits: "default"

mode: "optimization"

experiment:
  name: "optimize_ensemble"

training:
  enabled: false  # 跳过训练

optimization:
  enabled: true
  ensemble:
    enabled: true
    temperature_range:
      type: "range"
      start: 0.5
      end: 10.0
      step: 0.5
    alpha_values:
      type: "range"
      start: 0.0
      end: 1.0
      step: 0.05
```

## Python API

在代码中使用配置：

```python
from src.utils.config_manager import ConfigManager

# 加载配置
config = ConfigManager.load("configs/experiments/my_exp.yaml")

# 访问配置
lr = config.training.lr
datasets = config.data.id_datasets

# 安全访问（带默认值）
batch_size = config.get('data.batch_size', 32)

# 转换为字典
config_dict = config.to_dict()

# 转换为args（用于脚本）
args = config.to_args()
```

## 高级功能

### 环境变量

配置中支持环境变量：

```yaml
paths:
  data_root: ${DATA_ROOT}  # 从环境变量读取
```

### 动态配置

在Python中动态修改：

```python
config = ConfigManager.load("configs/exp.yaml")

# 动态修改
config._config['training']['lr'] = 5e-5

# 保存
config.save("configs/modified_exp.yaml")
```

### 配置对比

```python
from src.utils.config_manager import ConfigManager

config1 = ConfigManager.load("configs/exp1.yaml")
config2 = ConfigManager.load("configs/exp2.yaml")

# 找出差异
diff = {}
for key in config1._config:
    if config1._config[key] != config2._config[key]:
        diff[key] = (config1._config[key], config2._config[key])
```

## 最佳实践

### 1. 使用有意义的实验名称

```yaml
experiment:
  name: "lora_nsp_caltech_rank4_iter800"
  description: "LoRA-NSP on Caltech101 with rank=4, 800 iterations"
  tags: ["lora_nsp", "caltech101", "rank4"]
```

### 2. 注释配置

```yaml
# 使用大学习率因为数据集较小
training:
  lr: 5e-4  # 较大学习率适合小数据集
```

### 3. 版本控制配置

```bash
# 将配置加入git
git add configs/experiments/my_exp.yaml

# 提交
git commit -m "Add experiment: my_exp"
```

### 4. 复现实验

```bash
# 保存配置到实验目录
cp configs/experiments/my_exp.yaml outputs/my_exp_20240314_123456/

# 任何人都可以复现
python scripts/run_from_config.py \
    --config outputs/my_exp_20240314_123456/config.yaml
```

## 故障排除

### 错误：Config file not found

```
ConfigError: Config file not found: configs/exp.yaml
```

**解决**：检查路径是否正确

### 错误：Missing required field

```
ConfigError: Missing required field: experiment.name
```

**解决**：在配置中添加必填字段

### 错误：Invalid learning rate

```
ConfigError: Invalid learning rate: 1.0
```

**解决**：学习率应在(0,1)之间

## 参考

- [YAML语法](https://yaml.org/spec/)
- [Python argparse](https://docs.python.org/3/library/argparse.html)