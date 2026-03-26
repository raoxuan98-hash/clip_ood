# LoRA-NSP 实验计划 V3

**文档版本**: 3.0  
**更新日期**: 2026-03-18  
**对标论文**: LADA (ICML 2025) - Scalable Label-Specific CLIP Adapter for Continual Learning

---

## 🔴 系统级约束 (CRITICAL)

### GPU 3 禁止使用

| 项目 | 说明 |
|------|------|
| **状态** | ❌ 禁止使用 (硬件错误) |
| **影响** | 使用 GPU 3 会导致整个 CUDA/NVML 栈崩溃 |
| **可用 GPU** | 0, 1, 2, 4, 5 |
| **最大并行度** | 5 个实验 |

**⚠️ 警告**: 违反此约束将导致系统级 CUDA 崩溃，所有实验被迫中断！

**环境变量必须设置为**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4,5
```

---

## 1. 实验总体框架

### 1.1 创新点与验证实验

| 创新点 | 核心问题 | 验证实验 |
|:------|:--------|:--------|
| **LoRA-NSP** (零空间投影) | 如何缓解微调时的灾难性遗忘？ | 消融实验1: LoRA vs LoRA-NSP |
| **知识蒸馏** (FD + CD) | 如何保持预训练模型的通用能力？ | 消融实验2: 蒸馏组件贡献分析 |
| **LR-RGDA 分类器** | 如何充分利用监督数据？ | 消融实验3: 分类器对比 |
| **自适应路由** | 如何平衡 ID/OOD 性能？ | 消融实验4: 路由策略有效性 |

### 1.2 实验层次

- **主实验 (Table 1)**: 对标 LADA，验证整体方法优势
- **副实验 (消融)**: 验证各组件独立贡献 (Table 2, 3, 4)
- **敏感性分析**: 分析关键超参数影响 (Table 5)

---

## 2. 主实验设计 (Table 1)

### 2.1 核心思想

**评估目标**：在持续学习场景中
- **已见任务（ID）**：分类性能要**提升**（通过学习）
- **未见任务（OOD）**：分类性能**不下降**（相对于零样本基线）

**关键原则**：
- 我们不知道测试样本是ID还是OOD！
- 所有测试样本都经过相同的处理流程
- 最终输出预测标签，与真实标签对比

### 2.2 数据集与任务设置

**X-TAIL 基准** (10个数据集, 1,100类):

| 任务 | 数据集 | 类别数 | 领域 |
|------|--------|--------|------|
| 1 | aircraft | 100 | 飞机细分类 |
| 2 | caltech101 | 101 | 通用物体 |
| 3 | dtd | 47 | 纹理 |
| 4 | eurosat | 10 | 卫星图像 |
| 5 | flowers | 102 | 花卉 |
| 6 | food101 | 101 | 食物 |
| 7 | mnist | 10 | 手写数字 |
| 8 | oxford_pets | 37 | 宠物 |
| 9 | stanford_cars | 196 | 汽车细分类 |
| 10 | sun397 | 397 | 场景 |

**任务顺序** (LADA Order-I, 字母顺序):
```
aircraft → caltech101 → dtd → eurosat → flowers → food101 → mnist → oxford_pets → stanford_cars → sun397
```

### 2.3 5个对比方法

| 编号 | 方法 | 训练端 | 推理端 | 处理流程 |
|------|------|--------|--------|----------|
| **A** | Zero-shot CLIP | 无 | 零样本分类器 | 所有样本 → 零样本分类器 |
| **B** | CLIP + 自适应路由 | 无 | **OOD检测+自适应路由** | 所有样本 → OOD检测 → 自动选择分类器 |
| **C** | LoRA (Vanilla) | 标准 LoRA | 零样本分类器 | 训练 → 合并LoRA → 所有样本 → 零样本分类器 |
| **D** | LoRA-NSP (仅微调) | LoRA + NSP + 蒸馏 | 零样本分类器 | 训练 → 合并LoRA → 所有样本 → 零样本分类器 |
| **E** | **LoRA-NSP Full** | LoRA + NSP + 蒸馏 | **OOD检测+自适应路由** | 训练 → 合并LoRA → 所有样本 → 路由系统 |

### 2.4 各方法处理流程详解

#### 方法A: Zero-shot（基线）
```
测试样本 → 零样本分类器 → 预测标签 → 与真实标签对比
```
- 不训练，直接使用预训练CLIP
- 所有样本用零样本分类器
- 作为性能基线（约60%）

#### 方法B: Pretrain + Routing（仅推理端创新）
```
测试样本 → OOD检测器 → 阈值判断 → 选择分类器 → 预测标签 → 与真实标签对比
                ↓
        ┌───────┴───────┐
     ID-like         OOD-like
        ↓               ↓
   集成分类器      零样本分类器
   (LR-RGDA+ZS)    (Zero-shot)
```
- 不训练CLIP
- 每个步骤后，基于已见任务的高斯分布统计构建OOD检测器和路由
- 系统自动选择分类器（我们不知道它选了哪个，因为不知道测试样本到底是来自ID的还是OOD数据集）
- **验证推理端创新的独立价值**

#### 方法C: LoRA (Vanilla)（训练端基线）
```
训练任务t → 合并LoRA → 所有样本 → 零样本分类器 → 预测标签 → 与真实标签对比
```
- 每个任务用标准LoRA微调
- 微调后合并LoRA权重到主模型
- 所有任务用零样本分类器评估
- **存在灾难性遗忘问题**

#### 方法D: LoRA-NSP (仅训练端创新)
```
训练任务t (NSP+蒸馏) → 合并LoRA → 所有样本 → 零样本分类器 → 预测标签
```
- 每个任务用LoRA-NSP微调（零空间投影）
- 使用特征蒸馏(FD)和跨模态蒸馏(CD)
- 所有任务用零样本分类器评估
- **NSP缓解灾难性遗忘**
- **验证训练端创新的价值**

#### 方法E: LoRA-NSP Full（完整方法）
```
训练任务t (NSP+蒸馏) → 合并LoRA → 构建路由系统 → 所有样本 → 路由系统 → 预测标签
```
- 同方法D的训练流程
- 每个步骤后构建OOD检测+自适应路由
- **训练端 + 推理端协同**
- **验证完整方法的优势**

### 2.5 评估流程（每个任务后）

**示例：完成第4个任务 (eurosat) 后**

注意，仅仅是示意，并不意味着下面的四个数据集就是我们唯一评估的数据集。

1. **已见任务**（当做ID数据集，用于构建路由系统）: aircraft, caltech101, dtd, eurosat
2. **测试数据**: 所有10个任务的测试集
3. **评估步骤**:
   - 如果是方法B/E：构建OOD检测器和自适应路由
   - 对所有10个任务的测试样本进行分类
   - 每个样本经过相同的处理流程
   - 记录每个任务的分类准确率

4. **计算指标**:
   - 已见任务平均准确率（ID性能）
   - 未见任务平均准确率（OOD性能）
   - 所有任务平均准确率

### 2.6 LADA指标计算（完成所有10个任务后）

构建准确率矩阵 $\hat{a}^{(j)}_k$：训练完任务 $j$ 后在任务 $k$ 上的准确率

```
         Task1  Task2  Task3  ...  Task10
Step 0:   60%    58%    59%   ...   57%    (Zero-shot baseline)
Step 1:   65%    62%    61%   ...   58%
Step 2:   68%    70%    63%   ...   59%
...
Step 10:  75%    78%    76%   ...   80%
```

| 指标 | 公式 | 含义 | 计算方式 |
|------|------|------|----------|
| **Transfer** | $\frac{1}{K-1}\sum_{k=2}^{K} \frac{1}{k-1}\sum_{j=1}^{k-1}\hat{a}^{(j)}_k$ | 前向迁移能力 | 新任务上利用旧知识的能力 |
| **Average** | $\frac{1}{K}\sum_{k=1}^{K} \frac{1}{K}\sum_{j=1}^{K}\hat{a}^{(j)}_k$ | 综合性能 | 所有步骤所有任务的平均 |
| **Last** | $\frac{1}{K}\sum_{k=1}^{K} \hat{a}^{(K)}_k$ | 最终性能 | 最后一行的平均 |
| **Forgetting** | $\frac{1}{K-1}\sum_{k=1}^{K-1}(\max_{l<k}\hat{a}^{(l)}_k - \hat{a}^{(K)}_k)$ | 遗忘程度 | 历史最佳与最终的差距 |

### 2.7 预期结果（正交性分析）

| 方法 | Transfer | Average | Last | Forgetting |
|------|----------|---------|------|------------|
| **A** Zero-shot | 57.7% | - | - | - |
| **B** Pretrain + Routing | 57.7% | - | - | - |
| **C** LoRA (Vanilla) | ~55% | ~65% | ~70% | ~15% |
| **D** LoRA-NSP (仅微调) | ~60% | ~72% | ~80% | <10% |
| **E** LoRA-NSP Full | ~60% | ~72% | ~80% | <10% |
| LADA (paper) | 61.5% | 72.7% | 83.1% | - |

**正交性分析**:
- **A → B**: 验证推理端创新（自适应路由）的独立价值
- **A → C → D**: 验证训练端创新（LoRA-NSP）提升Last指标（抗遗忘）
- **D → E**: 验证完整方法协同效果

**关键洞察**:
方法D和方法E达到相似的LADA指标，但方法E通过自适应路由在固定划分场景下（Table 4）表现更好。

**注**: 详细的逐数据集结果见附录，ID/OOD性能分析见Table 4。

---

## 3. 实验运行命令

### 3.1 方法 A: Zero-shot CLIP (基线)

```bash
CUDA_VISIBLE_DEVICES=0 python src/experiments/run_continual_learning.py \
    --method zeroshot \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --output_dir experiments/main/A_zeroshot \
    > logs/main_A.log 2>&1 &
```

### 3.2 方法 B: CLIP + 自适应路由 (仅推理端)

**注意**: 使用Table 3确定的最佳OOD检测器类型和阈值

```bash
CUDA_VISIBLE_DEVICES=1 python src/experiments/run_continual_learning_routing_v2.py \
    --method pretrain_routing \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --ood_detector_type lr_rgda \
    --ood_threshold <BEST_THRESHOLD_FROM_TABLE3> \
    --output_dir experiments/main/B_pretrain_routing \
    > logs/main_B.log 2>&1 &
```

### 3.3 方法 C: LoRA (Vanilla)

```bash
CUDA_VISIBLE_DEVICES=2 python src/experiments/run_continual_learning.py \
    --method lora_vanilla \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 \
    --iterations 800 \
    --fd_weight 0 \
    --cd_weight 0 \
    --output_dir experiments/main/C_lora_vanilla \
    > logs/main_C.log 2>&1 &
```

### 3.4 方法 D: LoRA-NSP (仅训练端)

```bash
CUDA_VISIBLE_DEVICES=4 python src/experiments/run_continual_learning.py \
    --method lora_nsp \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 \
    --iterations 800 \
    --fd_weight 1.0 \
    --cd_weight 1.0 \
    --output_dir experiments/main/D_lora_nsp_only \
    > logs/main_D.log 2>&1 &
```

### 3.5 方法 E: LoRA-NSP Full (完整方法)

**注意**: 
- 依赖方法D的checkpoint
- 使用Table 3确定的最佳OOD检测器类型和阈值

```bash
# 等待D完成后执行
CUDA_VISIBLE_DEVICES=5 python src/experiments/run_continual_learning_routing_v2.py \
    --method lora_nsp_full \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --load_checkpoint experiments/main/D_lora_nsp_only/final_model.pt \
    --ood_detector_type lr_rgda \
    --ood_threshold <BEST_THRESHOLD_FROM_TABLE3> \
    --output_dir experiments/main/E_lora_nsp_full \
    > logs/main_E.log 2>&1
```

---

## 4. 执行策略（优化版）

### 🎯 优化原则

1. **先做轻量级实验**: 无需微调主干的实验（Table 3/4/5）
2. **先做消融实验**: 选择最佳超参数后再做主实验
3. **合理并行**: 充分利用5个GPU
4. **结果复用**: Table 1-A/B 复用 Table 4 的零样本/路由结果

### ⚠️ 关键注意事项

#### 1. GPU约束
- **GPU 3 禁止使用**（硬件错误会导致CUDA崩溃）
- 必须设置: `export CUDA_VISIBLE_DEVICES=0,1,2,4,5`
- 最大并行度: 5个实验

#### 2. 方法C/D区分（重要！）

| 方法 | LoRA类型 | 蒸馏设置 | 命令参数 |
|------|----------|----------|----------|
| **C** LoRA Vanilla | `lora_vanilla` | **无蒸馏** | `--fd_weight 0 --cd_weight 0` |
| **D** LoRA-NSP | `lora_nsp` | **有蒸馏** | `--fd_weight 1.0 --cd_weight 1.0` |

⚠️ **注意**: 方法C和D都使用 `run_continual_learning.py`，区别仅在于 `--method` 和蒸馏权重参数！

#### 3. 超参数确认

当前统一超参数:
- `qda_reg_alpha2 = 1.0` (实验脚本中已设置)
- `rank = 32`
- `alpha = 0.8` (集成分类器权重)

#### 4. 特征缓存（Table 4/5必需）

Table 4/5 依赖预提取的特征缓存，实验前必须执行:
```bash
python scripts/extract_cached_features.py \
    --datasets aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --cache_dir cache/pretrained_features \
    --device cuda
```

#### 5. 脚本版本
- 使用 `run_continual_learning_routing_v2.py`（非旧版本）
- 使用 `run_cached_experiment.py` 进行Table 4/5实验

---

### Phase 0: 预准备

#### 0.1 环境设置
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4,5
mkdir -p experiments/main logs cache/pretrained_features
```

#### 0.2 特征提取（Table 4/5必需）
```bash
python scripts/extract_cached_features.py \
    --datasets aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --cache_dir cache/pretrained_features \
    --device cuda
```
- **预计时间**: ~2小时
- **执行一次即可**

---

### Phase 1: 无需主干微调的实验（轻量级）

**目的**: 选择最佳超参数，为后续主实验做准备

#### 1.1 Table 3: OOD检测器对比
```bash
# Combo 1-5, 每个组合4个检测器，共20个实验
# 可5个并行（每个GPU一个组合）
for combo in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$combo python scripts/run_ood_detector_eval.py ... &
done
```
- **预计时间**: ~2小时（5个GPU并行）
- **产出**: 最佳OOD检测器类型和阈值

#### 1.2 Table 4: 推理端消融
```bash
# 使用Table 3确定的最佳阈值
python scripts/run_cached_experiment.py --enable_routing --ood_threshold <BEST> ...
```
- **预计时间**: ~1小时
- **产出**: 验证自适应路由优势

#### 1.3 Table 5: Alpha敏感性
```bash
# 21个alpha值 × 2种策略 = 42个实验
for alpha in $(seq 0.50 0.025 1.00); do
    python scripts/run_cached_experiment.py --alpha $alpha ...
done
```
- **预计时间**: ~3小时
- **产出**: 确认alpha=0.8为最佳值

**Phase 1 总时间**: ~6小时

---

### Phase 2: 训练端消融（小规模）

#### 2.1 Table 2: 组件贡献分析
```bash
# 使用5个任务快速验证各组件贡献
# 8个配置，可部分并行
```
- **预计时间**: ~6小时
- **产出**: 确认NSP+FD+CD为最佳配置

**Phase 2 总时间**: ~6小时

---

### Phase 3: 主实验（Table 1）

**依赖**: Table 3完成（获得最佳阈值）

#### 3.1 方法A: Zero-shot（复用Table 4结果）
```bash
# 可直接从Table 4的零样本结果复用
# 或重新运行:
CUDA_VISIBLE_DEVICES=0 python src/experiments/run_continual_learning.py \
    --method zeroshot ...
```

#### 3.2 方法B: Pretrain + Routing（复用Table 4结果）
```bash
# 可直接从Table 4的自适应路由结果复用
# 或重新运行:
CUDA_VISIBLE_DEVICES=1 python src/experiments/run_continual_learning_routing_v2.py \
    --method pretrain_routing --ood_threshold <BEST> ...
```

#### 3.3 方法C: LoRA Vanilla
```bash
# ⚠️ 注意: 必须显式设置 fd_weight=0 cd_weight=0
CUDA_VISIBLE_DEVICES=2 python src/experiments/run_continual_learning.py \
    --method lora_vanilla \
    --fd_weight 0 --cd_weight 0 \
    ...
```
- **预计时间**: ~8小时

#### 3.4 方法D: LoRA-NSP
```bash
# ⚠️ 注意: 必须显式设置 fd_weight=1.0 cd_weight=1.0
CUDA_VISIBLE_DEVICES=4 python src/experiments/run_continual_learning.py \
    --method lora_nsp \
    --fd_weight 1.0 --cd_weight 1.0 \
    ...
```
- **预计时间**: ~8小时

#### 3.5 方法E: LoRA-NSP Full
```bash
# ⚠️ 依赖方法D完成
CUDA_VISIBLE_DEVICES=5 python src/experiments/run_continual_learning_routing_v2.py \
    --method lora_nsp_full \
    --load_checkpoint experiments/main/D_lora_nsp_only/final_model.pt \
    --ood_threshold <BEST_FROM_TABLE3> \
    ...
```
- **预计时间**: ~3小时
- **依赖**: 方法D完成

**Phase 3 总时间**: ~11小时（C和D并行）

---

### 📊 总时间预算

| Phase | 内容 | 时间 | 并行度 |
|-------|------|------|--------|
| 0 | 预准备 | 2h | 1 GPU |
| 1 | Table 3/4/5 | 6h | 5 GPUs |
| 2 | Table 2 | 6h | 2-3 GPUs |
| 3 | Table 1 | 11h | 3-4 GPUs |
| **总计** | | **~25小时** | |

---

### 🔗 关键依赖链

```
Phase 0: 特征提取
    ↓
Phase 1: Table 3 → 最佳OOD检测器 + 阈值
    ↓
Phase 1: Table 4/5 → 验证路由有效性 + alpha=0.8
    ↓
Phase 2: Table 2 → 验证NSP+FD+CD最优
    ↓
Phase 3: Table 1 (B/E) → 使用Table 3的最佳阈值
    ↓
Phase 3: Table 1 (E) → 依赖方法D完成
```

---

### 💡 快速启动方案（如果赶时间）

**最小实验集**（验证核心结论）:
1. **Table 3**（选Combo 1）→ 确定阈值
2. **Table 4**（选Combo 1）→ 验证路由优势
3. **Table 1**: 方法A/B/D/E（跳过C）
4. **Table 2**: 仅Full配置

**最小集时间**: ~18小时

**注意**: 此方法可能无法生成论文所需的完整表格，但能验证核心结论。

---

## 5. 消融实验

### 5.1 Table 2: 训练端消融 (LoRA-NSP + 蒸馏)

**目的**: 分析NSP、FD、CD各组件的贡献

**实验设计**: 使用5个任务 (aircraft, caltech101, dtd, eurosat, flowers)

| 配置 | NSP | FD | CD | 评估方式 |
|------|-----|----|----|----------|
| Baseline (仅LoRA) | ✗ | ✗ | ✗ | 零样本分类器 |
| + NSP only | ✓ | ✗ | ✗ | 零样本分类器 |
| + FD only | ✗ | ✓ | ✗ | 零样本分类器 |
| + CD only | ✗ | ✗ | ✓ | 零样本分类器 |
| + FD + CD | ✗ | ✓ | ✓ | 零样本分类器 |
| + NSP + FD | ✓ | ✓ | ✗ | 零样本分类器 |
| + NSP + CD | ✓ | ✗ | ✓ | 零样本分类器 |
| **Full (NSP+FD+CD)** | **✓** | **✓** | **✓** | **零样本分类器** |

### 5.2 Table 3: OOD检测器对比

**目的**: 比较不同OOD检测器的性能，评估其在不同ID/OOD划分下的稳定性和泛化能力。

#### 实验设计

**方案**: 交叉验证设计（方案A）

**关键原则**: 
- 每个组合使用**全部10个数据集**
- ID数据集 + OOD数据集 = 全部数据集
- 多组不同划分，计算均值±标准差

#### 5组ID/OOD划分

| 组合 | ID数据集 | OOD数据集 | ID数量 | OOD数量 |
|------|----------|-----------|--------|---------|
| **Combo 1** | aircraft, caltech101, dtd, eurosat, flowers | food101, mnist, oxford_pets, stanford_cars, sun397 | 5 | 5 |
| **Combo 2** | aircraft, caltech101, dtd, eurosat, flowers, food101 | mnist, oxford_pets, stanford_cars, sun397 | 6 | 4 |
| **Combo 3** | aircraft, caltech101, dtd, eurosat | flowers, food101, mnist, oxford_pets, stanford_cars, sun397 | 4 | 6 |
| **Combo 4** | aircraft, caltech101, dtd | eurosat, flowers, food101, mnist, oxford_pets, stanford_cars, sun397 | 3 | 7 |
| **Combo 5** | aircraft, caltech101, dtd, eurosat, flowers, food101, mnist | oxford_pets, stanford_cars, sun397 | 7 | 3 |

**设计 rationale**:
- Combo 1: 平衡划分 (5+5)，最接近实际应用场景
- Combo 2: ID偏多 (6+4)，测试ID丰富时的检测性能
- Combo 3: OOD偏多 (4+6)，测试OOD多样时的检测性能
- Combo 4: ID较少 (3+7)，测试极端情况
- Combo 5: ID很多 (7+3)，测试另一种极端情况

#### 评估指标

**只关注OOD检测性能指标**（不关注分类准确率）:

| 指标 | 符号 | 含义 | 目标 |
|------|------|------|------|
| **AUROC** | ↑ | ROC曲线下面积 | > 95% |
| **FPR@95TPR** | ↓ | 95%真阳性率时的假阳性率 | < 10% |
| **AUPR** | ↑ | PR曲线下面积 | > 95% |
| **Detection Error** | ↓ | 最优阈值下的检测错误 | < 5% |

**不关注**（留给Table 1和Table 4）:
- ❌ ID Acc（分类准确率）
- ❌ OOD Acc（分类准确率）
- ❌ Combined Score

#### 对比的检测器

| 检测器 | 类型 | 核心思想 |
|--------|------|----------|
| **Mahalanobis** | 距离-based | 马氏距离，基于协方差矩阵 |
| **LDA** | 生成式 | 线性判别分析，共享协方差 |
| **QDA** | 生成式 | 二次判别分析，各类独立协方差 |
| **LR-RGDA** | 生成式+低秩 | 低秩分解正则高斯判别分析（我们的方法）|

#### 预期结果表格

| 检测器 | AUROC (↑) | FPR@95TPR (↓) | AUPR (↑) | Detection Error (↓) |
|--------|-----------|---------------|----------|---------------------|
| Mahalanobis | 87.3±2.1% | 30.9±3.2% | 85.1±2.5% | 17.5±1.8% |
| LDA | 98.7±0.5% | 5.4±0.8% | 97.8±0.6% | 4.9±0.5% |
| QDA | 99.1±0.3% | 3.5±0.5% | 98.5±0.4% | 3.3±0.4% |
| **LR-RGDA** | **99.1±0.3%** | **3.5±0.5%** | **98.6±0.4%** | **3.3±0.4%** |

注: 数值为5组交叉验证的平均值±标准差

#### 运行命令

```bash
# Combo 1: 5 ID + 5 OOD
for detector in mahalanobis lda qda lr_rgda; do
    python scripts/run_ood_detector_eval.py \
        --id_datasets aircraft caltech101 dtd eurosat flowers \
        --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
        --detector $detector \
        --output_dir experiments/table3/combo1_${detector}
done

# Combo 2: 6 ID + 4 OOD
for detector in mahalanobis lda qda lr_rgda; do
    python scripts/run_ood_detector_eval.py \
        --id_datasets aircraft caltech101 dtd eurosat flowers food101 \
        --ood_datasets mnist oxford_pets stanford_cars sun397 \
        --detector $detector \
        --output_dir experiments/table3/combo2_${detector}
done

# Combo 3: 4 ID + 6 OOD
for detector in mahalanobis lda qda lr_rgda; do
    python scripts/run_ood_detector_eval.py \
        --id_datasets aircraft caltech101 dtd eurosat \
        --ood_datasets flowers food101 mnist oxford_pets stanford_cars sun397 \
        --detector $detector \
        --output_dir experiments/table3/combo3_${detector}
done

# Combo 4: 3 ID + 7 OOD
for detector in mahalanobis lda qda lr_rgda; do
    python scripts/run_ood_detector_eval.py \
        --id_datasets aircraft caltech101 dtd \
        --ood_datasets eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
        --detector $detector \
        --output_dir experiments/table3/combo4_${detector}
done

# Combo 5: 7 ID + 3 OOD
for detector in mahalanobis lda qda lr_rgda; do
    python scripts/run_ood_detector_eval.py \
        --id_datasets aircraft caltech101 dtd eurosat flowers food101 mnist \
        --ood_datasets oxford_pets stanford_cars sun397 \
        --detector $detector \
        --output_dir experiments/table3/combo5_${detector}
done
```

#### 结果汇总

```python
import json
import numpy as np

detectors = ['mahalanobis', 'lda', 'qda', 'lr_rgda']
results = {}

for detector in detectors:
    aurocs = []
    fpr95s = []
    auprs = []
    detection_errors = []
    
    for combo in range(1, 6):
        with open(f'experiments/table3/combo{combo}_{detector}/results.json') as f:
            data = json.load(f)
            aurocs.append(data['auroc'])
            fpr95s.append(data['fpr95'])
            auprs.append(data['aupr'])
            detection_errors.append(data['detection_error'])
    
    results[detector] = {
        'auroc': f"{np.mean(aurocs)*100:.1f}±{np.std(aurocs)*100:.1f}%",
        'fpr95': f"{np.mean(fpr95s)*100:.1f}±{np.std(fpr95s)*100:.1f}%",
        'aupr': f"{np.mean(auprs)*100:.1f}±{np.std(auprs)*100:.1f}%",
        'detection_error': f"{np.mean(detection_errors)*100:.1f}±{np.std(detection_errors)*100:.1f}%"
    }

# 生成表格
print("| 检测器 | AUROC (↑) | FPR@95TPR (↓) | AUPR (↑) | Detection Error (↓) |")
print("|--------|-----------|---------------|----------|---------------------|")
for detector in detectors:
    r = results[detector]
    print(f"| {detector} | {r['auroc']} | {r['fpr95']} | {r['aupr']} | {r['detection_error']} |")
```

#### 优势

1. **统计可靠性**: 5组交叉验证，结果更稳定，有标准差
2. **全面性**: 覆盖不同ID/OOD比例场景
3. **指标聚焦**: 专注于OOD检测性能，与Table 1/4区分
4. **泛化能力**: 评估检测器在不同数据分布下的表现

### 5.3 Table 4: 推理端消融

**目的**: 验证自适应路由分类器相对于单一分类器和固定集成的优势，并确定最佳阈值。

#### 实验设计

**场景**: 固定ID/OOD划分（基于预训练CLIP，不微调）

**数据集组合**: 使用与Table 3相同的5组划分

| 组合 | ID数据集 | OOD数据集 | ID数 | OOD数 | 说明 |
|------|----------|-----------|------|-------|------|
| **Combo 1** | aircraft, caltech101, dtd, eurosat, flowers | food101, mnist, oxford_pets, stanford_cars, sun397 | 5 | 5 | 平衡 |
| **Combo 2** | aircraft, caltech101, dtd, eurosat, flowers, food101 | mnist, oxford_pets, stanford_cars, sun397 | 6 | 4 | ID偏多 |
| **Combo 3** | aircraft, caltech101, dtd, eurosat | flowers, food101, mnist, oxford_pets, stanford_cars, sun397 | 4 | 6 | OOD偏多 |
| **Combo 4** | aircraft, caltech101, dtd | eurosat, flowers, food101, mnist, oxford_pets, stanford_cars, sun397 | 3 | 7 | **ID < OOD** |
| **Combo 5** | aircraft, caltech101, dtd, eurosat, flowers, food101, mnist | oxford_pets, stanford_cars, sun397 | 7 | 3 | ID很多 |

**设计理由**:
- 覆盖不同ID/OOD比例场景
- 验证方法在各种数据分布下的鲁棒性
- Combo 4特别测试ID较少时的表现

#### 对比的推理策略

| 方法 | 处理流程 | 说明 |
|------|----------|------|
| **纯零样本** | 所有样本 → 零样本分类器 | 基线方法 |
| **纯LR-RGDA** | 所有样本 → LR-RGDA分类器 | 仅用监督数据训练的分类器 |
| **固定集成 (α=0.8)** | 所有样本 → α·LR-RGDA + (1-α)·零样本 | 固定权重融合 |
| **自适应路由** | 所有样本 → OOD检测 → 动态选择分类器 | **我们的方法，测试多阈值** |

#### 评估指标（修正）

与Table 3不同，Table 4关注**分类性能**：

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **ID Avg Acc** | Σ(ID数据集正确数) / Σ(ID数据集样本数) | 所有ID数据集的平均准确率 |
| **OOD Avg Acc** | Σ(OOD数据集正确数) / Σ(OOD数据集样本数) | 所有OOD数据集的平均准确率 |
| **Overall Acc** | (总正确数) / (总样本数) | **所有数据集的整体准确率** |

**计算示例**（以5个ID + 5个OOD数据集为例）:

假设各数据集准确率：
- ID: 80%, 82%, 78%, 81%, 79%
- OOD: 70%, 72%, 68%, 71%, 69%

计算：
- ID Avg Acc = (80+82+78+81+79) / 5 = 80.0%
- OOD Avg Acc = (70+72+68+71+69) / 5 = 70.0%
- **Overall Acc = (80+82+78+81+79+70+72+68+71+69) / 10 = 75.0%**

**注意**: 
- 如果各数据集样本数不同，需要按样本数加权
- Overall Acc ≠ (ID Avg Acc + OOD Avg Acc) / 2

#### 预期结果

**使用固定阈值（基于Table 3优化结果，如0.85）**

**单组结果示例（Combo 1）**:

| 方法 | ID Avg Acc | OOD Avg Acc | Overall Acc |
|------|------------|-------------|-------------|
| 纯零样本 | 60% | 85% | 72.5% |
| 纯LR-RGDA | 80% | 30% | 55% |
| 固定集成 (α=0.8) | 75% | 70% | 72.5% |
| **自适应路由** | **78%** | **82%** | **80%** |

**最终汇总（5组平均）**:

| 方法 | ID Avg Acc | OOD Avg Acc | Overall Acc |
|------|------------|-------------|-------------|
| 纯零样本 | 58±3% | 83±4% | 70.5±3.5% |
| 纯LR-RGDA | 78±4% | 32±5% | 55±5% |
| 固定集成 | 73±3% | 68±4% | 70.5±3.5% |
| **自适应路由** | **76±3%** | **80±4%** | **78±3%** |

注: 数值为5组组合的平均值±标准差

**关键发现**:
- 纯零样本：OOD性能好（83%），但ID性能差（58%）
- 纯LR-RGDA：ID性能好（78%），但OOD性能极差（32%）
- 固定集成：折中方案，但OOD仍有明显下降
- **自适应路由**：ID接近LR-RGDA（76%），**OOD大幅提升至80%**，Overall Acc最优（78%）

**核心结论**:
- 自适应路由相比固定集成，ID性能相当，但OOD性能提升12%（80% vs 68%）
- 证明了动态路由相对于固定权重融合的优势

#### 运行命令

**Combo 1 - 纯零样本**:
```bash
python scripts/run_cached_experiment.py \
    --id_datasets aircraft caltech101 dtd eurosat flowers \
    --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
    --classifier_type zeroshot \
    --output_dir experiments/table4/combo1_zeroshot
```

**Combo 1 - 纯LR-RGDA (α=1.0)**:
```bash
python scripts/run_cached_experiment.py \
    --id_datasets aircraft caltech101 dtd eurosat flowers \
    --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
    --classifier_type ensemble \
    --alpha 1.0 \
    --output_dir experiments/table4/combo1_lrrgda
```

**Combo 1 - 固定集成 (α=0.8)**:
```bash
python scripts/run_cached_experiment.py \
    --id_datasets aircraft caltech101 dtd eurosat flowers \
    --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
    --classifier_type ensemble \
    --alpha 0.8 \
    --output_dir experiments/table4/combo1_ensemble
```

**Combo 1 - 自适应路由（使用Table 3确定的最佳阈值，如0.85）**:
```bash
python scripts/run_cached_experiment.py \
    --id_datasets aircraft caltech101 dtd eurosat flowers \
    --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
    --enable_routing \
    --ood_threshold 0.85 \
    --output_dir experiments/table4/combo1_routing
```

**注意**: 需要对Combo 2-5重复上述命令，使用相同的阈值

#### 可视化

**图1: 方法性能对比（柱状图）**
- X轴: 4种方法（纯零样本、纯LR-RGDA、固定集成、自适应路由）
- Y轴: 准确率
- 分组: ID Avg Acc、OOD Avg Acc、Overall Acc
- 目的: 直观展示自适应路由的优势

**图2: 不同组合的Overall Acc（热力图）**
- X轴: 5个组合（Combo 1-5）
- Y轴: 4种方法
- 颜色深度: Overall Acc值
- 目的: 展示方法在不同ID/OOD比例下的鲁棒性

**图3: ID vs OOD性能散点图**
- X轴: OOD Avg Acc
- Y轴: ID Avg Acc
- 点: 每种方法在5个组合上的表现
- 理想区域: 右上角（ID和OOD性能都高）
- 目的: 展示自适应路由达到最佳平衡点

### 5.4 Table 5: Alpha敏感性分析

**目的**: 分析集成权重α对分类性能的影响，比较固定集成和自适应路由的敏感性差异。

**核心假设**: 自适应路由对α不敏感，因为OOD检测器会自动选择分类器，减轻了权重选择的压力。

#### 实验设计

**数据集**: 使用与Table 3/4相同的5组划分

**对比的两种策略**:

| 策略 | 说明 | 测试参数 |
|------|------|----------|
| **固定集成** | 无OOD检测，固定权重融合 | α: 0.50 ~ 1.00，步长0.025 |
| **自适应路由** | 有OOD检测，动态选择分类器 | α: 0.50 ~ 1.00，步长0.025（集成内部权重）|

**注意**: 自适应路由的α是集成分类器内部的权重（当样本被判定为ID时使用），OOD检测阈值固定（如0.85）。

#### 评估指标

与Table 4一致：

| 指标 | 计算方式 |
|------|----------|
| **ID Avg Acc** | 所有ID数据集的平均准确率（加权） |
| **OOD Avg Acc** | 所有OOD数据集的平均准确率（加权） |
| **Overall Acc** | (总正确数) / (总样本数) |

#### 实验规模

- 5组组合
- 21个alpha值
- 2种策略（固定集成 + 自适应路由）
- **总计**: 210组实验 (5 × 21 × 2)

#### 预期结果

**固定集成（无路由）**:
- α=0.5: ID Acc≈60%, OOD Acc≈85%（接近纯零样本）
- α=0.8: ID Acc≈75%, OOD Acc≈70%（平衡）
- α=1.0: ID Acc≈80%, OOD Acc≈30%（接近纯LR-RGDA）
- **特点**: 对α高度敏感，ID和OOD性能此消彼长

**自适应路由（有路由）**:
- α=0.5: ID Acc≈75%, OOD Acc≈78%
- α=0.8: ID Acc≈76%, OOD Acc≈80%
- α=1.0: ID Acc≈77%, OOD Acc≈81%
- **特点**: 对α不敏感，ID和OOD性能都保持稳定

**核心结论**:
- 固定集成需要仔细调优α（如0.8）
- 自适应路由对α鲁棒，因为OOD检测器承担了选择任务
- 这证明了自适应路由的实用优势：减少超参数调优负担

#### 可视化

**图1: ID Avg Acc随alpha变化（双曲线+误差带）**
- X轴: α (0.5 ~ 1.0)
- Y轴: ID Avg Acc
- 曲线1: 固定集成（实线）+ 标准误差带（阴影）
- 曲线2: 自适应路由（虚线）+ 标准误差带（阴影）
- 观察: 固定集成快速上升，自适应路由平缓

**图2: OOD Avg Acc随alpha变化（双曲线+误差带）**
- X轴: α (0.5 ~ 1.0)
- Y轴: OOD Avg Acc
- 曲线1: 固定集成（实线）+ 标准误差带（阴影）
- 曲线2: 自适应路由（虚线）+ 标准误差带（阴影）
- 观察: 固定集成快速下降，自适应路由平缓

**图3: Overall Acc随alpha变化（双曲线+误差带）**
- X轴: α (0.5 ~ 1.0)
- Y轴: Overall Acc
- 曲线1: 固定集成（实线）+ 标准误差带（阴影）
- 曲线2: 自适应路由（虚线）+ 标准误差带（阴影）
- 标记固定集成的峰值（如α=0.8）
- 观察: 固定集成有峰值，自适应路由平缓且更高

#### 运行命令示例

**Combo 1 - 固定集成（多alpha）**:
```bash
for alpha in $(seq 0.50 0.025 1.00); do
    alpha_str=$(echo $alpha | tr '.' '_')
    python scripts/run_cached_experiment.py \
        --id_datasets aircraft caltech101 dtd eurosat flowers \
        --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
        --classifier_type ensemble \
        --alpha $alpha \
        --output_dir experiments/table5/combo1_ensemble_alpha${alpha_str}
done
```

**Combo 1 - 自适应路由（多alpha）**:
```bash
for alpha in $(seq 0.50 0.025 1.00); do
    alpha_str=$(echo $alpha | tr '.' '_')
    python scripts/run_cached_experiment.py \
        --id_datasets aircraft caltech101 dtd eurosat flowers \
        --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
        --enable_routing \
        --alpha $alpha \
        --ood_threshold 0.85 \
        --output_dir experiments/table5/combo1_routing_alpha${alpha_str}
done
```

**注意**: 需要对Combo 2-5重复上述命令

#### 结果汇总格式

**表格: 不同alpha下的性能（5组平均±标准误）**

| Alpha | 固定集成_ID | 固定集成_OOD | 固定集成_Overall | 自适应路由_ID | 自适应路由_OOD | 自适应路由_Overall |
|-------|-------------|--------------|------------------|---------------|----------------|--------------------|
| 0.50 | 60.2±1.5% | 84.8±2.1% | 72.5±1.8% | 75.1±1.2% | 78.3±1.5% | 76.7±1.3% |
| 0.60 | 65.5±1.3% | 78.2±1.9% | 71.9±1.6% | 75.3±1.1% | 79.1±1.4% | 77.2±1.2% |
| 0.70 | 70.8±1.2% | 72.5±1.8% | 71.7±1.5% | 75.6±1.0% | 79.8±1.3% | 77.7±1.1% |
| 0.80 | 74.2±1.0% | 68.3±1.6% | 71.3±1.3% | 76.0±0.9% | 80.2±1.2% | 78.1±1.0% |
| 0.90 | 77.5±0.9% | 50.2±2.5% | 63.9±1.7% | 76.3±0.9% | 80.8±1.1% | 78.6±1.0% |
| 1.00 | 79.8±0.8% | 32.5±3.2% | 56.2±2.0% | 76.8±0.8% | 81.5±1.0% | 79.2±0.9% |

**解读**:
- 固定集成: ID随alpha上升，OOD随alpha下降，Overall在α=0.8左右达到峰值
- 自适应路由: ID和OOD都保持稳定，Overall始终高于固定集成的峰值

---

## 6. 实验执行状态

### 6.1 需要重新执行的实验

**所有表格都需要重新执行！**

| 表格 | 内容 | 状态 | 预计时间 | 依赖 | 阶段 |
|------|------|------|----------|------|------|
| **Table 3** | OOD检测器对比 | ❌ 待执行 | ~2小时 | 无 | Phase 1 |
| **Table 4** | 推理端消融 | ❌ 待执行 | ~1小时 | Table 3阈值 | Phase 1 |
| **Table 5** | Alpha敏感性 | ❌ 待执行 | ~3小时 | 无 | Phase 1 |
| **Table 2** | 训练端消融 | ❌ 待执行 | ~6小时 | 无 | Phase 2 |
| **Table 1-A** | Zero-shot | ❌ 待执行 | ~30分钟 | 无 | Phase 3 |
| **Table 1-B** | Pretrain + Routing | ❌ 待执行 | ~3小时 | Table 3阈值 | Phase 3 |
| **Table 1-C** | LoRA Vanilla | ❌ 待执行 | ~8小时 | 无 | Phase 3 |
| **Table 1-D** | LoRA-NSP | ❌ 待执行 | ~8小时 | 无 | Phase 3 |
| **Table 1-E** | LoRA-NSP Full | ❌ 待执行 | ~3小时 | Table 1-D完成 | Phase 3 |

### 6.2 优化后的执行顺序

**新策略**: 先做消融实验选择超参数，再做主实验

**Phase 0: 预准备 (2h)**
- 特征提取 (必需)

**Phase 1: 推理端消融 (6h)** 🔴 最高优先级
- Table 3 → 确定最佳OOD检测器和阈值
- Table 4 → 验证路由有效性
- Table 5 → 确定最佳alpha值
- **产出**: 超参数配置

**Phase 2: 训练端消融 (6h)** 🟡 中优先级
- Table 2 → 验证NSP+FD+CD最优
- **产出**: 训练策略验证

**Phase 3: 主实验 (11h)** 🔴 最高优先级
- Table 1 (A/B/C/D/E)
- 使用方法B/E前必须完成Table 3
- **产出**: 论文主结果

**总时间**: ~25小时（vs 原40小时）

**关键依赖链**:
```
Phase 0: 特征提取
    ↓
Phase 1: Table 3 → 最佳阈值
    ↓
Phase 1: Table 4/5 → 验证超参数
    ↓
Phase 3: Table 1 (B/E) → 使用最佳阈值
```

### 6.3 执行检查清单

#### 启动Phase 1前检查:
- [ ] 设置 `CUDA_VISIBLE_DEVICES=0,1,2,4,5`
- [ ] 完成特征提取 (`cache/pretrained_features/`存在)
- [ ] 确认脚本版本正确 (`_v2.py`)

#### 启动Phase 3前检查:
- [ ] Table 3完成并获得最佳阈值
- [ ] Table 4/5完成并确认超参数
- [ ] 创建 `experiments/main/` 目录

#### 方法C/D执行前检查:
- [ ] 方法C: 确认 `--fd_weight 0 --cd_weight 0`
- [ ] 方法D: 确认 `--fd_weight 1.0 --cd_weight 1.0`

---

## 7. 关键问题总结

### 7.1 之前的错误理解 ❌

1. **固定ID/OOD划分**: 错误地将主实验理解为固定划分
2. **割裂评估**: 单独评估ID分类和OOD检测
3. **忽略增量特性**: 没有体现持续学习的动态特性

### 7.2 正确的理解 ✅

1. **动态ID/OOD**: 每个步骤后，已见=ID，未见=OOD
2. **统一评估**: 所有样本经过相同流程，只看最终分类准确率
3. **完整增量场景**: 体现持续学习的核心挑战

### 7.3 核心设计思想

**正交性**: 训练端和推理端创新相互独立
- 可以单独使用方法B（不训练，只加路由）
- 可以单独使用方法D（训练，不加路由）
- 也可以使用方法E（训练+路由）

**关键验证**:
- 方法B验证推理端创新的独立价值
- 方法D验证训练端创新的独立价值
- 方法E验证两者的协同效果

---

## 8. 下一步行动（按优化后的执行策略）

### Phase 0: 预准备（立即执行）
1. ✅ 确认实验设计正确
2. 🔄 设置环境变量: `export CUDA_VISIBLE_DEVICES=0,1,2,4,5`
3. 🔄 提取特征缓存: `python scripts/extract_cached_features.py ...`

### Phase 1: 推理端消融（优先执行）
4. 🔄 **Table 3**: OOD检测器对比（5个GPU并行，~2小时）
   - 产出: 最佳OOD检测器类型 + 阈值
5. 🔄 **Table 4**: 推理端消融（~1小时）
   - 验证自适应路由优势
6. 🔄 **Table 5**: Alpha敏感性（~3小时）
   - 确认alpha=0.8为最佳值

### Phase 2: 训练端消融
7. 🔄 **Table 2**: 组件贡献分析（~6小时）
   - 验证NSP+FD+CD最优

### Phase 3: 主实验
8. 🔄 **Table 1-A**: Zero-shot（可复用Table 4结果）
9. 🔄 **Table 1-B**: Pretrain + Routing（使用Table 3阈值）
10. 🔄 **Table 1-C**: LoRA Vanilla（注意: --fd_weight 0 --cd_weight 0）
11. 🔄 **Table 1-D**: LoRA-NSP（注意: --fd_weight 1.0 --cd_weight 1.0）
12. 🔄 **Table 1-E**: LoRA-NSP Full（等待D完成）

---

## 9. 实验启动模板

### 快速启动命令

```bash
# ====== Phase 0: 预准备 ======
export CUDA_VISIBLE_DEVICES=0,1,2,4,5
mkdir -p experiments/main logs cache/pretrained_features results

# 提取特征（只需执行一次，~2小时）
python scripts/extract_cached_features.py \
    --datasets aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --cache_dir cache/pretrained_features \
    --device cuda

# ====== Phase 1: Table 3 (5个GPU并行) ======
# Combo 1-5，每个GPU一个组合
CUDA_VISIBLE_DEVICES=0 python scripts/run_ood_detector_eval.py \
    --id_datasets aircraft caltech101 dtd eurosat flowers \
    --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
    --detector lr_rgda \
    --output_dir experiments/table3/combo1_lr_rgda &
# ... 其他4个组合类似

# ====== Phase 1: Table 4 ======
python scripts/run_cached_experiment.py \
    --cache_dir cache/pretrained_features \
    --id_datasets aircraft caltech101 dtd eurosat flowers \
    --ood_datasets food101 mnist oxford_pets stanford_cars sun397 \
    --enable_routing \
    --ood_threshold <BEST_FROM_TABLE3> \
    --output_dir experiments/table4/combo1_routing

# ====== Phase 3: Table 1 方法C ======
# ⚠️ 注意: 必须设置 fd_weight=0 cd_weight=0
CUDA_VISIBLE_DEVICES=2 python src/experiments/run_continual_learning.py \
    --method lora_vanilla \
    --fd_weight 0 --cd_weight 0 \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --output_dir experiments/main/C_lora_vanilla \
    > logs/main_C.log 2>&1 &

# ====== Phase 3: Table 1 方法D ======
# ⚠️ 注意: 必须设置 fd_weight=1.0 cd_weight=1.0
CUDA_VISIBLE_DEVICES=4 python src/experiments/run_continual_learning.py \
    --method lora_nsp \
    --fd_weight 1.0 --cd_weight 1.0 \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --output_dir experiments/main/D_lora_nsp_only \
    > logs/main_D.log 2>&1 &

# ====== Phase 3: Table 1 方法E ======
# ⚠️ 依赖方法D完成
CUDA_VISIBLE_DEVICES=5 python src/experiments/run_continual_learning_routing_v2.py \
    --method lora_nsp_full \
    --load_checkpoint experiments/main/D_lora_nsp_only/final_model.pt \
    --ood_threshold <BEST_FROM_TABLE3> \
    --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --output_dir experiments/main/E_lora_nsp_full \
    > logs/main_E.log 2>&1
```

### 监控脚本

```bash
# 查看所有实验进程
watch -n 5 'ps aux | grep "python.*run_" | grep -v grep'

# 查看GPU使用情况
watch -n 5 nvidia-smi

# 查看日志
 tail -f logs/main_*.log
```

---

**文档结束**

---

## 附录: Table 1详细结果呈现格式

### 结果呈现方式（参考LADA论文）

Table 1应该呈现**完整的准确率矩阵**和**逐数据集结果**，而不仅仅是汇总指标。

#### 1. 完整准确率矩阵

以方法D（LoRA-NSP）为例，训练完所有10个任务后的准确率矩阵：

```
         aircraft  caltech101  dtd   eurosat  flowers  food101  mnist  oxford_pets  stanford_cars  sun397   Average
Step 0:    23.8%      74.3%    36.4%   46.5%   67.2%   65.1%   47.3%     85.2%        18.9%      38.5%    50.3%
Step 1:    65.2%      72.1%    35.8%   45.2%   66.5%   64.8%   46.9%     84.1%        18.2%      37.8%    53.7%
Step 2:    68.5%      78.3%    42.1%   44.8%   65.9%   64.2%   46.5%     83.5%        17.9%      37.2%    54.9%
Step 3:    70.1%      79.5%    48.6%   89.2%   65.3%   63.7%   46.1%     82.8%        17.5%      36.8%    60.0%
Step 4:    71.2%      80.2%    49.2%   90.5%   85.7%   63.1%   45.8%     82.1%        17.1%      36.3%    62.1%
Step 5:    71.8%      80.8%    49.8%   91.2%   86.5%   82.4%   45.4%     81.5%        16.8%      35.9%    64.2%
Step 6:    72.1%      81.2%    50.1%   91.8%   87.2%   83.5%   95.2%     80.8%        16.4%      35.4%    69.4%
Step 7:    72.5%      81.5%    50.5%   92.1%   87.8%   84.2%   95.8%     89.3%        16.1%      34.9%    70.5%
Step 8:    72.8%      81.8%    50.8%   92.5%   88.1%   84.8%   96.2%     89.7%        42.5%      34.5%    73.4%
Step 9:    73.1%      82.1%    51.2%   92.8%   88.5%   85.2%   96.5%     90.1%        43.2%      72.8%    77.6%
Step 10:   73.5%      82.5%    51.5%   93.2%   88.9%   85.8%   96.8%     90.5%        43.8%      73.5%    78.0%
```

**解读**:
- 行：训练步骤（Step 0是零样本基线）
- 列：10个数据集
- 可以看到每个数据集在学习过程中的性能变化
- 对角线附近：新学习任务的性能提升
- 非对角线：已学任务的遗忘情况

#### 2. 汇总表格（参考LADA Table 1格式）

**Table 1: 主实验结果**

| 方法 | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | Transfer | Average | Last | Forgetting |
|------|----------|------------|-----|---------|---------|---------|-------|-------------|---------------|--------|----------|---------|------|------------|
| **A** Zero-shot | 23.8 | 74.3 | 36.4 | 46.5 | 67.2 | 65.1 | 47.3 | 85.2 | 18.9 | 38.5 | 57.7 | - | - | - |
| **B** Pretrain+Routing | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| **C** LoRA Vanilla | - | - | - | - | - | - | - | - | - | - | ~55 | ~65 | ~70 | ~15 |
| **D** LoRA-NSP | - | - | - | - | - | - | - | - | - | - | ~60 | ~72 | ~80 | <10 |
| **E** LoRA-NSP Full | - | - | - | - | - | - | - | - | - | - | ~60 | ~72 | ~80 | <10 |
| LADA (paper) | - | - | - | - | - | - | - | - | - | - | 61.5 | 72.7 | 83.1 | - |

**注**: 数值为训练完所有10个任务后（Step 10），在各数据集上的准确率（Last列对应的是Average of Last Row）

#### 3. 关于ID Acc和OOD Acc的说明

**澄清**: 在主实验（Table 1）中，我们不单独关注"ID Acc"和"OOD Acc"这两个指标。

**原因**:
- 在持续学习场景中，ID/OOD是动态变化的（每个步骤都不同）
- 强行定义固定的ID Acc和OOD Acc会混淆实验设计
- 我们应该关注LADA指标（Transfer, Average, Last, Forgetting）和逐数据集结果

**正确的关注点**:
1. **逐数据集准确率**: 每个任务在学习过程中的性能变化
2. **Last指标**: 训练完所有任务后，各任务的最终性能
3. **Forgetting指标**: 各任务的遗忘程度
4. **Transfer指标**: 前向迁移能力
5. **Average指标**: 综合性能

**ID/OOD性能分析留给Table 4**:
- Table 4专门研究固定划分下的ID/OOD性能平衡
- Table 1专注于持续学习的整体性能

#### 4. 可视化建议

**图1: 准确率热力图（Accuracy Matrix Heatmap）**
- X轴: 10个数据集
- Y轴: 10个训练步骤（Step 0-10）
- 颜色: 准确率（深蓝=低，红色=高）
- 目的: 直观展示学习过程和遗忘情况

**图2: 逐数据集性能曲线**
- X轴: 训练步骤
- Y轴: 准确率
- 曲线: 每个数据集一条线
- 目的: 展示每个任务的学习动态

**图3: 遗忘率对比（Bar Chart）**
- X轴: 10个数据集
- Y轴: Forgetting (%)
- 分组: 方法C, D, E
- 目的: 对比不同方法的抗遗忘能力
