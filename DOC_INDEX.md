# 文档索引

快速找到您需要的文档

---

## 🚀 我想开始运行实验

| 场景 | 推荐阅读 |
|------|----------|
| 第一次使用项目 | [README.md](./README.md) |
| 运行完整流程 | [scripts/README.md](./scripts/README.md) |
| 运行增量学习 | [scripts/workflows/README.md](./scripts/workflows/README.md) |

---

## 📝 我想了解技术细节（论文写作）

| 内容 | 推荐阅读 |
|------|----------|
| **LoRA-NSP算法详解** | [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) |
| **实现笔记（快速参考）** | [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) ⭐ |
| **配置管理系统** | [configs/README.md](./configs/README.md) ⭐ |
| 数学公式和推导 | TECHNICAL_DOCUMENTATION.md 第2章 |
| 算法流程 | TECHNICAL_DOCUMENTATION.md 第3章 |
| 实现细节 | TECHNICAL_DOCUMENTATION.md 第4章 |
| 配置系统使用 | configs/README.md |
| 使用场景速查 | IMPLEMENTATION_NOTES.md 第6章 |
| 实验设计建议 | TECHNICAL_DOCUMENTATION.md 第5章 |
| 对比相关工作 | TECHNICAL_DOCUMENTATION.md 第6章 |

---

## 🔧 我想了解项目架构

| 内容 | 推荐阅读 |
|------|----------|
| 整体架构设计 | [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) 第2-3章 |
| 模块依赖关系 | PROJECT_DOCUMENTATION.md 第10章 |
| API参考 | PROJECT_DOCUMENTATION.md 第9章 |
| 设计决策讨论 | PROJECT_DOCUMENTATION.md 第6.2节 |

---

## ✅ 我想了解完成进度

| 内容 | 推荐阅读 |
|------|----------|
| 已完成功能 | [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) 第4章 |
| 待完成任务 | PROJECT_DOCUMENTATION.md 第5章 |
| 下一步计划 | PROJECT_DOCUMENTATION.md 第6.4节 |

---

## 📊 文档对应论文章节

| 论文章节 | 参考文档 | 参考章节 |
|----------|----------|----------|
| Introduction | PROJECT_DOCUMENTATION.md | 第1章 |
| **Method** | **TECHNICAL_DOCUMENTATION.md** | **全文** |
| - LoRA基础 | TECHNICAL_DOCUMENTATION.md | 2.1节 |
| - 零空间参数化 | TECHNICAL_DOCUMENTATION.md | 2.2节 |
| - 协方差维护 | TECHNICAL_DOCUMENTATION.md | 2.3节 |
| - 增量学习流程 | TECHNICAL_DOCUMENTATION.md | 第3章 |
| **Implementation** | **TECHNICAL_DOCUMENTATION.md** | **第4章** |
| Experiments | TECHNICAL_DOCUMENTATION.md | 第5章 |
| - 消融实验设计 | TECHNICAL_DOCUMENTATION.md | 5.3节 |
| Related Work | TECHNICAL_DOCUMENTATION.md | 第6章 |

---

## 🔍 快速搜索

### 关键词索引

| 关键词 | 所在文档 | 位置 |
|--------|----------|------|
| LoRA | TECHNICAL_DOCUMENTATION.md | 2.1节 |
| Null Space | TECHNICAL_DOCUMENTATION.md | 2.2节 |
| 协方差 | TECHNICAL_DOCUMENTATION.md | 2.3节, 4.2节 |
| 滑动平均 | TECHNICAL_DOCUMENTATION.md | 2.3.2节 |
| 投影矩阵 | TECHNICAL_DOCUMENTATION.md | 2.2.2节, 4.1节 |
| 增量学习 | TECHNICAL_DOCUMENTATION.md | 第3章 |
| 灾难性遗忘 | TECHNICAL_DOCUMENTATION.md | 1.1节 |
| 零样本分类 | PROJECT_DOCUMENTATION.md | 1.2节 |
| LR-RGDA | PROJECT_DOCUMENTATION.md | 3.2节 |
| OOD检测 | PROJECT_DOCUMENTATION.md | 3.3节 |
| 自适应路由 | PROJECT_DOCUMENTATION.md | 3.4节 |
| 统计分布 | PROJECT_DOCUMENTATION.md | 3.1节 |
| 核心脚本 | scripts/README.md | 核心脚本部分 |
| 工作流 | scripts/workflows/README.md | 全文 |

---

## 📊 我想了解优化结果

| 内容 | 推荐阅读 |
|------|----------|
| **超参数优化报告** | [OPTIMIZATION_REPORT.md](./OPTIMIZATION_REPORT.md) ⭐ |
| 优化实验记录 | [OPTIMIZATION_LOG.md](./OPTIMIZATION_LOG.md) |

## 🧪 我想运行论文实验

| 内容 | 推荐阅读 |
|------|----------|
| **实验计划与方案** | [EXPERIMENTAL_PLAN.md](./EXPERIMENTAL_PLAN.md) ⭐ |
| 实验配置文件 | [configs/experiments/README.md](./configs/experiments/README.md) |
| 批量运行脚本 | `scripts/run_paper_experiments.py` |

---

## 📁 文件清单

```
clip_ood/
├── DOC_INDEX.md                   <- 本文档（导航索引）
├── README.md                      # 项目简介和快速开始
├── PROJECT_DOCUMENTATION.md       # 项目整体架构和进度
├── TECHNICAL_DOCUMENTATION.md     # 技术细节和论文素材 ⭐
├── OPTIMIZATION_REPORT.md         # 超参数优化报告 ⭐
├── OPTIMIZATION_LOG.md            # 优化实验记录
│
├── scripts/
│   ├── README.md                  # 脚本使用指南
│   ├── workflows/
│   │   └── README.md              # 增量学习详细说明
│   └── ...
│
└── ...
```

---

## 💡 使用建议

### 如果你是开发者
1. 先看 [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) 了解架构
2. 再看 [scripts/README.md](./scripts/README.md) 学习使用脚本
3. 查阅 [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) 快速参考
4. 深入阅读 [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) 理解算法

### 如果你是论文作者
1. **直接阅读** [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) 全文
2. 第2-3章提供方法部分的完整描述
3. 第4章提供实现细节
4. 第5章提供实验设计建议
5. 第6章提供相关工作对比

### 如果你是审阅者
1. 阅读 [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) 第1章了解项目目标
2. 阅读 [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) 第2-3章评估方法创新性
3. 查看代码实现与文档的一致性

---

*最后更新: 2026-03-14*
