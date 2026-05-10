"""
共享工具函数：被 main_joint.py 和 main_incremental.py 共同使用

包含：
- fix_random_seed: 设置随机种子
- get_zeroshot_classifier: 构建零样本分类器权重
- evaluate_dataset: 在单个数据集上评估 ZS / RGDA / Ensemble 准确率
- get_full_stats: 从准确率矩阵计算 Transfer/Average/Last 指标
- print_paper_metrics: 打印论文风格的指标报告
"""

import os
import random
import numpy as np
import torch


def fix_random_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    print(f"Setting fixed seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_zeroshot_classifier(model, processor, class_names, device):
    """构建零样本分类器"""
    templates = [lambda x: f"a photo of a {x}."]
    zeroshot_weights = []
    with torch.no_grad():
        for classname in class_names:
            classname = classname.replace('_', ' ')
            texts = [template(classname) for template in templates]
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            class_embeddings = model.get_text_features(**text_inputs)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1).to(device)


def evaluate_dataset(args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                     current_num_classes, eval_label_offset,
                     alpha_sensitivity=False, n_alpha_samples=21):
    """
    在单个数据集上评估 ZS / RGDA / Ensemble 准确率

    Args:
        args: 全局参数
        d_name: 数据集名称
        model: CLIP 模型
        zeroshot_classifier: 零样本分类器权重 [D, num_classes]
        lr_rgda_classifier: LR-RGDA 分类器实例
        current_num_classes: 当前 ID 类别总数（用于 ensemble 时只在 ID 区域叠加 RGDA）
        eval_label_offset: 评估时的标签偏移量
        alpha_sensitivity: 是否对 alpha 做敏感性分析（枚举多个 alpha 值）
        n_alpha_samples: 敏感性分析时 alpha 的采样点数（默认 21，即 0, 0.05, ..., 1.0）

    Returns:
        (zs_acc, rgda_acc, ens_acc, num_classes_in_dataset, sensitivity_list)
        其中 sensitivity_list 为 [(alpha, acc), ...] 或 None
    """
    from utils_data import get_xtail_trainloader, get_transforms
    from src.utils.feature_extractor import extract_features

    _, test_transform = get_transforms(d_name)
    _, te_loader, _, c_names = get_xtail_trainloader(
        root=args.root, dataset_name=d_name,
        transform_train=None, transform_test=test_transform,
        num_shots=args.num_shots, batch_size=args.batch_size)

    features, labels = extract_features(model, te_loader, args.device)

    # 确保特征严格进行 L2 归一化
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.to(args.device)
    labels = (labels + eval_label_offset).to(args.device)

    with torch.no_grad():
        # Zero-shot 预测（不乘 logit_scale，与 debug_classifier_router.py 一致）
        zs_logits = features @ zeroshot_classifier
        zs_logits_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
        zs_preds = zs_logits_norm.argmax(dim=1)
        zs_acc = zs_preds.eq(labels).float().mean().item() * 100

        # 2. 纯 LR-RGDA 预测
        rgda_logits = lr_rgda_classifier.forward(features)
        rgda_logits_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
        rgda_preds = rgda_logits_norm.argmax(dim=1)
        rgda_acc = rgda_preds.eq(labels).float().mean().item() * 100

        # 3. Ensemble 预测 (1-alpha)*ZS + alpha*RGDA
        ensemble_logits = zs_logits_norm * (1 - args.alpha)
        ensemble_logits[:, :current_num_classes] += args.alpha * rgda_logits_norm
        ens_preds = ensemble_logits.argmax(dim=1)
        ens_acc = ens_preds.eq(labels).float().mean().item() * 100

        # 4. Alpha 敏感性分析（可选）
        sensitivity_list = None
        if alpha_sensitivity:
            sensitivity_list = []
            for alpha in torch.linspace(0, 1.0, n_alpha_samples):
                ens_logits = zs_logits_norm * (1 - alpha)
                ens_logits[:, :current_num_classes] += alpha * rgda_logits_norm
                ens_preds = ens_logits.argmax(dim=1)
                ens_acc_alpha = ens_preds.eq(labels).float().mean().item() * 100
                sensitivity_list.append((round(alpha.item(), 3), round(ens_acc_alpha, 2)))

    return zs_acc, rgda_acc, ens_acc, len(c_names), sensitivity_list


def batch_evaluate_datasets(
    dataset_names, model, processor, zeroshot_classifier, lr_rgda_classifier,
    num_id_classes, args=None, root=None, num_shots=16, batch_size=32, device='cuda',
    alpha=0.5, alpha_sensitivity=False, n_alpha_samples=21
):
    """
    批处理评估：先收集所有数据集的 feature 和 label 拼到一起，再统一评估。
    参考 debug_classifier_router.py 的评估逻辑。

    Args:
        dataset_names: 数据集名称列表
        model: CLIP 模型
        processor: CLIP processor
        zeroshot_classifier: 零样本分类器权重 [D, num_classes]
        lr_rgda_classifier: LR-RGDA 分类器实例
        num_id_classes: ID 类别总数
        args: 可选，全局参数对象（与下面独立参数二选一）
        root: 数据根目录
        num_shots: few-shot 数
        batch_size: batch size
        device: 设备
        alpha: 集成权重
        alpha_sensitivity: 是否做 α 敏感性分析
        n_alpha_samples: α 采样点数

    Returns:
        {
            "per_dataset": {d_name: {"zs": zs, "rgda": rgda, "ens": ens}, ...},
            "overall": {"zs": zs, "rgda": rgda, "ens": ens},
            "sensitivity": [(alpha, overall_acc), ...] | None
        }
    """
    # 解析参数（兼容 args 对象和独立传参）
    if args is not None:
        root = args.root
        num_shots = args.num_shots
        batch_size = args.batch_size
        device = args.device
        alpha = args.alpha

    from utils_data import get_xtail_trainloader, get_transforms
    from src.utils.feature_extractor import extract_features

    # 收集所有数据集的 features 和 labels
    all_features, all_labels = [], []
    per_dataset = {}
    offset = 0

    for d_name in dataset_names:
        _, test_transform = get_transforms(d_name)
        _, te_loader, _, c_names = get_xtail_trainloader(
            root=root, dataset_name=d_name,
            transform_train=None, transform_test=test_transform,
            num_shots=num_shots, batch_size=batch_size
        )
        features, labels = extract_features(model, te_loader, device)
        features = features / features.norm(dim=-1, keepdim=True)

        all_features.append(features)
        all_labels.append(labels + offset)
        offset += len(c_names)

    all_features = torch.cat(all_features).to(device)
    all_labels = torch.cat(all_labels).to(device)

    with torch.no_grad():
        # Zero-shot logits（不乘 logit_scale，与 debug_classifier_router.py 一致）
        zs_logits = all_features @ zeroshot_classifier
        zs_logits_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
        zs_preds = zs_logits_norm.argmax(dim=1)
        zs_overall = zs_preds.eq(all_labels).float().mean().item() * 100

        # 2. LR-RGDA logits
        rgda_logits = lr_rgda_classifier.forward(all_features)
        rgda_logits_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
        rgda_preds = rgda_logits_norm.argmax(dim=1)
        rgda_overall = rgda_preds.eq(all_labels).float().mean().item() * 100

        # 3. Ensemble
        ensemble_logits = zs_logits_norm * (1 - alpha)
        ensemble_logits[:, :num_id_classes] += alpha * rgda_logits_norm
        ens_preds = ensemble_logits.argmax(dim=1)
        ens_overall = ens_preds.eq(all_labels).float().mean().item() * 100

        # 4. Alpha 敏感性分析（在合并特征上统一扫描）
        sensitivity = None
        if alpha_sensitivity:
            sensitivity = []
            for a in torch.linspace(0, 1.0, n_alpha_samples):
                ens_logits = zs_logits_norm * (1 - a)
                ens_logits[:, :num_id_classes] += a * rgda_logits_norm
                ens_preds = ens_logits.argmax(dim=1)
                acc = ens_preds.eq(all_labels).float().mean().item() * 100
                sensitivity.append((round(a.item(), 3), round(acc, 2)))

    return {
        "overall": {"zs": zs_overall, "rgda": rgda_overall, "ens": ens_overall},
        "sensitivity": sensitivity,
    }


def get_full_stats(matrix):
    """
    从准确率矩阵计算 Transfer/Average/Last 指标

    Args:
        matrix: 二维列表
                - 单行 (联合微调): [[acc1, acc2, ..., accN]]
                - 多行 (增量微调): [[...], [...], ..., [...]] (T x N)

    Returns:
        dict 包含 raw_matrix, transfer, transfer_total_avg,
              average_per_task, average_total_avg, last, last_total_avg
    """
    num_rows = len(matrix)
    if num_rows == 1:
        # 联合微调：只有一行，所有指标相同
        data_row = matrix[0]
        return {
            "raw_matrix": matrix,
            "transfer": data_row,
            "transfer_total_avg": sum(data_row) / len(data_row),
            "average_per_task": data_row,
            "average_total_avg": sum(data_row) / len(data_row),
            "last": data_row,
            "last_total_avg": sum(data_row) / len(data_row)
        }
    else:
        # 增量学习：多行矩阵
        trans = [matrix[k][k] for k in range(num_rows)]
        lasts = matrix[-1]
        avgs = [sum(matrix[i][j] for i in range(j, num_rows)) / (num_rows - j)
                for j in range(num_rows)]
        return {
            "raw_matrix": matrix,
            "transfer": trans,
            "transfer_total_avg": sum(trans) / num_rows,
            "average_per_task": avgs,
            "average_total_avg": sum(avgs) / num_rows,
            "last": lasts,
            "last_total_avg": sum(lasts) / num_rows
        }


def print_paper_metrics(matrix, name, headers):
    """打印论文风格的指标报告"""
    stats = get_full_stats(matrix)
    num_cols = len(matrix[0])

    print(f"\n" + "-" * 110)
    print(f"[{name} 指标报告]")
    # 自动截取对应的表头
    header_str = " | ".join([f"{h[:8]:<8}" for h in headers[:num_cols]])
    print(f"指标类型   | " + header_str + " | [平均总分]")
    print("-" * 110)

    print(f"Transfer  | " + " | ".join([f"{x:8.1f}" for x in stats['transfer']])
          + f" | [{stats['transfer_total_avg']:.1f}]")
    print(f"Average   | " + " | ".join([f"{x:8.1f}" for x in stats['average_per_task']])
          + f" | [{stats['average_total_avg']:.1f}]")
    print(f"Last      | " + " | ".join([f"{x:8.1f}" for x in stats['last']])
          + f" | [{stats['last_total_avg']:.1f}]")
    print("-" * 110)
