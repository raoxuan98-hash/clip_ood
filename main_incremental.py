"""
增量微调 (Incremental Fine-tuning) 入口脚本

功能：
1. 按 dataset_sequence 逐任务增量训练
2. 每个任务进行 LoRA-NSP 训练 → 零空间投影 → 特征提取 → LR-RGDA 分类器构建
3. 自动评估所有已学任务，记录 Transfer / Average / Last 指标矩阵
4. 保存结果 JSON

用法示例：
    python main_incremental.py \\
        --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \\
        --num_shots 16 --batch_size 32 --iterations 800 \\
        --lora_type lora_nsp --alpha 0.5
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import argparse
import json
import logging
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.detectors.ood_detector import build_stats_dict_from_features
from src.utils.reference_loader import load_reference_dataset
from src.utils.main_utils import (
    fix_random_seed,
    get_zeroshot_classifier,
    evaluate_dataset,
    get_full_stats,
    print_paper_metrics,
)
from utils_data import get_xtail_trainloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Fine-tuning for CLIP Continual Learning")

    # 数据集相关参数
    parser.add_argument("--id_datasets", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="List of all ID datasets (for reference).")
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL",
                        help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16,
                        help="Number of shots for few-shot learning.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and testing.")
    parser.add_argument("--max_num_per_test_dataset", type=int, default=1000,
                        help="Maximum number of samples per test dataset.")

    # 增量学习特定参数
    parser.add_argument("--dataset_sequence", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="Sequence of datasets, one task per dataset. "
                             "Each element is a single dataset name.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training.")
    parser.add_argument("--iterations", type=int, default=800,
                        help="Number of training iterations per task.")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=3e-5,
                        help="Weight decay for optimizer.")

    # LoRA 相关参数
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_type", type=str, default="lora_nsp",
                        choices=["lora_sgp", "lora_nsp"],
                        help="Type of LoRA adaptation.")
    parser.add_argument("--nsp_eps", type=float, default=0.05,
                        help="Epsilon parameter for NSP.")
    parser.add_argument("--nsp_weight", type=float, default=0.02,
                        help="Weight parameter for NSP.")
    parser.add_argument("--weight_temp", type=float, default=1.0,
                        help="Temperature parameter for weight.")
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0,
                        help="P parameter for weight function.")

    # 参考数据集参数
    parser.add_argument("--reference_dataset", type=str, default="flickr8k",
                        help="Reference dataset for training.")
    parser.add_argument("--reference_batch_size", type=int, default=32,
                        help="Batch size for reference dataset.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")

    # 损失函数权重参数
    parser.add_argument("--fd_weight", type=float, default=1.0,
                        help="Weight for feature distillation loss.")
    parser.add_argument("--cd_weight", type=float, default=1.0,
                        help="Weight for cross-modal distillation loss.")

    # 分类器参数
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for LR-RGDA classifier in ensemble.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for zero-shot classifier.")

    # LR-RGDA 构建参数
    parser.add_argument("--rgda_rank", type=int, default=32,
                        help="Rank for LR-RGDA low-rank decomposition.")
    parser.add_argument("--rgda_alpha1", type=float, default=0.6,
                        help="qda_reg_alpha1 for LR-RGDA.")
    parser.add_argument("--rgda_alpha2", type=float, default=1.0,
                        help="qda_reg_alpha2 for LR-RGDA.")
    parser.add_argument("--rgda_alpha3", type=float, default=0.5,
                        help="qda_reg_alpha3 for LR-RGDA.")

    args = parser.parse_args()
    # 将 dataset_sequence 转换为嵌套列表格式 [[d1], [d2], ...]
    args.dataset_sequence = [[d] for d in args.dataset_sequence]
    return args


def main(args):
    if args.seed is not None:
        fix_random_seed(args.seed)

    # ========== 1. 初始化 ==========
    logging.info("\n=== Initializing Incremental Learning ===")
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                              processor, args.device)

    # ========== 2. 增量学习循环 ==========
    history_class_names = []  # 记录所有已学类名列表的列表
    global_stats_dict = {}

    acc_matrix_zs = []
    acc_matrix_rgda = []
    acc_matrix_ens = []

    for i, task_datasets in enumerate(args.dataset_sequence):
        print(f"\n" + "=" * 50)
        print(f"=== Task {i+1}: {task_datasets} ===")
        print("=" * 50)

        # --- 2a. 准备训练数据 ---
        train_loaders = []
        task_class_names = []
        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            train_loaders.append(tr_loader)
            task_class_names.extend(c_names)

        merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
        # 用于提取协方差的 loader 不打乱顺序
        cov_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=False)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)

        # --- 2b. 训练模型 (LoRA-NSP) ---
        model = trainer.train(merged_loader, task_class_names, reference_loader)

        # --- 2c. 零空间投影 (NSP) 抗遗忘 ---
        print("\n=== Applying Null-Space Projection (NSP) ===")
        covariances = trainer.extract_layer_covariances(cov_loader)
        trainer.update_covariance_history(covariances)
        trainer.finalize_task_for_incremental()

        # --- 2d. 提取特征并构建统计字典 ---
        task_features = []
        task_labels = []
        label_offset = sum(len(c_names) for c_names in history_class_names)

        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            from src.utils.feature_extractor import extract_features
            features, labels = extract_features(model, tr_loader, args.device)
            features = features / features.norm(dim=-1, keepdim=True)
            task_features.append(features)
            task_labels.append(labels + label_offset)

        task_features = torch.cat(task_features)
        task_labels = torch.cat(task_labels)

        task_stats_dict = build_stats_dict_from_features(task_features, task_labels)

        # 累加统计字典
        global_stats_dict.update(task_stats_dict)
        history_class_names.append(task_class_names)

        # --- 2e. 构建分类器 ---
        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=global_stats_dict,
            device=args.device,
            rank=args.rgda_rank,
            qda_reg_alpha1=args.rgda_alpha1,
            qda_reg_alpha2=args.rgda_alpha2,
            qda_reg_alpha3=args.rgda_alpha3,
            temperature=1.0
        )

        flat_class_names = [name for sublist in history_class_names for name in sublist]
        current_num_classes = len(flat_class_names)
        zeroshot_classifier = get_zeroshot_classifier(model, processor,
                                                      flat_class_names, args.device)

        # --- 2f. 评估所有已学任务 ---
        print("\n=== Evaluating Task ===")
        step_accs_zs, step_accs_rgda, step_accs_ens = [], [], []

        eval_label_offset = 0
        for j in range(i + 1):
            eval_datasets = args.dataset_sequence[j]
            d_name = eval_datasets[0]  # 每个 Task 只有一个数据集

            zs_acc, rgda_acc, ens_acc, c_len, _ = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                current_num_classes, eval_label_offset
            )
            eval_label_offset += c_len

            print(f"[Tested on Task {j+1}: {d_name:<10s}] -> "
                  f"Zero-shot: {zs_acc:5.1f}% | LR-RGDA: {rgda_acc:5.1f}% | "
                  f"Ensemble: {ens_acc:5.1f}%")

            step_accs_zs.append(zs_acc)
            step_accs_rgda.append(rgda_acc)
            step_accs_ens.append(ens_acc)

        acc_matrix_zs.append(step_accs_zs)
        acc_matrix_rgda.append(step_accs_rgda)
        acc_matrix_ens.append(step_accs_ens)

    # ========== 3. 打印最终结果 ==========
    print("\n=== Training and Evaluation Completed ===")
    print("\n" + "=" * 80)
    print("Final Results Mapping to Paper Tables")
    print("=" * 80)

    task_names = [d[0] for d in args.dataset_sequence]

    print_paper_metrics(acc_matrix_zs, "Zero-shot Baseline", task_names)
    print_paper_metrics(acc_matrix_rgda, "LR-RGDA Only", task_names)
    print_paper_metrics(acc_matrix_ens, f"Ours Ensemble (alpha={args.alpha})", task_names)

    # ========== 4. 保存结果 JSON ==========
    save_results = {
        "mode": "Incremental Learning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_order": task_names,
        "arguments": vars(args),
        "metrics": {
            "zero_shot": get_full_stats(acc_matrix_zs),
            "lr_rgda": get_full_stats(acc_matrix_rgda),
            "ours_ensemble": get_full_stats(acc_matrix_ens),
        }
    }

    save_dir = "experiments"
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"incremental_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(save_dir, file_name)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=4, ensure_ascii=False)

    logging.info(f"\n增量学习结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
