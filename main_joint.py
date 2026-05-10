"""
联合微调 (Joint Fine-tuning) 入口脚本

功能：
1. 将 id_datasets 拼接成大数据集进行联合训练
2. 训练结束后，在 id_datasets 和 ood_datasets 上同时评估：
   - 零样本分类器 (Zero-shot) 准确率
   - 集成分类器 (Ensemble) 准确率
3. 保存结果 JSON

用法示例：
    python main_joint.py \\
        --id_datasets aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \\
        --ood_datasets dtd eurosat mnist sun397 \\
        --num_shots 16 --batch_size 32 --iterations 800 \\
        --lora_type lora_nsp --alpha 0.5
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # 屏蔽 transformers 版本提示

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
    batch_evaluate_datasets,
    get_full_stats,
    print_paper_metrics,
)
from utils_data import get_xtail_trainloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Fine-tuning for CLIP Continual Learning")

    # 数据集相关参数
    parser.add_argument("--id_datasets", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="List of ID datasets for training and ID evaluation.")
    parser.add_argument("--ood_datasets", type=str, nargs='+',
                        default=["dtd", "eurosat", "mnist", "sun397"],
                        help="List of OOD datasets for OOD evaluation.")
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL",
                        help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16,
                        help="Number of shots for few-shot learning.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and testing.")
    parser.add_argument("--max_num_per_test_dataset", type=int, default=1000,
                        help="Maximum number of samples per test dataset.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training.")
    parser.add_argument("--iterations", type=int, default=800,
                        help="Number of training iterations.")

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

    # 模型微调开关
    parser.add_argument("--tune_student", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to fine-tune the student model (default: True). "
                             "Set to False to skip training and only evaluate.")

    # Alpha 敏感性分析
    parser.add_argument("--alpha_sensitivity", action='store_true', default=False,
                        help="If set, evaluate ensemble accuracy at multiple alpha values "
                             "(0 to 1.0) to analyze sensitivity.")

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
    return args


def main(args):
    if args.seed is not None:
        fix_random_seed(args.seed)

    # ========== 1. 初始化 ==========
    logging.info("\n=== Initializing Joint Fine-tuning ===")
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    tune_student = args.tune_student

    # ========== 2. 微调模型（可选） ==========
    all_class_names = []
    if tune_student:
        # 只有在微调时才加载参考数据集
        reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                                  processor, args.device)

        # 2a. 准备联合训练数据
        logging.info("\n=== Preparing Joint Training Data ===")

        class LocalShiftDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, shift):
                self.base = base_dataset
                self.shift = shift
            def __getitem__(self, idx):
                img, label = self.base[idx]
                return img, label + self.shift
            def __len__(self):
                return len(self.base)

        all_shifted_datasets = []
        current_offset = 0

        for d_name in args.id_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            shifted_ds = LocalShiftDataset(tr_loader.dataset, current_offset)
            all_shifted_datasets.append(shifted_ds)
            all_class_names.extend(c_names)
            current_offset += len(c_names)

        merged_dataset = ConcatDataset(all_shifted_datasets)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)

        # 2b. 训练模型
        logging.info("\n=== Training (Joint Fine-tuning) ===")
        model = trainer.train(merged_loader, all_class_names, reference_loader)

        # 2c. 合并 LoRA 权重
        logging.info("\n=== Merging LoRA Weights for Joint Evaluation ===")
        trainer.finalize_task_for_incremental()

        # 2d. 保存 checkpoint
        checkpoint_dir = f"experiments/checkpoints/joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        trainer.save_checkpoint(checkpoint_path, all_class_names)
        logging.info(f"\nCheckpoint saved to: {checkpoint_path}")
    else:
        # 不微调，仅获取类别名
        logging.info("\n=== Skipping Fine-tuning (tune_student=False) ===")
        for d_name in args.id_datasets:
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            all_class_names.extend(c_names)

    # ========== 3. 提取特征，构建 LR-RGDA 分类器 ==========
    logging.info("\n=== Extracting Features for LR-RGDA ===")
    all_features = []
    all_labels = []
    feat_label_offset = 0

    for d_name in args.id_datasets:
        train_transform, _ = get_transforms(d_name)
        tr_loader, _, _, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=train_transform, transform_test=None,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        from src.utils.feature_extractor import extract_features
        features, labels = extract_features(model, tr_loader, args.device)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features)
        all_labels.append(labels + feat_label_offset)
        feat_label_offset += len(c_names)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    stats_dict = build_stats_dict_from_features(all_features, all_labels)

    lr_rgda_classifier = LRRGDAClassifier(
        stats_dict=stats_dict, device=args.device,
        rank=args.rgda_rank,
        qda_reg_alpha1=args.rgda_alpha1,
        qda_reg_alpha2=args.rgda_alpha2,
        qda_reg_alpha3=args.rgda_alpha3,
        temperature=1.0
    )

    num_id_classes = len(all_class_names)
    zeroshot_classifier = get_zeroshot_classifier(model, processor,
                                                  all_class_names, args.device)

    # ========== 4. 评估 ID 数据集 ==========
    logging.info("\n=== Evaluating ID Datasets ===")
    id_zs_accs = []
    id_rgda_accs = []
    id_ens_accs = []

    # 预先构建数据集 -> 标签偏移的映射表
    id_dataset_offset = {}
    id_dataset_nclasses = {}
    eval_offset = 0
    for d_name in args.id_datasets:
        _, _, _, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=None, transform_test=None,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        id_dataset_offset[d_name] = eval_offset
        id_dataset_nclasses[d_name] = len(c_names)
        eval_offset += len(c_names)

    eval_offset = 0
    for d_name in args.id_datasets:
        zs_acc, rgda_acc, ens_acc, c_len, _ = evaluate_dataset(
            args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
            num_id_classes, eval_offset
        )
        id_zs_accs.append(zs_acc)
        id_rgda_accs.append(rgda_acc)
        id_ens_accs.append(ens_acc)
        eval_offset += c_len
        logging.info(f"[ID] {d_name:<12s} | ZS: {zs_acc:5.1f}% | "
                     f"RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%")

    # ========== 5. 评估 OOD 数据集 ==========
    logging.info("\n=== Evaluating OOD Datasets ===")
    ood_zs_accs = []
    ood_rgda_accs = []
    ood_ens_accs = []

    # OOD 数据集的标签偏移：若该数据集同时也是 ID 数据集，使用其在 ID 中的已知偏移；
    # 若是真正 novel 的数据集，使用当前累积的总偏移（新建独立的标签空间）
    novel_offset = sum(id_dataset_nclasses.values())
    for d_name in args.ood_datasets:
        if d_name in id_dataset_offset:
            ood_eval_offset = id_dataset_offset[d_name]
            c_len = id_dataset_nclasses[d_name]
        else:
            # 对于 novel 数据集，需要获取其类别数
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            c_len = len(c_names)
            ood_eval_offset = novel_offset
            novel_offset += c_len

        zs_acc, rgda_acc, ens_acc, _, _ = evaluate_dataset(
            args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
            num_id_classes, ood_eval_offset
        )
        ood_zs_accs.append(zs_acc)
        ood_rgda_accs.append(rgda_acc)
        ood_ens_accs.append(ens_acc)
        logging.info(f"[OOD] {d_name:<12s} | ZS: {zs_acc:5.1f}% | "
                     f"RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%")

    # ========== 6. 打印指标报告 ==========
    task_names = list(args.id_datasets)

    acc_matrix_zs = [id_zs_accs]
    acc_matrix_rgda = [id_rgda_accs]
    acc_matrix_ens = [id_ens_accs]

    print("\n" + "=" * 80)
    print("JOINT FINE-TUNING RESULTS")
    print("=" * 80)

    print_paper_metrics(acc_matrix_zs, "Joint Zero-shot", task_names)
    print_paper_metrics(acc_matrix_rgda, "Joint LR-RGDA", task_names)
    print_paper_metrics(acc_matrix_ens, "Joint Ours Ensemble", task_names)

    # 打印 OOD 评估结果（简化格式，非矩阵形式）
    print("\n" + "-" * 110)
    print("[OOD 评估结果]")
    print(f"{'数据集':<12s} | {'ZS准确率':>8s} | {'RGDA准确率':>10s} | {'Ensemble准确率':>12s}")
    print("-" * 110)
    for d_name, zs, rgda, ens in zip(args.ood_datasets,
                                     ood_zs_accs, ood_rgda_accs, ood_ens_accs):
        print(f"{d_name:<12s} | {zs:8.1f}% | {rgda:10.1f}% | {ens:12.1f}%")

    id_zs_avg = sum(id_zs_accs) / len(id_zs_accs)
    id_rgda_avg = sum(id_rgda_accs) / len(id_rgda_accs)
    id_ens_avg = sum(id_ens_accs) / len(id_ens_accs)
    ood_zs_avg = sum(ood_zs_accs) / len(ood_zs_accs)
    ood_rgda_avg = sum(ood_rgda_accs) / len(ood_rgda_accs)
    ood_ens_avg = sum(ood_ens_accs) / len(ood_ens_accs)

    print("-" * 110)
    print(f"{'ID平均':<12s} | {id_zs_avg:8.1f}% | {id_rgda_avg:10.1f}% | {id_ens_avg:12.1f}%")
    print(f"{'OOD平均':<12s} | {ood_zs_avg:8.1f}% | {ood_rgda_avg:10.1f}% | {ood_ens_avg:12.1f}%")
    print("-" * 110)

    # ========== 7. Alpha 敏感性分析（批处理，如 debug_classifier_router.py）==========
    if args.alpha_sensitivity:
        print("\n" + "=" * 110)
        print("[Alpha 敏感性分析] 所有 ID 数据集的 feature 拼在一起统一扫描")
        print("=" * 110)

        id_result = batch_evaluate_datasets(
            args.id_datasets, model, processor, zeroshot_classifier,
            lr_rgda_classifier, num_id_classes, args=args,
            alpha_sensitivity=True, n_alpha_samples=21
        )

        alphas = [s[0] for s in id_result["sensitivity"]]
        accs = [s[1] for s in id_result["sensitivity"]]

        # 找最佳 alpha
        best_idx = max(range(len(accs)), key=lambda i: accs[i])
        best_alpha, best_acc = alphas[best_idx], accs[best_idx]

        print(f"{'Alpha':>10} | {'Overall Acc (ID合拼)':>20s}")
        print("-" * 40)
        for a, acc in zip(alphas, accs):
            marker = " ← best" if a == best_alpha else ""
            print(f"{a:10.2f} | {acc:18.2f}%{marker}")
        print("-" * 40)
        print(f"最佳 Alpha: {best_alpha:.2f}, 最高准确率: {best_acc:.2f}%")

        # 同时也拼 OOD 做敏感性分析
        if args.ood_datasets:
            ood_result = batch_evaluate_datasets(
                args.ood_datasets, model, processor, zeroshot_classifier,
                lr_rgda_classifier, num_id_classes, args=args,
                alpha_sensitivity=True, n_alpha_samples=21
            )
            ood_accs = [s[1] for s in ood_result["sensitivity"]]
            best_ood_idx = max(range(len(ood_accs)), key=lambda i: ood_accs[i])
            print(f"\n{'Alpha':>10} | {'Overall Acc (OOD合拼)':>20s}")
            print("-" * 40)
            for a, acc in zip(alphas, ood_accs):
                marker = " ← best" if a == alphas[best_ood_idx] else ""
                print(f"{a:10.2f} | {acc:18.2f}%{marker}")
            print("-" * 40)

    # ========== 8. 保存结果 JSON ==========
    save_results = {
        "mode": "Joint Fine-tuning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "id_datasets": list(args.id_datasets),
        "ood_datasets": list(args.ood_datasets),
        "metrics": {
            "id": {
                "zero_shot": get_full_stats(acc_matrix_zs),
                "lr_rgda": get_full_stats(acc_matrix_rgda),
                "ours_ensemble": get_full_stats(acc_matrix_ens),
            },
            "ood": {
                "per_dataset": {
                    "zero_shot": {d: v for d, v in zip(args.ood_datasets, ood_zs_accs)},
                    "lr_rgda": {d: v for d, v in zip(args.ood_datasets, ood_rgda_accs)},
                    "ours_ensemble": {d: v for d, v in zip(args.ood_datasets, ood_ens_accs)},
                },
                "average": {
                    "zero_shot": ood_zs_avg,
                    "lr_rgda": ood_rgda_avg,
                    "ours_ensemble": ood_ens_avg,
                }
            }
        }
    }

    # 如果有敏感性分析，追加保存
    if args.alpha_sensitivity:
        id_result = batch_evaluate_datasets(
            args.id_datasets, model, processor, zeroshot_classifier,
            lr_rgda_classifier, num_id_classes, args=args,
            alpha_sensitivity=True, n_alpha_samples=21
        )
        sens_data = {
            "id": {
                "overall": id_result["overall"],
                "sweep": [(a, acc) for a, acc in id_result["sensitivity"]],
            }
        }
        if args.ood_datasets:
            ood_result = batch_evaluate_datasets(
                args.ood_datasets, model, processor, zeroshot_classifier,
                lr_rgda_classifier, num_id_classes, args=args,
                alpha_sensitivity=True, n_alpha_samples=21
            )
            sens_data["ood"] = {
                "overall": ood_result["overall"],
                "sweep": [(a, acc) for a, acc in ood_result["sensitivity"]],
            }
        save_results["alpha_sensitivity"] = sens_data

    os.makedirs("experiments", exist_ok=True)
    save_path = f"experiments/joint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=4)
    logging.info(f"\n联合微调结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
