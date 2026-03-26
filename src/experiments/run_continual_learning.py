"""
持续学习主实验运行脚本
支持 LADA 论文的评估协议
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.trainer import Trainer
from src.utils.continual_metrics import ContinualLearningMetrics, calculate_forgetting
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


# LADA 论文的 10 个数据集（字母顺序）
LADA_TASK_SEQUENCE = [
    "aircraft",
    "caltech101", 
    "dtd",
    "eurosat",
    "flowers",
    "food101",
    "mnist",
    "oxford_pets",
    "stanford_cars",
    "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="X-TAIL Continual Learning Experiments")
    
    # 数据集配置
    parser.add_argument("--root", type=str, 
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="X-TAIL 数据集根目录")
    parser.add_argument("--task_sequence", type=str, nargs='+',
                       default=LADA_TASK_SEQUENCE,
                       help="任务序列（默认使用 LADA 字母顺序）")
    parser.add_argument("--num_shots", type=int, default=16,
                       help="每类别的训练样本数 (16-shot 或 full)")
    parser.add_argument("--full_shot", action="store_true",
                       help="使用 full-shot 设置（覆盖 num_shots）")
    
    # 训练配置
    parser.add_argument("--method", type=str, 
                       choices=["zeroshot", "lora_vanilla", "lora_sgp", "lora_nsp"],
                       default="lora_nsp",
                       help="训练方法: zeroshot, lora_vanilla (基线), lora_sgp, lora_nsp")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    
    # LoRA 参数
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--nsp_eps", type=float, default=0.05)
    parser.add_argument("--nsp_weight", type=float, default=0.02)
    parser.add_argument("--weight_temp", type=float, default=1.0)
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0)
    
    # 蒸馏参数
    parser.add_argument("--fd_weight", type=float, default=1.0)
    parser.add_argument("--cd_weight", type=float, default=1.0)
    parser.add_argument("--reference_dataset", type=str, default="flickr8k")
    parser.add_argument("--reference_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # OOD 检测配置
    parser.add_argument("--enable_ood_eval", action="store_true", default=False,
                       help="启用 OOD 检测评估（目前简化版暂不支持）")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="experiments/continual_learning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def prepare_args_for_method(args):
    """根据方法类型准备 args"""
    if args.method == "zeroshot":
        # Zero-shot 不需要训练
        pass
    elif args.method == "lora_vanilla":
        # 普通 LoRA 基线（无 SGP/NSP）
        args.lora_type = "lora_vanilla"
        args.lora_alpha = getattr(args, 'lora_alpha', args.lora_rank)
    elif args.method == "lora_sgp":
        # LoRA + SGP（软投影）
        args.lora_type = "lora_sgp"
    elif args.method == "lora_nsp":
        # LoRA + NSP（硬投影）
        args.lora_type = "lora_nsp"
    
    return args


def evaluate_all_tasks(trainer, task_sequence: List[str], root: str, 
                       batch_size: int, max_samples: int = 1000) -> Dict:
    """在所有任务上评估模型"""
    results = {'accuracies': {}}
    
    _, test_transform = get_transforms("aircraft")
    
    for task_name in task_sequence:
        test_loader, class_names, _ = get_xtail_testloader(
            root=root,
            dataset_sequence=[task_name],
            transform_test=test_transform,
            batch_size=batch_size,
            max_num_per_dataset=max_samples
        )
        
        accuracy = trainer.evaluate(test_loader, class_names)
        results['accuracies'][task_name] = accuracy
        print(f"  {task_name}: Acc = {accuracy*100:.1f}%")
    
    return results


def run_continual_learning(args):
    """运行持续学习实验"""
    print("="*80)
    print("X-TAIL Continual Learning Experiment")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Task Sequence: {args.task_sequence}")
    print(f"Num Shots: {'full' if args.full_shot else args.num_shots}")
    print("="*80)
    
    # 准备参数
    args = prepare_args_for_method(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估指标记录器
    metrics_tracker = ContinualLearningMetrics(args.task_sequence)
    
    # 初始化训练器
    trainer = Trainer(args)
    
    # Step 0: 评估 Zero-shot 性能
    print("\n[Step 0] Evaluating Zero-shot Performance...")
    zeroshot_results = evaluate_all_tasks(
        trainer, args.task_sequence, args.root, args.batch_size
    )
    
    zeroshot_accs = {task: zeroshot_results['accuracies'].get(task, 0) 
                     for task in args.task_sequence}
    metrics_tracker.update(0, zeroshot_accs)
    
    print("\nZero-shot Accuracies:")
    for task, acc in zeroshot_accs.items():
        print(f"  {task}: {acc*100:.1f}%")
    
    # 如果只需要 zero-shot，直接结束
    if args.method == "zeroshot":
        print("\nZero-shot evaluation completed.")
        metrics_tracker.print_summary()
        metrics_tracker.save(os.path.join(args.output_dir, "zeroshot_results.json"))
        return
    
    # 增量学习循环
    for step, task_name in enumerate(args.task_sequence):
        print(f"\n{'='*80}")
        print(f"[Step {step+1}/{len(args.task_sequence)}] Training on: {task_name}")
        print(f"{'='*80}")
        
        # 获取当前任务的数据加载器
        train_transform, test_transform = get_transforms(task_name)
        num_shots = None if args.full_shot else args.num_shots
        
        train_loader, _, _, class_names = get_xtail_trainloader(
            root=args.root,
            dataset_name=task_name,
            transform_train=train_transform,
            transform_test=test_transform,
            num_shots=num_shots,
            batch_size=args.batch_size
        )
        
        # 训练当前任务
        print(f"\nTraining {task_name}...")
        
        # 准备参考数据加载器（如果需要蒸馏）
        reference_loader = None
        if hasattr(trainer, 'initialize_reference_loader'):
            from utils_data import Flickr8kDataset
            if args.reference_dataset == "flickr8k":
                reference_dataset = Flickr8kDataset(root="/data1/open_datasets/flickr8k/")
                reference_loader = trainer.initialize_reference_loader(reference_dataset)
        
        trainer.train(train_loader, class_names, reference_loader)
        
        # 增量学习关键步骤：将LoRA参数归并到主模型参数
        print(f"\nMerging LoRA weights into base model...")
        if hasattr(trainer.model.vision_model, 'merge_lora_weights'):
            trainer.model.vision_model.merge_lora_weights()
            print(f"  ✅ LoRA weights merged successfully")
        else:
            print(f"  ⚠️  Model does not support merge_lora_weights")
        
        # 在所有任务上评估
        print(f"\nEvaluating on all {len(args.task_sequence)} tasks...")
        eval_results = evaluate_all_tasks(
            trainer, args.task_sequence, args.root, args.batch_size
        )
        
        # 更新指标记录器
        metrics_tracker.update(step, eval_results['accuracies'])
        
        # 保存中间结果
        intermediate_results = {
            'step': step + 1,
            'task': task_name,
            'accuracies': {k: float(v) for k, v in eval_results['accuracies'].items()}
        }
        
        with open(os.path.join(args.output_dir, f"step_{step+1}_{task_name}.json"), 'w') as f:
            json.dump(intermediate_results, f, indent=2)
    
    # 打印最终总结
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    metrics_tracker.print_summary()
    
    # 计算遗忘率
    forgetting_rate = calculate_forgetting(metrics_tracker.get_accuracy_matrix())
    print(f"\nForgetting Rate: {forgetting_rate:.1f}%")
    
    # 保存最终结果
    final_results = {
        'args': {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v 
                 for k, v in vars(args).items()},
        'metrics': metrics_tracker.get_summary(),
        'per_task_metrics': metrics_tracker.calculate_per_task_metrics(),
        'forgetting_rate': float(forgetting_rate),
        'accuracy_matrix': metrics_tracker.get_accuracy_matrix().tolist()
    }
    
    output_file = os.path.join(args.output_dir, f"{args.method}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nAll results saved to: {args.output_dir}")
    
    return final_results


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 运行实验
    results = run_continual_learning(args)
    
    return results


if __name__ == "__main__":
    main()
