#!/usr/bin/env python3
"""
持续学习实验（带自适应路由）- 修正版
支持方法 B (Pretrain + Routing) 和 方法 E (LoRA-NSP Full)

核心修正：
- 评估所有10个任务的分类准确率（不只是ID任务）
- 通过自适应路由统一处理所有样本
- 目标是：ID性能提升，OOD性能不下降（相对于零样本）
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transformers import CLIPModel, CLIPProcessor
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector, MahalanobisOODDetector, build_stats_dict_from_features
from src.routing.adaptive_router import AdaptiveRouter
from src.utils.continual_metrics import ContinualLearningMetrics, calculate_forgetting
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning with Adaptive Routing")
    
    parser.add_argument("--method", type=str, required=True,
                       choices=["pretrain_routing", "lora_nsp_full"],
                       help="pretrain_routing: 方法B, lora_nsp_full: 方法E")
    parser.add_argument("--task_sequence", type=str, nargs='+',
                       default=["aircraft", "caltech101", "dtd", "eurosat", "flowers", 
                               "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"])
    parser.add_argument("--root", type=str, default="/home/raoxuan/projects/data/X-TAIL/")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Load checkpoint for method E")
    parser.add_argument("--ood_detector_type", type=str, default="lr_rgda",
                       choices=["mahalanobis", "lda", "qda", "lr_rgda"])
    parser.add_argument("--ood_threshold", type=float, default=0.993)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device):
    """提取图像特征"""
    model.eval()
    all_features = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        labels = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        
        images = images.to(device)
        features = model.get_image_features(pixel_values=images)
        features = features / features.norm(dim=-1, keepdim=True)
        
        all_features.append(features.cpu())
        if labels is not None:
            all_labels.append(labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels))
    
    features = torch.cat(all_features)
    labels = torch.cat(all_labels) if all_labels else torch.zeros(len(features))
    return features, labels


def get_zeroshot_classifier(model, processor, class_names, device):
    """构建零样本分类器"""
    templates = [lambda x: f"a photo of a {x}."]
    zeroshot_weights = []
    
    with torch.no_grad():
        for classname in tqdm(class_names, desc="Building zero-shot classifier", leave=False):
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


def evaluate_all_tasks_with_routing(model, processor, task_sequence: List[str], 
                                    seen_tasks: List[str], args) -> Dict:
    """
    使用自适应路由评估所有任务的分类性能
    
    Returns:
        accuracies: {task_name: accuracy}
    """
    device = args.device
    unseen_tasks = [t for t in task_sequence if t not in seen_tasks]
    
    print(f"\n  Building adaptive router from {len(seen_tasks)} seen tasks...")
    
    # Step 1: 收集已见任务的训练特征（用于构建分类器和OOD检测器）
    id_train_features = []
    id_train_labels = []
    id_class_names = []
    label_offset = 0
    
    for task_name in seen_tasks:
        train_transform, test_transform = get_transforms(task_name)
        train_loader, _, _, class_names = get_xtail_trainloader(
            root=args.root, dataset_name=task_name,
            transform_train=train_transform, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        
        features, labels = extract_features(model, train_loader, device)
        id_train_features.append(features)
        id_train_labels.append(labels + label_offset)
        id_class_names.extend(class_names)
        label_offset += len(class_names)
    
    # 如果没有已见任务（步骤0），使用零样本分类器评估
    if len(seen_tasks) == 0:
        print("  No seen tasks, using zero-shot classifier for all tasks")
        all_class_names = []
        for task_name in task_sequence:
            _, test_transform = get_transforms(task_name)
            _, class_names, _ = get_xtail_testloader(
                root=args.root, dataset_sequence=[task_name],
                transform_test=test_transform, batch_size=args.batch_size
            )
            all_class_names.extend(class_names)
        
        zeroshot_classifier = get_zeroshot_classifier(model, processor, all_class_names, device)
        
        accuracies = {}
        for task_name in task_sequence:
            _, test_transform = get_transforms(task_name)
            test_loader, _, _ = get_xtail_testloader(
                root=args.root, dataset_sequence=[task_name],
                transform_test=test_transform, batch_size=args.batch_size,
                max_num_per_dataset=1000
            )
            features, labels = extract_features(model, test_loader, device)
            logits = features.to(device) @ zeroshot_classifier * model.logit_scale.exp()
            predictions = logits.argmax(dim=1).cpu()
            accuracy = (predictions == labels).float().mean().item()
            accuracies[task_name] = accuracy
            print(f"    {task_name}: {accuracy*100:.1f}%")
        
        return accuracies
    
    # 有已见任务，构建完整的自适应路由系统
    id_train_features = torch.cat(id_train_features)
    id_train_labels = torch.cat(id_train_labels)
    
    # Step 2: 构建分类器和检测器
    stats_dict = build_stats_dict_from_features(id_train_features, id_train_labels)
    
    # 零样本分类器（用于所有任务的类别）
    all_class_names = []
    for task_name in task_sequence:
        _, test_transform = get_transforms(task_name)
        _, class_names, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[task_name],
            transform_test=test_transform, batch_size=args.batch_size
        )
        all_class_names.extend(class_names)
    
    zeroshot_classifier = get_zeroshot_classifier(model, processor, all_class_names, device)
    
    # LR-RGDA分类器（仅用于已见任务的类别）
    lr_rgda_classifier = LRRGDAClassifier(
        stats_dict=stats_dict, device=device, rank=32,
        qda_reg_alpha1=0.6, qda_reg_alpha2=1.0, qda_reg_alpha3=0.5, temperature=1.0
    )
    
    # 集成分类器
    ensemble_classifier = EnsembleClassifier(
        zeroshot_classifier, lr_rgda_classifier, alpha=0.8, temperature=1.0
    )
    
    # OOD检测器
    if args.ood_detector_type == "mahalanobis":
        ood_detector = MahalanobisOODDetector.from_stats_dict(stats_dict, alpha=0.2, device=device)
    else:
        ood_detector = ClassifierBasedOODDetector(
            stats_dict=stats_dict, classifier_type=args.ood_detector_type,
            device=device, rank=32, qda_reg_alpha1=0.6, qda_reg_alpha2=1.0, qda_reg_alpha3=0.5
        )
    
    # 自适应路由器
    router = AdaptiveRouter(zeroshot_classifier, ensemble_classifier, ood_detector, 
                           threshold=args.ood_threshold)
    
    # Step 3: 评估所有任务
    print(f"\n  Evaluating all {len(task_sequence)} tasks with routing...")
    accuracies = {}
    
    for task_name in task_sequence:
        _, test_transform = get_transforms(task_name)
        test_loader, _, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[task_name],
            transform_test=test_transform, batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, labels = extract_features(model, test_loader, device)
        predictions, is_ood = router.predict(features.to(device), model.logit_scale.exp())
        
        accuracy = (predictions.cpu() == labels).float().mean().item()
        accuracies[task_name] = accuracy
        
        task_type = "ID" if task_name in seen_tasks else "OOD"
        print(f"    [{task_type}] {task_name}: {accuracy*100:.1f}%")
    
    return accuracies


def run_continual_learning_with_routing(args):
    """主实验流程"""
    print("="*80)
    print(f"Continual Learning with Adaptive Routing - Method {args.method.upper()}")
    print("="*80)
    print(f"Task Sequence: {args.task_sequence}")
    print("="*80)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("\n[Setup] Loading CLIP model...")
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"  Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
        model = CLIPModel.from_pretrained(args.model_name).to(args.device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    else:
        model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    # 初始化指标记录器
    metrics_tracker = ContinualLearningMetrics(args.task_sequence)
    
    # Step 0: 零样本基线评估
    print("\n" + "="*80)
    print("[Step 0] Zero-shot Baseline Evaluation")
    print("="*80)
    
    zeroshot_accuracies = evaluate_all_tasks_with_routing(
        model, processor, args.task_sequence, [], args
    )
    metrics_tracker.update(0, zeroshot_accuracies)
    
    avg_zeroshot = np.mean(list(zeroshot_accuracies.values()))
    print(f"\n  Average Zero-shot Accuracy: {avg_zeroshot*100:.1f}%")
    
    # 保存零样本结果
    with open(os.path.join(args.output_dir, "step_0_zeroshot.json"), 'w') as f:
        json.dump({'accuracies': zeroshot_accuracies, 'average': avg_zeroshot}, f, indent=2)
    
    # 如果是纯零样本方法（A），直接结束
    if args.method == "zeroshot":
        metrics_tracker.print_summary()
        return
    
    # 增量学习循环
    for step, task_name in enumerate(args.task_sequence):
        print(f"\n{'='*80}")
        print(f"[Step {step+1}/{len(args.task_sequence)}] Task: {task_name}")
        print(f"{'='*80}")
        
        seen_tasks = args.task_sequence[:step+1]
        
        # 评估当前步骤（使用自适应路由）
        accuracies = evaluate_all_tasks_with_routing(
            model, processor, args.task_sequence, seen_tasks, args
        )
        
        # 更新指标
        metrics_tracker.update(step, accuracies)
        
        # 计算统计信息
        seen_accs = [accuracies[t] for t in seen_tasks]
        unseen_tasks = [t for t in args.task_sequence if t not in seen_tasks]
        unseen_accs = [accuracies[t] for t in unseen_tasks] if unseen_tasks else []
        
        print(f"\n  Summary:")
        print(f"    Seen tasks ({len(seen_tasks)}): avg = {np.mean(seen_accs)*100:.1f}%")
        if unseen_accs:
            print(f"    Unseen tasks ({len(unseen_tasks)}): avg = {np.mean(unseen_accs)*100:.1f}%")
        print(f"    Overall: avg = {np.mean(list(accuracies.values()))*100:.1f}%")
        
        # 保存中间结果
        step_result = {
            'step': step + 1,
            'task': task_name,
            'seen_tasks': seen_tasks,
            'unseen_tasks': unseen_tasks,
            'accuracies': accuracies,
            'seen_avg': float(np.mean(seen_accs)),
            'unseen_avg': float(np.mean(unseen_accs)) if unseen_accs else None,
            'overall_avg': float(np.mean(list(accuracies.values())))
        }
        
        with open(os.path.join(args.output_dir, f"step_{step+1}_{task_name}.json"), 'w') as f:
            json.dump(step_result, f, indent=2)
    
    # 最终总结
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    metrics_tracker.print_summary()
    
    forgetting_rate = calculate_forgetting(metrics_tracker.get_accuracy_matrix())
    print(f"\nForgetting Rate: {forgetting_rate:.1f}%")
    
    # 保存最终结果
    final_results = {
        'method': args.method,
        'task_sequence': args.task_sequence,
        'metrics': metrics_tracker.get_summary(),
        'forgetting_rate': float(forgetting_rate),
        'accuracy_matrix': metrics_tracker.get_accuracy_matrix().tolist()
    }
    
    output_file = os.path.join(args.output_dir, f"{args.method}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return final_results


def main():
    args = parse_args()
    run_continual_learning_with_routing(args)


if __name__ == "__main__":
    main()
