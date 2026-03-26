#!/usr/bin/env python3
"""
持续学习实验（带自适应路由）
支持方法 B (Pretrain + Routing) 和 方法 E (LoRA-NSP Full)

关键特性:
- 在每个任务步骤后动态划分ID/OOD
- 构建OOD检测器和自适应路由
- 计算LADA指标 (Transfer, Average, Last, Forgetting)
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transformers import CLIPModel, CLIPProcessor
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector, MahalanobisOODDetector, build_stats_dict_from_features
from src.routing.adaptive_router import AdaptiveRouter
from src.utils.evaluation import calculate_ood_metrics, calculate_classification_accuracy
from src.utils.continual_metrics import ContinualLearningMetrics, calculate_forgetting
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning with Adaptive Routing")
    
    # 方法选择
    parser.add_argument("--method", type=str, required=True,
                       choices=["pretrain_routing", "lora_nsp_full"],
                       help="pretrain_routing: 方法B, lora_nsp_full: 方法E")
    
    # 任务序列
    parser.add_argument("--task_sequence", type=str, nargs='+',
                       default=["aircraft", "caltech101", "dtd", "eurosat", "flowers", 
                               "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                       help="Task sequence for continual learning")
    
    # 数据集配置
    parser.add_argument("--root", type=str,
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="Dataset root directory")
    parser.add_argument("--num_shots", type=int, default=16,
                       help="Number of shots per class")
    parser.add_argument("--batch_size", type=int, default=32)
    
    # 模型配置
    parser.add_argument("--model_name", type=str,
                       default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Load checkpoint for method E (path to D's final model)")
    
    # OOD检测器配置
    parser.add_argument("--ood_detector_type", type=str, default="lr_rgda",
                       choices=["mahalanobis", "lda", "qda", "lr_rgda"])
    parser.add_argument("--ood_threshold", type=float, default=0.993)
    
    # 训练配置（仅用于方法E，如果需要微调）
    parser.add_argument("--iterations", type=int, default=0,
                       help="Training iterations (0 for no training)")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device):
    """提取图像特征"""
    model.eval()
    all_features = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Extracting features"):
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
        for classname in tqdm(class_names, desc="Building zero-shot classifier"):
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


def evaluate_step(model, processor, seen_tasks: List[str], unseen_tasks: List[str], 
                  args, ood_detector_type: str = "lr_rgda", threshold: float = 0.993):
    """
    在每个任务步骤后评估
    
    Returns:
        dict: {
            'id_accuracy': float,      # ID任务上的分类准确率
            'ood_auroc': float,        # OOD检测AUROC
            'ood_fpr95': float,        # FPR@95TPR
            'ood_accuracy': float,     # OOD任务上的准确率（使用自适应路由）
            'combined_score': float    # 综合得分
        }
    """
    device = args.device
    
    # Step 1: 收集已见任务（ID）的特征和标签
    print(f"\n  Collecting features from {len(seen_tasks)} seen tasks (ID)...")
    id_train_features = []
    id_train_labels = []
    id_class_names = []
    label_offset = 0
    
    for task_name in seen_tasks:
        train_transform, test_transform = get_transforms(task_name)
        train_loader, _, _, class_names = get_xtail_trainloader(
            root=args.root,
            dataset_name=task_name,
            transform_train=train_transform,
            transform_test=test_transform,
            num_shots=args.num_shots,
            batch_size=args.batch_size
        )
        
        features, labels = extract_features(model, train_loader, device)
        id_train_features.append(features)
        id_train_labels.append(labels + label_offset)
        id_class_names.extend(class_names)
        label_offset += len(class_names)
    
    id_train_features = torch.cat(id_train_features)
    id_train_labels = torch.cat(id_train_labels)
    
    print(f"    ID train samples: {len(id_train_features)}, classes: {len(id_class_names)}")
    
    # Step 2: 收集未见任务（OOD）的测试特征
    print(f"\n  Collecting features from {len(unseen_tasks)} unseen tasks (OOD)...")
    ood_test_features = []
    ood_test_labels = []
    
    for task_name in unseen_tasks:
        _, test_transform = get_transforms(task_name)
        test_loader, class_names, _ = get_xtail_testloader(
            root=args.root,
            dataset_sequence=[task_name],
            transform_test=test_transform,
            batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, labels = extract_features(model, test_loader, device)
        ood_test_features.append(features)
        ood_test_labels.append(labels)
    
    if ood_test_features:
        ood_test_features = torch.cat(ood_test_features)
        ood_test_labels = torch.cat(ood_test_labels)
        print(f"    OOD test samples: {len(ood_test_features)}")
    else:
        ood_test_features = torch.empty(0, 512)
        ood_test_labels = torch.empty(0)
        print(f"    No OOD tasks (last task)")
    
    # Step 3: 构建stats_dict和分类器
    stats_dict = build_stats_dict_from_features(id_train_features, id_train_labels)
    
    # 零样本分类器
    zeroshot_classifier = get_zeroshot_classifier(model, processor, id_class_names, device)
    
    # LR-RGDA分类器
    lr_rgda_classifier = LRRGDAClassifier(
        stats_dict=stats_dict,
        device=device,
        rank=32,
        qda_reg_alpha1=0.6,
        qda_reg_alpha2=1.0,
        qda_reg_alpha3=0.5,
        temperature=1.0
    )
    
    # 集成分类器
    ensemble_classifier = EnsembleClassifier(
        zeroshot_classifier,
        lr_rgda_classifier,
        alpha=0.8,
        temperature=1.0
    )
    
    # Step 4: 构建OOD检测器
    if ood_detector_type == "mahalanobis":
        ood_detector = MahalanobisOODDetector.from_stats_dict(
            stats_dict=stats_dict,
            alpha=0.2,
            device=device
        )
    else:
        ood_detector = ClassifierBasedOODDetector(
            stats_dict=stats_dict,
            classifier_type=ood_detector_type,
            device=device,
            rank=32,
            qda_reg_alpha1=0.6,
            qda_reg_alpha2=1.0,
            qda_reg_alpha3=0.5
        )
    
    # 自适应路由器
    router = AdaptiveRouter(
        zeroshot_classifier,
        ensemble_classifier,
        ood_detector,
        threshold=threshold
    )
    
    # Step 5: 在ID任务上评估分类准确率
    print(f"\n  Evaluating on ID tasks...")
    id_correct = 0
    id_total = 0
    
    for task_idx, task_name in enumerate(seen_tasks):
        _, test_transform = get_transforms(task_name)
        test_loader, class_names, _ = get_xtail_testloader(
            root=args.root,
            dataset_sequence=[task_name],
            transform_test=test_transform,
            batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, labels = extract_features(model, test_loader, device)
        predictions, is_ood = router.predict(features.to(device), model.logit_scale.exp())
        
        correct = (predictions.cpu() == labels).sum().item()
        id_correct += correct
        id_total += len(labels)
        
        print(f"    {task_name}: {correct}/{len(labels)} = {correct/len(labels)*100:.1f}%")
    
    id_accuracy = id_correct / id_total if id_total > 0 else 0.0
    
    # Step 6: 在OOD任务上评估OOD检测性能
    ood_auroc = 0.0
    ood_fpr95 = 1.0
    ood_accuracy = 0.0
    
    if len(unseen_tasks) > 0 and len(ood_test_features) > 0:
        print(f"\n  Evaluating OOD detection...")
        
        # 收集ID测试特征
        id_test_features_list = []
        for task_name in seen_tasks:
            _, test_transform = get_transforms(task_name)
            test_loader, _, _ = get_xtail_testloader(
                root=args.root,
                dataset_sequence=[task_name],
                transform_test=test_transform,
                batch_size=args.batch_size,
                max_num_per_dataset=1000
            )
            features, _ = extract_features(model, test_loader, device)
            id_test_features_list.append(features)
        
        id_test_features = torch.cat(id_test_features_list).to(device)
        
        # 计算OOD分数
        id_scores = ood_detector.predict_score(id_test_features)
        ood_scores = ood_detector.predict_score(ood_test_features.to(device))
        
        # 计算OOD检测指标
        ood_metrics = calculate_ood_metrics(
            id_scores.cpu().numpy().tolist(),
            ood_scores.cpu().numpy().tolist()
        )
        ood_auroc = ood_metrics['auroc']
        ood_fpr95 = ood_metrics['fpr_at_95_tpr']
        
        print(f"    AUROC: {ood_auroc:.4f}, FPR@95TPR: {ood_fpr95:.4f}")
        
        # Step 7: 在OOD任务上评估自适应路由准确率
        print(f"\n  Evaluating OOD classification with routing...")
        ood_correct = 0
        ood_total = 0
        
        for task_name in unseen_tasks:
            _, test_transform = get_transforms(task_name)
            test_loader, class_names, _ = get_xtail_testloader(
                root=args.root,
                dataset_sequence=[task_name],
                transform_test=test_transform,
                batch_size=args.batch_size,
                max_num_per_dataset=1000
            )
            
            features, labels = extract_features(model, test_loader, device)
            predictions, is_ood = router.predict(features.to(device), model.logit_scale.exp())
            
            # 对于OOD样本，使用零样本分类器的预测
            correct = (predictions.cpu() == labels).sum().item()
            ood_correct += correct
            ood_total += len(labels)
            
            print(f"    {task_name}: {correct}/{len(labels)} = {correct/len(labels)*100:.1f}%")
        
        ood_accuracy = ood_correct / ood_total if ood_total > 0 else 0.0
    
    # Step 8: 计算综合得分
    combined_score = (id_accuracy + ood_accuracy) / 2 if len(unseen_tasks) > 0 else id_accuracy
    
    results = {
        'id_accuracy': id_accuracy,
        'ood_auroc': ood_auroc,
        'ood_fpr95': ood_fpr95,
        'ood_accuracy': ood_accuracy,
        'combined_score': combined_score,
        'num_seen_tasks': len(seen_tasks),
        'num_unseen_tasks': len(unseen_tasks)
    }
    
    print(f"\n  Summary:")
    print(f"    ID Accuracy: {id_accuracy:.4f}")
    print(f"    OOD AUROC: {ood_auroc:.4f}")
    print(f"    OOD Accuracy: {ood_accuracy:.4f}")
    print(f"    Combined Score: {combined_score:.4f}")
    
    return results


def run_continual_learning_with_routing(args):
    """主实验流程"""
    print("="*80)
    print(f"Continual Learning with Adaptive Routing - Method {args.method.upper()}")
    print("="*80)
    print(f"Task Sequence: {args.task_sequence}")
    print(f"OOD Detector: {args.ood_detector_type}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载CLIP模型
    print("\n[Setup] Loading CLIP model...")
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        # 方法E: 加载预训练的模型
        print(f"  Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
        model = CLIPModel.from_pretrained(args.model_name).to(args.device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    else:
        # 方法B: 使用预训练CLIP（不微调）
        model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    # 初始化指标记录器
    metrics_tracker = ContinualLearningMetrics(args.task_sequence)
    
    # 记录每个步骤的结果
    step_results = []
    
    # 增量学习循环
    for step, task_name in enumerate(args.task_sequence):
        print(f"\n{'='*80}")
        print(f"[Step {step+1}/{len(args.task_sequence)}] Task: {task_name}")
        print(f"{'='*80}")
        
        # 划分ID和OOD
        seen_tasks = args.task_sequence[:step+1]
        unseen_tasks = args.task_sequence[step+1:]
        
        print(f"Seen tasks (ID): {seen_tasks}")
        print(f"Unseen tasks (OOD): {unseen_tasks if unseen_tasks else 'None (last task)'}")
        
        # 如果是方法E且需要训练，则训练当前任务
        if args.method == "lora_nsp_full" and args.iterations > 0:
            # TODO: 如果需要训练，这里添加训练代码
            # 目前假设模型已经训练好（复用D的模型）
            pass
        
        # 评估当前步骤
        eval_results = evaluate_step(
            model, processor, seen_tasks, unseen_tasks,
            args, args.ood_detector_type, args.ood_threshold
        )
        
        # 保存步骤结果
        step_results.append({
            'step': step + 1,
            'task': task_name,
            'seen_tasks': seen_tasks,
            'unseen_tasks': unseen_tasks,
            'results': eval_results
        })
        
        # 更新持续学习指标
        # 构建准确率字典（用于LADA指标计算）
        # 对于每个已见任务，使用id_accuracy作为近似
        accuracies = {task: eval_results['id_accuracy'] for task in seen_tasks}
        # 对于未见任务，使用ood_accuracy作为近似
        for task in unseen_tasks:
            accuracies[task] = eval_results['ood_accuracy']
        
        metrics_tracker.update(step, accuracies)
        
        # 保存中间结果
        with open(os.path.join(args.output_dir, f"step_{step+1}_{task_name}.json"), 'w') as f:
            json.dump(step_results[-1], f, indent=2)
    
    # 最终总结
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    metrics_tracker.print_summary()
    
    # 计算遗忘率
    forgetting_rate = calculate_forgetting(metrics_tracker.get_accuracy_matrix())
    print(f"\nForgetting Rate: {forgetting_rate:.1f}%")
    
    # 保存最终结果
    final_results = {
        'method': args.method,
        'task_sequence': args.task_sequence,
        'ood_detector_type': args.ood_detector_type,
        'ood_threshold': args.ood_threshold,
        'metrics': metrics_tracker.get_summary(),
        'forgetting_rate': float(forgetting_rate),
        'accuracy_matrix': metrics_tracker.get_accuracy_matrix().tolist(),
        'step_results': step_results
    }
    
    output_file = os.path.join(args.output_dir, f"{args.method}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Final results: {output_file}")
    
    return final_results


def main():
    args = parse_args()
    run_continual_learning_with_routing(args)


if __name__ == "__main__":
    main()
