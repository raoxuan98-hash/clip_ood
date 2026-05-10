#!/usr/bin/env python3
"""
Table 4/5: 分类评估脚本

功能:
- 评估分类性能（不是OOD检测性能）
- 支持纯零样本、纯LR-RGDA、固定集成、自适应路由
- 正确计算Overall Acc（所有数据集的整体准确率）

输出指标:
- ID Avg Acc: ID数据集的平均准确率
- OOD Avg Acc: OOD数据集的平均准确率  
- Overall Acc: 所有数据集的整体准确率（不是(ID+OOD)/2）
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import CLIPModel, CLIPProcessor
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector, MahalanobisOODDetector, build_stats_dict_from_features
from src.routing.adaptive_router import AdaptiveRouter
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Classification Evaluation for Table 4/5")
    
    # 数据集配置
    parser.add_argument("--id_datasets", type=str, nargs='+', required=True,
                       help="ID datasets")
    parser.add_argument("--ood_datasets", type=str, nargs='+', required=True,
                       help="OOD datasets")
    parser.add_argument("--root", type=str,
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="Dataset root directory")
    
    # 评估策略
    parser.add_argument("--strategy", type=str, required=True,
                       choices=["zeroshot", "lrrgda", "ensemble", "routing"],
                       help="Evaluation strategy")
    
    # 固定集成参数
    parser.add_argument("--alpha", type=float, default=0.8,
                       help="Ensemble alpha (for ensemble and routing strategies)")
    
    # 自适应路由参数
    parser.add_argument("--ood_threshold", type=float, default=0.85,
                       help="OOD threshold for routing (from Table 3)")
    parser.add_argument("--ood_detector_type", type=str, default="lr_rgda",
                       choices=["mahalanobis", "lda", "qda", "lr_rgda"],
                       help="OOD detector type for routing")
    
    # 模型配置
    parser.add_argument("--model_name", type=str,
                       default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练配置
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    
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


def evaluate_classification(args):
    """评估分类性能"""
    print("="*80)
    print(f"Classification Evaluation - Strategy: {args.strategy.upper()}")
    print("="*80)
    print(f"ID datasets: {args.id_datasets}")
    print(f"OOD datasets: {args.ood_datasets}")
    print(f"Strategy: {args.strategy}")
    if args.strategy in ["ensemble", "routing"]:
        print(f"Alpha: {args.alpha}")
    if args.strategy == "routing":
        print(f"OOD threshold: {args.ood_threshold}")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载CLIP模型
    print("\n[1/3] Loading CLIP model...")
    model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    # 收集所有类别名称
    all_class_names = []
    for dataset_name in args.id_datasets:
        _, test_transform = get_transforms(dataset_name)
        _, class_names, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[dataset_name],
            transform_test=test_transform, batch_size=args.batch_size
        )
        all_class_names.extend(class_names)
    
    print(f"Total classes: {len(all_class_names)}")
    
    # 构建分类器
    print(f"\n[2/3] Building classifiers for strategy: {args.strategy}...")
    
    # 零样本分类器（所有策略都需要）
    zeroshot_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
    
    # 根据策略构建分类器
    if args.strategy == "zeroshot":
        # 纯零样本
        classifier = zeroshot_classifier
        router = None
        
    elif args.strategy == "lrrgda":
        # 纯LR-RGDA
        # 提取ID训练特征
        id_train_features = []
        id_train_labels = []
        label_offset = 0
        
        for dataset_name in args.id_datasets:
            train_transform, test_transform = get_transforms(dataset_name)
            train_loader, _, _, class_names = get_xtail_trainloader(
                root=args.root, dataset_name=dataset_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            
            features, labels = extract_features(model, train_loader, args.device)
            id_train_features.append(features)
            id_train_labels.append(labels + label_offset)
            label_offset += len(class_names)
        
        id_train_features = torch.cat(id_train_features)
        id_train_labels = torch.cat(id_train_labels)
        
        stats_dict = build_stats_dict_from_features(id_train_features, id_train_labels)
        
        lrrgda_classifier = LRRGDAClassifier(
            stats_dict=stats_dict, device=args.device, rank=32,
            qda_reg_alpha1=0.6, qda_reg_alpha2=1.0, qda_reg_alpha3=0.5, temperature=1.0
        )
        
        # 使用纯LR-RGDA (alpha=1.0)
        classifier = EnsembleClassifier(
            zeroshot_classifier, lrrgda_classifier, alpha=1.0, temperature=1.0
        )
        router = None
        
    elif args.strategy in ["ensemble", "routing"]:
        # 固定集成或自适应路由
        # 提取ID训练特征
        id_train_features = []
        id_train_labels = []
        label_offset = 0
        
        for dataset_name in args.id_datasets:
            train_transform, test_transform = get_transforms(dataset_name)
            train_loader, _, _, class_names = get_xtail_trainloader(
                root=args.root, dataset_name=dataset_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            
            features, labels = extract_features(model, train_loader, args.device)
            id_train_features.append(features)
            id_train_labels.append(labels + label_offset)
            label_offset += len(class_names)
        
        id_train_features = torch.cat(id_train_features)
        id_train_labels = torch.cat(id_train_labels)
        
        stats_dict = build_stats_dict_from_features(id_train_features, id_train_labels)
        
        lrrgda_classifier = LRRGDAClassifier(
            stats_dict=stats_dict, device=args.device, rank=32,
            qda_reg_alpha1=0.6, qda_reg_alpha2=1.0, qda_reg_alpha3=0.5, temperature=1.0
        )
        
        # 集成分类器
        ensemble = EnsembleClassifier(
            zeroshot_classifier, lrrgda_classifier, alpha=args.alpha, temperature=1.0
        )
        
        if args.strategy == "ensemble":
            # 固定集成
            classifier = ensemble
            router = None
        else:
            # 自适应路由
            # 构建OOD检测器
            if args.ood_detector_type == "mahalanobis":
                ood_detector = MahalanobisOODDetector.from_stats_dict(
                    stats_dict=stats_dict, alpha=0.2, device=args.device
                )
            else:
                ood_detector = ClassifierBasedOODDetector(
                    stats_dict=stats_dict, classifier_type=args.ood_detector_type,
                    device=args.device, rank=32,
                    qda_reg_alpha1=0.6, qda_reg_alpha2=1.0, qda_reg_alpha3=0.5
                )
            
            router = AdaptiveRouter(
                zeroshot_classifier, ensemble, ood_detector,
                threshold=args.ood_threshold
            )
            classifier = None
    
    # 评估所有数据集
    print(f"\n[3/3] Evaluating all datasets...")
    
    id_results = []  # (correct, total) for each ID dataset
    ood_results = []  # (correct, total) for each OOD dataset
    all_results = []  # (correct, total) for all datasets
    
    # 评估ID数据集
    for dataset_name in args.id_datasets:
        _, test_transform = get_transforms(dataset_name)
        test_loader, _, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[dataset_name],
            transform_test=test_transform, batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, labels = extract_features(model, test_loader, args.device)
        
        if router is not None:
            predictions, _ = router.predict(features.to(args.device), model.logit_scale.exp())
        else:
            logits = classifier.forward(features.to(args.device))
            predictions = logits.argmax(dim=1)
        
        correct = (predictions.cpu() == labels).sum().item()
        total = len(labels)
        
        id_results.append((correct, total))
        all_results.append((correct, total))
        
        print(f"  [ID] {dataset_name}: {correct}/{total} = {correct/total*100:.1f}%")
    
    # 评估OOD数据集
    for dataset_name in args.ood_datasets:
        _, test_transform = get_transforms(dataset_name)
        test_loader, _, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[dataset_name],
            transform_test=test_transform, batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, labels = extract_features(model, test_loader, args.device)
        
        if router is not None:
            predictions, _ = router.predict(features.to(args.device), model.logit_scale.exp())
        else:
            logits = classifier.forward(features.to(args.device))
            predictions = logits.argmax(dim=1)
        
        correct = (predictions.cpu() == labels).sum().item()
        total = len(labels)
        
        ood_results.append((correct, total))
        all_results.append((correct, total))
        
        print(f"  [OOD] {dataset_name}: {correct}/{total} = {correct/total*100:.1f}%")
    
    # 计算指标
    id_correct = sum(c for c, t in id_results)
    id_total = sum(t for c, t in id_results)
    id_avg_acc = id_correct / id_total if id_total > 0 else 0.0
    
    ood_correct = sum(c for c, t in ood_results)
    ood_total = sum(t for c, t in ood_results)
    ood_avg_acc = ood_correct / ood_total if ood_total > 0 else 0.0
    
    all_correct = sum(c for c, t in all_results)
    all_total = sum(t for c, t in all_results)
    overall_acc = all_correct / all_total if all_total > 0 else 0.0
    
    # 打印结果
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"ID Avg Acc:   {id_avg_acc*100:.2f}% ({id_correct}/{id_total})")
    print(f"OOD Avg Acc:  {ood_avg_acc*100:.2f}% ({ood_correct}/{ood_total})")
    print(f"Overall Acc:  {overall_acc*100:.2f}% ({all_correct}/{all_total})")
    print("="*80)
    
    # 保存结果
    results = {
        'experiment': {
            'table': 'Table 4/5',
            'strategy': args.strategy,
            'alpha': args.alpha if args.strategy in ['ensemble', 'routing'] else None,
            'ood_threshold': args.ood_threshold if args.strategy == 'routing' else None,
        },
        'configuration': {
            'id_datasets': args.id_datasets,
            'ood_datasets': args.ood_datasets,
        },
        'metrics': {
            'id_avg_acc': float(id_avg_acc),
            'ood_avg_acc': float(ood_avg_acc),
            'overall_acc': float(overall_acc),
            'id_correct': int(id_correct),
            'id_total': int(id_total),
            'ood_correct': int(ood_correct),
            'ood_total': int(ood_total),
            'all_correct': int(all_correct),
            'all_total': int(all_total),
        },
        'detailed_results': {
            'id': [{'dataset': name, 'correct': c, 'total': t, 'accuracy': c/t} 
                   for name, (c, t) in zip(args.id_datasets, id_results)],
            'ood': [{'dataset': name, 'correct': c, 'total': t, 'accuracy': c/t} 
                    for name, (c, t) in zip(args.ood_datasets, ood_results)],
        }
    }
    
    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    args = parse_args()
    evaluate_classification(args)


if __name__ == "__main__":
    main()
