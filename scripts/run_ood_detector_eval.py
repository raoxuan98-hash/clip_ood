#!/usr/bin/env python3
"""
Table 3: OOD检测器对比实验

功能:
- 评估不同OOD检测器的性能
- 支持交叉验证（多组ID/OOD划分）
- 只输出OOD检测指标（不关注分类准确率）

输出指标:
- AUROC: ROC曲线下面积
- FPR@95TPR: 95%真阳性率时的假阳性率
- AUPR: PR曲线下面积
- Detection Error: 最优阈值下的检测错误
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
from src.detectors.ood_detector import (
    ClassifierBasedOODDetector, 
    MahalanobisOODDetector, 
    build_stats_dict_from_features
)
from src.utils.evaluation import calculate_ood_metrics
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detector Evaluation for Table 3")
    
    # 数据集配置
    parser.add_argument("--id_datasets", type=str, nargs='+', required=True,
                       help="ID datasets for training OOD detector")
    parser.add_argument("--ood_datasets", type=str, nargs='+', required=True,
                       help="OOD datasets for testing OOD detector")
    parser.add_argument("--root", type=str, 
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="Dataset root directory")
    
    # OOD检测器配置
    parser.add_argument("--detector_type", type=str, required=True,
                       choices=["mahalanobis", "lda", "qda", "lr_rgda"],
                       help="Type of OOD detector")
    
    # 模型配置
    parser.add_argument("--model_name", type=str,
                       default="openai/clip-vit-base-patch16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    # 训练配置
    parser.add_argument("--num_shots", type=int, default=16,
                       help="Number of shots for training data")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for feature extraction")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
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


def evaluate_ood_detector(args):
    """评估OOD检测器"""
    print("="*80)
    print(f"Table 3: OOD Detector Evaluation")
    print("="*80)
    print(f"Detector: {args.detector_type}")
    print(f"ID datasets: {args.id_datasets}")
    print(f"OOD datasets: {args.ood_datasets}")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载CLIP模型
    print("\n[1/4] Loading CLIP model...")
    model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    # 提取ID训练特征
    print(f"\n[2/4] Extracting ID training features from {len(args.id_datasets)} datasets...")
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
        
        print(f"  {dataset_name}: {len(features)} samples, {len(class_names)} classes")
    
    id_train_features = torch.cat(id_train_features)
    id_train_labels = torch.cat(id_train_labels)
    print(f"  Total: {len(id_train_features)} samples, {label_offset} classes")
    
    # 构建stats_dict
    print("\n[3/4] Building stats dict...")
    stats_dict = build_stats_dict_from_features(id_train_features, id_train_labels)
    
    # 构建OOD检测器
    print(f"\n[4/4] Building {args.detector_type} OOD detector...")
    if args.detector_type == "mahalanobis":
        detector = MahalanobisOODDetector.from_stats_dict(
            stats_dict=stats_dict,
            alpha=0.2,
            device=args.device
        )
    else:
        detector = ClassifierBasedOODDetector(
            stats_dict=stats_dict,
            classifier_type=args.detector_type,
            device=args.device,
            rank=32,
            qda_reg_alpha1=0.6,
            qda_reg_alpha2=1.0,
            qda_reg_alpha3=0.5
        )
    
    # 提取ID测试特征（用于计算ID分数）
    print("\nEvaluating on ID test sets...")
    id_test_scores = []
    id_test_info = []
    
    for dataset_name in args.id_datasets:
        _, test_transform = get_transforms(dataset_name)
        test_loader, _, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[dataset_name],
            transform_test=test_transform, batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, _ = extract_features(model, test_loader, args.device)
        scores = detector.predict_score(features.to(args.device))
        
        id_test_scores.extend(scores.cpu().numpy().tolist())
        id_test_info.append({
            'dataset': dataset_name,
            'num_samples': len(features),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std())
        })
    
    print(f"  Total ID test samples: {len(id_test_scores)}")
    
    # 提取OOD测试特征（用于计算OOD分数）
    print("\nEvaluating on OOD test sets...")
    ood_test_scores = []
    ood_test_info = []
    
    for dataset_name in args.ood_datasets:
        _, test_transform = get_transforms(dataset_name)
        test_loader, _, _ = get_xtail_testloader(
            root=args.root, dataset_sequence=[dataset_name],
            transform_test=test_transform, batch_size=args.batch_size,
            max_num_per_dataset=1000
        )
        
        features, _ = extract_features(model, test_loader, args.device)
        scores = detector.predict_score(features.to(args.device))
        
        ood_test_scores.extend(scores.cpu().numpy().tolist())
        ood_test_info.append({
            'dataset': dataset_name,
            'num_samples': len(features),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std())
        })
    
    print(f"  Total OOD test samples: {len(ood_test_scores)}")
    
    # 计算OOD检测指标
    print("\n" + "="*80)
    print("OOD Detection Metrics")
    print("="*80)
    
    ood_metrics = calculate_ood_metrics(id_test_scores, ood_test_scores)
    
    print(f"AUROC: {ood_metrics['auroc']:.4f}")
    print(f"FPR@95TPR: {ood_metrics['fpr_at_95_tpr']:.4f}")
    print(f"Detection Error: {ood_metrics['detection_error']:.4f}")
    print(f"Optimal Threshold: {ood_metrics['optimal_threshold']:.4f}")
    print("="*80)
    
    # 保存结果
    results = {
        'experiment': {
            'table': 'Table 3',
            'name': f'OOD Detector - {args.detector_type}',
            'detector_type': args.detector_type,
            'timestamp': str(torch.randn(1).item()),  # placeholder
        },
        'configuration': {
            'id_datasets': args.id_datasets,
            'ood_datasets': args.ood_datasets,
            'num_shots': args.num_shots,
            'model_name': args.model_name,
        },
        'results': {
            'ood_metrics': ood_metrics,
            'id_test_info': id_test_info,
            'ood_test_info': ood_test_info,
        },
        'metrics': {
            'auroc': float(ood_metrics['auroc']),
            'fpr_at_95_tpr': float(ood_metrics['fpr_at_95_tpr']),
            'detection_error': float(ood_metrics['detection_error']),
            'optimal_threshold': float(ood_metrics['optimal_threshold']),
        },
        'paths': {
            'raw_data': os.path.join(args.output_dir, 'raw_scores.json'),
            'log': os.path.join(args.output_dir, 'eval.log'),
        }
    }
    
    # 保存主要结果
    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存原始分数（用于后续分析）
    raw_scores = {
        'id_scores': id_test_scores,
        'ood_scores': ood_test_scores,
    }
    with open(os.path.join(args.output_dir, 'raw_scores.json'), 'w') as f:
        json.dump(raw_scores, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    args = parse_args()
    evaluate_ood_detector(args)


if __name__ == "__main__":
    main()
