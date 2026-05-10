#!/usr/bin/env python3
"""
使用缓存特征运行推理端实验
支持 Table 3/4/5 的快速实验
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import CLIPModel, CLIPProcessor
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector, MahalanobisOODDetector, build_stats_dict_from_features
from src.routing.adaptive_router import AdaptiveRouter
from src.utils.evaluation import calculate_ood_metrics, calculate_classification_accuracy


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with cached features")
    
    # 数据集配置
    parser.add_argument("--id_datasets", type=str, nargs='+',
                       default=["caltech101", "flowers", "oxford_pets"],
                       help="ID datasets")
    parser.add_argument("--ood_datasets", type=str, nargs='+',
                       default=["dtd", "eurosat", "mnist"],
                       help="OOD datasets")
    parser.add_argument("--root", type=str,
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="Dataset root directory")
    
    # 缓存配置
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Directory containing cached features")
    
    # 模型配置
    parser.add_argument("--model_name", type=str,
                       default="openai/clip-vit-base-patch16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    # 分类器配置
    parser.add_argument("--classifier_type", type=str, default=None,
                       choices=["zeroshot", "ensemble"],
                       help="Classifier type for ablation")
    parser.add_argument("--alpha", type=float, default=0.8,
                       help="Ensemble weight for LR-RGDA")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for zero-shot classifier")
    
    # OOD检测器配置
    parser.add_argument("--ood_detector_type", type=str, default="lr_rgda",
                       choices=["mahalanobis", "lda", "qda", "lr_rgda"],
                       help="OOD detector type")
    parser.add_argument("--ood_threshold", type=float, default=0.993,
                       help="OOD detection threshold for routing")
    parser.add_argument("--enable_routing", action="store_true",
                       help="Enable adaptive routing")
    parser.add_argument("--mahalanobis_alpha", type=float, default=0.2,
                       help="Alpha for Mahalanobis detector")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def load_cached_features(cache_dir, dataset_name):
    """加载缓存的特征"""
    cache_file = os.path.join(cache_dir, f"{dataset_name}_features.pkl")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    return data


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


def evaluate_with_cached_features(args):
    """使用缓存特征进行评估"""
    
    print("="*80)
    print("Cached Feature Experiment")
    print("="*80)
    print(f"ID datasets: {args.id_datasets}")
    print(f"OOD datasets: {args.ood_datasets}")
    print(f"Classifier type: {args.classifier_type or 'adaptive_routing'}")
    print(f"OOD detector: {args.ood_detector_type}")
    print(f"Cache directory: {args.cache_dir}")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载CLIP模型（仅用于零样本分类器文本编码）
    print("\nLoading CLIP model...")
    model = CLIPModel.from_pretrained(args.model_name).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    # 收集所有类别的名称（ID + OOD）
    all_class_names = []
    for dataset_name in args.id_datasets:
        cache_data = load_cached_features(args.cache_dir, dataset_name)
        all_class_names.extend(cache_data['class_names'])
    
    # 也收集OOD数据集的类别名（用于构建完整的零样本分类器）
    for dataset_name in args.ood_datasets:
        try:
            cache_data = load_cached_features(args.cache_dir, dataset_name)
            all_class_names.extend(cache_data['class_names'])
        except FileNotFoundError:
            print(f"Warning: Cache not found for OOD dataset {dataset_name}")
    
    print(f"\nTotal classes: {len(all_class_names)}")
    
    # 收集所有ID特征和标签
    print("\nLoading ID features from cache...")
    all_train_features = []
    all_train_labels = []
    label_offset = 0
    
    for dataset_name in args.id_datasets:
        cache_data = load_cached_features(args.cache_dir, dataset_name)
        
        train_features = cache_data['train_features']
        train_labels = cache_data['train_labels'] + label_offset
        
        all_train_features.append(train_features)
        all_train_labels.append(train_labels)
        label_offset += len(cache_data['class_names'])
        
        print(f"  {dataset_name}: {train_features.shape[0]} train samples, "
              f"{cache_data['test_features'].shape[0]} test samples")
    
    all_train_features = torch.cat(all_train_features)
    all_train_labels = torch.cat(all_train_labels)
    
    print(f"\nTotal train features: {all_train_features.shape}")
    
    # 构建stats_dict
    print("\nBuilding stats dict...")
    stats_dict = build_stats_dict_from_features(all_train_features, all_train_labels)
    
    # 构建零样本分类器
    print("\nBuilding zero-shot classifier...")
    zeroshot_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
    
    # 根据分类器类型进行评估
    results = {}
    
    if args.classifier_type == "zeroshot":
        # 仅使用零样本分类器
        print("\nEvaluating with Zero-shot Classifier...")
        id_acc, ood_acc = evaluate_zeroshot(
            args, model, zeroshot_classifier, all_class_names
        )
        results['classifier_type'] = 'zeroshot'
        results['id_accuracy'] = id_acc
        results['ood_accuracy'] = ood_acc
        
    elif args.classifier_type == "ensemble":
        # 使用集成分类器（无路由）
        print(f"\nEvaluating with Ensemble Classifier (alpha={args.alpha})...")
        id_acc, ood_acc = evaluate_ensemble(
            args, model, zeroshot_classifier, stats_dict, all_class_names
        )
        results['classifier_type'] = 'ensemble'
        results['alpha'] = args.alpha
        results['id_accuracy'] = id_acc
        results['ood_accuracy'] = ood_acc
        
    elif args.enable_routing:
        # 使用自适应路由
        print(f"\nEvaluating with Adaptive Routing (detector={args.ood_detector_type})...")
        id_acc, ood_acc, ood_metrics = evaluate_with_routing(
            args, model, processor, zeroshot_classifier, stats_dict, all_class_names
        )
        results['classifier_type'] = 'adaptive_routing'
        results['ood_detector_type'] = args.ood_detector_type
        results['ood_threshold'] = args.ood_threshold
        results['id_accuracy'] = id_acc
        results['ood_accuracy'] = ood_acc
        results['ood_metrics'] = ood_metrics
    else:
        print("\nWarning: No classifier type specified. Use --classifier_type or --enable_routing")
        return
    
    # 计算综合得分
    results['combined_score'] = (results['id_accuracy'] + results['ood_accuracy']) / 2
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)
    
    # 打印结果
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"ID Accuracy:  {results['id_accuracy']:.4f}")
    print(f"OOD Accuracy: {results['ood_accuracy']:.4f}")
    print(f"Combined Score: {results['combined_score']:.4f}")
    if 'ood_metrics' in results:
        print(f"AUROC: {results['ood_metrics']['auroc']:.4f}")
        print(f"FPR@95TPR: {results['ood_metrics']['fpr_at_95_tpr']:.4f}")
    print("="*80)
    print(f"Results saved to: {output_file}")


def evaluate_zeroshot(args, model, zeroshot_classifier, class_names):
    """使用零样本分类器评估"""
    
    id_correct = 0
    id_total = 0
    
    # 评估ID数据集 - 需要应用label_offset
    label_offset = 0
    for dataset_name in args.id_datasets:
        cache_data = load_cached_features(args.cache_dir, dataset_name)
        test_features = cache_data['test_features'].to(args.device)
        # 应用label_offset以匹配全局分类器
        test_labels = cache_data['test_labels'] + label_offset
        
        # 计算相似度
        logits = test_features @ zeroshot_classifier * model.logit_scale.exp()
        predictions = logits.argmax(dim=1).cpu()
        
        id_correct += (predictions == test_labels).sum().item()
        id_total += len(test_labels)
        
        # 更新offset
        label_offset += len(cache_data['class_names'])
    
    id_acc = id_correct / id_total if id_total > 0 else 0
    
    # 评估OOD数据集（使用零样本分类器）- OOD数据集使用全局zeroshot分类器
    ood_correct = 0
    ood_total = 0
    
    for dataset_name in args.ood_datasets:
        try:
            cache_data = load_cached_features(args.cache_dir, dataset_name)
            test_features = cache_data['test_features'].to(args.device)
            # OOD数据集使用全局zeroshot分类器，需要计算正确的label_offset
            # 首先计算该数据集在全局类别列表中的起始位置
            ood_label_offset = 0
            for id_ds in args.id_datasets:
                id_cache = load_cached_features(args.cache_dir, id_ds)
                ood_label_offset += len(id_cache['class_names'])
            # 然后加上该数据集在OOD列表中的位置
            for ood_ds in args.ood_datasets:
                if ood_ds == dataset_name:
                    break
                ood_cache = load_cached_features(args.cache_dir, ood_ds)
                ood_label_offset += len(ood_cache['class_names'])
            
            test_labels = cache_data['test_labels'] + ood_label_offset
            
            # 使用零样本分类器计算分类准确率
            logits = test_features @ zeroshot_classifier * model.logit_scale.exp()
            predictions = logits.argmax(dim=1).cpu()
            
            ood_correct += (predictions == test_labels).sum().item()
            ood_total += len(test_labels)
        except FileNotFoundError:
            print(f"  Warning: Cache not found for {dataset_name}")
            continue
    
    ood_acc = ood_correct / ood_total if ood_total > 0 else 0
    
    return id_acc, ood_acc


def evaluate_ensemble(args, model, zeroshot_classifier, stats_dict, class_names):
    """使用集成分类器评估（无路由）"""
    
    # 获取logit_scale
    logit_scale = model.logit_scale.exp()
    
    # 构建LR-RGDA分类器
    lr_rgda_classifier = LRRGDAClassifier(
        stats_dict=stats_dict,
        device=args.device,
        rank=32,
        qda_reg_alpha1=0.6,
        qda_reg_alpha2=1.0,
        qda_reg_alpha3=0.5,
        temperature=1.0
    )
    
    # 计算ID类别数量
    num_id_classes = len(stats_dict)
    
    # 构建集成分类器（传入num_id_classes以正确处理ID/OOD类别）
    ensemble = EnsembleClassifier(
        zeroshot_classifier,
        lr_rgda_classifier,
        alpha=args.alpha,
        temperature=args.temperature,
        num_id_classes=num_id_classes
    )
    
    # 评估ID数据集 - 应用label_offset
    id_correct = 0
    id_total = 0
    label_offset = 0
    
    for dataset_name in args.id_datasets:
        cache_data = load_cached_features(args.cache_dir, dataset_name)
        test_features = cache_data['test_features'].to(args.device)
        test_labels = cache_data['test_labels'] + label_offset
        
        predictions = ensemble.predict(test_features, logit_scale)
        id_correct += (predictions.cpu() == test_labels).sum().item()
        id_total += len(test_labels)
        
        label_offset += len(cache_data['class_names'])
    
    id_acc = id_correct / id_total if id_total > 0 else 0
    
    # 评估OOD数据集（使用集成分类器）- 集成分类器只包含ID类别，对OOD效果不好
    # 这里仍然进行评估，但预期OOD准确率会很低
    ood_correct = 0
    ood_total = 0
    
    # 计算OOD label offset
    ood_label_offset = 0
    for id_ds in args.id_datasets:
        id_cache = load_cached_features(args.cache_dir, id_ds)
        ood_label_offset += len(id_cache['class_names'])
    
    for dataset_name in args.ood_datasets:
        try:
            cache_data = load_cached_features(args.cache_dir, dataset_name)
            test_features = cache_data['test_features'].to(args.device)
            # OOD样本的labels在全局空间中
            current_ood_offset = ood_label_offset
            for ood_ds in args.ood_datasets:
                if ood_ds == dataset_name:
                    break
                ood_cache = load_cached_features(args.cache_dir, ood_ds)
                current_ood_offset += len(ood_cache['class_names'])
            test_labels = cache_data['test_labels'] + current_ood_offset
            
            # 使用集成分类器计算分类准确率
            predictions = ensemble.predict(test_features, logit_scale)
            ood_correct += (predictions.cpu() == test_labels).sum().item()
            ood_total += len(test_labels)
        except FileNotFoundError:
            continue
    
    ood_acc = ood_correct / ood_total if ood_total > 0 else 0
    
    return id_acc, ood_acc


def evaluate_with_routing(args, model, processor, zeroshot_classifier, stats_dict, class_names):
    """使用自适应路由评估"""
    
    # 构建LR-RGDA分类器
    lr_rgda_classifier = LRRGDAClassifier(
        stats_dict=stats_dict,
        device=args.device,
        rank=32,
        qda_reg_alpha1=0.6,
        qda_reg_alpha2=1.0,
        qda_reg_alpha3=0.5,
        temperature=1.0
    )
    
    # 计算ID类别数量
    num_id_classes = len(stats_dict)
    
    # 构建集成分类器（传入num_id_classes以正确处理ID/OOD类别）
    ensemble = EnsembleClassifier(
        zeroshot_classifier,
        lr_rgda_classifier,
        alpha=args.alpha,
        temperature=args.temperature,
        num_id_classes=num_id_classes
    )
    
    # 构建OOD检测器
    if args.ood_detector_type == "mahalanobis":
        ood_detector = MahalanobisOODDetector.from_stats_dict(
            stats_dict=stats_dict,
            alpha=args.mahalanobis_alpha,
            device=args.device
        )
    else:
        ood_detector = ClassifierBasedOODDetector(
            stats_dict=stats_dict,
            classifier_type=args.ood_detector_type,
            device=args.device,
            rank=32,
            qda_reg_alpha1=0.6,
            qda_reg_alpha2=1.0,
            qda_reg_alpha3=0.5
        )
    
    # 构建自适应路由器
    router = AdaptiveRouter(
        zeroshot_classifier,
        ensemble,
        ood_detector,
        threshold=args.ood_threshold
    )
    
    # 评估ID数据集 - 应用label_offset
    id_correct = 0
    id_total = 0
    label_offset = 0
    
    for dataset_name in args.id_datasets:
        cache_data = load_cached_features(args.cache_dir, dataset_name)
        test_features = cache_data['test_features'].to(args.device)
        test_labels = cache_data['test_labels'] + label_offset
        
        predictions, is_ood = router.predict(test_features, model.logit_scale.exp())
        id_correct += (predictions.cpu() == test_labels).sum().item()
        id_total += len(test_labels)
        
        label_offset += len(cache_data['class_names'])
    
    id_acc = id_correct / id_total if id_total > 0 else 0
    
    # 评估OOD数据集（使用路由分类器计算分类准确率）- 应用label_offset
    ood_correct = 0
    ood_total = 0
    
    # 计算OOD label offset
    ood_label_offset = 0
    for id_ds in args.id_datasets:
        id_cache = load_cached_features(args.cache_dir, id_ds)
        ood_label_offset += len(id_cache['class_names'])
    
    for dataset_name in args.ood_datasets:
        try:
            cache_data = load_cached_features(args.cache_dir, dataset_name)
            test_features = cache_data['test_features'].to(args.device)
            
            # 计算该OOD数据集在全局类别中的offset
            current_ood_offset = ood_label_offset
            for ood_ds in args.ood_datasets:
                if ood_ds == dataset_name:
                    break
                ood_cache = load_cached_features(args.cache_dir, ood_ds)
                current_ood_offset += len(ood_cache['class_names'])
            test_labels = cache_data['test_labels'] + current_ood_offset
            
            # 使用路由分类器（自动为OOD样本选择零样本分类器）
            predictions, is_ood = router.predict(test_features, model.logit_scale.exp())
            ood_correct += (predictions.cpu() == test_labels).sum().item()
            ood_total += len(test_labels)
        except FileNotFoundError:
            continue
    
    ood_acc = ood_correct / ood_total if ood_total > 0 else 0
    
    # 收集OOD分数用于OOD检测指标评估
    ood_scores_list = []
    id_scores_list = []
    
    for dataset_name in args.ood_datasets:
        try:
            cache_data = load_cached_features(args.cache_dir, dataset_name)
            test_features = cache_data['test_features'].to(args.device)
            
            ood_score = ood_detector.predict_score(test_features)
            ood_scores_list.extend(ood_score.cpu().numpy())
        except FileNotFoundError:
            continue
    
    # 收集ID分数用于OOD检测评估
    for dataset_name in args.id_datasets:
        cache_data = load_cached_features(args.cache_dir, dataset_name)
        test_features = cache_data['test_features'].to(args.device)
        
        id_score = ood_detector.predict_score(test_features)
        id_scores_list.extend(id_score.cpu().numpy())
    
    # 计算OOD检测指标
    ood_metrics = calculate_ood_metrics(id_scores_list, ood_scores_list)
    
    return id_acc, ood_acc, ood_metrics


if __name__ == "__main__":
    import json
    args = parse_args()
    evaluate_with_cached_features(args)
