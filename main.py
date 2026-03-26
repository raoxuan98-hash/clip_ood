import os
import torch
import random
import argparse
import numpy as np
from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector, MahalanobisOODDetector, build_stats_dict_from_features
from src.routing.adaptive_router import AdaptiveRouter
from src.utils.feature_extractor import extract_features
from src.utils.evaluation import calculate_ood_metrics, calculate_classification_accuracy
from src.utils.reference_loader import load_reference_dataset
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-shot Classification Continual Learning")
    
    # 数据集相关参数
    parser.add_argument("--id_datasets", type=list, default=["caltech101", "flowers", "oxford_pets", "stanford_cars", "food101"], help="List of ID datasets for training.")
    parser.add_argument("--ood_datasets", type=list, default=["dtd", "eurosat", "mnist", "sun397"], help="List of OOD datasets for evaluation.")
    parser.add_argument("--root", type=str, default="/home/raoxuan/projects/data/X-TAIL/", help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16, help="Number of shots for few-shot learning.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--max_num_per_test_dataset", type=int, default=1000, help="Maximum number of samples per test dataset.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--iterations", type=int, default=800, help="Number of training iterations.")
    
    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="Weight decay for optimizer.")
    
    # LoRA相关参数
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_type", type=str, default="lora_nsp", choices=["lora_sgp", "lora_nsp"], help="Type of LoRA adaptation.")
    parser.add_argument("--nsp_eps", type=float, default=0.05, help="Epsilon parameter for NSP.")
    parser.add_argument("--nsp_weight", type=float, default=0.02, help="Weight parameter for NSP.")
    parser.add_argument("--weight_temp", type=float, default=1.0, help="Temperature parameter for weight.")
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0, help="P parameter for weight function.")

    # 参考数据集参数
    parser.add_argument("--reference_dataset", type=str, default="flickr8k", help="Reference dataset for training.")
    parser.add_argument("--reference_batch_size", type=int, default=32, help="Batch size for reference dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    
    # 损失函数权重参数
    parser.add_argument("--fd_weight", type=float, default=1.0, help="Weight for feature distillation loss.")
    parser.add_argument("--cd_weight", type=float, default=1.0, help="Weight for cross-modal distillation loss.")
    
    # 分类器参数
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for LR-RGDA classifier in ensemble.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for zero-shot classifier.")
    
    # OOD检测参数
    parser.add_argument("--ood_detector_type", type=str, default="lr_rgda", choices=["mahalanobis", "lda", "lr_rgda", "qda"], help="Type of OOD detector.")
    parser.add_argument("--ood_threshold", type=float, default=0.5, help="Threshold for OOD detection.")
    parser.add_argument("--mahalanobis_alpha", type=float, default=0.2, help="Alpha parameter for Mahalanobis OOD detector.")
    
    # 推理端消融实验参数 (Table 4)
    parser.add_argument("--classifier_type", type=str, default=None, choices=["zeroshot", "ensemble"], help="Classifier type for evaluation: zeroshot or ensemble.")
    parser.add_argument("--enable_routing", action="store_true", help="Enable adaptive routing for OOD detection.")
    parser.add_argument("--use_cached_features", action="store_true", help="Use cached features for evaluation.")
    parser.add_argument("--cache_dir", type=str, default="cache/features", help="Directory for cached features.")
    
    # 增量学习模式
    parser.add_argument("--incremental_mode", type=bool, default=False, help="Whether to use incremental learning mode.")
    parser.add_argument("--dataset_sequence", type=list, default=[["caltech101"], ["flowers"], ["oxford_pets"], ["stanford_cars"], ["food101"]], help="Sequence of datasets for incremental learning.")

    args = parser.parse_args()
    return args

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

def main(args):
    """主程序"""
    # 设置随机种子
    if args.seed is not None:
        fix_random_seed(args.seed)

    # 初始化训练器
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    # 加载参考数据集
    reference_loader = load_reference_dataset(args, trainer.model_pretrain, processor, args.device)

    if not args.incremental_mode:
        # 第一种学习模式：联合微调
        print("\n=== Starting Joint Fine-tuning ===")
        
        # 准备训练数据
        train_loaders = []
        all_class_names = []
        label_offset = 0
        
        for d_name in args.id_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            train_loaders.append(tr_loader)
            all_class_names.extend(c_names)
        
        # 合并训练数据
        from torch.utils.data import ConcatDataset, DataLoader
        merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
        
        # 训练模型
        model = trainer.train(merged_loader, all_class_names, reference_loader)
        
        # 提取特征用于训练分类器和OOD检测器
        print("\n=== Extracting Features ===")
        all_features = []
        all_labels = []
        label_offset = 0
        
        for i, d_name in enumerate(args.id_datasets):
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            
            features, labels = extract_features(model, tr_loader, args.device)
            all_features.append(features)
            all_labels.append(labels + label_offset)
            label_offset += len(c_names)
        
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        
        # 构建类别统计分布字典
        print("\n=== Building Stats Dict ===")
        from src.detectors.ood_detector import build_stats_dict_from_features
        stats_dict = build_stats_dict_from_features(all_features, all_labels)
        
        # 构建LR-RGDA分类器
        print("\n=== Building LR-RGDA Classifier ===")
        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=stats_dict,
            device=args.device,
            rank=32,
            qda_reg_alpha1=0.6,
            qda_reg_alpha2=1.0,
            qda_reg_alpha3=0.5,
            temperature=1.0
        )
        
        # 构建零样本分类器
        print("\n=== Building Zero-shot Classifier ===")
        zeroshot_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
        
        # 构建集成分类器
        print("\n=== Building Ensemble Classifier ===")
        ensemble_classifier = EnsembleClassifier(
            zeroshot_classifier, 
            lr_rgda_classifier, 
            alpha=args.alpha, 
            temperature=args.temperature
        )
        
        # 构建OOD检测器
        print("\n=== Building OOD Detector ===")
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
        
        # 构建自适应路由分类器
        print("\n=== Building Adaptive Router ===")
        router = AdaptiveRouter(
            zeroshot_classifier, 
            ensemble_classifier, 
            ood_detector, 
            threshold=args.ood_threshold
        )
        
        # 评估
        print("\n=== Evaluating ===")
        all_accuracies = {}
        
        # 评估ID数据集
        for d_name in args.id_datasets:
            _, test_transform = get_transforms(d_name)
            _, te_loader, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=None, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            
            features, labels = extract_features(model, te_loader, args.device)
            predictions, is_ood = router.predict(features.to(args.device), model.logit_scale)
            accuracy = calculate_classification_accuracy(predictions.cpu().numpy(), labels.numpy())
            all_accuracies[f"{d_name}_id"] = accuracy
            print(f"{d_name} ID accuracy: {accuracy:.4f}")
        
        # 评估OOD数据集
        ood_scores = []
        id_scores = []
        
        for d_name in args.ood_datasets:
            _, test_transform = get_transforms(d_name)
            _, te_loader, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name, 
                transform_train=None, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            
            features, labels = extract_features(model, te_loader, args.device)
            predictions, is_ood = router.predict(features.to(args.device), model.logit_scale)
            
            # 计算OOD分数
            ood_score = ood_detector.predict_score(features.to(args.device))
            ood_scores.extend(ood_score.cpu().numpy())
            
            # 对于ID数据集，我们也需要计算OOD分数以评估检测性能
            for id_d_name in args.id_datasets:
                _, test_transform = get_transforms(id_d_name)
                _, te_loader, _, c_names = get_xtail_trainloader(
                    root=args.root, dataset_name=id_d_name, 
                    transform_train=None, transform_test=test_transform,
                    num_shots=args.num_shots, batch_size=args.batch_size
                )
                features, labels = extract_features(model, te_loader, args.device)
                id_score = ood_detector.predict_score(features.to(args.device))
                id_scores.extend(id_score.cpu().numpy())
        
        # 计算OOD检测指标
        ood_metrics = calculate_ood_metrics(id_scores, ood_scores)
        print("\nOOD Detection Metrics:")
        print(f"AUROC: {ood_metrics['auroc']:.4f}")
        print(f"FPR@95TPR: {ood_metrics['fpr_at_95_tpr']:.4f}")
        print(f"Detection Error: {ood_metrics['detection_error']:.4f}")
        
    else:
        # 第二种学习模式：增量学习
        print("\n=== Starting Incremental Learning ===")
        
        # 初始化历史记录
        history_classifiers = []
        history_ood_detectors = []
        history_class_names = []
        
        for i, task_datasets in enumerate(args.dataset_sequence):
            print(f"\n=== Task {i+1}: {task_datasets} ===")
            
            # 准备训练数据
            train_loaders = []
            task_class_names = []
            label_offset = sum(len(c_names) for c_names in history_class_names)
            
            for d_name in task_datasets:
                train_transform, test_transform = get_transforms(d_name)
                tr_loader, _, _, c_names = get_xtail_trainloader(
                    root=args.root, dataset_name=d_name, 
                    transform_train=train_transform, transform_test=test_transform,
                    num_shots=args.num_shots, batch_size=args.batch_size
                )
                train_loaders.append(tr_loader)
                task_class_names.extend(c_names)
            
            # 合并训练数据
            from torch.utils.data import ConcatDataset, DataLoader
            merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
            merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
            
            # 训练模型
            model = trainer.train(merged_loader, task_class_names, reference_loader)
            
            # 提取特征
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
                
                features, labels = extract_features(model, tr_loader, args.device)
                task_features.append(features)
                task_labels.append(labels + label_offset)
            
            task_features = torch.cat(task_features)
            task_labels = torch.cat(task_labels)
            
            # 构建类别统计分布字典
            from src.detectors.ood_detector import build_stats_dict_from_features
            task_stats_dict = build_stats_dict_from_features(task_features, task_labels)
            
            # 构建LR-RGDA分类器
            lr_rgda_classifier = LRRGDAClassifier(
                stats_dict=task_stats_dict,
                device=args.device,
                rank=32,
                qda_reg_alpha1=0.6,
                qda_reg_alpha2=1.0,
                qda_reg_alpha3=0.5,
                temperature=1.0
            )
            
            # 构建零样本分类器
            all_class_names = history_class_names + [task_class_names]
            flat_class_names = [name for sublist in all_class_names for name in sublist]
            zeroshot_classifier = get_zeroshot_classifier(model, processor, flat_class_names, args.device)
            
            # 构建集成分类器
            ensemble_classifier = EnsembleClassifier(
                zeroshot_classifier, 
                lr_rgda_classifier, 
                alpha=args.alpha, 
                temperature=args.temperature
            )
            
            # 构建OOD检测器
            if args.ood_detector_type == "mahalanobis":
                ood_detector = MahalanobisOODDetector.from_stats_dict(
                    stats_dict=task_stats_dict,
                    alpha=args.mahalanobis_alpha,
                    device=args.device
                )
            else:
                ood_detector = ClassifierBasedOODDetector(
                    stats_dict=task_stats_dict,
                    classifier_type=args.ood_detector_type,
                    device=args.device,
                    rank=32,
                    qda_reg_alpha1=0.6,
                    qda_reg_alpha2=1.0,
                    qda_reg_alpha3=0.5
                )
            
            # 保存到历史记录
            history_classifiers.append(ensemble_classifier)
            history_ood_detectors.append(ood_detector)
            history_class_names.append(task_class_names)
            
            # 评估
            print("\n=== Evaluating Task ===")
            # 这里可以添加评估代码

    print("\n=== Training and Evaluation Completed ===")

if __name__ == "__main__":
    # 解析命令行参数
    command_line_args = parse_args()
    
    # 开始训练和评估
    main(command_line_args)