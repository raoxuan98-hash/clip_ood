import torch
import numpy as np
from sklearn.model_selection import ParameterGrid
from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier, EnsembleClassifier
from src.detectors.ood_detector import ClassifierBasedOODDetector
from src.routing.adaptive_router import AdaptiveRouter
from src.utils.feature_extractor import extract_features
from src.utils.evaluation import calculate_ood_metrics, calculate_classification_accuracy
from src.utils.reference_loader import load_reference_dataset
from utils_data import get_xtail_trainloader, get_transforms

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

def evaluate_combination(args, model, processor, all_features, all_labels, all_class_names, id_datasets, ood_datasets, reference_loader, alpha, temperature, ood_threshold):
    """评估特定超参数组合的性能"""
    device = args.device
    
    # 构建LR-RGDA分类器
    lr_rgda_classifier = LRRGDAClassifier(device)
    lr_rgda_classifier.fit(all_features, all_labels)
    
    # 构建零样本分类器
    zeroshot_classifier = get_zeroshot_classifier(model, processor, all_class_names, device)
    
    # 构建集成分类器
    ensemble_classifier = EnsembleClassifier(
        zeroshot_classifier, 
        lr_rgda_classifier, 
        alpha=alpha, 
        temperature=temperature
    )
    
    # 构建OOD检测器
    ood_detector = ClassifierBasedOODDetector(
        model, 
        device, 
        classifier_type=args.ood_detector_type
    )
    ood_detector.fit_from_features(all_features, all_labels)
    
    # 构建自适应路由分类器
    router = AdaptiveRouter(
        zeroshot_classifier, 
        ensemble_classifier, 
        ood_detector, 
        threshold=ood_threshold
    )
    
    # 评估ID数据集
    id_accuracies = []
    for d_name in id_datasets:
        _, test_transform = get_transforms(d_name)
        _, te_loader, _, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name, 
            transform_train=None, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        
        features, labels = extract_features(model, te_loader, device)
        predictions, is_ood = router.predict(features.to(device), model.logit_scale)
        accuracy = calculate_classification_accuracy(predictions.cpu().numpy(), labels.numpy())
        id_accuracies.append(accuracy)
    
    # 评估OOD检测性能
    ood_scores = []
    id_scores = []
    
    for d_name in ood_datasets:
        _, test_transform = get_transforms(d_name)
        _, te_loader, _, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name, 
            transform_train=None, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        
        features, labels = extract_features(model, te_loader, device)
        ood_score = ood_detector.predict_score(features.to(device))
        ood_scores.extend(ood_score.cpu().numpy())
    
    for d_name in id_datasets:
        _, test_transform = get_transforms(d_name)
        _, te_loader, _, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name, 
            transform_train=None, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        features, labels = extract_features(model, te_loader, device)
        id_score = ood_detector.predict_score(features.to(device))
        id_scores.extend(id_score.cpu().numpy())
    
    ood_metrics = calculate_ood_metrics(id_scores, ood_scores)
    
    # 计算综合得分
    avg_id_accuracy = np.mean(id_accuracies)
    auroc = ood_metrics['auroc']
    fpr_at_95_tpr = ood_metrics['fpr_at_95_tpr']
    
    # 综合得分：ID准确率 * 0.6 + AUROC * 0.4
    # 同时考虑FPR@95TPR，越低越好
    composite_score = avg_id_accuracy * 0.6 + auroc * 0.4 - fpr_at_95_tpr * 0.1
    
    return {
        'avg_id_accuracy': avg_id_accuracy,
        'auroc': auroc,
        'fpr_at_95_tpr': fpr_at_95_tpr,
        'composite_score': composite_score
    }

def optimize_hyperparameters(args):
    """优化超参数"""
    # 初始化训练器
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    # 加载参考数据集
    reference_loader = load_reference_dataset(args, trainer.model_pretrain, processor, args.device)
    
    # 准备训练数据
    train_loaders = []
    all_class_names = []
    
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
    
    # 提取特征
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
    
    # 定义超参数搜索空间
    param_grid = {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
        'temperature': [0.5, 1.0, 1.5, 2.0],
        'ood_threshold': [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    
    # 遍历所有超参数组合
    best_score = -float('inf')
    best_params = None
    best_results = None
    
    grid = ParameterGrid(param_grid)
    total_combinations = len(grid)
    print(f"Total hyperparameter combinations: {total_combinations}")
    
    for i, params in enumerate(grid):
        print(f"Evaluating combination {i+1}/{total_combinations}: {params}")
        
        results = evaluate_combination(
            args, model, processor, all_features, all_labels, all_class_names, 
            args.id_datasets, args.ood_datasets, reference_loader,
            params['alpha'], params['temperature'], params['ood_threshold']
        )
        
        print(f"Results: {results}")
        
        if results['composite_score'] > best_score:
            best_score = results['composite_score']
            best_params = params
            best_results = results
            print(f"New best: {best_params} with score: {best_score}")
    
    print("\n=== Hyperparameter Optimization Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best results: {best_results}")
    
    return best_params, best_results