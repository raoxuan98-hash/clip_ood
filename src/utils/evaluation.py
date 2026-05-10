import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

def calculate_ood_metrics(id_scores, ood_scores):
    """
    计算OOD检测的评估指标
    Args:
        id_scores: ID样本的OOD分数
        ood_scores: OOD样本的OOD分数
    Returns:
        metrics: 评估指标字典
    """
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # FPR@95TPR
    fpr_at_95_tpr = 1.0
    if np.any(tpr >= 0.95):
        idx = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95_tpr = fpr[idx]
    
    # Detection Error
    # 对于ID样本，错误是被分类为OOD；对于OOD样本，错误是被分类为ID
    # 取最佳阈值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    id_error = np.mean(id_scores > optimal_threshold)
    ood_error = np.mean(ood_scores <= optimal_threshold)
    detection_error = 0.5 * (id_error + ood_error)
    
    metrics = {
        'auroc': auroc,
        'fpr_at_95_tpr': fpr_at_95_tpr,
        'detection_error': detection_error,
        'optimal_threshold': optimal_threshold
    }
    
    return metrics

def calculate_classification_accuracy(predictions, labels):
    """
    计算分类准确率
    Args:
        predictions: 预测类别
        labels: 真实标签
    Returns:
        accuracy: 准确率
    """
    return accuracy_score(labels, predictions)

def evaluate_router(router, dataloader, logit_scale, device):
    """
    评估自适应路由分类器
    Args:
        router: 自适应路由分类器
        dataloader: 数据加载器
        logit_scale: CLIP的logit_scale
        device: 设备
    Returns:
        accuracy: 分类准确率
        ood_accuracy: OOD检测准确率
    """
    from src.utils.feature_extractor import extract_features
    
    features, labels = extract_features(dataloader.dataset.model, dataloader, device)
    predictions, is_ood = router.predict(features.to(device), logit_scale)
    
    # 计算分类准确率
    accuracy = calculate_classification_accuracy(predictions.cpu().numpy(), labels.numpy())
    
    # 计算OOD检测准确率
    # 这里假设dataloader中的样本是已知的ID或OOD样本
    # 实际应用中需要根据具体情况调整
    
    return accuracy