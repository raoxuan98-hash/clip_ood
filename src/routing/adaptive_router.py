import torch
import torch.nn.functional as F

class AdaptiveRouter:
    """
    自适应路由分类器
    根据OOD检测结果动态选择分类器
    """
    def __init__(self, zeroshot_classifier, ensemble_classifier, ood_detector, threshold=0.9):
        """
        Args:
            zeroshot_classifier: 零样本分类器
            ensemble_classifier: 集成分类器
            ood_detector: OOD检测器
            threshold: OOD检测阈值
        """
        self.zeroshot_classifier = zeroshot_classifier
        self.ensemble_classifier = ensemble_classifier
        self.ood_detector = ood_detector
        self.threshold = threshold
    
    @torch.no_grad()
    def predict_proba(self, features, logit_scale_zeroshot = 5):
        """
        预测概率
        Args:
            features: 图像特征
            logit_scale: CLIP的logit_scale
        Returns:
            probs: 预测概率
            is_ood: 是否为OOD样本
        """
        # 计算OOD分数
        ood_scores = self.ood_detector.predict_score(features)
        
        # 判断是否为OOD样本
        is_ood = ood_scores > self.threshold
        
        # 初始化概率矩阵
        batch_size = features.shape[0]
        num_classes = self.zeroshot_classifier.shape[1]
        probs = torch.zeros((batch_size, num_classes), device=features.device)
        
        # 处理ID样本
        id_mask = ~is_ood
        if id_mask.sum() > 0:
            id_features = features[id_mask]
            id_probs = self.ensemble_classifier.predict_proba(id_features, logit_scale_zeroshot)
            probs[id_mask] = id_probs
        
        # 处理OOD样本
        ood_mask = is_ood
        if ood_mask.sum() > 0:
            ood_features = features[ood_mask]
            # 零样本分类器预测 (logit_scale已经是exp后的值)
            zeroshot_logits = ood_features @ self.zeroshot_classifier
            zeroshot_logits = zeroshot_logits - zeroshot_logits.max(dim=-1, keepdim=True).values
            ood_probs = F.softmax(logit_scale_zeroshot * zeroshot_logits, dim=1)
            probs[ood_mask] = ood_probs
        
        return probs, is_ood
    
    @torch.no_grad()
    def predict(self, features, logit_scale):
        """
        预测类别
        Args:
            features: 图像特征
            logit_scale: CLIP的logit_scale
        Returns:
            predictions: 预测类别
            is_ood: 是否为OOD样本
        """
        probs, is_ood = self.predict_proba(features, logit_scale)
        predictions = torch.argmax(probs, dim=1)
        return predictions, is_ood
    
    def set_threshold(self, threshold):
        """
        设置OOD检测阈值
        Args:
            threshold: OOD检测阈值
        """
        self.threshold = threshold
    
    def get_threshold(self):
        """
        获取当前OOD检测阈值
        Returns:
            threshold: 当前OOD检测阈值
        """
        return self.threshold