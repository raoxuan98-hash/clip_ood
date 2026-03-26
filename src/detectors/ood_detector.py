import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from sklearn.covariance import EmpiricalCovariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm


class MahalanobisOODDetector:
    """
    基于类别统计分布的 Mahalanobis 距离 OOD 检测器
    
    完全基于各类别的均值和协方差矩阵构建，无需原始数据。
    OOD分数定义为：样本到最近类别的Mahalanobis距离
    
    Usage:
        # 1. 准备类别统计分布（均值和协方差）
        class_means = torch.stack([mean_0, mean_1, ...])  # [C, D]
        class_precisions = torch.stack([prec_0, prec_1, ...])  # [C, D, D]
        
        # 2. 构建检测器
        detector = MahalanobisOODDetector(class_means, class_precisions, device='cuda')
        
        # 3. 计算OOD分数
        ood_scores = detector.predict_score(features)
    """
    
    def __init__(
        self, 
        class_means: torch.Tensor, 
        class_precisions: torch.Tensor,
        device: str = "cuda"
    ):
        """
        Args:
            class_means: 类别均值 [C, D]
            class_precisions: 类别精度矩阵（协方差的逆）[C, D, D]
            device: 计算设备
        """
        self.device = device
        self.class_means = class_means.to(device)
        self.class_precisions = class_precisions.to(device)
        self.num_classes = class_means.shape[0]
        
        print(f"✓ Mahalanobis OOD Detector initialized with {self.num_classes} classes")
    
    @torch.no_grad()
    def predict_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算 OOD 分数（Mahalanobis距离）
        
        分数越高，越可能是OOD样本
        
        Args:
            features: 输入特征 [B, D]
            
        Returns:
            Mahalanobis距离 [B]
        """
        features = features.to(self.device)
        B, D = features.shape
        C = self.num_classes
        
        # 计算到每个类的Mahalanobis距离: (x - μ)^T Σ^{-1} (x - μ)
        diff = features.unsqueeze(1) - self.class_means.unsqueeze(0)  # [B, C, D]
        temp = torch.einsum('bcd,cde->bce', diff, self.class_precisions)  # [B, C, D]
        dists = (temp * diff).sum(dim=-1)  # [B, C]
        
        # 取最近类的距离作为OOD分数
        min_dists, _ = dists.min(dim=1)  # [B]
        
        return min_dists
    
    @classmethod
    def from_stats_dict(
        cls, 
        stats_dict: Dict[int, 'GaussianStatistics'], 
        alpha: float = 0.2,
        device: str = "cuda"
    ) -> 'MahalanobisOODDetector':
        """
        从GaussianStatistics字典构建Mahalanobis检测器
        
        Args:
            stats_dict: {class_id: GaussianStatistics}
            alpha: 类特定协方差的权重 (0=纯共享协方差, 1=纯类特定协方差)
            device: 计算设备
            
        Returns:
            MahalanobisOODDetector实例
        """
        class_ids = sorted(stats_dict.keys())
        num_classes = len(class_ids)
        
        # 获取维度
        first_stats = stats_dict[class_ids[0]]
        d = first_stats.mean.shape[0]
        
        # 收集均值
        means_list = []
        for cid in class_ids:
            means_list.append(stats_dict[cid].mean.to(device))
        class_means = torch.stack(means_list)  # [C, D]
        
        # 计算共享协方差
        global_cov = torch.zeros(d, d, device=device)
        for cid in class_ids:
            global_cov.add_(stats_dict[cid].cov.to(device))
        global_cov.div_(num_classes)
        
        # 计算每个类的插值协方差和精度矩阵
        precisions_list = []
        reg_epsilon = 1e-6 * torch.eye(d, device=device)
        
        for cid in class_ids:
            cov = stats_dict[cid].cov.to(device)
            # 插值: alpha * class_cov + (1-alpha) * global_cov
            sigma = alpha * cov + (1 - alpha) * global_cov
            sigma = sigma + reg_epsilon
            
            try:
                prec = torch.linalg.inv(sigma)
            except torch.linalg.LinAlgError:
                prec = torch.linalg.pinv(sigma)
            
            precisions_list.append(prec)
        
        class_precisions = torch.stack(precisions_list)  # [C, D, D]
        
        return cls(class_means, class_precisions, device)


class ClassifierBasedOODDetector:
    """
    基于类别统计分布的分类器 OOD 检测器
    
    该检测器完全基于各类别的高斯统计分布（均值和协方差）构建，
    无需原始数据或特征向量。OOD分数定义为：1 - max(后验概率)
    
    支持分类器类型: lda, lr_rgda, qda
    
    Usage:
        # 1. 准备类别统计分布
        stats_dict = {
            0: GaussianStatistics(mean_0, cov_0),
            1: GaussianStatistics(mean_1, cov_1),
            ...
        }
        
        # 2. 构建检测器
        detector = ClassifierBasedOODDetector(
            stats_dict, 
            classifier_type='lr_rgda',
            device='cuda'
        )
        
        # 3. 计算OOD分数
        ood_scores = detector.predict_score(features)
    """
    
    def __init__(
        self, 
        stats_dict: Dict[int, 'GaussianStatistics'],
        classifier_type: str = "lr_rgda",
        device: str = "cuda",
        **classifier_kwargs
    ):
        """
        Args:
            stats_dict: 类别统计分布字典 {class_id: GaussianStatistics}
            classifier_type: 分类器类型 ("lda", "lr_rgda", "qda")
            device: 计算设备
            classifier_kwargs: 分类器特定参数
                - lda: reg_alpha
                - lr_rgda/qda: qda_reg_alpha1/2/3, rank
        """
        self.device = device
        self.classifier_type = classifier_type
        self.classifier_kwargs = classifier_kwargs
        self.stats_dict = stats_dict
        self.num_classes = len(stats_dict)
        
        # 构建底层分类器
        self.classifier = self._build_classifier(stats_dict)
        
        print(f"✓ {classifier_type.upper()} OOD Detector initialized with {self.num_classes} classes")
    
    def _build_classifier(self, stats_dict: Dict[int, 'GaussianStatistics']):
        """根据类型构建分类器"""
        from src.classifiers.da_classifier_builder import LDAClassifierBuilder, LRRGDAClassifierBuilder, RegularQDAClassifierBuilder
        
        if self.classifier_type == "lda":
            builder = LDAClassifierBuilder(
                reg_alpha=self.classifier_kwargs.get('reg_alpha', 0.3),
                device=self.device
            )

        elif self.classifier_type == "lr_rgda":
            builder = LRRGDAClassifierBuilder(
                qda_reg_alpha1=self.classifier_kwargs.get('qda_reg_alpha1', 0.3),
                qda_reg_alpha2=self.classifier_kwargs.get('qda_reg_alpha2', 0.3),
                qda_reg_alpha3=self.classifier_kwargs.get('qda_reg_alpha3', 0.3),
                rank=self.classifier_kwargs.get('rank', 32),
                device=self.device
            )

        elif self.classifier_type == "qda":
            builder = RegularQDAClassifierBuilder(
                qda_reg_alpha1=self.classifier_kwargs.get('qda_reg_alpha1', 0.3),
                qda_reg_alpha2=self.classifier_kwargs.get('qda_reg_alpha2', 0.3),
                qda_reg_alpha3=self.classifier_kwargs.get('qda_reg_alpha3', 0.3),
                device=self.device
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        return builder.build(stats_dict)
    
    @torch.no_grad()
    def predict_score(self, features: torch.Tensor, temp=0.05) -> torch.Tensor:
        logits = self.classifier.forward(features)
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits / temp, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        ood_scores = 1.0 - max_probs

        # 简单的能量分数实现
        # nergy_scores = torch.logsumexp(logits, dim=1) 
        # AUC 计算（注意能量分数越高通常代表越像 ID，可能需要取负号作为 OOD Score）
        # auc = roc_auc_score(labels, -energy_scores) 
        return ood_scores
    
    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测后验概率（用于调试或可视化）
        
        Args:
            features: 输入特征 [B, D]
            
        Returns:
            后验概率 [B, num_classes]
        """
        return self.classifier.predict_proba(features)
    
    @property
    def class_ids(self):
        """获取类别ID列表"""
        return sorted(self.stats_dict.keys())


# ============== 辅助函数 ==============

def build_stats_dict_from_features(
    features: torch.Tensor, 
    labels: torch.Tensor
) -> Dict[int, 'GaussianStatistics']:
    """
    从特征和标签构建类别统计分布字典
    
    这是一个辅助函数，用于将原始特征转换为stats_dict格式，
    以便与 ClassifierBasedOODDetector 和 LRRGDAClassifier 兼容。
    
    Args:
        features: 特征向量 [N, D]
        labels: 标签 [N]
        
    Returns:
        类别统计分布字典 {class_id: GaussianStatistics}
    """
    from src.classifiers.gaussian_statistics import GaussianStatistics
    
    unique_classes = torch.unique(labels)
    stats_dict = {}
    
    for c in unique_classes:
        idx = (labels == c)
        data_c = features[idx]
        mu = torch.mean(data_c, dim=0)
        # 处理样本数不足的情况
        if data_c.shape[0] > 1:
            cov = torch.cov(data_c.T)
        else:
            # 单样本时，使用单位矩阵作为协方差
            cov = torch.eye(data_c.shape[1], device=data_c.device)
        
        stats_dict[int(c)] = GaussianStatistics(mu, cov)
    
    return stats_dict


def extract_stats_dict_from_model(
    model,
    dataset_names: list,
    args,
    device: str = "cuda"
) -> Dict[int, 'GaussianStatistics']:
    """
    从数据集和模型提取类别统计分布
    
    这是一个辅助函数，用于在训练流程中从数据构建stats_dict。
    
    Args:
        model: CLIP模型或其他特征提取器
        dataset_names: 数据集名称列表
        args: 参数对象，需包含root, num_shots等
        device: 计算设备
        
    Returns:
        类别统计分布字典 {class_id: GaussianStatistics}
    """
    from src.classifiers.gaussian_statistics import GaussianStatistics
    from utils_data import get_xtail_trainloader, get_transforms
    
    model.eval()
    stats_dict = {}
    label_offset = 0
    
    for d_name in dataset_names:
        transform, _ = get_transforms(d_name)
        tr_loader, _, _, c_names = get_xtail_trainloader(
            root=args.root, 
            dataset_name=d_name,
            transform_train=transform, 
            transform_test=None,
            num_shots=args.num_shots, 
            batch_size=32
        )
        
        # 收集特征
        all_feats = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, lbls in tqdm(tr_loader, desc=f"Extracting {d_name}", leave=False):
                imgs = imgs.to(device)
                feats = model.get_image_features(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())
                all_labels.append(lbls + label_offset)
        
        all_feats = torch.cat(all_feats)
        all_labels = torch.cat(all_labels)
        
        # 计算每个类的统计量
        unique_labels = torch.unique(all_labels)
        for c in unique_labels:
            idx = (all_labels == c)
            data_c = all_feats[idx]
            mu = torch.mean(data_c, dim=0)
            if data_c.shape[0] > 1:
                cov = torch.cov(data_c.T)
            else:
                cov = torch.eye(data_c.shape[1])
            
            stats_dict[int(c)] = GaussianStatistics(mu, cov)
        
        label_offset += len(c_names)
    
    return stats_dict
