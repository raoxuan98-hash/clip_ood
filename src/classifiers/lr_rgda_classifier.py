import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .da_classifier_builder import LRRGDAClassifierBuilder
from .gaussian_statistics import GaussianStatistics


class LRRGDAClassifier:
    """
    LR-RGDA分类器（基于类别统计分布构建）
    
    该类完全基于各类别的高斯统计分布（均值和协方差）构建，
    无需原始数据或特征向量，适用于增量学习场景。
    
    Usage:
        # 1. 准备类别统计分布
        stats_dict = {
            0: GaussianStatistics(mean_0, cov_0),
            1: GaussianStatistics(mean_1, cov_1),
            ...
        }
        
        # 2. 构建分类器
        classifier = LRRGDAClassifier(stats_dict, device='cuda')
        
        # 3. 预测
        predictions = classifier.predict(features)
    """
    
    def __init__(
        self, 
        stats_dict: Dict[int, GaussianStatistics], 
        device: str = 'cuda',
        rank: int = 32,  # 优化后的默认值
        qda_reg_alpha1: float = 0.3,
        qda_reg_alpha2: float = 0.3,
        qda_reg_alpha3: float = 0.3,
        temperature: float = 1.0
    ):
        """
        Args:
            stats_dict: 类别统计分布字典 {class_id: GaussianStatistics}
            device: 计算设备
            rank: 低秩分解的秩
            qda_reg_alpha1: 类内协方差权重
            qda_reg_alpha2: 全局协方差权重
            qda_reg_alpha3: 单位矩阵正则化权重
            temperature: 温度参数（用于概率输出的softmax）
        """
        self.device = device
        self.stats_dict = stats_dict
        
        # 构建LR-RGDA分类器
        builder = LRRGDAClassifierBuilder(
            rank=rank,
            qda_reg_alpha1=qda_reg_alpha1,
            qda_reg_alpha2=qda_reg_alpha2,
            qda_reg_alpha3=qda_reg_alpha3,
            temperature=temperature,
            device=device
        )
        
        self.classifier = builder.build(stats_dict)
        self.num_classes = len(stats_dict)
    
    def predict_proba(self, features: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """
        预测概率
        
        Args:
            features: 输入特征 [B, D]
            temperature: 可选的温度参数，覆盖初始化时的设置
            
        Returns:
            概率分布 [B, num_classes]
        """
        return self.classifier.predict_proba(features, temperature=temperature)
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            features: 输入特征 [B, D]
            
        Returns:
            预测类别 [B]
        """
        return self.classifier.predict(features)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播（返回logits）"""
        return self.classifier.forward(features)
    
    @property
    def class_ids(self):
        """获取类别ID列表"""
        return sorted(self.stats_dict.keys())

class EnsembleClassifier:
    def __init__(self, zeroshot_classifier, lr_rgda_classifier, alpha=0.5, num_id_classes=None):
        """
        集成分类器
        
        Args:
            zeroshot_classifier: 零样本分类器权重 [D, num_all_classes]
            lr_rgda_classifier: LR-RGDA分类器（只覆盖ID类别）
            alpha: 集成权重 (LR-RGDA对ID类别的贡献)
            num_id_classes: ID类别数量。如果为None，假设所有类别都是ID类别
        """
        self.zeroshot_classifier = zeroshot_classifier
        self.lr_rgda_classifier = lr_rgda_classifier
        self.alpha = alpha  # 集成权重
        self.num_id_classes = num_id_classes  # ID类别数量
        
        if num_id_classes is not None:
            # 分离ID类别和OOD类别的Zeroshot权重
            self.zeroshot_id = zeroshot_classifier[:, :num_id_classes]  # ID类别
            self.zeroshot_ood = zeroshot_classifier[:, num_id_classes:]  # OOD类别
    
    def predict_proba(self, features, logit_scale_zeroshot):
        """预测概率"""
        if self.num_id_classes is None:
            # 原始行为：所有类别都使用Ensemble
            zeroshot_logits = features @ self.zeroshot_classifier
            zeroshot_logits = zeroshot_logits - zeroshot_logits.max(dim=-1, keepdim=True).values
            zeroshot_logits = logit_scale_zeroshot * zeroshot_logits
            zeroshot_probs = F.softmax(zeroshot_logits, dim=-1)

            rgda_logits = self.lr_rgda_classifier.forward(features)
            rgda_logits = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
            rgda_probs = F.softmax(rgda_logits, dim=-1)
            
            ensemble_probs = self.alpha * rgda_probs + (1 - self.alpha) * zeroshot_probs
            
            return ensemble_probs
            
        else:
            # 扩展类别空间：所有类别都参与集成
            # ID类别: alpha * LR-RGDA + (1-alpha) * Zeroshot
            # OOD类别: (1-alpha) * Zeroshot (LR-RGDA对这些类别输出0)
            batch_size = features.shape[0]
            num_all_classes = self.zeroshot_classifier.shape[1]

            zeroshot_logits = features @ self.zeroshot_classifier
            zeroshot_logits = zeroshot_logits - zeroshot_logits.max(dim=-1, keepdim=True).values
            zeroshot_logits = logit_scale_zeroshot * zeroshot_logits
            zeroshot_probs = F.softmax(zeroshot_logits, dim=1)
            
            # 2. 计算LR-RGDA概率（只覆盖ID类别）
            lr_rgda_logits_id = self.lr_rgda_classifier.forward(features)
            lr_rgda_logits_id = lr_rgda_logits_id - lr_rgda_logits_id.max(dim=-1, keepdim=True).values
            lr_rgda_probs_id = F.softmax(lr_rgda_logits_id, dim=1)

            # 再套一层 softmax
            # 将LR-RGDA概率扩展到所有类别（OOD类别补0）
            lr_rgda_probs = torch.zeros(batch_size, num_all_classes, device=features.device)
            lr_rgda_probs[:, :self.num_id_classes] = lr_rgda_probs_id
            
            # 3. 集成：alpha * LR-RGDA + (1-alpha) * Zeroshot
            # 这对所有类别都适用：
            # - ID类别：正常融合
            # - OOD类别：LR-RGDA部分为0，只剩 (1-alpha) * Zeroshot
            ensemble_probs = torch.zeros_like(zeroshot_probs)
            ensemble_probs[:, :self.num_id_classes] = self.alpha * lr_rgda_probs[:, :self.num_id_classes] + (1 - self.alpha) * zeroshot_probs[:, :self.num_id_classes]
            ensemble_probs[:, self.num_id_classes:] = zeroshot_probs[:, self.num_id_classes:]
            # ensemble_probs[:, self.num_id_classes:] = zeroshot_probs[:, self.num_id_classes:] * torch.norm(ensemble_probs[:, :self.num_id_classes], dim=1, keepdim=True) / torch.norm(zeroshot_probs[:, self.num_id_classes:], dim=1, keepdim=True)
            return ensemble_probs
    
    def predict(self, features, logit_scale):
        """预测类别"""
        probs = self.predict_proba(features, logit_scale)
        return torch.argmax(probs, dim=1)