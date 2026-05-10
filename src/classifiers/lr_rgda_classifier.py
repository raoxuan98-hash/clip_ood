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
        self.zeroshot_classifier = zeroshot_classifier
        self.lr_rgda_classifier = lr_rgda_classifier
        self.alpha = alpha
        self.num_id_classes = num_id_classes

    def _get_ensemble_logits(self, features, logit_scale_zeroshot):
        """内部辅助函数：计算融合后的 Logits """
        # 1. 计算两边的原始 Logits 并进行数值稳定性处理 
        zs_logits = (features @ self.zeroshot_classifier) * logit_scale_zeroshot
        zs_logits = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
        
        rgda_logits = self.lr_rgda_classifier.forward(features)
        rgda_logits = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values

        if self.num_id_classes is None:
            # 场景1：全场融合
            return (1 - self.alpha) * zs_logits + self.alpha * rgda_logits
        else:
            # 场景2：ID区域
            ensemble_logits = zs_logits * (1 - self.alpha)
            # 只在 ID 对应的列上加上专家意见
            ensemble_logits[:, :self.num_id_classes] += self.alpha * rgda_logits
            return ensemble_logits

    def predict_proba(self, features, logit_scale_zeroshot):
        """如确实需要概率（比如算置信度）"""
        logits = self._get_ensemble_logits(features, logit_scale_zeroshot)
        return F.softmax(logits, dim=-1)

    def predict(self, features, logit_scale):
        """预测类别标签，直接对 Logits 进行 argmax"""
        logits = self._get_ensemble_logits(features, logit_scale)
        return torch.argmax(logits, dim=1)