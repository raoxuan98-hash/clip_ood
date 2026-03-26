# classifier/base_classifier_builder.py
from abc import ABC, abstractmethod
from typing import Dict
import torch.nn as nn
from .gaussian_statistics import GaussianStatistics

class BaseClassifierBuilder(ABC):
    """所有分类器构建器的统一接口"""

    @abstractmethod
    def build(self, stats_dict: Dict[int, GaussianStatistics]) -> nn.Module:
        pass
    
