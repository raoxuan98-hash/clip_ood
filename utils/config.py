"""
配置工具模块

该模块提供与旧版 YACS 配置的向后兼容，
并增加了向新 Pydantic 配置系统迁移的适配器。

推荐使用新配置系统:
    >>> from configs import ConfigManager
    >>> config = ConfigManager("config.yaml")

向后兼容 (旧代码):
    >>> from utils.config import _C
    >>> _C.merge_from_file("config.yaml")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# 尝试导入 YACS，如果不存在则提供警告
try:
    from yacs.config import CfgNode as CN
    YACS_AVAILABLE = True
except ImportError:
    YACS_AVAILABLE = False
    # 创建一个假的 CfgNode 类以保持向后兼容
    class CN(dict):
        """模拟 YACS CfgNode 的简化版本"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._frozen = False
        
        def merge_from_file(self, cfg_filename):
            raise ImportError("YACS is not installed. Use new config system: from configs import ConfigManager")
        
        def merge_from_list(self, cfg_list):
            raise ImportError("YACS is not installed. Use new config system: from configs import ConfigManager")
        
        def freeze(self):
            self._frozen = True
        
        def defrost(self):
            self._frozen = False
        
        def is_frozen(self):
            return self._frozen
        
        def clone(self):
            """创建深拷贝"""
            import copy
            return copy.deepcopy(self)
        
        def __getattr__(self, name):
            """支持点号访问"""
            if name.startswith('_'):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        def __setattr__(self, name, value):
            """支持点号设置"""
            if name.startswith('_'):
                super().__setattr__(name, value)
            else:
                self[name] = value


# 尝试导入新的配置系统
try:
    from configs import ConfigManager, ConfigSchema
    NEW_CONFIG_AVAILABLE = True
except ImportError:
    NEW_CONFIG_AVAILABLE = False
    ConfigManager = None
    ConfigSchema = None


# =============================================================================
# 旧版 YACS 配置定义 (向后兼容)
# =============================================================================

_C = CN()

_C.dataset = ""
_C.dataset_sequence = []
_C.root = ""
_C.num_shots = -1

_C.backbone = ""
_C.resolution = 224

_C.output_dir = None
_C.model_dir = None
_C.print_freq = 10

_C.seed = None
_C.deterministic = False
_C.gpu = None
_C.num_workers = 20
_C.prec = "amp"

_C.num_epochs = 10
_C.batch_size = 128
_C.lr = 0.01
_C.weight_decay = 5e-4
_C.loss_type = "CE"
_C.theta = 1.0
_C.lada_k = 16
_C.prototype_k = 1
_C.image_prototypes_weight_coef = 1.0

_C.t_full_tuning = False
_C.t_bias_tuning = False
_C.t_ln_tuning = False
_C.t_adapter = False
_C.t_adaptformer = False
_C.t_lora = False
_C.t_lora_mlp = False
_C.t_ssf_attn = False
_C.t_ssf_mlp = False
_C.t_ssf_ln = False
_C.t_mask = False
_C.t_partial = None
_C.t_adapter_dim = None
_C.t_mask_ratio = None
_C.t_mask_seed = None
_C.alpha = 1.0
_C.beta = 1.0

_C.zero_shot = False
_C.continue_train = False
_C.continue_train_first = False


# =============================================================================
# 配置转换函数
# =============================================================================

def cn_to_dict(cfg_node: CN) -> Dict[str, Any]:
    """将 YACS CfgNode 转换为字典"""
    result = {}
    for key, value in cfg_node.items():
        if isinstance(value, CN):
            result[key] = cn_to_dict(value)
        else:
            result[key] = value
    return result


def convert_to_new_config(cfg_node: CN) -> "ConfigManager":
    """将旧的 YACS 配置转换为新的配置管理器"""
    if not NEW_CONFIG_AVAILABLE:
        raise ImportError(
            "New config system is not available. "
            "Please install pydantic and pyyaml."
        )
    
    old_dict = cn_to_dict(cfg_node)
    
    new_dict = {
        "model": {
            "clip_variant": old_dict.get("backbone", "CLIP-ViT-B/16"),
            "resolution": old_dict.get("resolution", 224),
            "lora_k": old_dict.get("lada_k", 16),
            "prototype_k": old_dict.get("prototype_k", 1),
            "image_prototypes_weight_coef": old_dict.get("image_prototypes_weight_coef", 1.0),
            "peft_method": "adaptformer" if old_dict.get("t_adaptformer") else "lora" if old_dict.get("t_lora") else "none",
            "peft_dim": old_dict.get("t_adapter_dim"),
        },
        "training": {
            "num_epochs": old_dict.get("num_epochs", 10),
            "batch_size": old_dict.get("batch_size", 128),
            "lr": old_dict.get("lr", 0.01),
            "weight_decay": old_dict.get("weight_decay", 5e-4),
            "loss_type": old_dict.get("loss_type", "CE"),
        },
        "data": {
            "dataset_sequence": old_dict.get("dataset_sequence", []),
            "root": old_dict.get("root", ""),
            "num_shots": old_dict.get("num_shots", -1),
            "num_workers": old_dict.get("num_workers", 8),
        },
        "pipeline": {
            "output_dir": old_dict.get("output_dir"),
            "model_dir": old_dict.get("model_dir"),
            "print_freq": old_dict.get("print_freq", 10),
            "seed": old_dict.get("seed"),
            "deterministic": old_dict.get("deterministic", False),
            "gpu": old_dict.get("gpu"),
            "prec": old_dict.get("prec", "amp"),
            "zero_shot": old_dict.get("zero_shot", False),
            "continue_train": old_dict.get("continue_train", False),
            "continue_train_first": old_dict.get("continue_train_first", False),
        },
    }
    
    return ConfigManager(new_dict)


# =============================================================================
# 向后兼容的辅助函数
# =============================================================================

def merge_from_file(cfg: CN, cfg_file: str) -> None:
    """从文件合并配置（向后兼容）"""
    if YACS_AVAILABLE:
        cfg.merge_from_file(cfg_file)
    else:
        raise ImportError("YACS is not installed. Use new config system: from configs import ConfigManager")


def get_cfg_defaults() -> CN:
    """获取默认配置（向后兼容）"""
    return _C.clone()


# 为了完全向后兼容，导出所有内容
__all__ = [
    "_C",
    "CN",
    "cn_to_dict",
    "convert_to_new_config",
    "merge_from_file",
    "get_cfg_defaults",
    "YACS_AVAILABLE",
    "NEW_CONFIG_AVAILABLE",
]

# 如果新的配置系统可用，导出相关类
if NEW_CONFIG_AVAILABLE:
    __all__.extend([
        "ConfigManager",
        "ConfigSchema",
    ])
