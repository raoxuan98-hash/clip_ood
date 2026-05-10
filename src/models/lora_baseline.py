"""
普通 LoRA (Baseline) 实现
无 SGP/NSP 投影，作为最基础的 LoRA 基线
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional


class VanillaLoRALinear(nn.Module):
    """
    普通 LoRA 线性层
    无投影矩阵，仅使用低秩分解：W' = W + B @ A
    """
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        device = linear.weight.device
        dtype = linear.weight.dtype
        
        # LoRA 参数 A 和 B
        # A: (r, in_features), B: (out_features, r)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        
        # 初始化：A 使用 Kaiming，B 使用零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Bias
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：原始输出 + LoRA 分支"""
        # 原始线性输出
        original_output = self.linear(x)
        
        # LoRA 分支：x @ A^T @ B^T
        # x: (batch, seq_len, in_features)
        # lora_A: (r, in_features) -> 需要转置
        # lora_B: (out_features, r) -> 需要转置
        result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return original_output + result
    
    def merge_lora_weights(self):
        """将 LoRA 权重合并到原始权重中（用于推理加速）"""
        with torch.no_grad():
            # delta_W = B @ A * scaling
            delta_W = self.lora_B @ self.lora_A * self.scaling
            self.linear.weight.data += delta_W
            
            # 重置 LoRA 参数
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def get_lora_params(self):
        """获取 LoRA 可训练参数"""
        return [self.lora_A, self.lora_B]


class VanillaLoRACLIPVisionTransformer(nn.Module):
    """
    使用普通 LoRA 的 CLIP Vision Transformer
    作为最基础的 LoRA 基线，无 SGP/NSP 投影
    """
    def __init__(
        self,
        clip_vision_model: nn.Module,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        include_norm: bool = False
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        self.lora_alpha = lora_alpha
        self.feature_dim = clip_vision_model.embeddings.patch_embedding.out_channels
        
        # 冻结原始参数
        for n, p in clip_vision_model.named_parameters():
            if include_norm and ("norm" in n or "layernorm" in n.lower()):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        
        # 确定哪些层使用 LoRA
        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(clip_vision_model.encoder.layers)))
        self.lora_modules = nn.ModuleDict()
        
        # 遍历每一层 Transformer 添加 LoRA
        for idx, layer in enumerate(clip_vision_model.encoder.layers):
            if idx not in self.lora_layer:
                continue
            
            # === Self-Attention Projections ===
            for proj_name in ["k_proj", "v_proj", "q_proj", "out_proj"]:
                linear = getattr(layer.self_attn, proj_name)
                lora_mod = VanillaLoRALinear(linear, r, lora_alpha, lora_dropout)
                setattr(layer.self_attn, proj_name, lora_mod)
                self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod
            
            # === MLP ===
            for mlp_name in ["fc1", "fc2"]:
                linear = getattr(layer.mlp, mlp_name)
                lora_mod = VanillaLoRALinear(linear, r, lora_alpha, lora_dropout)
                setattr(layer.mlp, mlp_name, lora_mod)
                self.lora_modules[f"layer_{idx}_mlp_{mlp_name}"] = lora_mod
        
        self.clip_vision_model = clip_vision_model
    
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        return self.clip_vision_model(pixel_values, **kwargs)
    
    def get_params(self):
        """获取可训练参数"""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    def merge_lora_weights(self):
        """合并所有 LoRA 权重到原始权重"""
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()
    
    def get_module_names(self):
        """获取所有 LoRA 模块名称"""
        return list(self.lora_modules.keys())
    
    def regularization_loss(self) -> torch.Tensor:
        """普通 LoRA 无特殊正则化"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


def get_vanilla_lora_model(args):
    """
    获取普通 LoRA 模型（基线）
    
    Args:
        args: 包含 lora_rank, lora_alpha 等参数
        
    Returns:
        model, processor
    """
    from transformers import CLIPModel, CLIPProcessor
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
    
    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False
    
    # 获取 LoRA 参数
    rank = getattr(args, 'lora_rank', 4)
    alpha = getattr(args, 'lora_alpha', rank)  # 默认 alpha = rank
    dropout = getattr(args, 'lora_dropout', 0.0)
    
    # 替换 vision_model 为 VanillaLoRA 版本
    model.vision_model = VanillaLoRACLIPVisionTransformer(
        model.vision_model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout
    )
    
    return model, processor
