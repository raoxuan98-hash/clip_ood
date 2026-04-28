import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional
import logging

def compute_weights(x: torch.Tensor, weight_kind="log1p", beta=1.0, weight_p=1.0, weight_alpha=0.5, weight_kappa=2.0) -> torch.Tensor:
    if weight_kind == "exp":
        return torch.exp(-beta * x)

    elif weight_kind == "rational1":
        return 1.0 / (1.0 + beta * x)

    elif weight_kind == "rational2":
        return 1.0 / (1.0 + beta * (x ** 2))

    elif weight_kind == "sqrt_rational2":
        return 1.0 / torch.sqrt(1.0 + beta * (x ** 2))

    elif weight_kind == "log1p":
        return 1.0 / (1.0 + beta * torch.log1p(x**weight_p))

    elif weight_kind == "power_family":
        return (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)

    elif weight_kind == "stretched_exp":
        return torch.exp(- (beta * x) ** weight_kappa)

    else:
        raise ValueError(
            f"Unknown weight_kind='{weight_kind}'. "
            f"Choose from ['exp','rational1','rational2','sqrt_rational2','log1p','power_family','stretched_exp']")

class FixedProjection(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)

    def forward(self):
        return self.P

class SGPBaseLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        proj: nn.Module):

        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj

        device = linear.weight.device
        dtype = linear.weight.dtype
        
        self.A = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)

        self.register_buffer("lora_active", torch.tensor(True, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            P_scaled = self.P()
            A_eff = self.A @ P_scaled
            lora_delta = self.B @ A_eff
            adapted_weight = self.linear.weight + lora_delta
            return F.linear(x, adapted_weight, self.bias)
        else:
           return self.linear(x)

    def merge_lora_weights(self, lora_active: bool=True) -> None:
        """将 LoRA 权重合并到原始权重中"""
        with torch.no_grad():
            P_scaled = self.P()
            delta = self.B @ self.A @ P_scaled
            self.linear.weight.data.add_(delta)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B.data.zero_()
            self.lora_active = torch.tensor(lora_active)


class SGPBaseDoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        proj: nn.Module):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj

        # LoRA 参数
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # DoRA 参数：方向 + 幅度
        with torch.no_grad():
            weight = linear.weight.data
            weight_norm = weight.norm(p=2, dim=1, keepdim=True) + 1e-8
            self.weight_directions = nn.Parameter(
                weight / weight_norm, requires_grad=False)
            self.magnitude = nn.Parameter(weight_norm.clone(), requires_grad=True)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)

        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P_scaled = self.P()
        if self.lora_active:
            A_eff = self.A @ P_scaled
            lora_delta = self.B @ A_eff  # (out, in)
            adapted_weight = (self.weight_directions + lora_delta) * self.magnitude
        else:
            adapted_weight = self.weight_directions * self.magnitude
        return F.linear(x, adapted_weight, self.bias)

    def merge_lora_weights(self, lora_active: bool=True) -> None:
        with torch.no_grad():
            P_scaled = self.P()
            lora_delta = self.B @ self.A @ P_scaled
            self.weight_directions.data.add_(lora_delta)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B.data.zero_()
            self.lora_active = torch.tensor(lora_active)

class LoRACLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        clip_vision_model: nn.Module,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        use_soft_projection: bool = True,
        weight_temp: float = 1.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.02,
        lora_class: type = SGPBaseDoRA,
        include_norm: bool = False):

        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        self.feature_dim = clip_vision_model.embeddings.patch_embedding.out_channels  #768

        self.use_soft_projection = use_soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p

        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight

        for n, p in clip_vision_model.named_parameters():
            if include_norm and ("norm" in n or "layernorm" in n.lower()):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(clip_vision_model.encoder.layers)))
        self.lora_modules = nn.ModuleDict()

        # 设备和 dtype 推断
        dev = clip_vision_model.embeddings.patch_embedding.weight.device
        dtype = clip_vision_model.embeddings.patch_embedding.weight.dtype

        def make_placeholder(d):
            return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

        # 遍历每一层 Transformer
        for idx, layer in enumerate(clip_vision_model.encoder.layers):
            if idx not in self.lora_layer:
                continue

            # === Self-Attention Projections ===
            for proj_name in ["k_proj", "v_proj", "q_proj", "out_proj"]:
                linear = getattr(layer.self_attn, proj_name)
                proj = make_placeholder(linear.in_features)
                lora_mod = lora_class(linear, r, proj)
                setattr(layer.self_attn, proj_name, lora_mod)
                self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod

            # === MLP ===
            for mlp_name in ["fc1", "fc2"]:
                linear = getattr(layer.mlp, mlp_name)
                proj = make_placeholder(linear.in_features)
                lora_mod = lora_class(linear, r, proj)
                setattr(layer.mlp, mlp_name, lora_mod)
                self.lora_modules[f"layer_{idx}_mlp_{mlp_name}"] = lora_mod

        self.clip_vision_model = clip_vision_model

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue

            P = build_projection(
                cov,
                soft_projection=self.use_soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight)
            
            self.lora_modules[name].P = FixedProjection(P)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.clip_vision_model(pixel_values, kwargs)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def finalize_without_lora(self) -> None:
        self.eval()
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights(lora_active=False)

    def merge_lora_weights(self):
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()

    def get_params(self):
        params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            else:
                params.append(param)
        return params


def build_projection(
    cov: torch.Tensor,
    soft_projection: bool = True,
    weight_temp: float = 5.0,
    nsp_eps = 0.05, 
    nsp_weight = 0.0,
    *,
    weight_kind: str = "log1p",
    weight_alpha: float = 0.5,
    weight_p: float = 2.0,
    weight_kappa: float = 2 ) -> torch.Tensor:

# --- [修改点] 提升计算精度和稳定性 ---
    # 1. 强制转换为 float64 (双精度)，这是解决 MKL Argument 8 错误的核心
    cov_double = cov.to(torch.float64)
    
    # 2. 检查并清理 NaN 或 Inf，防止脏数据导致崩溃
    if torch.isnan(cov_double).any() or torch.isinf(cov_double).any():
        cov_double = torch.nan_to_num(cov_double, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 3. 确保矩阵绝对对称（理论上 cov 是对称的，但浮点误差会导致微小不对称，引发 eigh 报错）
    cov_double = (cov_double + cov_double.t()) / 2.0
    
    # 4. 增加正则化项 (岭系数)，将 1e-6 提高到 1e-4 以增强稳定性
    safe_eps = 1e-4
    cov_double = cov_double + safe_eps * torch.eye(cov_double.size(0), device=cov_double.device, dtype=torch.float64)
    
    try:
        # 5. 在双精度下进行特征值分解
        eigvals_double, eigvecs_double = torch.linalg.eigh(cov_double)
    except RuntimeError:
        # [备选方案] 如果 GPU 分解失败，尝试在 CPU 上分解（CPU 的 MKL 库通常比 GPU 更鲁棒）
        logging.warning("GPU eigh failed, falling back to CPU...")
        eigvals_double, eigvecs_double = torch.linalg.eigh(cov_double.cpu())
        eigvals_double = eigvals_double.to(cov.device)
        eigvecs_double = eigvecs_double.to(cov.device)
    
    # 6. 计算完成后，转回模型原本的精度（float16 或 float32）
    # [修改点] 将结果从双精度转回原精度，并移回 GPU (cuda)
    eigvals = eigvals_double.to(dtype=cov.dtype, device='cuda')
    eigvecs = eigvecs_double.to(dtype=cov.dtype, device='cuda')
    # --- [修改结束] ---
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    sum_vals = eigvals.sum()
    scale_ = d / (sum_vals + safe_eps)
    eigvals = eigvals * scale_


    if soft_projection:
        weights = compute_weights(eigvals, weight_kind, weight_temp, weight_p, weight_alpha, weight_kappa)
        max_weight = weights.max()
        weights = weights / max_weight
        diag_w = torch.diag(weights)
        P = eigvecs @ diag_w @ eigvecs.t()
    else:
        eps_hard = nsp_eps
        total = eigvals.sum()
        cumsum = torch.cumsum(eigvals, dim=0)
        ratio = cumsum / (total + 1e-12)
        idx = (ratio >= eps_hard).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        P = V_keep @ V_keep.t()
        I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
        P = (1 - nsp_weight) * P + nsp_weight * I
    
    # [修改点] 确保返回的 P 矩阵一定在显卡上，与模型权重设备对齐
    P = P.to(device='cuda')
    return P