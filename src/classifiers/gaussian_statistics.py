import torch

def cholesky_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    if matrix.dim() == 3:
        batch_size, n, _ = matrix.shape
        reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        reg_eye = reg_eye.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.linalg.cholesky(matrix + reg_eye)
    
    elif matrix.dim() == 2:
        reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return torch.linalg.cholesky(matrix + reg_eye)
    else:
        raise ValueError(f"不支持的矩阵维度: {matrix.dim()}。支持2D或3D张量。")


class GaussianStatistics:
    """Container for per-class Gaussian statistics."""
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-4, cholesky = False):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        if mean.dim() != 1:
            raise AssertionError("GaussianStatistics.mean 必须是 1D 向量")

        self.mean = mean
        self.cov = cov
        self.reg = reg

        if cholesky:
            self.L = cholesky_stable(cov, reg=reg)
        else:
            self.L = None

    def to(self, device):
        """Move statistics to the requested device."""

        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        if self.L is not None:
            self.L = self.L.to(device)
        return self

    def sample(
        self,
        n_samples = None,
        cached_eps = None,
    ) -> torch.Tensor:
        """Draw samples from the Gaussian distribution."""

        if self.L is None:
            self.L = cholesky_stable(self.cov, reg=self.reg)

        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            if n_samples is None:
                raise ValueError("n_samples 必须在未提供 cached_eps 时给定")
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        samples = self.mean.unsqueeze(0) + eps @ self.L.t()
        return samples

class LowRankGaussianStatistics:
    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        rank: int = 512,
        reg: float = 1e-8, 
        device=None
    ):
        if mean.dim() != 1:
            raise ValueError("mean must be a 1D vector")
        
        d = mean.size(0)
        if cov.shape != (d, d):
            raise ValueError("cov shape mismatch")

        self.mean = mean
        self.d = d
        self.rank = rank or min(100, d)  # default max rank=100
        self.reg = reg

        U, S, Vh = torch.svd_lowrank(cov, q=rank, niter=4, M=None)
        U = U[:, :self.rank]
        S = S[:self.rank]

        self.U = U
        self.S = torch.clamp(S, min=0.0)

        if device:
            self.to(device)
    @property
    def L(self):
        return self.U * torch.sqrt(self.S).unsqueeze(0)

    @property
    def cov(self) -> torch.Tensor:
        """Reconstruct full covariance (use sparingly for high d!)"""
        return self.L @ self.L.T

    def to(self, device):
        self.mean = self.mean.to(device)
        return self

    def sample(self, n_samples: int) -> torch.Tensor:
        """Efficient sampling without forming full covariance"""
        eps = torch.randn(n_samples, self.L.size(1), device=self.L.device, dtype=self.L.dtype)
        return self.mean.unsqueeze(0) + eps @ self.L.T  # (n_samples, d)