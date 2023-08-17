import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import drop_path, trunc_normal_
import math
import torch.utils.checkpoint as checkpoint
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torch import Tensor, LongTensor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Extractor(nn.Module):
    def __init__(self, dim_base=128, dim=384, omega=4096):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        self.proj = nn.Sequential(
            nn.Linear(12 * dim_base, 12 * dim_base),
            nn.LayerNorm(12 * dim_base),
            nn.GELU(),
            nn.Linear(12 * dim_base, dim),
        )
        self.omega = omega

    def forward(self, x, Lmax=None):
    
        if Lmax is not None:
            # take random subset of the sequence
            idx_start = torch.randint(0, x.shape[1] - Lmax, (1,)).item()
            idx_end = idx_start + Lmax
            print(idx_start, idx_end)
            x = x[:, idx_start:idx_end, :]
        
        x = self.emb(self.omega * x).flatten(-2) # (B, T, 12, dim_base) -> (B, T, 12 * dim_base)
        
        return x
        
        x = self.proj(x)
        return x