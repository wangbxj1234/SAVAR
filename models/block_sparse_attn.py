# Block Sparse Attention for VAR
# Optimized for training with fixed initial scales (FIXED_L = 66)
# This avoids computing attention for tokens that won't contribute to loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockSparseAttention(nn.Module):
    """
    Block diagonal attention mask for VAR training
    Only computes attention within each scale (block diagonal pattern)
    
    For VAR with patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16):
    - Scale 1: tokens 0-0 (1x1)
    - Scale 2: tokens 1-4 (2x2) 
    - Scale 3: tokens 5-13 (3x3)
    - ...
    - Scale 10: tokens 2084-2239 (16x16)
    
    Each scale only attends to itself, no cross-scale attention.
    """
    
    def __init__(self, patch_nums, fixed_l=66):
        super().__init__()
        self.patch_nums = patch_nums
        self.fixed_l = fixed_l
        self.L = sum(pn**2 for pn in patch_nums)
        
        # Compute block boundaries
        self.block_boundaries = []
        cur = 0
        for pn in patch_nums:
            size = pn * pn
            self.block_boundaries.append((cur, cur + size))
            cur += size
        
        # Pre-compute block diagonal mask
        # 1 where attention is allowed, 0 where it should be masked
        mask = torch.zeros(self.L, self.L, dtype=torch.bool)
        for start, end in self.block_boundaries:
            mask[start:end, start:end] = 1
        
        # Convert to attention bias: 0 for allowed, -inf for masked
        attn_bias = torch.where(mask, 0.0, float('-inf'))
        self.register_buffer('attn_bias', attn_bias)
        
    def get_attention_bias(self, B, H, device, dtype):
        """
        Returns attention bias for block diagonal pattern
        Shape: (1, 1, L, L) -> broadcasts to (B, H, L, L)
        """
        return self.attn_bias.to(device=device, dtype=dtype).view(1, 1, self.L, self.L)


class OptimizedSelfAttention(nn.Module):
    """
    Optimized self-attention with:
    1. Block sparse attention support
    2. Gradient checkpointing option
    3. Better memory layout
    """
    
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, 
        flash_if_available=True, patch_nums=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), 
                requires_grad=True
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            import math
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        
        # Import flash attention if available
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func if flash_if_available else None
        except ImportError:
            self.flash_attn_func = None
        
        self.using_flash = self.flash_attn_func is not None
        
        # Block sparse attention
        if patch_nums is not None:
            self.block_sparse = BlockSparseAttention(patch_nums)
        else:
            self.block_sparse = None
        
        # For inference only
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    def forward(self, x, attn_bias, use_block_sparse=False):
        """
        Args:
            x: (B, L, C)
            attn_bias: (1, 1, L, L) or None
            use_block_sparse: whether to use block sparse attention
        """
        B, L, C = x.shape
        
        # Compute Q, K, V
        qkv = F.linear(
            input=x, 
            weight=self.mat_qkv.weight, 
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        ).view(B, L, 3, self.num_heads, self.head_dim)
        
        main_type = qkv.dtype
        
        # Use block sparse attention during training
        if use_block_sparse and self.block_sparse is not None and self.training:
            attn_bias = self.block_sparse.get_attention_bias(B, self.num_heads, x.device, main_type)
        
        # Choose attention implementation
        using_flash = (
            self.using_flash and 
            attn_bias is None and 
            qkv.dtype != torch.float32 and
            not use_block_sparse  # flash doesn't support custom masks well
        )
        
        if using_flash:
            q, k, v = qkv.unbind(dim=2)  # BLHc
            dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            dim_cat = 2
        
        # L2 normalization
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        # KV caching (inference only)
        if self.caching:
            if self.cached_k is None:
                self.cached_k, self.cached_v = k, v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        
        # Attention computation
        if using_flash:
            oup = self.flash_attn_func(
                q.to(dtype=main_type), 
                k.to(dtype=main_type), 
                v.to(dtype=main_type),
                dropout_p=dropout_p, 
                softmax_scale=self.scale
            ).view(B, L, C)
        else:
            # Standard attention with mask
            attn = q.mul(self.scale) @ k.transpose(-2, -1)  # BHLc @ BHcL => BHLL
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            if dropout_p > 0:
                attn = F.dropout(attn, p=dropout_p, inplace=True)
            oup = (attn @ v).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        return (f'using_flash={self.using_flash}, attn_l2_norm={self.attn_l2_norm}, '
                f'block_sparse={self.block_sparse is not None}')
