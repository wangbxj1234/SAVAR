from typing import Optional

import torch
import torch.nn as nn

from models.varslip import VARSlip


class VARSlipFhatCond(VARSlip):
    """VARSlip variant whose AdaLN condition is derived directly from f_hat_k."""

    def __init__(self, *args, fhat_cond_dim: int = 1024, **kwargs):
        super().__init__(*args, n_cond_embed=fhat_cond_dim, **kwargs)
        self.fhat_cond_dim = int(fhat_cond_dim)
        self.fhat_cond_head = nn.Sequential(
            nn.Conv2d(self.Cvae, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.fhat_cond_dim),
            nn.LayerNorm(self.fhat_cond_dim),
        )
        self._init_fhat_cond_head()

    def _init_fhat_cond_head(self) -> None:
        for module in self.fhat_cond_head.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def make_condition_embedding(self, fhat_k: torch.Tensor) -> torch.Tensor:
        return self.fhat_cond_head(fhat_k.float())

    def forward(self, fhat_k: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        embed = self.make_condition_embedding(fhat_k)
        return super().forward(embed, fhat_k, x_BLCv_wo_first_l)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        fhat_k: torch.Tensor,
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        profile_decode: bool = False,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if fhat_k.shape[0] != B:
            if fhat_k.shape[0] != 1:
                raise ValueError(f"fhat_k batch must be 1 or B={B}, got {fhat_k.shape[0]}")
            fhat_k = fhat_k.expand(B, -1, -1, -1)
        label_B = self.make_condition_embedding(fhat_k)
        return super().autoregressive_infer_cfg(
            B=B,
            label_B=label_B,
            fhat_k=fhat_k,
            g_seed=g_seed,
            cfg=cfg,
            top_k=top_k,
            top_p=top_p,
            more_smooth=more_smooth,
            profile_decode=profile_decode,
            autocast_dtype=autocast_dtype,
        )
