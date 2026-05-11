import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from .var import VAR

class VARFixed(VAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_sos(self, fhat_k: torch.Tensor, B: int) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            embed2 = self.proj1(self.down1(fhat_k))
            embed2 = self.cond_proj2(embed2)
            sos = embed2.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        return sos

    def forward(self, embed, fhat_k: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        B = embed.shape[0]
        # 使用全序列长度 L
        ed = self.L
        
        with torch.cuda.amp.autocast(enabled=False):
            # 1. 【关键修复】保留原始 embed，只用于 shared_ada_lin
            embed_for_cond = self.cond_proj(embed)
            
            # 2. 生成 sos (Scale 1 的输入)
            sos = self.get_sos(fhat_k, B)

            # 3. 构造完整序列 (sos + teacher forcing inputs)
            # x_BLCv_wo_first_l 提供了 Scale 2-10 的输入
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            
            # 4. 【关键修复】还原原版层级嵌入和位置嵌入
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

        # 5. 运行 Transformer Blocks
        cond_BD = self.shared_ada_lin(embed_for_cond)
        
        # 兼容混合精度类型
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD = cond_BD.to(dtype=main_type)
        attn_bias = self.attn_bias_for_masking.to(dtype=main_type)

        if self.training and getattr(self, 'use_checkpoint', False):
            from torch.utils.checkpoint import checkpoint
            for b in self.blocks:
                x_BLC = checkpoint(b, x_BLC, cond_BD, attn_bias, use_reentrant=False)
        else:
            for b in self.blocks:
                x_BLC = b(x=x_BLC, cond_BD=cond_BD, attn_bias=attn_bias)
        
        # 6. 【关键修复】传投影后的 embed 给 get_logits
        return self.get_logits(x_BLC.float(), embed_for_cond)

    @torch.no_grad()
    def forward_at_step(self, x_BLC, embed, fhat_k, ed):
        # 推理时也需要正确的投影逻辑
        embed_proj = self.cond_proj(embed)
        cond_BD = self.shared_ada_lin(embed_proj)
        attn_bias = self.attn_bias_for_masking[:, : , :ed, :ed]

        for b in self.blocks:
            x_BLC = b(x=x_BLC, cond_BD=cond_BD, attn_bias=attn_bias)

        return self.get_logits(x_BLC, embed_proj)
