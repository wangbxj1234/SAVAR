"""VARNoSos: VAR with sos token removed from training forward.

Sequence length becomes L - first_l (skip scale 0).
Only overrides forward(); all weights/buffers inherited from VAR.
"""
import torch
import torch.nn as nn
from models.var import VAR
from models.basic_var import AdaLNSelfAttn


class VARNoSos(VAR):

    def forward(self, embed, fhat_k: torch.Tensor,
                x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        """Forward WITHOUT sos. Output shape: (B, L-first_l, V)."""
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = embed.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            indexs = torch.randperm(B)[:int(B * 0.1)]
            noise = self.noise(torch.tensor(0, device=embed.device)).unsqueeze(0)
            embed[indexs, :] = noise
            embed = self.cond_proj(embed)

            x_BLC = self.word_embed(x_BLCv_wo_first_l.float())
            x_BLC += (self.lvl_embed(self.lvl_1L[:, self.first_l:ed].expand(B, -1))
                      + self.pos_1LC[:, self.first_l:ed])

        attn_bias = self.attn_bias_for_masking[:, :, self.first_l:ed, self.first_l:ed]
        cond_BD_or_gss = self.shared_ada_lin(embed)

        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        if self.training and getattr(self, 'use_checkpoint', False):
            from torch.utils.checkpoint import checkpoint
            for i, b in enumerate(self.blocks):
                x_BLC = checkpoint(b, x_BLC, cond_BD_or_gss, attn_bias, use_reentrant=False)
        else:
            for i, b in enumerate(self.blocks):
                x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), embed)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC
