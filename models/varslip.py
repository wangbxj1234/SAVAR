import math
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class VARSlip(nn.Module):
    """
    VARSlip: 与 VAR 相同，但 condition embed 维度改为 1024（SigLIP），
    fhat_k 路径（down1 + proj1 → cond_proj2）保持不变（768 维）。
    """
    def __init__(
            self, vae_local: VQVAE,
            n_cond_embed=1024, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
            use_checkpoint=False,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.use_checkpoint = use_checkpoint

        self.patch_nums: Tuple[int] = patch_nums
        max_patch_num = max(patch_nums)
        fhat_k_spatial_size = max_patch_num

        # fhat_k 路径：完全不变，输出 768 维
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.proj1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * fhat_k_spatial_size * fhat_k_spatial_size, 768),
            nn.LayerNorm(768)
        )

        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        # Runtime debug stats: updated every forward for observability.
        self.last_cond_drop_count = 0
        self.last_cond_drop_ratio = 0.0
        self.prog_si = -1  # progressive training
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        init_std = math.sqrt(1 / self.C / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, std=0.02)

        # 2. condition embedding
        # noise for embed path (1024 维，匹配 condition model 输出)
        self.noise = nn.Embedding(1, n_cond_embed)
        nn.init.trunc_normal_(self.noise.weight.data, std=0.02)
        # noise for fhat path (768 维，匹配 proj1 输出)
        self.noise_fhat = nn.Embedding(1, 768)
        nn.init.trunc_normal_(self.noise_fhat.weight.data, std=0.02)

        # cond_proj: condition model 输出 (1024) → transformer 维度 (C)
        self.cond_proj = nn.Linear(n_cond_embed, self.C)
        # cond_proj2: fhat 路径 proj1 输出 (768) → transformer 维度 (C)，硬编码 768
        self.cond_proj2 = nn.Linear(768, self.C)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, std=0.02)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, std=0.02)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VARSlip config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g}, cond_drop_rate={cond_drop_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask
        d: torch.Tensor = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L,
                                                                                                              1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self, B: int, label_B: Optional[Union[int, torch.LongTensor]], fhat_k: torch.Tensor,
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False,
            profile_decode: bool = False,
            autocast_dtype: Optional[torch.dtype] = None,
            sos_source: str = "structural",
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: condition embedding (B, 1024) from condition model
        :param fhat_k: fhat from VAE (B, Cvae, H, H)
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax
        :param profile_decode: If True, fill ``self.last_decode_profile_ms`` with CUDA (or CPU) timings in ms.
        :param autocast_dtype: Optional CUDA autocast dtype (e.g. torch.bfloat16 / torch.float16).
        :param sos_source: "structural" uses fhat-derived SOS; "text" uses the AdaLN condition as SOS.
        :return: reconstructed image (B, 3, H, W) in [0, 1]
        """
        if sos_source not in {"structural", "text"}:
            raise ValueError(f"sos_source must be 'structural' or 'text', got {sos_source!r}")
        self.last_decode_profile_ms = None
        dev = label_B.device
        use_cuda_profile = profile_decode and dev.type == "cuda"
        use_cpu_profile = profile_decode and dev.type != "cuda"

        ev_chain: List[torch.cuda.Event] = []
        cpu_marks: List[float] = []

        def mark_cuda():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev_chain.append(e)

        def mark_cpu():
            cpu_marks.append(time.perf_counter())

        if use_cuda_profile:
            mark_cuda()
        elif use_cpu_profile:
            mark_cpu()

        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        # Fast path: when cfg==0, unconditional branch contributes nothing.
        use_cfg = abs(float(cfg)) > 1e-12
        fhat_embeds = self.proj1(self.down1(fhat_k))  # (B, 768)，不变
        if use_cfg:
            # noise for unconditional branch (不同维度)
            noise = self.noise(torch.tensor(0, device=label_B.device)).unsqueeze(0).expand(B, -1)          # (B, 1024)
            noise_fhat = self.noise_fhat(torch.tensor(0, device=label_B.device)).unsqueeze(0).expand(B, -1)  # (B, 768)
            sos_structural = self.cond_proj2(torch.cat((fhat_embeds, noise_fhat), dim=0))  # (2B, C)
            cond_BD = self.cond_proj(torch.cat((label_B, noise), dim=0))  # (2B, C)
            batch_mul = 2
        else:
            sos_structural = self.cond_proj2(fhat_embeds)  # (B, C)
            cond_BD = self.cond_proj(label_B)   # (B, C)
            batch_mul = 1
        sos = sos_structural if sos_source == "structural" else cond_BD

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(batch_mul * B, self.first_l, -1) + self.pos_start.expand(batch_mul * B, self.first_l,
                                                                                                  -1) + lvl_pos[:,
                                                                                                        :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        if use_cuda_profile:
            mark_cuda()
        elif use_cpu_profile:
            mark_cpu()

        if autocast_dtype is not None and dev.type == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
        else:
            amp_ctx = nullcontext()

        SN = len(self.patch_nums)
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn
            with amp_ctx:
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                x = next_token_map
                AdaLNSelfAttn.forward
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            if use_cuda_profile:
                mark_cuda()
            elif use_cpu_profile:
                mark_cpu()

            with amp_ctx:
                logits_BlV = self.get_logits(x, cond_BD)

            if use_cfg:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            if use_cuda_profile:
                mark_cuda()
            elif use_cpu_profile:
                mark_cpu()

            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                if use_cfg:
                    next_token_map = next_token_map.repeat(2, 1, 1)

            if use_cuda_profile:
                mark_cuda()
            elif use_cpu_profile:
                mark_cpu()

        for b in self.blocks: b.attn.kv_caching(False)
        out = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

        if use_cuda_profile:
            mark_cuda()
            torch.cuda.synchronize()
            prep_ms = ev_chain[0].elapsed_time(ev_chain[1])
            blocks_ms = head_ms = vq_ms = 0.0
            for si in range(SN):
                base = 1 + 3 * si
                blocks_ms += ev_chain[base].elapsed_time(ev_chain[base + 1])
                head_ms += ev_chain[base + 1].elapsed_time(ev_chain[base + 2])
                vq_ms += ev_chain[base + 2].elapsed_time(ev_chain[base + 3])
            finalize_ms = ev_chain[1 + 3 * SN].elapsed_time(ev_chain[-1])
            total_ms = prep_ms + blocks_ms + head_ms + vq_ms + finalize_ms
            self.last_decode_profile_ms = {
                "device": "cuda",
                "prep_ms": prep_ms,
                "transformer_blocks_ms": blocks_ms,
                "head_logits_cfg_sample_ms": head_ms,
                "vq_embed_next_ms": vq_ms,
                "finalize_kv_off_fhat_to_img_ms": finalize_ms,
                "total_profiled_ms": total_ms,
                "num_stages": SN,
            }
        elif use_cpu_profile:
            mark_cpu()
            m = cpu_marks
            prep_ms = (m[1] - m[0]) * 1000.0
            blocks_ms = head_ms = vq_ms = 0.0
            for si in range(SN):
                base = 1 + 3 * si
                blocks_ms += (m[base + 1] - m[base]) * 1000.0
                head_ms += (m[base + 2] - m[base + 1]) * 1000.0
                vq_ms += (m[base + 3] - m[base + 2]) * 1000.0
            finalize_ms = (m[-1] - m[1 + 3 * SN]) * 1000.0
            total_ms = prep_ms + blocks_ms + head_ms + vq_ms + finalize_ms
            self.last_decode_profile_ms = {
                "device": "cpu",
                "prep_ms": prep_ms,
                "transformer_blocks_ms": blocks_ms,
                "head_logits_cfg_sample_ms": head_ms,
                "vq_embed_next_ms": vq_ms,
                "finalize_kv_off_fhat_to_img_ms": finalize_ms,
                "total_profiled_ms": total_ms,
                "num_stages": SN,
            }

        return out

    def forward(self, embed, fhat_k: torch.Tensor,
                x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        """
        :param embed: condition embedding (B, 1024) from condition model (SigLIP-based)
        :param fhat_k: fhat from VAE (B, Cvae, H, H)，用于 SOS token
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = embed.shape[0]
        with torch.cuda.amp.autocast(enabled=False):

            embed2 = self.proj1(self.down1(fhat_k))  # (B, 768)，不变

            # CFG dropout: only for training, and follow configured cond_drop_rate.
            if self.training and self.cond_drop_rate > 0:
                drop_mask = torch.rand(B, device=embed.device) < self.cond_drop_rate
                if torch.any(drop_mask):
                    drop_count = int(drop_mask.sum().item())
                    noise = self.noise(torch.tensor(0, device=embed.device)).unsqueeze(0)            # (1, 1024)
                    noise_fhat = self.noise_fhat(torch.tensor(0, device=embed.device)).unsqueeze(0)  # (1, 768)
                    # Avoid mutating caller tensors in-place.
                    embed = embed.clone()
                    embed2 = embed2.clone()
                    embed[drop_mask, :] = noise
                    embed2[drop_mask, :] = noise_fhat
                else:
                    drop_count = 0
            else:
                drop_count = 0
            self.last_cond_drop_count = drop_count
            self.last_cond_drop_ratio = (drop_count / B) if B > 0 else 0.0

            embed = self.cond_proj(embed)    # (B, 1024) → (B, C)
            embed2 = self.cond_proj2(embed2)  # (B, 768) → (B, C)

            sos = embed2.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(embed)

        # hack: get the dtype if mixed precision is used
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
        # Keep DDP reduction graph stable when cond_drop_rate=0:
        # noise/noise_fhat may be intentionally unused in the main forward path.
        x_BLC[0, 0, 0] += self.noise.weight.view(-1)[0] * 0 + self.noise_fhat.weight.view(-1)[0] * 0
        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.trunc_normal_(m.weight.data)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
