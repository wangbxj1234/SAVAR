"""
traind36_pn512_v4.py
基于 v3，改用 Jina-CLIP-v2 (1024-dim) 替代 SigLIP。
主要变化：
  1. condition model 来自 output_jina（text_dim=1024, out_dim=1024）
  2. JinaCLIPv2Wrapper 替代 SigLIPWrapper
  3. VARSlip / trainer_3 完全复用（同为 1024 维接口）
  4. 初始 VAR 权重从 v2 (output_d36_pn512_3) 加载（v3 效果不如 v2）
"""
import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import dist
from utils import arg_util_jina as arg_util, misc
from utils.data import ImageNet as ImageNetBase
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from condition_model import CondAlignModelV2, CondAlignModelV4FhatGate
from transformers import AutoModel

# ============ Jina-CLIP-v2 配置 ============
JINA_CKPT = "jinaai/jina-clip-v2"
JINA_DIM = 1024
n_cond_embed = 1024
normalize_jina = True


# ============ 数据集：返回原始 caption 字符串 ============
class ImageNetRawCaption(ImageNetBase):
    """与 ImageNet 相同，但返回原始 caption 字符串供 Jina-CLIP processor 使用。"""
    def __getitem__(self, indices):
        img1, _ = super().__getitem__(indices)
        _, raw_caption = self.reader[indices]
        return img1, raw_caption


def imagenet_raw(root, final_reso, model, hflip, mid_reso):
    return ImageNetRawCaption(root, final_reso=final_reso, model=model, hflip=hflip, mid_reso=mid_reso)


# ============ Jina-CLIP-v2 Wrapper ============
class JinaCLIPv2Wrapper(nn.Module):
    """Jina-CLIP-v2 编码器封装：text/image -> 1024 维，与 SigLIPWrapper 接口一致。"""
    def __init__(self, device, normalize=False):
        super().__init__()
        self.device = device
        self.normalize = normalize
        self.model = AutoModel.from_pretrained(
            JINA_CKPT,
            trust_remote_code=True,
            use_text_flash_attn=False,
        )
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if normalize:
            print(f"normalize Jina-CLIP-v2 embeddings ({JINA_DIM}-dim)")

    def _to_tensor(self, emb):
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        return emb.to(self.device)

    @torch.no_grad()
    def encode_image(self, image):
        """image: [B, 3, H, W] tensor，范围 [-1, 1]。返回 1024 维特征。"""
        B = image.shape[0]
        x = (image.cpu().clamp(-1, 1) + 1) * 0.5
        pil_list = [to_pil_image(x[i]) for i in range(B)]
        emb = self.model.encode_image(pil_list)
        emb = self._to_tensor(emb)
        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def encode_text(self, captions):
        """captions: list of B strings。返回 1024 维特征。"""
        if isinstance(captions, str):
            captions = [captions]
        elif isinstance(captions, tuple):
            captions = list(captions)
        emb = self.model.encode_text(captions)
        emb = self._to_tensor(emb)
        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb


# ============ 构建 VARSlip 模型（与 v3 完全相同） ============
def build_vae_varslip(
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    n_cond_embed=1024, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    cond_drop_rate=0.1,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    use_checkpoint=False,
):
    from models.vqvae import VQVAE
    from models.varslip import VARSlip

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VARSlip(
        vae_local=vae_local,
        n_cond_embed=n_cond_embed, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        use_checkpoint=use_checkpoint,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)

    return vae_local, var_wo_ddp


def _collate_fn(batch):
    """自定义 collate_fn：字符串 caption 不需要 stack"""
    imgs = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return imgs, captions


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'),
            verbose=True)
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')

    # build data
    print(f'[build PT data] ...\n')

    dataset_train = imagenet_raw(
        args.data_path, final_reso=args.data_load_reso, model='train',
        hflip=args.hflip, mid_reso=args.mid_reso)

    dataset_val = imagenet_raw(
        args.data_path, final_reso=args.data_load_reso, model='val',
        hflip=args.hflip, mid_reso=args.mid_reso)

    types = str((type(dataset_train).__name__, type(dataset_val).__name__))

    ld_val = DataLoader(
        dataset_val, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size * 1.5),
        sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
        collate_fn=_collate_fn,
    )
    del dataset_val

    ld_train = DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks, shuffle=True, fill_last=True,
            rank=dist.get_rank(), world_size=dist.get_world_size(),
            start_ep=start_ep, start_it=start_it,
        ),
        collate_fn=_collate_fn,
    )
    del dataset_train

    [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    print(f'     [dataloader multi processing](*) finished! ({time.time() - stt:.2f}s)', flush=True, clean=True)
    print(
        f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')

    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models.varslip import VARSlip
    from models import VQVAE
    from trainer_3 import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params

    vae_local, var_wo_ddp = build_vae_varslip(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=dist.get_device(), patch_nums=args.patch_nums,
        n_cond_embed=n_cond_embed, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        cond_drop_rate=args.cdr,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        use_checkpoint=args.ckpt,
    )

    vae_ckpt = Path(__file__).parent / 'pretrained' / 'vae_ch160v4096z32.pth'
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget -P pretrained https://hf-mirror.com/FoundationVision/var/blob/main/vae_ch160v4096z32.pth')
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    # 如果是从头训练，优先加载 FoundationVision VAR-d20 预训练权重作为初始化
    if trainer_state is None or len(trainer_state) == 0:
        init_ckpt = os.environ.get(
            "VAR_INIT_CKPT",
            str(Path(__file__).parent / "pretrained" / "foundationvision_var" / "var_d20.pth"),
        )
        if os.path.exists(init_ckpt):
            print(f"[INIT] Loading initial VAR weights from {init_ckpt}...")
            ckpt = torch.load(init_ckpt, map_location='cpu')

            sd = None
            if isinstance(ckpt, dict) and 'trainer' in ckpt:
                trainer_dict = ckpt['trainer']
                if isinstance(trainer_dict, dict) and 'var_wo_ddp' in trainer_dict:
                    sd = trainer_dict['var_wo_ddp']
                    print(f"[INIT] Extracted 'var_wo_ddp' from checkpoint['trainer']")
                elif isinstance(trainer_dict, dict) and 'var' in trainer_dict:
                    sd = trainer_dict['var']
                    print(f"[INIT] Extracted 'var' from checkpoint['trainer']")
                else:
                    print(f"[INIT] Could not find 'var' or 'var_wo_ddp' in trainer dict, keys: {trainer_dict.keys()}")
            else:
                sd = ckpt
                print(f"[INIT] Using checkpoint directly as state_dict")

            if sd is not None:
                new_sd = {}
                for k, v in sd.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_sd[name] = v
                sd = new_sd

                model_sd = var_wo_ddp.state_dict()
                filtered_sd = {}
                skipped_keys = []
                for k, v in sd.items():
                    if k in model_sd:
                        if v.shape == model_sd[k].shape:
                            filtered_sd[k] = v
                        else:
                            skipped_keys.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}")
                    else:
                        skipped_keys.append(f"{k}: not in VARSlip (unexpected)")
                if skipped_keys:
                    print(f"[INIT] Skipped {len(skipped_keys)} keys due to shape mismatch or absence:")
                    for s in skipped_keys:
                        print(f"    {s}")

                try:
                    msg = var_wo_ddp.load_state_dict(filtered_sd, strict=False)
                    missing = getattr(msg, "missing_keys", [])
                    unexpected = getattr(msg, "unexpected_keys", [])
                    print(f"[INIT] Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

                    if len(missing) > 0:
                        print(f"[INIT] Missing keys (will be initialized):")
                        for i, k in enumerate(missing):
                            if i < 20:
                                print(f"    {k}")
                            elif i == 20:
                                print(f"    ... and {len(missing)-20} more")
                                break

                        initialized_count = 0
                        for name, param in var_wo_ddp.named_parameters():
                            if name in missing:
                                initialized_count += 1
                                if 'weight' in name:
                                    if len(param.shape) >= 2:
                                        torch.nn.init.trunc_normal_(param.data, std=0.02)
                                    else:
                                        param.data.normal_(std=0.02)
                                elif 'bias' in name:
                                    param.data.zero_()
                                elif 'embedding' in name.lower():
                                    torch.nn.init.normal_(param.data, std=0.02)
                                else:
                                    if param.requires_grad:
                                        torch.nn.init.normal_(param.data, std=0.02)
                        print(f"[INIT] Initialized {initialized_count} missing layers")

                        # Smart init: expand old 768d cond_proj.weight → 1024d
                        if 'cond_proj.weight' in sd and 'cond_proj.weight' not in filtered_sd:
                            old_w = sd['cond_proj.weight']
                            new_w = var_wo_ddp.cond_proj.weight.data
                            if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
                                new_w.zero_()
                                new_w[:, :old_w.shape[1]] = old_w
                                extra_dim = new_w.shape[1] - old_w.shape[1]
                                new_w[:, old_w.shape[1]:] = torch.randn(new_w.shape[0], extra_dim, device=new_w.device) * 0.001
                                print(f"[INIT] Smart init cond_proj.weight: expanded {tuple(old_w.shape)} -> {tuple(new_w.shape)}, "
                                      f"first {old_w.shape[1]}d from ckpt, last {extra_dim}d near-zero random")

                    if len(unexpected) > 0:
                        print(f"[INIT] Unexpected keys (ignored):")
                        for i, k in enumerate(unexpected):
                            if i < 20:
                                print(f"    {k}")
                            elif i == 20:
                                print(f"    ... and {len(unexpected)-20} more")
                                break

                    del ckpt, sd, new_sd
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[INIT] Failed to load weights from {init_ckpt}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[INIT] Failed to extract model state_dict from checkpoint")
        else:
            print(f"[INIT] {init_ckpt} not found! Training VARSlip from scratch.")
    else:
        print("[RESUME] Skipping initial VAR weights loading, will load from checkpoint.")

    # 加载 Jina-CLIP-v2
    jina = JinaCLIPv2Wrapper(dist.get_device(), normalize=normalize_jina)

    # 加载 condition model（Jina-CLIP-v2 1024 维版本）
    condition_model_path = os.environ.get(
        "CONDALIGN_CKPT",
        "/mnt/data/var_dec_scripts_small/VAR_dec/output_jinacondv3/condalign-ckpt-best.pth",
    )

    if condition_model_path and os.path.exists(condition_model_path):
        print(f"Loading condition model from {condition_model_path}")
        checkpoint = torch.load(condition_model_path, map_location='cuda')
        checkpoint = checkpoint["model_state_dict"]

        if isinstance(checkpoint, dict) and any(key.startswith('module.') for key in checkpoint.keys()):
            new_checkpoint = {}
            for key, value in checkpoint.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_checkpoint[new_key] = value
                else:
                    new_checkpoint[key] = value
            checkpoint = new_checkpoint
        # Auto-detect condition architecture by checkpoint key layout.
        # v4-fhatgate checkpoints contain fhat_gate_mlp.* instead of gate.*.
        if any(k.startswith("fhat_gate_mlp.") for k in checkpoint.keys()):
            print("Detected CondAlignModelV4FhatGate checkpoint layout.")
            condition_model = CondAlignModelV4FhatGate(image_dim=32, text_dim=1024, out_dim=1024).cuda()
        else:
            condition_model = CondAlignModelV2(image_dim=32, text_dim=1024, out_dim=1024).cuda()

        condition_model.load_state_dict(checkpoint)
        print(f"Condition model loaded successfully!")
    else:
        print(f"Warning: No condition model checkpoint found at {condition_model_path}.")
        print("Make sure you have trained the condition model first.")
        condition_model = CondAlignModelV2(image_dim=32, text_dim=1024, out_dim=1024).cuda()
    condition_model.eval().requires_grad_(False)

    vae_local = args.compile_model(vae_local, args.vfast)
    var_wo_ddp = args.compile_model(var_wo_ddp, args.tfast)
    var = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()],
                                                    find_unused_parameters=False, broadcast_buffers=False)

    print(f'[INIT] VARSlip model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder),
        ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VARSlip', var_wo_ddp),)]) + '\n\n')

    # build optimizer (differential LR: newly initialized layers get higher LR)
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    },
        high_lr_keys={'cond_proj.weight', 'cond_proj.bias', 'noise.weight', 'noise_fhat.weight'},
        high_lr_scale=20.0,
        base_lr_scale=0.1,
    )
    opt_clz = {
        'adam': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')

    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups

    # build trainer (复用 trainer_3，接口完全兼容)
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var, clip=jina,
        condition_model=condition_model,
        var_opt=var_optim, label_smooth=args.ls,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)

    del vae_local, var_wo_ddp, var, var_optim

    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def main_training():
    k = 4
    args: arg_util.Args = arg_util.init_dist_and_get_args()

    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)

    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1

    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)

        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, k, trainer
        )

        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats[
            'tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep + 1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = None, None, None, None
        # 每个 epoch 都验证
        val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val, k=k)
        best_updated = best_val_loss_tail > val_loss_tail
        best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail,
                                                                                             val_loss_tail)
        best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail,
                                                                                         val_acc_tail)
        AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean,
                          vacc_tail=val_acc_tail)
        args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
        print(
            f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')

        # 按 iter 在 train_one_ep 里定期存 last；验证后每个 epoch 也保存 last。
        need_save = True  # save ar-ckpt-last.pth after every epoch validation
        if need_save and dist.is_local_master():
            local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
            local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
            print(f'[saving ckpt] ...', end='', flush=True)
            torch.save({
                'epoch': ep + 1,
                'iter': 0,
                'trainer': trainer.state_dict(),
                'args': args.state_dict(),
            }, local_out_ckpt)
            if best_updated:
                shutil.copy(local_out_ckpt, local_out_ckpt_best)
            print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
        if need_save:
            dist.barrier()

        print(
            f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}',
            flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep + 1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep + 1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log();
        tb_lg.flush()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(
        f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')

    del stats
    del iters_train, ld_train
    gc.collect(), torch.cuda.empty_cache()

    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log();
    tb_lg.flush();
    tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger,
                 ld_or_itrt, iters_train: int, k: int, trainer):
    from trainer_3 import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer

    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    me.add_meter('tcl', misc.SmoothedValue(window_size=30, fmt='{median:.2f} ({global_avg:.2f})'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'

    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train

    for it, (inp, raw_captions) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 1000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()

        inp = inp.to(args.device, non_blocking=True)

        args.cur_it = f'{it + 1}/{iters_train}'

        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde,
            g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        if args.pg:
            if g_it <= wp_it:
                prog_si = args.pg0
            elif g_it >= max_it * args.pg:
                prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it * args.pg - wp_it), 0), 1)
                prog_si = args.pg0 + round(progress * delta)
        else:
            prog_si = -1

        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)

        grad_norm, scale_log2 = trainer.train_step(
            it=it, k=k, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, raw_captions=raw_captions, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )

        # Frequent iteration checkpoints limit progress loss if the job receives SIGTERM.
        if args.iter_ckpt_interval > 0 and (g_it + 1) % args.iter_ckpt_interval == 0 and dist.is_local_master():
            local_out_ckpt = os.path.join(args.local_out_dir_path, f'ar-ckpt-iter{g_it+1}.pth')
            local_out_last = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
            print(f'[saving ckpt at iter {g_it+1}] ...', end='', flush=True)
            state = {
                'epoch': ep,
                'iter': g_it + 1,
                'trainer': trainer.state_dict(),
                'args': args.state_dict(),
            }
            torch.save(state, local_out_ckpt)
            torch.save(state, local_out_last)
            print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)

        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)

        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(
        max_it - (g_it + 1) + (args.ep - ep) * 15)


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
