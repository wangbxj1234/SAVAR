import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import dist
from utils import arg_util_jina as arg_util, misc
from utils import data as data_module
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from models.vqvae import VQVAE
from condition_model import CondAlignModelV3
import math
from transformers import AutoModel

# Jina-CLIP-v2 配置（1024 维 embedding，支持 matryoshka 降维但这里用原生维度）
JINA_CKPT = "jinaai/jina-clip-v2"
JINA_DIM = 1024

# 训练配置
n_cond_embed = 1024
normalize_emb = True
PAIRWISE_WEIGHT = 1.0
CONTRASTIVE_WEIGHT = 0.04
CONTRASTIVE_WEIGHT_MIN = 0.004
CONTRASTIVE_ANNEAL_EPOCHS = 6
LOGIT_SCALE_INIT = 4.0
COSINE_WARMUP_EPOCHS = 5
COSINE_LR_MIN_RATIO = 0.05
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 2
PLATEAU_MIN_LR_SCALE = 0.02
PLATEAU_THRESHOLD = 1e-3
STOP_SIGNAL_FILENAME = "STOP_TRAINING"
MODEL_ARCH = "CondAlignModelV3"


class ImageNetRawCaption(data_module.ImageNet):
    """与 ImageNet 相同，但返回原始 caption 字符串供 Jina-CLIP processor 使用。"""
    def __getitem__(self, indices):
        img1, _ = super().__getitem__(indices)
        _, raw_caption = self.reader[indices]
        return img1, raw_caption


class ImageNetRawCaptionWithIdx(ImageNetRawCaption):
    """多返回样本索引，用于预计算 image embedding 的查表。"""
    def __getitem__(self, indices):
        img1, raw_caption = super().__getitem__(indices)
        return img1, raw_caption, indices


def imagenet_raw_caption(root, final_reso, model, hflip, mid_reso, with_idx=False):
    cls = ImageNetRawCaptionWithIdx if with_idx else ImageNetRawCaption
    return cls(root, final_reso=final_reso, model=model, hflip=hflip, mid_reso=mid_reso)


class JinaCLIPv2Wrapper(nn.Module):
    """
    Jina-CLIP-v2 编码器封装：image/text -> 1024 维，与 SigLIPWrapper 接口一致。
    内部使用 model.encode_text / model.encode_image 高级 API，自动处理预处理。
    """
    def __init__(self, device, normalize=False):
        super().__init__()
        self.device = device
        self.normalize = normalize
        # Keep text encoder on non-flash-attn path to avoid Triton API mismatch
        # (e.g., "triton.language has no attribute split") in this environment.
        self.model = AutoModel.from_pretrained(
            JINA_CKPT, trust_remote_code=True, use_text_flash_attn=False
        )
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if normalize:
            print(f"normalize Jina-CLIP-v2 embeddings ({JINA_DIM}-dim)")

    def _to_tensor(self, emb):
        """将 encode 返回值统一转为 GPU tensor。"""
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
    def encode_image_1024(self, image):
        """返回 1024 维 image 特征，用于预计算缓存。"""
        return self.encode_image(image)

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

    def load_precomputed_1024(self, emb_1024):
        """加载预计算的 1024 维特征（如果需要归一化就归一化）。"""
        if self.normalize:
            return emb_1024 / emb_1024.norm(dim=-1, keepdim=True)
        return emb_1024


def run_precompute_jina_emb(args):
    """预计算全体 train/val 的 Jina-CLIP-v2 image 特征（1024 维），存到 local_out_dir。支持断点续算与多卡并行。"""
    from torch.utils.data import Sampler
    device = dist.get_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    jina = JinaCLIPv2Wrapper(device, normalize=False)
    out_dir = Path(args.local_out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = 1000

    class ChunkSampler(Sampler):
        """每个 rank 分得一段连续的索引，保证最终拼接顺序正确。"""
        def __init__(self, dataset, rank, world_size):
            self.n = len(dataset)
            self.rank = rank
            self.world_size = world_size
            self.start = self.n * rank // world_size
            self.end = self.n * (rank + 1) // world_size
        def __iter__(self):
            return iter(range(self.start, self.end))
        def __len__(self):
            return self.end - self.start

    for split, model in [("train", "train"), ("val", "val")]:
        ds = imagenet_raw_caption(
            args.data_path, final_reso=args.data_load_reso, model=model,
            hflip=False, mid_reso=args.mid_reso)
        n = len(ds)
        chunk_sampler = ChunkSampler(ds, rank, world_size)
        ld = DataLoader(
            ds, batch_size=min(64, len(chunk_sampler)) if len(chunk_sampler) > 0 else 1,
            sampler=chunk_sampler, num_workers=args.workers, pin_memory=True)

        ckpt_path = out_dir / f"jina_img_emb_{split}_checkpoint_rank{rank}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu')
            emb_list = ckpt["emb_list"]
            start_batch = ckpt["last_batch_idx"] + 1
            print(f"[Rank {rank}] Resuming {split} from batch {start_batch}/{len(ld)} (checkpoint found)")
        else:
            emb_list = []
            start_batch = 0
            if dist.is_master():
                print(f"Precomputing Jina-CLIP-v2 image embeddings for {split} (n={n}, {world_size} GPUs) ...")
            print(f"[Rank {rank}] Processing {split} indices [{chunk_sampler.start}:{chunk_sampler.end})")

        for i, (img1, _) in enumerate(ld):
            if i < start_batch:
                continue
            img1 = img1.to(device, non_blocking=True)
            e = jina.encode_image_1024(img1)
            emb_list.append(e.cpu())

            if (i + 1) % 100 == 0:
                print(f"  [Rank {rank}] {split} batch {i+1}/{len(ld)}")

            if (i + 1) % checkpoint_interval == 0:
                torch.save({"emb_list": emb_list, "last_batch_idx": i}, ckpt_path)
                print(f"  [Rank {rank}] Checkpoint saved at batch {i+1}")

        emb_chunk = torch.cat(emb_list, dim=0) if emb_list else torch.empty(0, JINA_DIM)
        rank_path = out_dir / f"jina_img_emb_{split}_rank{rank}.pt"
        torch.save({"emb": emb_chunk, "start_idx": chunk_sampler.start}, rank_path)
        print(f"[Rank {rank}] Saved {rank_path} shape {emb_chunk.shape}")

        dist.barrier()

        if dist.is_master():
            print(f"  Merging {world_size} rank files for {split}...")
            all_chunks = []
            for r in range(world_size):
                rp = out_dir / f"jina_img_emb_{split}_rank{r}.pt"
                chunk_data = torch.load(rp, map_location='cpu')
                all_chunks.append((chunk_data["start_idx"], chunk_data["emb"]))
                rp.unlink()
            all_chunks.sort(key=lambda x: x[0])
            emb_cat = torch.cat([c[1] for c in all_chunks], dim=0)

            path = out_dir / f"jina_img_emb_{split}.pt"
            torch.save({
                "emb": emb_cat,
                "data_load_reso": args.data_load_reso,
                "n": emb_cat.shape[0],
            }, path)
            print(f"  Saved {path} shape {emb_cat.shape} (data_load_reso={args.data_load_reso})")

        if ckpt_path.exists():
            ckpt_path.unlink()
        dist.barrier()

    if dist.is_master():
        print("Precompute done. Run training without --precompute_jina_emb to use cache.")


class VFLossWithLearnableScale(nn.Module):
    """
    CLIP-style contrastive loss + optional pairwise cosine loss,
    with learnable logit scale.
    """

    def __init__(
        self,
        initial_logit_scale=3.6,
        pairwise_weight=0.0,
        contrastive_weight=0.05,
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_logit_scale)
        self.pairwise_weight = pairwise_weight
        self.contrastive_weight = contrastive_weight
        self.max_logit_scale = 100.0

    def forward(self, pred, target):
        pred_norm = torch.nn.functional.normalize(pred, dim=1)
        target_norm = torch.nn.functional.normalize(target, dim=1)

        logits = torch.matmul(pred_norm, target_norm.t()) * self.logit_scale.exp().clamp(max=self.max_logit_scale)
        labels = torch.arange(pred.shape[0], device=pred.device)

        loss_pred2target = torch.nn.functional.cross_entropy(logits, labels)
        loss_target2pred = torch.nn.functional.cross_entropy(logits.t(), labels)
        contrastive_loss = (loss_pred2target + loss_target2pred) / 2.0

        pairwise_loss = 0.0
        if self.pairwise_weight > 0:
            cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
            pairwise_core = (1 - cosine_sim).mean()
            pairwise_loss = pairwise_core * self.pairwise_weight

        total_loss = pairwise_loss + self.contrastive_weight * contrastive_loss
        return total_loss, contrastive_loss, pairwise_loss


def get_contrastive_weight_by_epoch(ep: int) -> float:
    if CONTRASTIVE_ANNEAL_EPOCHS <= 0:
        return CONTRASTIVE_WEIGHT
    ratio = min(max(ep, 0), CONTRASTIVE_ANNEAL_EPOCHS) / float(CONTRASTIVE_ANNEAL_EPOCHS)
    return CONTRASTIVE_WEIGHT - (CONTRASTIVE_WEIGHT - CONTRASTIVE_WEIGHT_MIN) * ratio


def get_lr_scale_by_epoch(ep: int, total_epochs: int = 100) -> float:
    if ep < COSINE_WARMUP_EPOCHS:
        return (ep + 1) / COSINE_WARMUP_EPOCHS
    progress = (ep - COSINE_WARMUP_EPOCHS) / max(1, total_epochs - COSINE_WARMUP_EPOCHS)
    return COSINE_LR_MIN_RATIO + 0.5 * (1.0 - COSINE_LR_MIN_RATIO) * (1 + math.cos(math.pi * progress))


def is_compatible_condition_checkpoint(checkpoint, ckpt_path):
    ckpt_arch = checkpoint.get("model_arch")
    if ckpt_arch == MODEL_ARCH:
        return True
    if dist.is_master():
        print(
            f"[WARN] Skip incompatible condition checkpoint {ckpt_path}: "
            f"model_arch={ckpt_arch!r}, expected {MODEL_ARCH!r}. "
            "Start from scratch or pass a matching --condalign_init_ckpt."
        )
    return False


def build_everything(args):
    os.makedirs(args.local_out_dir_path, exist_ok=True)

    start_ep, start_it = 0, 0
    auto_resume_info = [f"[auto_resume] disabled; using {MODEL_ARCH} model_arch-checked resume only"]
    tb_lg = misc.DistLogger(
        misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'),
        verbose=True) if dist.is_master() else misc.DistLogger(None, verbose=False)
    dist.barrier()

    # 是否使用预计算的 Jina-CLIP-v2 image embedding
    out_dir = Path(args.local_out_dir_path)
    cache_train = out_dir / "jina_img_emb_train.pt"
    cache_val = out_dir / "jina_img_emb_val.pt"
    use_jina_cache = cache_train.exists() and cache_val.exists()
    if use_jina_cache and dist.is_master():
        print(f"Using precomputed Jina-CLIP-v2 image embeddings from {out_dir} (train/val cache).")
    if use_jina_cache:
        args_hflip = False
    else:
        args_hflip = args.hflip

    dataset_train = imagenet_raw_caption(
        args.data_path, final_reso=args.data_load_reso, model='train',
        hflip=args_hflip, mid_reso=args.mid_reso, with_idx=use_jina_cache)
    dataset_val = imagenet_raw_caption(
        args.data_path, final_reso=args.data_load_reso, model='val',
        hflip=args_hflip, mid_reso=args.mid_reso, with_idx=use_jina_cache)

    iters_train = len(dataset_train) // args.glb_batch_size

    precomputed_emb_train = None
    precomputed_emb_val = None
    if use_jina_cache:
        ck_train = torch.load(cache_train, map_location='cpu')
        ck_val = torch.load(cache_val, map_location='cpu')
        if isinstance(ck_train, dict):
            precomputed_emb_train = ck_train["emb"]
            reso_train, n_cache_train = ck_train.get("data_load_reso"), ck_train.get("n")
        else:
            precomputed_emb_train = ck_train
            reso_train, n_cache_train = None, precomputed_emb_train.shape[0]
        if isinstance(ck_val, dict):
            precomputed_emb_val = ck_val["emb"]
            reso_val, n_cache_val = ck_val.get("data_load_reso"), ck_val.get("n")
        else:
            precomputed_emb_val = ck_val
            reso_val, n_cache_val = None, precomputed_emb_val.shape[0]
        n_train, n_val = len(dataset_train), len(dataset_val)
        if precomputed_emb_train.shape[0] != n_train or precomputed_emb_val.shape[0] != n_val:
            raise RuntimeError(
                f"Precomputed cache length mismatch: cache train={precomputed_emb_train.shape[0]} val={precomputed_emb_val.shape[0]}, "
                f"current dataset train={n_train} val={n_val}. Use same --data_path and --pn as precompute, or delete cache and re-precompute."
            )
        if reso_train is not None and reso_train != args.data_load_reso:
            raise RuntimeError(
                f"Precomputed cache was built with data_load_reso={reso_train}, current is {args.data_load_reso}. "
                f"Use same --pn as precompute, or delete cache and re-precompute."
            )
        if precomputed_emb_train.shape[1] != JINA_DIM:
            raise RuntimeError(
                f"Precomputed cache dim is {precomputed_emb_train.shape[1]}, expected {JINA_DIM}. Re-run precompute."
            )
        if dist.is_master():
            print(f"  Cache OK: train [n={n_train}, dim={JINA_DIM}], val [n={n_val}], data_load_reso={args.data_load_reso}")

    ld_val = DataLoader(
        dataset_val, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size * 1.5),
        sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    del dataset_val

    # VAE
    vae_ckpt = Path(__file__).parent / 'pretrained' / 'vae_ch160v4096z32.pth'
    V = 4096
    Cvae = 32
    ch = 160
    share_quant_resi = 4
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=args.patch_nums).to(dist.get_device())
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae_local.eval().to(dist.get_device())
    for p in vae_local.parameters():
        p.requires_grad = False

    device = dist.get_device()
    jina = JinaCLIPv2Wrapper(device, normalize=normalize_emb)

    if dist.is_master():
        print(f"CondAlignModelV3 training: fhat_k={args.fhat_k} (VAE img_to_fhat_k index)")
    model = CondAlignModelV3(image_dim=32, text_dim=1024, out_dim=1024).to(device)
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)

    loss_fn = VFLossWithLearnableScale(
        initial_logit_scale=LOGIT_SCALE_INIT,
        pairwise_weight=PAIRWISE_WEIGHT,
        contrastive_weight=CONTRASTIVE_WEIGHT,
    )

    base_lr_model = args.tlr * 0.1
    base_lr_loss = args.tlr * 0.01
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": base_lr_model, "initial_lr": base_lr_model, "weight_decay": 0.01},
            {"params": loss_fn.parameters(), "lr": base_lr_loss, "initial_lr": base_lr_loss, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=PLATEAU_THRESHOLD,
        min_lr=[
            base_lr_model * PLATEAU_MIN_LR_SCALE,
            base_lr_loss * PLATEAU_MIN_LR_SCALE,
        ],
    )

    init_path = getattr(args, 'condalign_init_ckpt', '') or ''
    if init_path:
        p = Path(init_path)
        if p.is_file():
            if dist.is_master():
                print(f"Loading CondAlign init weights from {p} (forced init; skipping last-resume)")
            ck = torch.load(p, map_location='cpu')
            if isinstance(ck, dict) and 'model_state_dict' in ck and not is_compatible_condition_checkpoint(ck, p):
                raise RuntimeError(f"Incompatible condition checkpoint for {MODEL_ARCH}: {p}")
            sd = ck.get('model_state_dict', ck)
            model.load_state_dict(sd, strict=True)
            if 'loss_fn_state_dict' in ck:
                loss_fn.load_state_dict(ck['loss_fn_state_dict'], strict=True)
            start_ep, start_it = 0, 0
            best_val_loss = ck.get('best_val_loss', ck.get('val_loss', float('inf')))
            best_val_pairwise = ck.get(
                'best_val_pairwise',
                ck.get('val_pairwise_loss', ck.get('best_val_loss', ck.get('val_loss', float('inf'))))
            )
            if dist.is_master():
                print(
                    "  Init load done: model"
                    + (" + loss_fn" if 'loss_fn_state_dict' in ck else "")
                    + f"; epoch/it reset to 0; best thresholds keep val_loss={best_val_loss:.4f}, "
                    + f"val_pairwise={best_val_pairwise:.4f}"
                )
        else:
            if dist.is_master():
                print(f"[WARN] condalign_init_ckpt not found: {p}")
            best_val_loss = float('inf')
            best_val_pairwise = float('inf')
            if dist.is_master():
                print("No valid init checkpoint; fallback to normal resume/scratch flow")
            last_ckpt_path = Path(args.local_out_dir_path) / 'condalign-ckpt-last.pth'
            if last_ckpt_path.exists():
                if dist.is_master():
                    print(f"Loading last checkpoint from {last_ckpt_path}")
                checkpoint = torch.load(last_ckpt_path, map_location='cpu')
                if is_compatible_condition_checkpoint(checkpoint, last_ckpt_path):
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
                    start_ep = checkpoint['epoch']
                    start_it = checkpoint.get('iter', 0)
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    best_val_pairwise = checkpoint.get('best_val_pairwise', best_val_loss)
                    if 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if dist.is_master():
                        print(
                            f"Resumed from epoch {start_ep}, iter {start_it}, "
                            f"best_val_loss: {best_val_loss:.4f}, best_val_pairwise: {best_val_pairwise:.4f}"
                        )
                else:
                    start_ep, start_it = 0, 0
                    best_val_loss = float('inf')
                    best_val_pairwise = float('inf')
    else:
        last_ckpt_path = Path(args.local_out_dir_path) / 'condalign-ckpt-last.pth'
        if last_ckpt_path.exists():
            if dist.is_master():
                print(f"Loading last checkpoint from {last_ckpt_path}")
            checkpoint = torch.load(last_ckpt_path, map_location='cpu')

            if is_compatible_condition_checkpoint(checkpoint, last_ckpt_path):
                model.load_state_dict(checkpoint['model_state_dict'])

                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])

                start_ep = checkpoint['epoch']
                start_it = checkpoint.get('iter', 0)
                best_val_loss = checkpoint.get('best_val_loss')
                if best_val_loss is None:
                    best_val_loss = float('inf')
                best_val_pairwise = checkpoint.get('best_val_pairwise', best_val_loss)
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                if dist.is_master():
                    print(
                        f"Resumed from epoch {start_ep}, iter {start_it}, "
                        f"best_val_loss: {best_val_loss:.4f}, best_val_pairwise: {best_val_pairwise:.4f}"
                    )
            else:
                start_ep, start_it = 0, 0
                best_val_loss = float('inf')
                best_val_pairwise = float('inf')
        else:
            best_val_loss = float('inf')
            best_val_pairwise = float('inf')
            if dist.is_master():
                print("No last checkpoint found, starting from scratch")

    # Sampler 必须用最终的 start_ep/start_it（init 权重会重置为 0，不能在 load 之前建 DataLoader）
    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers, pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep,
            start_it=start_it,
        ),
    )
    del dataset_train
    ld_train = iter(ld_train)

    if not use_jina_cache:
        precomputed_emb_train = None
        precomputed_emb_val = None

    return (
        model, vae_local, jina, optimizer, scheduler, loss_fn, ld_train, ld_val, tb_lg,
        start_ep, start_it, iters_train, best_val_loss, best_val_pairwise,
        precomputed_emb_train, precomputed_emb_val
    )


def eval_one_ep(model, vae_local, jina, loss_fn, ld_val, precomputed_emb_val=None, fhat_k: int = 4):
    model.eval()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_pairwise_loss = 0.0
    num_batches = 0
    device = dist.get_device()

    with torch.no_grad():
        for batch in ld_val:
            if precomputed_emb_val is not None:
                img1, caption, indices = batch
                jina_img_emb = jina.load_precomputed_1024(precomputed_emb_val[indices].to(device))
            else:
                img1, caption = batch
                jina_img_emb = jina.encode_image(img1.to(device))
            img1 = img1.to(device, non_blocking=True)

            fhat = vae_local.img_to_fhat_k(img1, k=fhat_k)
            text_emb = jina.encode_text(caption)

            pred = model(text_emb, fhat)
            loss, contrastive_loss, pairwise_loss = loss_fn(pred, jina_img_emb)

            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_pairwise_loss += pairwise_loss.item()
            num_batches += 1

    if dist.initialized() and dist.get_world_size() > 1:
        t = torch.tensor(
            [total_loss, total_contrastive_loss, total_pairwise_loss, float(num_batches)],
            device=device, dtype=torch.float64,
        )
        dist.allreduce(t)
        total_loss, total_contrastive_loss, total_pairwise_loss, num_batches = (
            t[0].item(), t[1].item(), t[2].item(), int(t[3].item())
        )

    nb = max(1, num_batches)
    avg_loss = total_loss / nb
    avg_contrastive_loss = total_contrastive_loss / nb
    avg_pairwise_loss = total_pairwise_loss / nb

    return avg_loss, avg_contrastive_loss, avg_pairwise_loss


def train_one_ep(
    ep,
    start_it,
    args,
    model,
    vae_local,
    jina,
    optimizer,
    scheduler,
    loss_fn,
    ld_train,
    iters_train,
    tb_lg,
    best_val_loss,
    best_val_pairwise,
    precomputed_emb_train=None,
):
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_pairwise_loss = 0.0
    device = dist.get_device()
    stop_requested = False
    stop_signal = Path(args.local_out_dir_path) / STOP_SIGNAL_FILENAME

    steps_to_run = iters_train - start_it
    for it_idx in range(steps_to_run):
        it = start_it + it_idx
        batch = next(ld_train)
        if precomputed_emb_train is not None:
            img1, caption, indices = batch
            with torch.no_grad():
                jina_img_emb = jina.load_precomputed_1024(precomputed_emb_train[indices].to(device))
        else:
            img1, caption = batch
            with torch.no_grad():
                jina_img_emb = jina.encode_image(img1.to(device))
        img1 = img1.to(device, non_blocking=True)

        with torch.no_grad():
            fhat = vae_local.img_to_fhat_k(img1, k=args.fhat_k)
            text_emb = jina.encode_text(caption)

        pred = model(text_emb, fhat)
        loss, contrastive_loss, pairwise_loss = loss_fn(pred, jina_img_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_pairwise_loss += pairwise_loss.item()

        if dist.is_master() and (it % 50 == 0 or it == iters_train - 1):
            logit_scale = loss_fn.logit_scale.item()
            print(f"[Ep {ep}][It {it}/{iters_train}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Contrastive: {contrastive_loss.item():.4f} | "
                  f"Pairwise: {pairwise_loss.item():.4f} | "
                  f"Logit_scale: {logit_scale:.2f}")

            tb_lg.update(head='condalign_loss', loss=loss.item(), step=ep * iters_train + it)
            tb_lg.update(head='contrastive_loss', contrastive_loss=contrastive_loss.item(),
                         step=ep * iters_train + it)
            tb_lg.update(head='pairwise_loss', pairwise_loss=pairwise_loss.item(), step=ep * iters_train + it)
            tb_lg.update(head='logit_scale', logit_scale=logit_scale, step=ep * iters_train + it)
            tb_lg.flush()

        if dist.is_master() and (it + 1) % 1000 == 0:
            checkpoint = {
                'model_arch': MODEL_ARCH,
                'epoch': ep,
                'iter': it + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_pairwise': best_val_pairwise,
                'fhat_k': args.fhat_k,
                'trainer': {},
                'args': {},
            }
            ckpt_path = os.path.join(args.local_out_dir_path, 'condalign-ckpt-last.pth')
            torch.save(checkpoint, ckpt_path)
            print(f" >>> [Auto-Save] Iteration {it+1} checkpoint saved to {ckpt_path}")

        if it % 50 == 0 or it == iters_train - 1:
            should_stop_now = torch.tensor(
                [1 if stop_signal.exists() else 0],
                device=device,
                dtype=torch.int32,
            )
            if dist.initialized():
                dist.allreduce(should_stop_now)
            if int(should_stop_now.item()) > 0:
                if dist.is_master():
                    checkpoint = {
                        'model_arch': MODEL_ARCH,
                        'epoch': ep,
                        'iter': it + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss_fn_state_dict': loss_fn.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_pairwise': best_val_pairwise,
                        'fhat_k': args.fhat_k,
                        'trainer': {},
                        'args': {},
                    }
                    ckpt_path = os.path.join(args.local_out_dir_path, 'condalign-ckpt-last.pth')
                    torch.save(checkpoint, ckpt_path)
                    if stop_signal.exists():
                        stop_signal.unlink()
                    print(f" >>> [Stop-Signal] Saved checkpoint at iter {it+1}: {ckpt_path}")
                stop_requested = True
                break

    denom = max(1, steps_to_run)
    avg_loss = total_loss / denom
    avg_contrastive_loss = total_contrastive_loss / denom
    avg_pairwise_loss = total_pairwise_loss / denom

    return avg_loss, avg_contrastive_loss, avg_pairwise_loss, stop_requested


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if getattr(args, 'precompute_jina_emb', 0):
        if dist.is_master():
            print("Precompute-only mode: computing Jina-CLIP-v2 image embeddings and exiting.")
        run_precompute_jina_emb(args)
        return

    (
        model, vae_local, jina, optimizer, scheduler, loss_fn, ld_train, ld_val, tb_lg,
        start_ep, start_it, iters_train, best_val_loss, best_val_pairwise,
        precomputed_emb_train, precomputed_emb_val
    ) = build_everything(args)

    log_file = None
    if dist.is_master():
        log_file = open('train_log.txt', 'a')
        print(f"Starting training with {iters_train} iterations per epoch")
        print(
            f"Starting from epoch {start_ep}, best_val_loss so far: {best_val_loss:.4f}, "
            f"best_val_pairwise so far: {best_val_pairwise:.4f}"
        )

    total_epochs = args.ep
    for ep in range(start_ep, total_epochs):
        loss_fn.contrastive_weight = get_contrastive_weight_by_epoch(ep)
        lr_scale = get_lr_scale_by_epoch(ep, total_epochs)
        for pg_idx, pg in enumerate(optimizer.param_groups):
            pg['lr'] = pg['initial_lr'] * lr_scale
        if dist.is_master():
            print(
                f"[Ep {ep}] contrastive_weight={loss_fn.contrastive_weight:.4f} "
                f"| lr_scale={lr_scale:.4f} | model_lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        current_start_it = start_it if ep == start_ep else 0
        avg_loss, avg_contrastive_loss, avg_pairwise_loss, stop_requested = train_one_ep(
            ep, current_start_it, args, model, vae_local, jina, optimizer, scheduler, loss_fn, ld_train, iters_train, tb_lg,
            best_val_loss=best_val_loss,
            best_val_pairwise=best_val_pairwise,
            precomputed_emb_train=precomputed_emb_train,
        )
        if stop_requested:
            should_stop = torch.ones(1, dtype=torch.int32, device=dist.get_device())
            if dist.initialized():
                dist.broadcast(should_stop, src_rank=0)
            break

        val_loss, val_contrastive_loss, val_pairwise_loss = eval_one_ep(
            model, vae_local, jina, loss_fn, ld_val, precomputed_emb_val=precomputed_emb_val,
            fhat_k=args.fhat_k,
        )
        scheduler.step(val_pairwise_loss)

        dist.barrier()

        should_stop = torch.zeros(1, dtype=torch.int32, device=dist.get_device())
        if dist.is_master():
            print(f"[Ep {ep}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"[Ep {ep}] Val Pairwise Loss: {val_pairwise_loss:.4f}")
            tb_lg.update(head='val_loss', val_loss=val_loss, step=ep)
            tb_lg.update(head='val_contrastive_loss', val_contrastive_loss=val_contrastive_loss, step=ep)
            tb_lg.update(head='val_pairwise_loss', val_pairwise_loss=val_pairwise_loss, step=ep)

            prev_best_val = best_val_loss
            prev_best_pairwise = best_val_pairwise
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_pairwise_loss < best_val_pairwise:
                best_val_pairwise = val_pairwise_loss

            checkpoint = {
                'model_arch': MODEL_ARCH,
                'epoch': ep + 1,
                'iter': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_pairwise': best_val_pairwise,
                'avg_loss': avg_loss,
                'avg_pairwise_loss': avg_pairwise_loss,
                'val_loss': val_loss,
                'val_pairwise_loss': val_pairwise_loss,
                'fhat_k': args.fhat_k,
                'trainer': {},
                'args': {},
            }

            last_ckpt_path = os.path.join(args.local_out_dir_path, 'condalign-ckpt-last.pth')
            torch.save(checkpoint, last_ckpt_path)

            if val_pairwise_loss < prev_best_pairwise:
                best_ckpt_path = os.path.join(args.local_out_dir_path, 'condalign-ckpt-best.pth')
                torch.save(checkpoint, best_ckpt_path)
                print(
                    f"New best checkpoint: val_loss={best_val_loss:.4f}, "
                    f"val_pairwise={best_val_pairwise:.4f} saved to {best_ckpt_path}"
                )

            tb_lg.flush()
            if log_file:
                log_file.write(
                    f"Epoch {ep} finished. Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Val Pairwise: {val_pairwise_loss:.4f} | Best val: {best_val_loss:.4f}\n"
                )
                log_file.flush()

        if dist.initialized():
            dist.broadcast(should_stop, src_rank=0)
        if int(should_stop.item()) != 0:
            break

    if dist.is_master() and log_file:
        log_file.close()
    tb_lg.close()


if __name__ == "__main__":
    main_training()
