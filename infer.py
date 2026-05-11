"""
infer.py
VARSlip + Jina-CLIP-v2 (1024-dim) inference.
"""
import os, json, argparse, math, time, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image as PImage
from pathlib import Path

from models.vqvae import VQVAE
from models.varslip import VARSlip
from condition_model import CondAlignModelV2
from transformers import AutoModel

# ========== 配置 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DEPTH = 36
PATCH_NUMS = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
FINAL_RESO = 512
DEFAULT_FHAT_K = 4
TOKEN_PATCH_NUMS = PATCH_NUMS[: DEFAULT_FHAT_K + 1]
TOKEN_COUNT = sum(pn * pn for pn in TOKEN_PATCH_NUMS)

JINA_CKPT = "jinaai/jina-clip-v2"
JINA_DIM = 1024

KODAK_DATA = [
    ("kodim01.png", "a brick building with red doors and windows"),
    ("kodim02.png", "a red door with a metal door knocker on it"),
    ("kodim03.png", "a group of hats on the side of a wall"),
    ("kodim04.png", "a woman wearing a red hat and a red dress"),
    ("kodim05.png", "a group of people on dirt bikes in a race"),
    ("kodim06.png", "a boat floating in the water in the ocean"),
    ("kodim07.png", "a pink flower in front of a window"),
    ("kodim08.png", "a group of older buildings in a city"),
    ("kodim09.png", "a group of small sailboats in the water"),
    ("kodim10.png", "a group of sailboats in the water"),
    ("kodim11.png", "a boat in the water next to a pier"),
    ("kodim12.png", "a man and a woman walking on the beach"),
    ("kodim13.png", "a stream of water with trees and mountains in the background"),
    ("kodim14.png", "a group of people in a raft on a river"),
    ("kodim15.png", "a young girl with paint on her face"),
    ("kodim16.png", "a large body of water with palm trees on an island"),
    ("kodim17.png", "a statue of a woman holding a coconut"),
    ("kodim18.png", "a woman in a dress holding an umbrella"),
    ("kodim19.png", "a lighthouse next to a white picket fence"),
    ("kodim20.png", "a small plane sitting on the grass in a field"),
    ("kodim21.png", "a lighthouse on a rocky island in the ocean"),
    ("kodim22.png", "a red barn sitting next to a body of water"),
    ("kodim23.png", "two colorful parrots standing next to each other"),
    ("kodim24.png", "a house with a painting on the side of it"),
]

def _load_data_manifest(path: str):
    if not path:
        return None
    manifest_path = os.path.expanduser(path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"--data_manifest not found: {manifest_path}")

    def _to_entry(obj, idx):
        if not isinstance(obj, dict):
            raise TypeError(f"Manifest row #{idx} must be an object/dict")
        raw_img = (
            obj.get("image")
            or obj.get("img")
            or obj.get("img_path")
            or obj.get("path")
            or obj.get("file")
        )
        if not raw_img:
            raise KeyError(f"Manifest row #{idx} missing image/img_path/path/file")
        raw_caption = obj.get("caption", obj.get("txt", obj.get("text", "")))
        raw_name = obj.get("name", obj.get("basename", obj.get("id", "")))
        img_path = os.path.expanduser(str(raw_img))
        name = os.path.basename(str(raw_name).strip()) if str(raw_name).strip() else os.path.basename(img_path)
        if not name:
            raise ValueError(f"Manifest row #{idx} has invalid name/img_path")
        return {
            "name": name,
            "img_path": img_path,
            "caption": str(raw_caption),
        }

    ext = os.path.splitext(manifest_path)[1].lower()
    rows = []
    if ext in {".jsonl", ".jsonlines"}:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append(_to_entry(json.loads(line), i))
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            data = data["data"]
        if not isinstance(data, list):
            raise TypeError("Manifest JSON must be a list of objects, or {'data': [...]} structure")
        rows = [_to_entry(obj, i + 1) for i, obj in enumerate(data)]
    if not rows:
        raise ValueError(f"--data_manifest has no valid rows: {manifest_path}")
    return rows

def _apply_caption_mode(rows, mode: str, generic_caption: str):
    if mode == "correct":
        return rows
    if mode == "none":
        return [{**row, "caption": ""} for row in rows]
    if mode == "generic":
        return [{**row, "caption": generic_caption} for row in rows]
    if mode in {"shuffle", "wrong"}:
        captions = [row["caption"] for row in rows]
        if len(captions) > 1:
            captions = captions[1:] + captions[:1]
        return [{**row, "caption": captions[i]} for i, row in enumerate(rows)]
    raise ValueError(f"Unsupported caption_mode: {mode}")

def _apply_caption_length_controls(rows, keep_ratio: float, max_words: int):
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    out = []
    for row in rows:
        text = str(row["caption"])
        words = text.split()
        if words:
            keep_n = len(words)
            if keep_ratio < 1.0:
                keep_n = max(1, int(math.ceil(len(words) * keep_ratio)))
            if max_words > 0:
                keep_n = min(keep_n, max_words)
            text = " ".join(words[:keep_n])
        out.append({**row, "caption": text})
    return out

def _apply_caption_overrides(rows, override_json_path: str):
    if not override_json_path:
        return rows
    with open(os.path.expanduser(override_json_path), "r", encoding="utf-8") as f:
        override_obj = json.load(f)
    if not isinstance(override_obj, dict):
        raise TypeError("--caption_override_json must be a JSON object: {name_or_basename: caption}")
    out = []
    for row in rows:
        name = row["name"]
        base = os.path.basename(name)
        if name in override_obj:
            caption = str(override_obj[name])
        elif base in override_obj:
            caption = str(override_obj[base])
        else:
            caption = row["caption"]
        out.append({**row, "caption": caption})
    return out

def _save_image_tensor(img_tensor, out_path: str) -> None:
    grid = torchvision.utils.make_grid(img_tensor.unsqueeze(0), nrow=1, normalize=False)
    img_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    PImage.fromarray(img_np).save(out_path)

vae_ckpt = 'pretrained/vae_ch160v4096z32.pth'
var_ckpt = 'output_jina_var_v6/ar-ckpt-best.pth'
condition_ckpt = 'output_jinacondv3/condalign-ckpt-last.pth'

kodak_dir = "../kodak"
output_dir = "output"

num_samples = 1
cfg = 0.0
top_k = 900
top_p = 0.95
more_smooth = False
seed = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Kodak inference (VAR + Jina)."
    )
    parser.add_argument("--var_ckpt", type=str, default=var_ckpt)
    parser.add_argument("--condition_ckpt", type=str, default=condition_ckpt)
    parser.add_argument("--vae_ckpt", type=str, default=vae_ckpt)
    parser.add_argument("--kodak_dir", type=str, default=kodak_dir)
    parser.add_argument(
        "--data_manifest",
        type=str,
        default="",
        help="Optional JSON/JSONL entries with image path + caption; when set, overrides hardcoded Kodak list.",
    )
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--num_samples", type=int, default=num_samples)
    parser.add_argument("--cfg", type=float, default=cfg)
    parser.add_argument("--top_k", type=int, default=top_k)
    parser.add_argument("--top_p", type=float, default=top_p)
    parser.add_argument("--more_smooth", action="store_true", default=more_smooth)
    parser.add_argument("--no_more_smooth", action="store_false", dest="more_smooth")
    parser.add_argument("--seed", type=int, default=seed, help="Sampling seed; default None keeps stochastic behavior.")
    parser.add_argument("--final_reso", type=int, default=FINAL_RESO)
    parser.add_argument(
        "--profile_max_images", type=int, default=0,
        help="If >0, only run the first N Kodak entries (for quick profiling). 0 = all.",
    )
    parser.add_argument(
        "--ar_precision", type=str, default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="Autocast precision for autoregressive decode path.",
    )
    parser.add_argument(
        "--compile_ar", type=str, default="none",
        choices=["none", "reduce-overhead", "max-autotune", "default"],
        help="Optional torch.compile mode for autoregressive decode call.",
    )
    parser.add_argument(
        "--text_emb_table",
        type=str,
        default="",
        help="Optional .pt dict[basename e.g. kodim01.png -> FloatTensor[1,1024]] Jina-normalized text emb; skips loading Jina text tower.",
    )
    parser.add_argument(
        "--dump_text_emb_table",
        type=str,
        default="",
        help="Encode KODAK_DATA captions once with Jina, save .pt, exit (no VAR/VAE run).",
    )
    parser.add_argument(
        "--fhat_k",
        type=int,
        default=DEFAULT_FHAT_K,
        help="VAE img_to_fhat_k index. Codec uses first k+1 scales.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Alias for --fhat_k; if set, overrides --fhat_k.",
    )
    parser.add_argument(
        "--sos_source",
        type=str,
        default="structural",
        choices=["structural", "text"],
        help="Ablation switch: structural uses fhat-derived SOS; text uses AdaLN condition as SOS.",
    )
    parser.add_argument(
        "--adaln_source",
        type=str,
        default="aligned",
        choices=["aligned", "text"],
        help="Ablation switch: aligned uses CondAlignModelV2(text, fhat); text uses raw Jina text embedding.",
    )
    parser.add_argument(
        "--caption_mode",
        type=str,
        default="correct",
        choices=["correct", "none", "generic", "shuffle", "wrong"],
        help="Text robustness ablation: correct/empty/generic/wrong(shuffle) caption.",
    )
    parser.add_argument(
        "--generic_caption",
        type=str,
        default="an image",
        help="Caption string used when --caption_mode generic.",
    )
    parser.add_argument(
        "--caption_keep_ratio",
        type=float,
        default=1.0,
        help="Keep first ceil(N*ratio) caption words after caption_mode. 1.0 keeps full caption.",
    )
    parser.add_argument(
        "--caption_max_words",
        type=int,
        default=0,
        help="If >0, truncate caption to at most this many words after ratio truncation.",
    )
    parser.add_argument(
        "--caption_override_json",
        type=str,
        default="",
        help="Optional JSON dict to override captions by filename (name or basename).",
    )
    parser.add_argument(
        "--text_emb_mode",
        type=str,
        default="normal",
        choices=["normal", "zero"],
        help="Text embedding mode: normal Jina embedding or all-zero embedding.",
    )
    parser.add_argument(
        "--disable_var_flash_attn",
        action="store_true",
        help="Disable VAR flash-attention kernels for stability.",
    )
    return parser.parse_args()

args = parse_args()
_fhat_k = args.k if args.k is not None else args.fhat_k
if _fhat_k < 0 or _fhat_k >= len(PATCH_NUMS):
    raise SystemExit(f"--fhat_k/--k must be in [0, {len(PATCH_NUMS) - 1}], got {_fhat_k}")
TOKEN_PATCH_NUMS = PATCH_NUMS[: _fhat_k + 1]
TOKEN_COUNT = sum(pn * pn for pn in TOKEN_PATCH_NUMS)
fhat_k = _fhat_k

vae_ckpt = args.vae_ckpt
var_ckpt = args.var_ckpt
condition_ckpt = args.condition_ckpt
kodak_dir = args.kodak_dir
data_manifest_path = (args.data_manifest or "").strip()
output_dir = args.output_dir
num_samples = args.num_samples
cfg = args.cfg
top_k = args.top_k
top_p = args.top_p
more_smooth = args.more_smooth
seed = args.seed
FINAL_RESO = args.final_reso
profile_max_images = max(0, args.profile_max_images)
ar_precision = args.ar_precision
compile_ar = args.compile_ar
text_emb_table_path = (args.text_emb_table or "").strip()
dump_text_emb_table_path = (args.dump_text_emb_table or "").strip()
sos_source = args.sos_source
adaln_source = args.adaln_source
caption_mode = args.caption_mode
generic_caption = args.generic_caption
caption_keep_ratio = float(args.caption_keep_ratio)
caption_max_words = max(0, int(args.caption_max_words))
caption_override_json = (args.caption_override_json or "").strip()
text_emb_mode = args.text_emb_mode
disable_var_flash_attn = bool(args.disable_var_flash_attn)
os.makedirs(output_dir, exist_ok=True)
print(
    f"[infer STREAM] fhat_k={fhat_k}  TOKEN_PATCH_NUMS={TOKEN_PATCH_NUMS}  token_count={TOKEN_COUNT}"
)

if data_manifest_path:
    eval_data = _load_data_manifest(data_manifest_path)
    print(f"[infer STREAM] Using data manifest: {data_manifest_path} ({len(eval_data)} entries)")
else:
    eval_data = [
        {"name": fname, "img_path": os.path.join(kodak_dir, fname), "caption": caption}
        for fname, caption in KODAK_DATA
    ]
    print(f"[infer STREAM] Using built-in Kodak list ({len(eval_data)} entries)")
eval_data = _apply_caption_mode(eval_data, caption_mode, generic_caption)
eval_data = _apply_caption_length_controls(eval_data, keep_ratio=caption_keep_ratio, max_words=caption_max_words)
eval_data = _apply_caption_overrides(eval_data, caption_override_json)
print(
    "[infer STREAM] Ablation switches: "
    f"sos_source={sos_source}, adaln_source={adaln_source}, caption_mode={caption_mode}, "
    f"text_emb_mode={text_emb_mode}, keep_ratio={caption_keep_ratio:g}, max_words={caption_max_words}"
)

if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

def resolve_ar_autocast_dtype(mode: str):
    if device != "cuda":
        return None
    if mode == "fp32":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def maybe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def encode_text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")

def decode_text_from_bytes(payload: bytes) -> str:
    return payload.decode("utf-8")

def gzip_caption_bits(text: str) -> int:
    raw = text.encode("utf-8")
    return len(gzip.compress(raw, compresslevel=9)) * 8

def pack_token_scales_to_bytes(ms_idx_Bl):
    flat = torch.cat([idx.reshape(-1).to(torch.int32).cpu() for idx in ms_idx_Bl], dim=0).numpy().astype(np.uint16)
    return flat.tobytes()

def unpack_token_scales_from_bytes(token_bytes: bytes, token_lengths, batch_size: int, device):
    flat = np.frombuffer(token_bytes, dtype=np.uint16).astype(np.int64, copy=True)
    expected = batch_size * sum(token_lengths)
    if flat.size != expected:
        raise RuntimeError(f"Token payload size mismatch: got {flat.size}, expected {expected}.")
    cur = 0
    out = []
    for l in token_lengths:
        chunk = flat[cur:cur + batch_size * l].reshape(batch_size, l)
        out.append(torch.from_numpy(chunk).to(device=device, dtype=torch.long))
        cur += batch_size * l
    return out

def reconstruct_fhat_from_token_scales(vae: VQVAE, ms_idx_Bl):
    if not ms_idx_Bl:
        raise ValueError("Empty token scales.")
    B = ms_idx_Bl[0].shape[0]
    C = vae.Cvae
    H = W = vae.quantize.v_patch_nums[-1]
    SN_full = len(vae.quantize.v_patch_nums)

    f_hat = torch.zeros((B, C, H, W), device=ms_idx_Bl[0].device, dtype=torch.float32)
    for si, idx_Bl in enumerate(ms_idx_Bl):
        l = idx_Bl.shape[1]
        pn = int(round(l ** 0.5))
        if pn * pn != l:
            raise RuntimeError(f"Invalid token length {l}, not square.")
        idx_Bhw = idx_Bl.view(B, pn, pn)
        h_BChw = vae.quantize.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
        h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
        h_BChw = vae.quantize.quant_resi[si / (SN_full - 1)](h_BChw)
        f_hat.add_(h_BChw)
    return f_hat

class JinaCLIPv2Wrapper(nn.Module):
    def __init__(self, device, normalize=False):
        super().__init__()
        self.device = device
        self.normalize = normalize
        load_errors = []
        model = None
        for extra_kwargs in (
            {"use_text_flash_attn": False},
            {"use_text_flash_attn": False, "attn_implementation": "eager"},
            {"attn_implementation": "eager"},
            {"use_flash_attn": False},
            {},
        ):
            try:
                model = AutoModel.from_pretrained(
                    JINA_CKPT, trust_remote_code=True, **extra_kwargs
                )
                if extra_kwargs:
                    print(f"[JinaCLIPv2Wrapper] loaded with kwargs={extra_kwargs}")
                break
            except Exception as e:
                load_errors.append(f"kwargs={extra_kwargs}: {e}")
        if model is None:
            raise RuntimeError(
                "Failed to load jinaai/jina-clip-v2. Tried eager/flash fallback paths:\n"
                + "\n".join(load_errors)
            )
        self.model = model
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _to_tensor(self, emb):
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        return emb.to(self.device)

    @torch.no_grad()
    def encode_text(self, captions):
        if isinstance(captions, str):
            captions = [captions]
        elif isinstance(captions, tuple):
            captions = list(captions)
        emb = self.model.encode_text(captions)
        emb = self._to_tensor(emb)
        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

def build_varslip_for_infer(device, patch_nums, depth=36, shared_aln=False, flash_if_available=True):
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm,
                nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    vae_local = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                      share_quant_resi=4, v_patch_nums=patch_nums).to(device)
    var = VARSlip(
        vae_local=vae_local,
        n_cond_embed=1024, depth=depth, embed_dim=width, num_heads=heads,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=True, patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=True,
    ).to(device)
    return vae_local, var

def infer_shared_aln_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    has_shared = any(k.startswith("shared_ada_lin") for k in keys)
    has_block = any(".ada_lin." in k and k.startswith("blocks.") for k in keys)

    if has_shared and not has_block:
        return True
    if has_block and not has_shared:
        return False
    if has_shared and has_block:
        raise RuntimeError("Checkpoint contains both shared_ada_lin and per-block ada_lin keys; cannot infer shared_aln.")
    raise RuntimeError("Cannot infer shared_aln from checkpoint keys.")

if dump_text_emb_table_path:
    print(f"[dump_text_emb_table] Jina encode for {len(eval_data)} captions -> {dump_text_emb_table_path}")
    _jtmp = JinaCLIPv2Wrapper("cpu", normalize=True)
    _table = {row["name"]: _jtmp.encode_text([row["caption"]]).detach().cpu() for row in eval_data}
    torch.save(_table, dump_text_emb_table_path)
    print(f"[dump_text_emb_table] Done, saved {len(_table)} tensors [1,1024].")
    raise SystemExit(0)

text_emb_table = None
if text_emb_table_path:
    text_emb_table = torch.load(text_emb_table_path, map_location="cpu")
    if not isinstance(text_emb_table, dict):
        raise TypeError("--text_emb_table must be a .pt dict[str, Tensor]")
    print(
        f"Using --text_emb_table ({len(text_emb_table)} keys); Jina text encoder will not be loaded."
    )

def get_text_emb_tensor(fname: str, caption: str):
    if text_emb_mode == "zero":
        return torch.zeros((1, 1024), device=device, dtype=torch.float32)
    if text_emb_table is not None:
        if fname not in text_emb_table:
            raise KeyError(f"{fname} missing in text_emb_table (have {len(text_emb_table)} keys)")
        te = text_emb_table[fname]
        if not isinstance(te, torch.Tensor):
            raise TypeError(f"text_emb_table[{fname}] must be Tensor")
        te = te.to(device, non_blocking=True)
    else:
        te = jina.encode_text([caption])
    return te

def build_adaln_condition(text_emb: torch.Tensor, fhat: torch.Tensor) -> torch.Tensor:
    if adaln_source == "aligned":
        if condition_model is None:
            raise RuntimeError("condition_model is required when --adaln_source aligned")
        with torch.autocast(device_type="cuda", enabled=False):
            cond = condition_model(text_emb.float(), fhat.float())
        if not torch.isfinite(cond).all():
            cond = text_emb.float()
    else:
        cond = text_emb
    cond = F.normalize(cond, dim=1)
    if not torch.isfinite(cond).all():
        cond = F.normalize(text_emb.float(), dim=1)
    return cond

var_checkpoint = torch.load(var_ckpt, map_location='cpu')
var_weights = var_checkpoint['trainer']['var_wo_ddp'] if 'trainer' in var_checkpoint else var_checkpoint
var_weights = {k.replace("module.", ""): v for k, v in var_weights.items()}

shared_aln = infer_shared_aln_from_state_dict(var_weights)
print(f"Building VARSlip (depth={MODEL_DEPTH}, Jina-CLIP-v2 1024-dim, shared_aln={shared_aln})...")
if disable_var_flash_attn and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
    print("Flash SDP disabled via --disable_var_flash_attn")
vae, var = build_varslip_for_infer(
    device,
    PATCH_NUMS,
    MODEL_DEPTH,
    shared_aln=shared_aln,
    flash_if_available=(not disable_var_flash_attn),
)

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
load_msg = var.load_state_dict(var_weights, strict=False)
if load_msg.missing_keys or load_msg.unexpected_keys:
    raise RuntimeError(
        "VAR checkpoint/model mismatch:\n"
        f"  missing_keys={len(load_msg.missing_keys)}\n"
        f"  unexpected_keys={len(load_msg.unexpected_keys)}\n"
        f"  sample_missing={load_msg.missing_keys[:10]}\n"
        f"  sample_unexpected={load_msg.unexpected_keys[:10]}"
    )

vae.eval().to(device)
var.eval().to(device)
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
if hasattr(var, 'rng'):
    var.rng = torch.Generator(device=device)

flash_blocks = sum(int(getattr(b.attn, "using_flash", False)) for b in var.blocks)
xform_blocks = sum(int(getattr(b.attn, "using_xform", False)) for b in var.blocks)
print(f"Attention backend: flash={flash_blocks}/{len(var.blocks)}, xformers={xform_blocks}/{len(var.blocks)}")

jina = None
if text_emb_table is None and text_emb_mode != "zero":
    print("Loading Jina-CLIP-v2...")
    jina = JinaCLIPv2Wrapper(device, normalize=True)
elif text_emb_mode == "zero":
    print("Skipping Jina-CLIP-v2 load: --text_emb_mode zero")

condition_model = None
if adaln_source == "aligned":
    print("Loading condition model (1024-dim)...")
    condition_model = CondAlignModelV2(image_dim=32, text_dim=1024, out_dim=1024).to(device)
    cond_ckpt = torch.load(condition_ckpt, map_location=device)
    if 'model_state_dict' in cond_ckpt:
        cond_ckpt = cond_ckpt['model_state_dict']
    condition_model.load_state_dict({k.replace('module.', ''): v for k, v in cond_ckpt.items()}, strict=False)
    condition_model.eval()
    for p in condition_model.parameters(): p.requires_grad_(False)
else:
    print("Skipping condition model: --adaln_source text")

def _resize_shorter_side_center_crop(image: PImage.Image, size: int) -> PImage.Image:
    w, h = image.size
    if w < 1 or h < 1:
        raise ValueError("empty image")
    scale = size / min(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    image = image.resize((nw, nh), PImage.Resampling.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return image.crop((left, top, left + size, top + size))

image_preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image_to_tensor = T.Compose([
    T.ToTensor(),
])

print(f"\nStarting VAR + Jina-CLIP-v2 {FINAL_RESO} Inference ({num_samples} samples per image)...")

token_lengths = [pn * pn for pn in TOKEN_PATCH_NUMS]
token_bits_per_symbol = int(math.ceil(math.log2(4096)))
ar_autocast_dtype = resolve_ar_autocast_dtype(ar_precision)
ar_decode_fn = var.autoregressive_infer_cfg
ar_compile_enabled = False
if compile_ar != "none":
    if hasattr(torch, "compile"):
        try:
            ar_decode_fn = torch.compile(ar_decode_fn, mode=compile_ar)
            ar_compile_enabled = True
        except Exception as e:
            print(f"[WARN] torch.compile for AR decode failed, fallback to eager: {e}")
    else:
        print("[WARN] torch.compile is unavailable in current PyTorch build; fallback to eager.")

print(
    "AR runtime setup: "
    f"precision={ar_precision} ({'fp32' if ar_autocast_dtype is None else str(ar_autocast_dtype).split('.')[-1]}), "
    f"compile={compile_ar} ({'on' if ar_compile_enabled else 'off'})"
)

for image_index, row in enumerate(eval_data):
    if profile_max_images > 0 and image_index >= profile_max_images:
        break
    try:
        fname = row["name"]
        caption = row["caption"]
        img_path = row["img_path"]
        if not os.path.exists(img_path):
            continue

        img_name_raw = Path(fname).stem
        save_sub_dir = os.path.join(output_dir, img_name_raw)
        os.makedirs(save_sub_dir, exist_ok=True)

        print(f"Processing {fname} -> {save_sub_dir}")
        image = PImage.open(img_path).convert("RGB")
        image_512 = _resize_shorter_side_center_crop(image, FINAL_RESO)
        image_tensor = image_preprocess(image_512).unsqueeze(0).to(device)

        with torch.no_grad():
            maybe_cuda_sync()
            all_scales_idx = vae.img_to_idxBl(image_tensor)
            token_scales = all_scales_idx[: len(TOKEN_PATCH_NUMS)]
            token_payload = pack_token_scales_to_bytes(token_scales)
            text_payload = encode_text_to_bytes(caption)

            caption_decoded = decode_text_from_bytes(text_payload)
            token_scales_decoded = unpack_token_scales_from_bytes(
                token_payload, token_lengths=token_lengths, batch_size=1, device=device
            )
            fhat_k_codec = reconstruct_fhat_from_token_scales(vae, token_scales_decoded)
            text_emb = get_text_emb_tensor(fname, caption_decoded)
            embed_fused = build_adaln_condition(text_emb, fhat_k_codec)

        with torch.inference_mode():
            maybe_cuda_sync()
            recon = ar_decode_fn(
                B=num_samples,
                label_B=embed_fused.expand(num_samples, -1),
                fhat_k=fhat_k_codec.expand(num_samples, -1, -1, -1),
                cfg=cfg, top_k=top_k, top_p=top_p,
                g_seed=seed,
                more_smooth=more_smooth,
                profile_decode=False,
                autocast_dtype=ar_autocast_dtype,
                sos_source=sos_source,
            )
            maybe_cuda_sync()

        recon_01 = recon.clamp(0, 1)

        for i in range(num_samples):
            sample_path = os.path.join(save_sub_dir, f"sample_{i+1:04d}.png")
            img_tensor = recon_01[i].unsqueeze(0).cpu()
            grid = torchvision.utils.make_grid(img_tensor, nrow=1, normalize=False)
            img_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
            PImage.fromarray(img_np).save(sample_path)

        # Save the first sample as sample_best.png for easy reference
        _save_image_tensor(recon_01[0].cpu(), os.path.join(save_sub_dir, "sample_best.png"))

        print(f"  Saved {num_samples} samples to {save_sub_dir}")

    except Exception as e:
        print(f"  Failed on {fname}: {e}")
        import traceback
        traceback.print_exc()

print("\nVAR + Jina-CLIP-v2 512 Inference completed!")
