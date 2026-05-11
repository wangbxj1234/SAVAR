import torch
import torch.nn as nn
import torch.nn.functional as F

class Injector(nn.Module):
    def __init__(self, image_dim, text_dim, num_attn_head=8):
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_attn_head = num_attn_head
        
        # 文本特征投影到图像维度
        self.text_feature_dim_proj = nn.Linear(text_dim, image_dim)
        nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)
        
        # LayerNorm
        self.image_norm = nn.LayerNorm(image_dim)
        self.text_norm = nn.LayerNorm(image_dim)
        
        # 交叉注意力：图像作为query，文本作为key和value
        self.cross_attn = nn.MultiheadAttention(image_dim, num_attn_head, batch_first=True)
        
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.zeros(image_dim), requires_grad=True)
        
    def forward(self, fhat, text_emb):
        # fhat: [B, C, H, W], text_emb: [B, D]
        B, C, H, W = fhat.shape
        
        # 将图像特征reshape为序列格式
        image_features = fhat.contiguous().flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # 投影文本特征到图像维度
        text_features = self.text_feature_dim_proj(text_emb).unsqueeze(1)  # [B, 1, C]
        
        # LayerNorm
        image_features = self.image_norm(image_features)
        text_features = self.text_norm(text_features)
        
        # 交叉注意力：图像query，文本key/value
        attn_out, attn_weights = self.cross_attn(image_features, text_features, text_features)
        
        # 残差连接 + 可学习缩放
        image_features = image_features + self.gamma.unsqueeze(0).unsqueeze(0) * attn_out
        
        # 恢复原始形状
        output = image_features.contiguous().permute(0, 2, 1).view(B, C, H, W)
        
        return output

class Extractor(nn.Module):
    def __init__(self, image_dim, text_dim, num_attn_head=8):
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_attn_head = num_attn_head
        
        # 图像特征投影到文本维度
        self.image_feature_dim_proj = nn.Linear(image_dim, text_dim)
        nn.init.kaiming_normal_(self.image_feature_dim_proj.weight)
        
        # LayerNorm
        self.image_norm = nn.LayerNorm(text_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        # 交叉注意力：文本作为query，图像作为key和value
        self.cross_attn = nn.MultiheadAttention(text_dim, num_attn_head, batch_first=True)
        
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.zeros(text_dim), requires_grad=True)
        
    def forward(self, text_emb, inj_out):
        # text_emb: [B, D], inj_out: [B, C, H, W]
        B, C, H, W = inj_out.shape
        
        # 将图像特征reshape并投影到文本维度
        image_features = inj_out.contiguous().flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        image_features = self.image_feature_dim_proj(image_features)  # [B, H*W, text_dim]
        
        # 确保文本特征有正确的形状
        if len(text_emb.shape) == 2:
            text_features = text_emb.unsqueeze(1)  # [B, 1, D]
        else:
            text_features = text_emb  # [B, seq_len, D]
        
        # LayerNorm
        image_features = self.image_norm(image_features)
        text_features = self.text_norm(text_features)
        
        # 交叉注意力：文本query，图像key/value
        attn_out, attn_weights = self.cross_attn(text_features, image_features, image_features)
        
        # 残差连接 + 可学习缩放
        text_features = text_features + self.gamma.unsqueeze(0).unsqueeze(0) * attn_out
        
        # 如果原始输入是2D，则输出也是2D
        if len(text_emb.shape) == 2:
            return text_features.squeeze(1)  # [B, D]
        else:
            return text_features  # [B, seq_len, D]

class InjExtraCondAlignModel(nn.Module):
    def __init__(self, image_dim=32, text_dim=768, out_dim=768, num_attn_head=8, dropout=0.1):
        super().__init__()
        self.injector = Injector(image_dim, text_dim, num_attn_head)
        self.extractor = Extractor(image_dim, text_dim, num_attn_head)
        
        # 最终自注意力层
        self.final_norm = nn.LayerNorm(text_dim)
        self.final_self_attn = nn.MultiheadAttention(text_dim, num_attn_head, dropout=dropout, batch_first=True)
        self.final_gamma = nn.Parameter(torch.zeros(text_dim), requires_grad=True)
        
        # FFN层
        self.ffn_norm = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 4, text_dim),
            nn.Dropout(dropout)
        )
        self.ffn_gamma = nn.Parameter(torch.zeros(text_dim), requires_grad=True)
        
        # 最终投影层
        self.proj = nn.Linear(text_dim, out_dim)

    def forward(self, text_emb, fhat):
        # text_emb: [B, D], fhat: [B, C, H, W]
        inj_out = self.injector(fhat, text_emb)  # [B, C, H, W] - 保持形状不变
        extra_out = self.extractor(text_emb, inj_out)  # [B, D] - 保持形状不变
        
        # 添加自注意力层进行最终细化
        # 需要添加序列维度进行自注意力
        features = extra_out.unsqueeze(1)  # [B, 1, D]
        
        # 自注意力层
        normed_features = self.final_norm(features)
        self_attn_out, _ = self.final_self_attn(normed_features, normed_features, normed_features)
        features = features + self.final_gamma.unsqueeze(0).unsqueeze(0) * self_attn_out
        
        # FFN层
        normed_features = self.ffn_norm(features)
        ffn_out = self.ffn(normed_features)
        features = features + self.ffn_gamma.unsqueeze(0).unsqueeze(0) * ffn_out
        
        # 移除序列维度并投影到输出维度
        features = features.squeeze(1)  # [B, D]
        out = self.proj(features)  # [B, out_dim]
        
        return out

class FusionBlock(nn.Module):
    """Cross-attn (query→context) + self-attn (query↔query) + FFN, all pre-norm."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, context):
        q = self.cross_norm_q(queries)
        kv = self.cross_norm_kv(context)
        queries = queries + self.cross_attn(q, kv, kv)[0]

        q = self.self_norm(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class CondAlignModelV3(nn.Module):
    """
    Pooled-text-safe condition alignment model.

    Spatial fhat tokens are first compressed into a small set of learned image
    summary tokens. The single pooled text token then queries those summaries,
    avoiding raw H*W image tokens competing with one text token in a concatenated
    context.
    """
    def __init__(self, image_dim=32, text_dim=1024, out_dim=1024,
                 hidden_dim=512, num_heads=8,
                 num_image_summary_tokens=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_image_summary_tokens = num_image_summary_tokens

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.image_summary_tokens = nn.Parameter(
            torch.randn(1, num_image_summary_tokens, hidden_dim) * 0.02)

        self.summary_norm_q = nn.LayerNorm(hidden_dim)
        self.summary_norm_kv = nn.LayerNorm(hidden_dim)
        self.image_summary_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.text_norm_q = nn.LayerNorm(hidden_dim)
        self.text_norm_kv = nn.LayerNorm(hidden_dim)
        self.text_to_image_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, text_emb, fhat, return_diagnostics=False):
        B, C, H, W = fhat.shape

        img_tokens = self.image_proj(
            fhat.flatten(2).permute(0, 2, 1))           # [B, H*W, hidden]
        txt_tokens = self.text_proj(
            text_emb).unsqueeze(1)                       # [B, 1, hidden]

        summary_queries = self.image_summary_tokens.expand(B, -1, -1)
        image_summaries, image_summary_attn = self.image_summary_attn(
            self.summary_norm_q(summary_queries),
            self.summary_norm_kv(img_tokens),
            self.summary_norm_kv(img_tokens),
        )                                                     # [B, M, hidden]

        spatial_update, text_to_image_attn = self.text_to_image_attn(
            self.text_norm_q(txt_tokens),
            self.text_norm_kv(image_summaries),
            self.text_norm_kv(image_summaries),
        )                                                     # [B, 1, hidden]

        fusion_gate = self.gate(torch.cat([txt_tokens, spatial_update], dim=-1))
        fused = txt_tokens + fusion_gate * spatial_update
        fused = fused + self.fusion_ffn(self.fusion_norm(fused))

        out = self.output_proj(self.output_norm(fused).squeeze(1))  # [B, out_dim]
        if not return_diagnostics:
            return out

        diagnostics = {
            "image_summary_attention": image_summary_attn,
            "text_to_image_attention": text_to_image_attn,
            "fusion_gate": fusion_gate,
            "image_summary_attention_mass": image_summary_attn.sum(dim=-1),
            "text_to_image_attention_mass": text_to_image_attn.sum(dim=-1),
            "fusion_gate_mean": fusion_gate.mean(dim=-1),
        }
        return out, diagnostics


class CondAlignModelV2(CondAlignModelV3):
    """
    Backward-compatible name for the redesigned pooled-text-safe aligner.
    """
    def __init__(self, image_dim=32, text_dim=1024, out_dim=1024,
                 hidden_dim=512, num_layers=4, num_heads=8,
                 num_query_tokens=8, dropout=0.1):
        super().__init__(
            image_dim=image_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_image_summary_tokens=num_query_tokens,
            dropout=dropout,
        )
        self.num_layers = num_layers
        self.num_query_tokens = num_query_tokens


class CondAlignModelV4FhatGate(nn.Module):
    """
    Variant of V3 that uses the text-driven cross-attention result to gate
    spatial fhat tokens, instead of gating the text token.
    """
    def __init__(self, image_dim=32, text_dim=1024, out_dim=1024,
                 hidden_dim=512, num_heads=8,
                 num_image_summary_tokens=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_image_summary_tokens = num_image_summary_tokens

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.image_summary_tokens = nn.Parameter(
            torch.randn(1, num_image_summary_tokens, hidden_dim) * 0.02)

        self.summary_norm_q = nn.LayerNorm(hidden_dim)
        self.summary_norm_kv = nn.LayerNorm(hidden_dim)
        self.image_summary_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.text_norm_q = nn.LayerNorm(hidden_dim)
        self.text_norm_kv = nn.LayerNorm(hidden_dim)
        self.text_to_image_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.fhat_gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, text_emb, fhat, return_diagnostics=False):
        B, _, _, _ = fhat.shape

        img_tokens = self.image_proj(
            fhat.flatten(2).permute(0, 2, 1))           # [B, H*W, hidden]
        txt_tokens = self.text_proj(
            text_emb).unsqueeze(1)                       # [B, 1, hidden]

        summary_queries = self.image_summary_tokens.expand(B, -1, -1)
        image_summaries, image_summary_attn = self.image_summary_attn(
            self.summary_norm_q(summary_queries),
            self.summary_norm_kv(img_tokens),
            self.summary_norm_kv(img_tokens),
        )                                                # [B, M, hidden]

        spatial_update, text_to_image_attn = self.text_to_image_attn(
            self.text_norm_q(txt_tokens),
            self.text_norm_kv(image_summaries),
            self.text_norm_kv(image_summaries),
        )                                                # [B, 1, hidden]

        # Broadcast the CA result to each spatial token and gate per token.
        spatial_context = spatial_update.expand(-1, img_tokens.size(1), -1)
        fhat_gate = self.fhat_gate_mlp(
            torch.cat([img_tokens, spatial_context], dim=-1))
        gated_fhat_tokens = img_tokens + fhat_gate * spatial_context

        pooled = gated_fhat_tokens.mean(dim=1, keepdim=True)
        fused = pooled + self.fusion_ffn(self.fusion_norm(pooled))
        out = self.output_proj(self.output_norm(fused).squeeze(1))  # [B, out_dim]
        if not return_diagnostics:
            return out

        diagnostics = {
            "image_summary_attention": image_summary_attn,
            "text_to_image_attention": text_to_image_attn,
            "spatial_update": spatial_update,
            "fhat_gate": fhat_gate,
            "gated_fhat_tokens": gated_fhat_tokens,
            "image_summary_attention_mass": image_summary_attn.sum(dim=-1),
            "text_to_image_attention_mass": text_to_image_attn.sum(dim=-1),
            "fhat_gate_mean": fhat_gate.mean(dim=(1, 2)),
        }
        return out, diagnostics


if __name__ == "__main__":
    B, C, H, W = 2, 32, 8, 8
    text_dim = 1024
    out_dim = 1024

    for name, cls, kw in [
        ("V1 (InjExtraCondAlignModel)",
         InjExtraCondAlignModel,
         dict(image_dim=C, text_dim=text_dim, out_dim=out_dim)),
        ("V2 (CondAlignModelV2)",
         CondAlignModelV2,
         dict(image_dim=C, text_dim=text_dim, out_dim=out_dim)),
    ]:
        m = cls(**kw)
        text_emb = torch.randn(B, text_dim)
        fhat = torch.randn(B, C, H, W)
        out = m(text_emb, fhat)
        n_params = sum(p.numel() for p in m.parameters())
        out.sum().backward()
        print(f"[{name}]  params={n_params:,}  output={out.shape}  grad OK")
