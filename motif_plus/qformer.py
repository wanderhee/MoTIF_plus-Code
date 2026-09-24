import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class QFormerConfig:
    d_visual: int = 1152
    num_query: int = 32
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 3072
    dropout: float = 0.1
    cross_attention_frequency: int = 1


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"{dim} not divisible by {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        B, Lq, _ = query.shape
        B, Lk, _ = key.shape

        q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Lq, self.dim)
        return self.out_proj(out)


class QFormerLayer(nn.Module):
    def __init__(self, config: QFormerConfig, has_cross_attention: bool):
        super().__init__()
        self.has_cross_attention = has_cross_attention
        dim = config.hidden_size

        self.self_attn = MultiHeadAttention(dim, config.num_heads, config.dropout)
        if has_cross_attention:
            self.cross_attn = MultiHeadAttention(dim, config.num_heads, config.dropout)
            self.cross_ln = nn.LayerNorm(dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, visual, key_padding_mask=None):
        residual = query
        query = self.ln1(query)
        query = self.self_attn(query, query, query)
        query = residual + self.dropout(query)

        if self.has_cross_attention:
            residual = query
            query = self.cross_ln(query)
            query = self.cross_attn(query, visual, visual, key_padding_mask)
            query = residual + self.dropout(query)

        residual = query
        query = self.ln2(query)
        query = self.ffn(query)
        query = residual + self.dropout(query)
        return query


class QFormer(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.config = config

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query, config.hidden_size)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.visual_proj = nn.Linear(config.d_visual, config.hidden_size)

        self.layers = nn.ModuleList(
            QFormerLayer(
                config,
                has_cross_attention=(i % config.cross_attention_frequency == 0),
            )
            for i in range(config.num_layers)
        )
        self.ln_final = nn.LayerNorm(config.hidden_size)

    def forward(self, visual_features, key_padding_mask=None):
        B = visual_features.shape[0]
        visual = self.visual_proj(visual_features)
        query = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            query = layer(query, visual, key_padding_mask)

        return self.ln_final(query)
