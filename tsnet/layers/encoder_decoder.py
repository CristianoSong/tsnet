"""
This module implements the encoder and decoder blocks for TSNet.
It includes both encoder and decoder stacks, each consisting of multiple blocks.
"""

import torch
import torch.nn as nn


class TSNetEncoderBlock(nn.Module):
    """
    A single encoder block: decomposition -> attention -> FFN with residuals
    """
    def __init__(self, d_model, attention_layer, decomposition_layer, dropout=0.1):
        super().__init__()
        self.decomposition = decomposition_layer
        self.attention = attention_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        """
        Args:
            x: encoder input with shape [B, L, D]
        """
        seasonal, trend = self.decomposition(x)
        x = self.attention(seasonal) + x
        x = self.norm1(x)
        x = self.ffn(x) + x
        x = self.norm2(x)
        trend = self.trend_proj(trend)
        return x + trend


class TSNetEncoder(nn.Module):
    """
    A stack of encoder blocks for TSNet.
    """
    def __init__(self, d_model, attention_layer_fn, decomposition_layer_fn, n_layers=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSNetEncoderBlock(
                d_model=d_model,
                attention_layer=attention_layer_fn(),
                decomposition_layer=decomposition_layer_fn(),
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TSNetDecoderBlock(nn.Module):
    """
    A single decoder block for TSNet:
    decomposition -> self-attention -> FFN1 -> cross-attention -> FFN2
    """
    def __init__(self, d_model, self_attn_layer, cross_attn_layer, decomposition_layer, dropout=0.1):
        super().__init__()
        self.decomposition = decomposition_layer
        self.self_attention = self_attn_layer
        self.cross_attention = cross_attn_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x, enc_output=None):
        """
        Args:
            x: decoder input with shape [B, L, D]
            enc_output: optional encoder output [B, L', D]
        """
        assert x.dim() == 3, f"TSNetDecoderBlock expected x of shape [B, L, D], got {x.shape}"

        seasonal, trend = self.decomposition(x)

        # Self-attention + residual + norm
        x = self.self_attention(seasonal) + x
        x = self.norm1(x)

        # FFN1 + residual + norm
        x = self.ffn(x) + x
        x = self.norm2(x)

        # Cross-attention + residual + norm (if encoder output available)
        if enc_output is not None:
            x = self.cross_attention(x, enc_output, enc_output) + x
        x = self.norm3(x)

        trend = self.trend_proj(trend)
        return x + trend


class TSNetDecoderStack(nn.Module):
    """
    A stack of decoder blocks for TSNet.
    """
    def __init__(self, d_model, self_attn_fn, cross_attn_fn, decomposition_layer_fn, n_layers=1, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSNetDecoderBlock(
                d_model=d_model,
                self_attn_layer=self_attn_fn(),
                cross_attn_layer=cross_attn_fn(),
                decomposition_layer=decomposition_layer_fn(),
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x, enc_output=None):
        for block in self.blocks:
            x = block(x, enc_output)
        return x
