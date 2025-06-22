"""
This module provides the input embedding layer for the TSNet model.
Input embeddings include value embeddings, positional embeddings, and temporal embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    """
    Positional Embedding using sine and cosine functions.

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum sequence length supported.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)].to(x.device)


class ValueEmbedding(nn.Module):
    """
    Projects scalar input values from input channels to d_model embedding space.

    Args:
        input_dim (int): Number of input features (channels).
        d_model (int): Embedding dimension.
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.proj(x)


class TemporalEmbedding(nn.Module):
    """
    Encodes timestamp information like hour-of-day, day-of-week as embeddings.
    Implemented as either learned or fixed embeddings.
    Inspired by the FEDformer paper.

    Args:
        d_model (int): Embedding dimension.
        embed_type (str): 'fixed' or 'learned' positional encoding.
    """
    def __init__(self, d_model, embed_type='learned'):
        super().__init__()
        Embed = nn.Embedding if embed_type == 'learned' else FixedEmbedding
        self.minute_embed = Embed(60, d_model)  # 0-59 (60 minutes)
        self.hour_embed = Embed(24, d_model)    # 0-23 (24 hours)
        self.weekday_embed = Embed(7, d_model)  # 0-6 (Monday-Sunday)
        self.day_embed = Embed(32, d_model)     # 1-31
        self.month_embed = Embed(13, d_model)   # 1-12

    def forward(self, x_mark):
        """
        Args:
            x_mark: Tensor of shape (batch_size, seq_len, 5),
                    where the last dim is (minute, hour, weekday, day, month)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        minute_x = self.minute_embed(x_mark[:, :, 0].long())
        hour_x = self.hour_embed(x_mark[:, :, 1].long())
        weekday_x = self.weekday_embed(x_mark[:, :, 2].long())
        day_x = self.day_embed(x_mark[:, :, 3].long())
        month_x = self.month_embed(x_mark[:, :, 4].long())
        temporal_x = minute_x + hour_x + weekday_x + day_x + month_x
        if temporal_x.shape[0] == 1:
            # If batch size is 1, unsqueeze to match expected shape
            temporal_x = temporal_x.squeeze(0)
        return temporal_x


class FixedEmbedding(nn.Module):
    """
    Sinusoidal embedding (non-trainable) for temporal features.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, vocab_size, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.pe[:, x]


class DataEmbedding(nn.Module):
    """
    Combines value, positional, and temporal embeddings as data embeddings.

    Args:
        input_dim (int): Number of input features (channels).
        d_model (int): Embedding dimension.
        embed_type (str): 'fixed' or 'learned' for temporal embeddings.
        dropout (float): Dropout rate.
    """
    def __init__(self, input_dim, d_model, embed_type='learned', dropout=0.1):
        super().__init__()
        self.value_embedding = ValueEmbedding(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model, embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            x_mark: Optional Tensor of shape (batch_size, seq_len, 5)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        x_val = self.value_embedding(x)
        x_pos = self.position_embedding(x_val)
        if x_mark is not None:
            x_time = self.temporal_embedding(x_mark)
            return self.dropout(x_val + x_pos + x_time)
        else:
            return self.dropout(x_val + x_pos)

class PatchEmbedding(nn.Module):
    """
    Applies patch segmentation and embedding for time series sequence.
    Inspired by PatchTST. Each patch is projected to d_model, with positional and temporal encoding.
    The patch embedding layer handles multi-channel inputs by channel independence.
    Projection to d_model is on patch level, not channel level.

    Args:
        patch_len (int): Length of each patch.
        stride (int): Stride between patches.
        d_model (int): Dimension to project each patch into.
        dropout (float): Dropout rate.
        use_temporal (bool): Whether to include temporal embeddings.
        embed_type (str): 'learned' or 'fixed' for temporal embeddings.
    """
    def __init__(self, patch_len, stride=0, d_model=256, dropout=0.1, use_temporal=True, embed_type='learned'):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len  # Default stride is patch_len
        assert patch_len > 0, "patch_len must be positive"
        self.d_model = d_model
        self.use_temporal = use_temporal

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.num_patch = 0

        if use_temporal:
            self.temporal_embedding = TemporalEmbedding(d_model, embed_type)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: Tensor of shape (B, L, C)
            x_mark: Optional temporal marks of shape (B, L, 5)

        Returns:
            patch_tokens: Tensor of shape (B * C, N, d_model)
            N: Number of patches per sequence
        """
        B, L, C = x.shape
        N = (L - self.patch_len) // self.stride + 1
        needed_len = (N - 1) * self.stride + self.patch_len
        if L < needed_len:
            pad_len = needed_len - L
            x = F.pad(x, (0, 0, pad_len, 0))  # pad front on seq_len dim
            if x_mark is not None:
                x_mark = F.pad(x_mark, (0, 0, pad_len, 0))
        elif (L - self.patch_len) % self.stride != 0:
            # Truncate from front to ensure only valid full patches
            new_L = (N - 1) * self.stride + self.patch_len
            x = x[:, -new_L:, :]
            if x_mark is not None:
                x_mark = x_mark[:, -new_L:, :]

        N = (x.shape[1] - self.patch_len) // self.stride + 1
        self.num_patch = N

        # Create patches: unfold produces (B, N, patch_len, C), then reshape to (B*C, N, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, N, C, patch_len)
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, C, N, patch_len)
        B, C, N, P = patches.shape
        patches = patches.view(B * C, N, P)  # (B*C, N, patch_len)

        patch_tokens = self.patch_embedding(patches)  # (B*C, N, d_model)
        patch_tokens = self.position_embedding(patch_tokens)

        if self.use_temporal and x_mark is not None:
            x_mark_patch = x_mark.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, N, 5, patch_len)
            mid_idx = self.patch_len // 2
            x_mark_patch = x_mark_patch[:, :, :, mid_idx]  # (B, N, 5)
            x_mark_patch = x_mark_patch.repeat_interleave(C, dim=0)  # (B*C, N, 5)
            temporal_enc = self.temporal_embedding(x_mark_patch)
            patch_tokens = patch_tokens + temporal_enc

        return self.dropout(patch_tokens), N
