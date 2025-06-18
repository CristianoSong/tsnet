"""
This module provides the input embedding layer for the TSNet model.
Input embeddings include value embeddings, positional embeddings, and temporal embeddings.
"""

import numpy as np
import torch
import torch.nn as nn

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
