"""
This module implements multiple projection heads for TSNet outputs, 
which should be concatenated after the TSNet backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastingHead(nn.Module):
    """
    ForecastingHead projects TSNet embeddings to future time steps.

    Args:
        x: Tensor of shape [B, L, D], where
           B = batch size,
           L = sequence length,
           D = model dimension

    Output:
        Tensor of shape [B, pred_len, C], where
           C = number of output channels (same as input_dim),
           pred_len = forecast horizon
    """
    def __init__(self, d_model, out_len, out_channels):
        super().__init__()
        self.out_len = out_len
        self.projection = nn.Linear(d_model, out_channels)

    def forward(self, x):
        if x.size(1) < self.out_len:
            raise ValueError(f"Input sequence too short for out_len={self.out_len}. Use an autoregressive wrapper.")
        x = x[:, -self.out_len:, :]  # take last `out_len` time steps
        return self.projection(x)    # [B, pred_len, C]]


class ImputationHead(nn.Module):
    """
    ImputationHead projects embeddings back to reconstruct the original time sequence.

    Args:
        x: Tensor of shape [B, L, D], where L is the masked input length

    Output:
        Tensor of shape [B, C, L], where C = number of channels
    """
    def __init__(self, d_model, out_channels):
        super().__init__()
        self.projection = nn.Linear(d_model, out_channels)

    def forward(self, x):
        return self.projection(x).transpose(1, 2)  # [B, C, L]


class ClassificationHead(nn.Module):
    """
    ClassificationHead pools the TSNet sequence embeddings and outputs class probabilities.

    Args:
        x: Tensor of shape [B, L, D], where
           L = sequence length,
           D = model dimension

    Output:
        Tensor of shape [B, class_dim] (class probabilities)
    """
    def __init__(self, d_model, class_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # mean over time
        self.classifier = nn.Linear(d_model, class_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.pool(x).squeeze(-1)  # [B, D]
        logits = self.classifier(x)  # [B, class_dim]
        return F.softmax(logits, dim=-1)

    def predict(self, x, threshold=None):
        """
        Inference method:
        - If threshold is None: returns class index with highest probability.
        - If threshold is a float: returns class indices with prob >= threshold.

        Returns:
            List[List[int]] if threshold is set,
            else Tensor of shape [B] as index of options
        """
        probs = self.forward(x)
        if threshold is None:
            return torch.argmax(probs, dim=-1)
        else:
            return [torch.nonzero(row >= threshold, as_tuple=True)[0].tolist() for row in probs]

