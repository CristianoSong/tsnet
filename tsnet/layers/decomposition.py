"""
This module implements a series decomposition layer that separates a time series 
into seasonal and trend components.
It also allows for multi-kernel decomposition, where multiple smoothing filters 
are applied to capture different levels of trend.
"""

import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """
    Decomposes input sequence into seasonal and trend components using average pooling.

    Args:
        kernel_size (int): Size of the moving average window.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2, 
            count_include_pad=False
            )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            seasonal: x - trend, shape (batch_size, seq_len, d_model)
            trend: moving average on x, shape (batch_size, seq_len, d_model)
        """
        # Permute to match AvgPool1d input: [B, D, L]
        x_t = x.permute(0, 2, 1)
        trend = self.moving_avg(x_t)
        trend = trend.permute(0, 2, 1)  # Back to [B, L, D]
        seasonal = x - trend
        return seasonal, trend


class MultiSeriesDecomposition(nn.Module):
    """
    Decomposes input sequence into seasonal and trend components using multiple smoothing filters.
    The kernel sizes are combined with learnable weights, adapting smoothing levels to data.

    Args:
        kernel_sizes (List[int]): List of moving average kernel sizes.
    """
    def __init__(self, kernel_sizes):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.moving_avgs = nn.ModuleList([
            nn.AvgPool1d(
                kernel_size=k, 
                stride=1, 
                padding=0, 
                count_include_pad=False
                )
            for k in kernel_sizes
        ])
        self.weight_logits = nn.Parameter(torch.zeros(len(kernel_sizes)))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            seasonal: x - trend, shape (batch_size, seq_len, d_model)
            weighted_trend: smoothed trend component after applying multiple kernels,
                            shape (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        x_t = x.permute(0, 2, 1)  # [B, D, L]
        trends = []

        for avg, k in zip(self.moving_avgs, self.kernel_sizes):
            pad_left = (k - 1) // 2
            pad_right = k - 1 - pad_left
            front = x_t[:, :, 0:1].repeat(1, 1, pad_left)
            back = x_t[:, :, -1:].repeat(1, 1, pad_right)
            padded = torch.cat([front, x_t, back], dim=2)  # [B, D, L + pad]
            t = avg(padded).permute(0, 2, 1)  # [B, L, D]
            trends.append(t.unsqueeze(-1))

        trend_stack = torch.cat(trends, dim=-1)  # [B, L, D, K]
        weights = torch.softmax(self.weight_logits, dim=0)  # [K]
        weighted_trend = torch.sum(
            trend_stack * weights.view(1, 1, 1, -1), 
            dim=-1
            )  # [B, L, D]
        seasonal = x - weighted_trend
        return seasonal, weighted_trend
    

