"""
This module implements a frequency-domain attention mechanism with sparse sampling.
"""

import torch
import torch.nn as nn
import torch.fft


class FrequencyAttention(nn.Module):
    """
    Implements Frequency-Enhanced Attention as described in the TSNet paper.

    This module performs attention entirely in the frequency domain:
    1. Query, Key, and Value are projected into frequency space via rFFT.
    2. Top-k frequency bins are selected dynamically based on the energy of the query spectrum.
    3. Attention weights are computed using real-valued dot products in frequency space.
    4. The attended values are used to replace the top-k bins in the output spectrum.
    5. Remaining frequency bins are filled with the mean of the projected query.
    6. Inverse FFT is applied to return to the time domain.
    """
    def __init__(self, d_model, n_heads=8, k_top=32):
        """
        Args:
            d_model (int): Model dimensions, must be divisible by n_heads.
            n_heads (int): Number of attention heads.
            k_top (int): Number of top frequency bins to select from the query spectrum.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k_top = k_top
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert k_top > 0, "k_top must be positive"
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None):
        """
        Args:
            query: Tensor of shape (B, L_q, d_model)
            key: Tensor of shape (B, L_k, d_model)
            value: Tensor of shape (B, L_k, d_model)

        Returns:
            out: Tensor of shape (B, L_q, d_model)
        """
        device = query.device
        B, L_q, D = query.shape
        key = query if key is None else key
        value = query if value is None else value
        L_k = key.shape[1]
        assert L_k == value.shape[1], "Key and value must have same sequence length"

        # Calculate frequency sampled lengths
        F_q = L_q // 2 + 1
        F_k = L_k // 2 + 1
        k_top = min(self.k_top, F_q)

        # Linear projections of query, key, value to multi-head format
        q = self.q_proj(query).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)

        # Fourier transform of query, key, value to frequency domain
        # Using rFFT to get real-valued frequency components
        q_freq = torch.fft.rfft(q, dim=2)  # [B, H, F_q, d_head]
        k_freq = torch.fft.rfft(k, dim=2)  # [B, H, F_k, d_head]
        v_freq = torch.fft.rfft(v, dim=2)  # [B, H, F_k, d_head]

        # Select top-k frequency bins from query based on energy
        q_energy = q_freq.abs().mean(dim=-1)  # [B, H, F_q]
        idx = torch.topk(q_energy, k_top, dim=-1).indices  # [B, H, k_top]

        # Interpolate corresponding bins in K and V
        # Map indices from query to key frequency bins
        # Using linear interpolation to map indices from query to key frequency bins
        mapped_idx = (idx.float() * (F_k - 1) / (F_q - 1)).round().long().clamp(0, F_k - 1)
        q_top = torch.gather(q_freq, dim=2, index=idx.unsqueeze(-1).expand(-1, -1, -1, q_freq.shape[-1]))
        k_top_freq = torch.gather(k_freq, dim=2, index=mapped_idx.unsqueeze(-1).expand(-1, -1, -1, k_freq.shape[-1]))
        v_top = torch.gather(v_freq, dim=2, index=mapped_idx.unsqueeze(-1).expand(-1, -1, -1, v_freq.shape[-1]))

        # Attention weights in frequency domain 
        eps = 1e-8
        attn_score = (q_top.real * k_top_freq.real).sum(-1) / (self.d_head ** 0.5 + eps)
        attn = attn_score.softmax(dim=-1).unsqueeze(-1)  # [B, H, k_top, 1]

        # Fill full frequency spectrum with projected frequency-domain mean
        q_mean_freq = q_freq.mean(dim=2, keepdim=True)  # [B, H, 1, d_head]
        out_freq = q_mean_freq.expand(-1, -1, F_q, -1).clone()  # [B, H, F_q, d_head]

        # Create index tensor for scattering
        expand_idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)  # [B, H, k_top, d_head]

        # Compute weighted values to be inserted
        attn_weighted_v = (attn * v_top.real).to(out_freq.dtype)  # [B, H, k_top, d_head]

        # Scatter into full frequency tensor
        out_freq = out_freq.scatter(dim=2, index=expand_idx, src=attn_weighted_v)

        # Inverse Fourier transform to return to time domain
        out_time = torch.fft.irfft(out_freq, n=L_q, dim=2)
        out = out_time.transpose(1, 2).reshape(B, L_q, D)
        return self.out_proj(out)


if __name__ == "__main__":
    # Example case for self-attention
    batch_size = 4
    seq_len = 96
    d_model = 64
    n_heads = 4
    k_top = 16
    x = torch.randn(batch_size, seq_len, d_model)  # Query, Key, and Value are the same
    attention_layer = FrequencyAttention(d_model=d_model, n_heads=n_heads, k_top=k_top)
    self_attention_output = attention_layer(x)  # Self-attention

    print("Self-Attention Output shape:", self_attention_output.shape)  # Should be (batch_size, seq_len, d_model)

    # Example case for cross-attention
    src_len = seq_len * 2 + 5
    src = torch.randn(batch_size, src_len, d_model)  # Key and Value are different from Query
    cross_attention_output = attention_layer(x, src, src)  # Cross-attention

    print("Cross-Attention Output shape:", cross_attention_output.shape)  # Should be (batch_size, seq_len, d_model)
