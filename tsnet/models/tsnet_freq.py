import torch
import torch.nn as nn

from tsnet.layers.embedding import DataEmbedding
from tsnet.layers.attention import FrequencyAttention
from tsnet.layers.decomposition import MultiSeriesDecomposition
from tsnet.layers.encoder_decoder import TSNetEncoder, TSNetDecoder
from tsnet.layers.projection_head import ForecastingHead, ImputationHead, ClassificationHead


class TSNetFreqConfig:
    def __init__(self,
                 task='forecasting',
                 d_model=64,
                 in_channels=1,
                 out_channels=1,
                 seq_len=96,
                 pred_len=24,
                 label_len=48,
                 class_dim=10,
                 dropout=0.1,
                 k_top=16,
                 n_heads=4,
                 enc_layers=2,
                 dec_layers=2,
                 kernel_sizes=[5, 15, 25],
                 channel_independence=True,
                 batch_size=1):
        self.task = task
        self.d_model = d_model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.class_dim = class_dim
        self.dropout = dropout
        self.k_top = k_top
        self.n_heads = n_heads
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.kernel_sizes = kernel_sizes
        self.channel_independence = channel_independence
        self.batch_size = batch_size
        if self.channel_independence:
            self.out_channels = self.in_channels


class TSNetFreq(nn.Module):
    """
    TSNetFreq: Transformer-based time series forecasting model using frequency-aware attention.

    Key parameters:
    - seq_len: Length of the input sequence to the encoder.
    - label_len: Length of known historical input passed to the decoder.
    - pred_len: Number of time steps to forecast (prediction horizon).
    - channel_independence: If True, applies channel-wise modeling.
    - The decoder input is [last label_len steps] + [zero padding of length pred_len].

    The output of the model is of shape [B, pred_len, C], where:
    - B = batch size
    - C = number of output channels (equal to input channels if channel_independence=True)
    """
    def __init__(self, configs: TSNetFreqConfig):
        super().__init__()
        self.configs = configs
        input_dim = 1 if configs.channel_independence else configs.in_channels
        self.enc_embedding = DataEmbedding(input_dim, configs.d_model, configs.seq_len)
        self.dec_embedding = DataEmbedding(input_dim, configs.d_model, configs.seq_len)

        def attn_layer():
            return FrequencyAttention(configs.d_model, configs.n_heads, configs.k_top)

        def decomp_layer():
            return MultiSeriesDecomposition(configs.kernel_sizes)

        self.encoder = TSNetEncoder(
            d_model=configs.d_model,
            attention_layer_fn=attn_layer,
            decomposition_layer_fn=decomp_layer,
            n_layers=configs.enc_layers,
            dropout=configs.dropout
        )

        self.decoder = TSNetDecoder(
            d_model=configs.d_model,
            self_attn_fn=attn_layer,
            cross_attn_fn=attn_layer,
            decomposition_layer_fn=decomp_layer,
            n_layers=configs.dec_layers,
            dropout=configs.dropout
        )

        if configs.task == 'forecasting':
            if configs.channel_independence:
                self.head = ForecastingHead(configs.d_model, configs.pred_len, 1)
            else:
                self.head = ForecastingHead(configs.d_model, configs.pred_len, configs.out_channels)
        elif configs.task == 'imputation':
            self.head = ImputationHead(configs.d_model, configs.out_channels)
        elif configs.task == 'classification':
            self.head = ClassificationHead(configs.d_model, configs.class_dim)
        else:
            raise ValueError(f"Unknown task type: {configs.task}")

    def forward_backbone(self, x, x_mark=None):
        # Input shapes:
        # x: [B, seq_len, C]
        # x_mark: [B, seq_len, time_feature_dim]
        # Output: dec_out: [B*C, label_len+pred_len, D] or [B, label_len+pred_len, D]
        B, L, C_in = x.shape
        self.configs.batch_size = B
        if self.configs.channel_independence:
            x_reshaped = x.permute(0, 2, 1).reshape(B * C_in, L, 1)
            x_mark_reshaped = x_mark.repeat_interleave(C_in, dim=0) if x_mark is not None else None
        else:
            x_reshaped = x
            x_mark_reshaped = x_mark

        enc_in = self.enc_embedding(x_reshaped, x_mark_reshaped)

        label_len = self.configs.label_len
        pred_len = self.configs.pred_len
        C_in_dec = 1 if self.configs.channel_independence else self.configs.in_channels

        # Construct decoder input:
        # - Use last `label_len` steps from input
        # - Append zero padding of length `pred_len`
        # - Resulting shape: [B*C, label_len + pred_len, C]
        dec_input = x_reshaped[:, -label_len:, :]
        zero_pad = torch.zeros(x_reshaped.size(0), pred_len, C_in_dec, device=x_reshaped.device)
        dec_input = torch.cat([dec_input, zero_pad], dim=1)

        if x_mark_reshaped is not None:
            dec_x_mark = x_mark_reshaped[:, -label_len:, :]
            dec_x_mark_zeros = torch.zeros_like(x_mark_reshaped[:, :pred_len, :])
            dec_x_mark = torch.cat([dec_x_mark, dec_x_mark_zeros], dim=1)
        else:
            dec_x_mark = None

        dec_in = self.dec_embedding(dec_input, dec_x_mark)
        enc_out = self.encoder(enc_in) if self.configs.enc_layers > 0 else enc_in
        dec_out = self.decoder(dec_in, enc_output=enc_out) if self.configs.dec_layers > 0 else enc_out
        # Return only decoder output; projection head is applied in forward_head()
        return dec_out

    def forward_head(self, dec_out):
        head_out = self.head(dec_out)
        B = self.configs.batch_size
        C = self.configs.out_channels
        if self.configs.task == 'forecasting':
            if self.configs.channel_independence:
                head_out = head_out.reshape(B, C, self.configs.pred_len).transpose(1, 2)

        elif self.configs.task == 'imputation':
            raise NotImplementedError("Imputation task not implemented yet.")
        elif self.configs.task == 'classification':
            raise NotImplementedError("Classification task not implemented yet.")

        return head_out

    def forward(self, x, x_mark=None):
        dec_out = self.forward_backbone(x, x_mark)
        res = self.forward_head(dec_out)
        return res

    def minft(self):
        """
        Apply minimal fine-tuning by freezing core backbone layers and 
        enabling training only on embeddings, norm layers, and the projection head.
        """
        for name, param in self.named_parameters():
            lname = name.lower()
            if (
                'embed' in lname
                or 'norm' in lname
                or 'head' in lname
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False


if __name__ == "__main__":
    configs = TSNetFreqConfig(
        task='forecasting',
        d_model=64,
        in_channels=1,
        out_channels=1,
        seq_len=96,
        pred_len=24,
        label_len=48,
        enc_layers=6,
        dec_layers=3,
        n_heads=4,
        k_top=16,
        kernel_sizes=[5, 15, 25]
    )

    model = TSNetFreq(configs)

    BATCH_SIZE = 10
    x = torch.randn(BATCH_SIZE, configs.seq_len, configs.in_channels)  # [B, L, C_in]

    x_mark = torch.cat([
        torch.randint(0, 60, (BATCH_SIZE, configs.seq_len, 1)),   # minute
        torch.randint(0, 24, (BATCH_SIZE, configs.seq_len, 1)),   # hour
        torch.randint(0, 7, (BATCH_SIZE, configs.seq_len, 1)),    # weekday
        torch.randint(1, 32, (BATCH_SIZE, configs.seq_len, 1)),   # day of month
        torch.randint(1, 13, (BATCH_SIZE, configs.seq_len, 1))    # month
    ], dim=-1)

    with torch.no_grad():
        out = model(x, x_mark)

    print("Output shape:", out.shape)