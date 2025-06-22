import torch
import torch.nn as nn

from tsnet.layers.embedding import PatchEmbedding
from tsnet.layers.encoder_decoder import TransformerEncoder, TransformerDecoder
from tsnet.layers.projection_head import PatchForecastingHead


class TSNetPatchConfig:
    def __init__(self,
                 task='forecasting',
                 d_model=64,
                 in_channels=3,
                 out_channels=3,
                 seq_len=96,
                 pred_len=24,
                 label_len=48,
                 class_dim=10,
                 dropout=0.1,
                 n_heads=4,
                 enc_layers=2,
                 channel_independence=True,
                 batch_size=1,
                 patch_len=16,
                 stride=16):
        self.task = task
        self.d_model = d_model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.class_dim = class_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.enc_layers = enc_layers
        self.channel_independence = channel_independence
        self.batch_size = batch_size
        self.patch_len = patch_len
        self.stride = stride
        if self.channel_independence:
            self.out_channels = self.in_channels


class TSNetPatch(nn.Module):
    """
    TSNetFreq: Time Series Transformer with Patch Embedding.
        - Applies patch embedding to both encoder and decoder input sequences.
        - Uses standard Transformer encoder and decoder blocks with MultiheadAttention.
    Output shape: [B, pred_len, C]
    """

    def __init__(self, configs: TSNetPatchConfig):
        super().__init__()
        self.configs = configs

        self.enc_embedding = PatchEmbedding(
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=configs.d_model,
            dropout=configs.dropout
        )
        self.dec_embedding = PatchEmbedding(
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        self.encoder = TransformerEncoder(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            n_layers=configs.enc_layers,
            dropout=configs.dropout
        )

        self.decoder = TransformerDecoder(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            n_layers=configs.enc_layers,
            dropout=configs.dropout
        )

        if configs.task == 'forecasting':
            if configs.channel_independence:
                self.head = PatchForecastingHead(configs.d_model, configs.pred_len)
            else:
                raise NotImplementedError("Patch forecasting only supports independent channels.")
        elif configs.task == 'imputation':
            raise NotImplementedError("Patch imputation is not implemented yet.")
        elif configs.task == 'classification':
            raise NotImplementedError("Patch classification is not implemented yet.")
        else:
            raise NotImplementedError("Unsupported task: {}".format(configs.task))

    def forward_backbone(self, x, x_mark=None):
        B, L, C_in = x.shape
        self.configs.batch_size = B

        label_len = self.configs.label_len
        patch_len = self.configs.patch_len
        label_len = (label_len // patch_len) * patch_len
        self.configs.label_len = label_len

        x_enc = x
        x_dec = x[:, -label_len:, :]

        if self.configs.channel_independence:
            x_enc = x_enc.permute(0, 2, 1).reshape(B * C_in, L, 1)
            x_dec = x_dec.permute(0, 2, 1).reshape(B * C_in, label_len, 1)
        else:
            pass

        enc_in, N_enc = self.enc_embedding(x_enc, x_mark)
        dec_in, N_dec = self.dec_embedding(x_dec, x_mark[:, -label_len:, :])

        enc_out = self.encoder(enc_in)
        dec_out = self.decoder(dec_in, enc_output=enc_out)

        return dec_out

    def forward_head(self, x):
        head_out = self.head(x)  # [B*C, pred_len]
        B = self.configs.batch_size
        C = self.configs.out_channels
        if self.configs.channel_independence:
            head_out = head_out.reshape(B, C, self.configs.pred_len).transpose(1, 2)  # [B, pred_len, C]
        return head_out

    def forward(self, x, x_mark=None):
        x = self.forward_backbone(x, x_mark)
        return self.forward_head(x)

    def minft(self):
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
    configs = TSNetPatchConfig(
        task='forecasting',
        d_model=64,
        in_channels=1,
        out_channels=1,
        seq_len=96,
        pred_len=24,
        label_len=48,
        enc_layers=6,
        n_heads=4,
        patch_len=16,
        stride=8,
    )

    model = TSNetPatch(configs)

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