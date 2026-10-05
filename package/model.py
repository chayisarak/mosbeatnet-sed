from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio



class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.score_proj = nn.Linear(dim, 1)

    def forward(self, query, value):
        q = self.query_proj(query)                  # [B, T, D]
        v = self.value_proj(value)                  # [B, T, D]
        scores = self.score_proj(torch.tanh(q + v)).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)        # [B, T]
        context = weights.unsqueeze(-1) * value    # [B, T, D]
        return context, weights


class Mosbeatnet(nn.Module):
    def __init__(self, n_timesteps, n_outputs):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=50)
        self.norm1 = nn.InstanceNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, dilation=2, padding=32)
        self.norm2 = nn.InstanceNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=25)
        self.norm3 = nn.InstanceNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.attn = AdditiveAttention(128)
        self.layer_norm = nn.LayerNorm(128)
        self.feature_conv = nn.Conv1d(128, 128, kernel_size=1)

        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(256, n_outputs)

    def forward(self, x):
        # x: [B, T, 1, L]
        batch_size, num_segments, channels, length = x.shape
        x = x.view(batch_size * num_segments, channels, length)

        x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))

        x = self.global_pool(x).squeeze(-1)                 # [B*T, 64]
        x = x.view(batch_size, num_segments, 64)            # [B, T, 64]

        x, _ = self.lstm1(x)                                # [B, T, 128]
        x, _ = self.lstm2(x)                                # [B, T, 128]

        context, _ = self.attn(x, x)
        x = self.layer_norm(x + context)

        x = x.permute(0, 2, 1)                              # [B, 128, T]
        x = self.feature_conv(x)
        x = x.permute(0, 2, 1)                              # [B, T, 128]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)                            # [B, T, n_outputs]
        return x



# Mosbeatnet_V2 : 

class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=7,
        stride=1,
        dilation=1,
        dropout=0.1,
    ):
        super().__init__()

        padding = dilation * (kernel_size // 2)

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)



class MosbeatnetV2(nn.Module):
    """
    Improved Mosbeatnet.

    Input:
        x: [B, T, 1, L]

    Output:
        logits: [B, T, n_outputs]

    Main changes from original Mosbeatnet:
        1. Stronger CNN encoder with BatchNorm + GELU.
        2. 2-layer BiLSTM so LSTM dropout actually works.
        3. Temporal Conv1D after attention to smooth segment-level context.
        4. Keeps the same output shape as the original training loop.
    """

    def __init__(self, n_timesteps, n_outputs):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_outputs = n_outputs

        # --------------------------------------------------
        # Segment-level waveform encoder
        # --------------------------------------------------
        self.encoder = nn.Sequential(
            ConvBlock1D(
                in_ch=1,
                out_ch=32,
                kernel_size=15,
                stride=2,
                dilation=1,
                dropout=0.1,
            ),
            ConvBlock1D(
                in_ch=32,
                out_ch=32,
                kernel_size=9,
                stride=1,
                dilation=2,
                dropout=0.1,
            ),
            nn.MaxPool1d(kernel_size=3, stride=3),

            ConvBlock1D(
                in_ch=32,
                out_ch=64,
                kernel_size=9,
                stride=2,
                dilation=1,
                dropout=0.1,
            ),
            ConvBlock1D(
                in_ch=64,
                out_ch=64,
                kernel_size=7,
                stride=1,
                dilation=2,
                dropout=0.1,
            ),
            nn.MaxPool1d(kernel_size=3, stride=3),

            ConvBlock1D(
                in_ch=64,
                out_ch=128,
                kernel_size=7,
                stride=1,
                dilation=1,
                dropout=0.1,
            ),
            ConvBlock1D(
                in_ch=128,
                out_ch=128,
                kernel_size=5,
                stride=1,
                dilation=2,
                dropout=0.1,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --------------------------------------------------
        # Sequence-level temporal modeling
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # --------------------------------------------------
        # Attention over segment sequence
        # --------------------------------------------------
        self.attn = AdditiveAttention(128)
        self.layer_norm = nn.LayerNorm(128)

        # --------------------------------------------------
        # Local temporal smoothing across segments
        # --------------------------------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # --------------------------------------------------
        # Segment-level classifier
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        # x: [B, T, 1, L]
        batch_size, num_segments, channels, length = x.shape

        # Encode each 0.5 s segment independently
        x = x.reshape(batch_size * num_segments, channels, length)
        x = self.encoder(x)                              # [B*T, 128, L']
        x = self.global_pool(x).squeeze(-1)              # [B*T, 128]

        # Restore sequence shape
        x = x.reshape(batch_size, num_segments, 128)     # [B, T, 128]

        # BiLSTM temporal modeling
        x, _ = self.lstm(x)                              # [B, T, 128]

        # Additive attention + residual connection
        context, _ = self.attn(x, x)                     # [B, T, 128]
        x = self.layer_norm(x + context)                 # [B, T, 128]

        # Temporal Conv over segment axis
        x = x.permute(0, 2, 1)                           # [B, 128, T]
        x = self.temporal_conv(x)                        # [B, 128, T]
        x = x.permute(0, 2, 1)                           # [B, T, 128]

        # Classify each segment
        logits = self.classifier(x)                      # [B, T, n_outputs]
        return logits


# mosbeatnet_beta

class AdditiveAttentionBeta(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.score_proj = nn.Linear(dim, 1)

    def forward(self, query, value):
        # query, value: [B, T, D]

        q = self.query_proj(query).unsqueeze(2)   # [B, T, 1, D]
        v = self.value_proj(value).unsqueeze(1)   # [B, 1, T, D]

        scores = self.score_proj(torch.tanh(q + v)).squeeze(-1) # [B, T, T]

        weights = F.softmax(scores, dim=-1)       # attention over all segments

        context = torch.bmm(weights, value)       # [B, T, D]

        return context, weights


class Mosbeatnet_beta(nn.Module):
    def __init__(self, n_timesteps, n_outputs):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_outputs = n_outputs

        self.encoder = nn.Sequential(
            ConvBlock1D(1, 32, kernel_size=15, stride=2, dilation=1, dropout=0.1),
            ConvBlock1D(32, 32, kernel_size=9, stride=1, dilation=2, dropout=0.1),
            nn.MaxPool1d(kernel_size=3, stride=3),

            ConvBlock1D(32, 64, kernel_size=9, stride=2, dilation=1, dropout=0.1),
            ConvBlock1D(64, 64, kernel_size=7, stride=1, dilation=2, dropout=0.1),
            nn.MaxPool1d(kernel_size=3, stride=3),

            ConvBlock1D(64, 128, kernel_size=7, stride=1, dilation=1, dropout=0.1),
            ConvBlock1D(128, 128, kernel_size=5, stride=1, dilation=2, dropout=0.1),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.attn = AdditiveAttentionBeta(128)
        self.layer_norm = nn.LayerNorm(128)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        # x: [B, T, 1, L]

        batch_size, num_segments, channels, length = x.shape

        x = x.reshape(batch_size * num_segments,channels,length)

        x = self.encoder(x)                    # [B*T, 128, L']
        x = self.global_pool(x).squeeze(-1)    # [B*T, 128]

        x = x.reshape(batch_size,num_segments,128)   # [B, T, 128]

        x, _ = self.lstm(x)                    # [B, T, 128]

        context, attn_weights = self.attn(x, x) # [B,T,128], [B,T,T]

        x = self.layer_norm(x + context)        # [B, T, 128]

        x = x.permute(0, 2, 1)                 # [B, 128, T]
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)                 # [B, T, 128]

        logits = self.classifier(x)             # [B, T, n_outputs]

        return logits

# class MosbeatnetV2_in(nn.Module):
#     """
#     Improved Mosbeatnet.

#     Input:
#         x: [B, T, 1, L]

#     Output:
#         logits: [B, T, n_outputs]

#     Main changes from original Mosbeatnet:
#         1. Stronger CNN encoder with BatchNorm + GELU.
#         2. 2-layer BiLSTM so LSTM dropout actually works.
#         3. Temporal Conv1D after attention to smooth segment-level context.
#         4. Keeps the same output shape as the original training loop.
#     """

#     def __init__(self, n_timesteps, n_outputs):
#         super().__init__()

#         self.n_timesteps = n_timesteps
#         self.n_outputs = n_outputs

#         # --------------------------------------------------
#         # Segment-level waveform encoder
#         # --------------------------------------------------
#         self.encoder = nn.Sequential(
#             ConvBlock1D(
#                 in_ch=1,
#                 out_ch=32,
#                 kernel_size=15,
#                 stride=2,
#                 dilation=1,
#                 dropout=0.1,
#             ),
#             ConvBlock1D(
#                 in_ch=32,
#                 out_ch=32,
#                 kernel_size=9,
#                 stride=1,
#                 dilation=2,
#                 dropout=0.1,
#             ),
#             nn.MaxPool1d(kernel_size=3, stride=3),

#             ConvBlock1D(
#                 in_ch=32,
#                 out_ch=64,
#                 kernel_size=9,
#                 stride=2,
#                 dilation=1,
#                 dropout=0.1,
#             ),
#             ConvBlock1D(
#                 in_ch=64,
#                 out_ch=64,
#                 kernel_size=7,
#                 stride=1,
#                 dilation=2,
#                 dropout=0.1,
#             ),
#             nn.MaxPool1d(kernel_size=3, stride=3),

#             ConvBlock1D(
#                 in_ch=64,
#                 out_ch=128,
#                 kernel_size=7,
#                 stride=1,
#                 dilation=1,
#                 dropout=0.1,
#             ),
#             ConvBlock1D(
#                 in_ch=128,
#                 out_ch=128,
#                 kernel_size=5,
#                 stride=1,
#                 dilation=2,
#                 dropout=0.1,
#             ),
#         )

#         self.global_pool = nn.AdaptiveAvgPool1d(1)

#         # --------------------------------------------------
#         # Sequence-level temporal modeling
#         # --------------------------------------------------
#         self.lstm = nn.LSTM(
#             input_size=128,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.3,
#         )

#         # --------------------------------------------------
#         # Attention over segment sequence
#         # --------------------------------------------------
#         self.attn = AdditiveAttention(128)
#         self.layer_norm = nn.LayerNorm(128)

#         # --------------------------------------------------
#         # Local temporal smoothing across segments
#         # --------------------------------------------------
#         self.temporal_conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=128,
#                 out_channels=128,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.InstanceNorm1d(128, affine=True),
#             nn.GELU(),
#             nn.Dropout(0.2),
#         )

#         # --------------------------------------------------
#         # Segment-level classifier
#         # --------------------------------------------------
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, n_outputs),
#         )

#     def forward(self, x):
#         # x: [B, T, 1, L]
#         batch_size, num_segments, channels, length = x.shape

#         # Encode each 0.5 s segment independently
#         x = x.reshape(batch_size * num_segments, channels, length)
#         x = self.encoder(x)                              # [B*T, 128, L']
#         x = self.global_pool(x).squeeze(-1)              # [B*T, 128]

#         # Restore sequence shape
#         x = x.reshape(batch_size, num_segments, 128)     # [B, T, 128]

#         # BiLSTM temporal modeling
#         x, _ = self.lstm(x)                              # [B, T, 128]

#         # Additive attention + residual connection
#         context, _ = self.attn(x, x)                     # [B, T, 128]
#         x = self.layer_norm(x + context)                 # [B, T, 128]

#         # Temporal Conv over segment axis
#         x = x.permute(0, 2, 1)                           # [B, 128, T]
#         x = self.temporal_conv(x)                        # [B, 128, T]
#         x = x.permute(0, 2, 1)                           # [B, T, 128]

#         # Classify each segment
#         logits = self.classifier(x)                      # [B, T, n_outputs]
#         return logits



# class SEBlock1D(nn.Module):
#     """
#     Self-implemented 1D Squeeze-and-Excitation block.

#     This is adapted for 1D CNN feature maps:
#         input:  [B, C, L]
#         output: [B, C, L]

#     It is based on the SE idea from:
#     Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

#     This code is not copied from the official GitHub source.
#     It is a simple PyTorch 1D implementation.
#     """

#     def __init__(self, channels, reduction=8):
#         super().__init__()

#         hidden = max(channels // reduction, 8)

#         self.squeeze = nn.AdaptiveAvgPool1d(1)

#         self.excitation = nn.Sequential(
#             nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
#             nn.GELU(),
#             nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # x: [B, C, L]
#         scale = self.squeeze(x)          # [B, C, 1]
#         scale = self.excitation(scale)   # [B, C, 1]
#         return x * scale                 # [B, C, L]


# class MosbeatnetV2_SE(nn.Module):
#     """
#     MosbeatnetV2 with lightweight SE channel attention.

#     Input:
#         x: [B, T, 1, L]

#     Output:
#         logits: [B, T, n_outputs]

#     Changes from your MosbeatnetV2:
#         1. Add SEBlock1D after deeper CNN stages.
#         2. Use AvgPool + MaxPool before projection.
#         3. Keep LSTM, attention, temporal conv, classifier mostly same.

#     References:
#         - Hu et al., "Squeeze-and-Excitation Networks", CVPR 201
#         https://doi.org/10.48550/arXiv.1709.01507
#         https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
#         https: //github.com/hujie-frank/SENet
#     """


#     def __init__(self, n_timesteps, n_outputs):
#         super().__init__()

#         self.n_timesteps = n_timesteps
#         self.n_outputs = n_outputs

#         # --------------------------------------------------
#         # Segment-level waveform encoder
#         # --------------------------------------------------
#         self.encoder = nn.Sequential(
#             ConvBlock1D(
#                 in_ch=1,
#                 out_ch=32,
#                 kernel_size=15,
#                 stride=2,
#                 dilation=1,
#                 dropout=0.1,
#             ),
#             ConvBlock1D(
#                 in_ch=32,
#                 out_ch=32,
#                 kernel_size=9,
#                 stride=1,
#                 dilation=2,
#                 dropout=0.1,
#             ),
#             nn.MaxPool1d(kernel_size=3, stride=3),

#             ConvBlock1D(
#                 in_ch=32,
#                 out_ch=64,
#                 kernel_size=9,
#                 stride=2,
#                 dilation=1,
#                 dropout=0.1,
#             ),
#             ConvBlock1D(
#                 in_ch=64,
#                 out_ch=64,
#                 kernel_size=7,
#                 stride=1,
#                 dilation=2,
#                 dropout=0.1,
#             ),

#             # Added SE attention
#             SEBlock1D(channels=64, reduction=8),

#             nn.MaxPool1d(kernel_size=3, stride=3),

#             ConvBlock1D(
#                 in_ch=64,
#                 out_ch=128,
#                 kernel_size=7,
#                 stride=1,
#                 dilation=1,
#                 dropout=0.1,
#             ),
#             ConvBlock1D(
#                 in_ch=128,
#                 out_ch=128,
#                 kernel_size=5,
#                 stride=1,
#                 dilation=2,
#                 dropout=0.1,
#             ),

#             # Added SE attention
#             SEBlock1D(channels=128, reduction=8),
#         )

#         # --------------------------------------------------
#         # Better segment pooling
#         # --------------------------------------------------
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.global_max_pool = nn.AdaptiveMaxPool1d(1)

#         self.pool_proj = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.GELU(),
#             nn.Dropout(0.1),
#         )

#         # --------------------------------------------------
#         # Sequence-level temporal modeling
#         # --------------------------------------------------
#         self.lstm = nn.LSTM(
#             input_size=128,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.3,
#         )

#         # --------------------------------------------------
#         # Attention over segment sequence
#         # --------------------------------------------------
#         self.attn = AdditiveAttention(128)
#         self.layer_norm = nn.LayerNorm(128)

#         # --------------------------------------------------
#         # Local temporal smoothing across segments
#         # --------------------------------------------------
#         self.temporal_conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=128,
#                 out_channels=128,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm1d(128),
#             nn.GELU(),
#             nn.Dropout(0.2),
#         )

#         # --------------------------------------------------
#         # Segment-level classifier
#         # --------------------------------------------------
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(128),
#             nn.Linear(128, 256),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, n_outputs),
#         )

#     def forward(self, x):
#         # x: [B, T, 1, L]
#         batch_size, num_segments, channels, length = x.shape

#         # Encode each segment independently
#         x = x.reshape(batch_size * num_segments, channels, length)
#         x = self.encoder(x)                              # [B*T, 128, L']

#         # Avg pooling captures overall pattern
#         x_avg = self.global_avg_pool(x).squeeze(-1)      # [B*T, 128]

#         # Max pooling captures sharp detection peaks
#         x_max = self.global_max_pool(x).squeeze(-1)      # [B*T, 128]

#         # Combine both
#         x = torch.cat([x_avg, x_max], dim=1)             # [B*T, 256]
#         x = self.pool_proj(x)                            # [B*T, 128]

#         # Restore sequence shape
#         x = x.reshape(batch_size, num_segments, 128)     # [B, T, 128]

#         # BiLSTM temporal modeling
#         x, _ = self.lstm(x)                              # [B, T, 128]

#         # Additive attention + residual connection
#         context, _ = self.attn(x, x)                     # [B, T, 128]
#         x = self.layer_norm(x + context)                 # [B, T, 128]

#         # Temporal Conv over segment axis
#         x = x.permute(0, 2, 1)                           # [B, 128, T]
#         x = self.temporal_conv(x)                        # [B, 128, T]
#         x = x.permute(0, 2, 1)                           # [B, T, 128]

#         # Classify each segment
#         logits = self.classifier(x)                      # [B, T, n_outputs]

#         return logits


class MosqPlusModel(nn.Module):
    def __init__(self, n_timesteps, n_outputs):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=100, stride=4)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=64, stride=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=64, stride=3)
        self.pool = nn.MaxPool1d(kernel_size=3)

        flat_size = self._infer_flat_size(n_timesteps)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def _infer_flat_size(self, n_timesteps):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_timesteps)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            return x.numel()

    def forward(self, x):
        # x: [B, T, 1, L]
        batch_size, num_segments, _, _ = x.size()
        x = x.view(batch_size * num_segments, 1, -1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, num_segments, -1)            # [B, T, n_outputs]
        return x




class SEDNetSegmentLevel(nn.Module):
    def __init__(self, sr=8000, hop_len=512, segment_sec=0.5, n_classes=5, dropout_rate=0.3):
        super().__init__()

        self.n_classes = n_classes

        segment_length = int(sr * segment_sec) 
        n_fft = 1024

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=40,
            center=False,
            power=1.0,
        )

        self.spec_bn = nn.BatchNorm2d(1)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_rate),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_rate),
        )

        self.rnn = nn.GRU(
            input_size=64 * 40,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B, T, 1, L]
        batch_size, num_segments, channels, length = x.shape

        # join all 0.5-sec segments into one continuous waveform
        x = x.reshape(batch_size, num_segments * length)
        # [B, total_samples]

        x = self.mel_spectrogram(x)
        # [B, mel, time]

        x = x.unsqueeze(1)
        # [B, 1, mel, time]

        x = self.spec_bn(x)

        x = self.cnn1(x)
        x = self.cnn2(x)
        # [B, C, mel, time']

        # make temporal dimension match original segments
        x = torch.nn.functional.adaptive_avg_pool2d(x,(x.shape[2], num_segments))
        # [B, C, mel, 20]

        batch_size, channels, mel_bins, time_steps = x.shape

        x = x.permute(0, 3, 1, 2)
        # [B, 20, C, mel]

        x = x.reshape(batch_size,time_steps,channels * mel_bins,)
        # [B, 20, C*mel]

        x, _ = self.rnn(x)
        # [B, 20, 128]

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # [B, 20, n_classes]

        return x




# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _check_odd_kernel(kernel_size: int) -> None:
    if int(kernel_size) % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd to preserve temporal length, got {kernel_size}."
        )


def _as_segment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert segment input to [N, 1, T].

    Accepted:
        [N, T]
        [N, 1, T]
    """
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(f"Expected channel dimension 1, got shape {tuple(x.shape)}")
        return x
    raise ValueError(f"Expected [N, T] or [N, 1, T], got shape {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# Plain ResNet9 baselines from the original folder code
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Plain Conv1D residual block from the original ResNet9 baseline."""

    def __init__(self, channels: int, kernel_size: int, padding: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(num_features=channels)
        self.bn2 = nn.BatchNorm1d(num_features=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.conv1(x))
        out = self.bn1(out)
        out = F.gelu(self.conv2(out))
        out = self.bn2(out)
        return out + residual


class ResNet9Segment(nn.Module):
    """
    Plain ResNet9 segment classifier.

    This is useful as a non-Fourier baseline alongside CF-ResNet-1D.

    Input:
        [N, T] or [N, 1, T]

    Output:
        [N, num_classes]
    """

    CHANNELS: Dict[str, tuple[int, int, int, int]] = {
        "small": (32, 64, 96, 128),
        "medium": (32, 64, 128, 256),
        "large": (64, 128, 256, 512),
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
    ):
        super().__init__()
        _check_odd_kernel(kernel_size)
        if variant not in self.CHANNELS:
            raise ValueError(f"variant must be one of {sorted(self.CHANNELS)}, got {variant}")

        c1, c2, c3, c4 = self.CHANNELS[variant]
        padding = kernel_size // 2

        self.variant = variant
        self.pool_size = int(pool_size)
        self.kernel_size = int(kernel_size)

        self.conv1 = nn.Conv1d(1, c1, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(c1)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(c2)

        self.rb1 = ResidualBlock(c2, kernel_size=kernel_size, padding=padding, stride=1)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn3 = nn.BatchNorm1d(c3)

        self.conv4 = nn.Conv1d(c3, c4, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn4 = nn.BatchNorm1d(c4)

        self.rb2 = ResidualBlock(c4, kernel_size=kernel_size, padding=padding, stride=1)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c4, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_segment_batch(x)
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb1(x)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb2(x)

        x = self.gap(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class ResNet9Sequence(nn.Module):
    """
    Sequence wrapper for plain ResNet9.

    Input:
        [B, S, 1, T]

    Output:
        [B, S, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
    ):
        super().__init__()
        self.segment_model = ResNet9Segment(
            num_classes=num_classes,
            variant=variant,
            pool_size=pool_size,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            b, s, c, t = x.shape
            if c != 1:
                raise ValueError(f"Expected channel dimension 1, got shape {tuple(x.shape)}")
            x_flat = x.reshape(b * s, c, t)
            logits = self.segment_model(x_flat)
            return logits.reshape(b, s, -1)

        # Also allow direct segment batches for debugging or standalone use.
        if x.ndim in (2, 3):
            return self.segment_model(x)

        raise ValueError(f"Expected [B, S, 1, T], [N, 1, T], or [N, T], got {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# CF-ResNet-1D baseline adapted from the original paper code
# -----------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """
    1D Fourier spectral convolution.

    Original source reference:
    https://github.com/szbela87/insect_wingbeat_classification_fno/blob/main/model.py

    Original comment in the source says this layer follows the Fourier Neural
    Operator idea: FFT, learnable linear transform in the Fourier domain, then
    inverse FFT.

    MOSBNET adaptation:
    used_modes = min(requested modes, available rFFT bins)
    so the layer remains safe after repeated temporal pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        if int(modes1) <= 0:
            raise ValueError(f"modes1 must be positive, got {modes1}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)

        self.scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels,
                self.out_channels,
                self.modes1,
                2,
                dtype=torch.float,
            )
        )

    @staticmethod
    def compl_mul1d(input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication.

        input_ft: [batch, in_channel, modes]
        weights:  [in_channel, out_channel, modes]
        output:   [batch, out_channel, modes]
        """
        return torch.einsum("bix,iox->box", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        signal_length = x.size(-1)

        x_ft = torch.fft.rfft(x)
        n_freq = signal_length // 2 + 1
        used_modes = min(self.modes1, n_freq)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            n_freq,
            device=x.device,
            dtype=torch.cfloat,
        )

        weights = torch.view_as_complex(self.weights1)
        out_ft[:, :, :used_modes] = self.compl_mul1d(
            x_ft[:, :, :used_modes],
            weights[:, :, :used_modes],
        )

        return torch.fft.irfft(out_ft, n=signal_length)


class FourierLayer(nn.Module):
    """
    Convolutional Fourier Layer from CF-ResNet-1D.

    output = Conv1D(x) + SpectralConv1D(x)

    This mirrors the original code structure:
        x1 = self.conv1(x)
        x2 = self.conv_fno1(x)
        out = x1 + x2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        modes: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
        )
        self.conv_fno1 = SpectralConv1d(in_channels, out_channels, modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv_fno1(x)


class ResidualBlockFNO(nn.Module):
    """Residual block using two Convolutional Fourier Layers."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        modes: int,
    ):
        super().__init__()
        self.fn1 = FourierLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            modes=modes,
        )
        self.fn2 = FourierLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            modes=modes,
        )
        self.bn1 = nn.BatchNorm1d(num_features=channels)
        self.bn2 = nn.BatchNorm1d(num_features=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.fn1(x))
        out = self.bn1(out)
        out = F.gelu(self.fn2(out))
        out = self.bn2(out)
        return out + residual


class CFResNet1DSegment(nn.Module):
    """
    CF-ResNet-1D segment classifier.

    This is the original-style CF-ResNet classifier adapted for robust segment
    input shapes. It predicts one label per segment.

    Variants:
        small:  1 -> 32 -> 64 -> 96  -> 128
        medium: 1 -> 32 -> 64 -> 128 -> 256
        large:  1 -> 64 -> 128 -> 256 -> 512

    Input:
        [N, T] or [N, 1, T]

    Output:
        [N, num_classes]

    Notes for faithful reporting:
        The original uploaded/source code uses pool_size=5 and kernel_size=11.
        The original small FNO class default uses modes=64, while the medium
        class default uses modes=16. For the MOSBNET segment pipeline, modes=16
        is a safer default because the segment length is 4000 and repeated
        pool_size=5 reduces the final temporal dimension to about 32 samples.
        You may still pass modes=64 for a closer small-model reproduction because
        SpectralConv1d safely clamps used modes at runtime.
    """

    CHANNELS: Dict[str, tuple[int, int, int, int]] = {
        "small": (32, 64, 96, 128),
        "medium": (32, 64, 128, 256),
        "large": (64, 128, 256, 512),
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
        modes: int = 16,
    ):
        super().__init__()
        _check_odd_kernel(kernel_size)
        if variant not in self.CHANNELS:
            raise ValueError(f"variant must be one of {sorted(self.CHANNELS)}, got {variant}")

        c1, c2, c3, c4 = self.CHANNELS[variant]
        padding = kernel_size // 2

        self.variant = variant
        self.pool_size = int(pool_size)
        self.kernel_size = int(kernel_size)
        self.modes = int(modes)

        self.conv1 = FourierLayer(1, c1, kernel_size, padding, stride=1, modes=modes)
        self.bn1 = nn.BatchNorm1d(num_features=c1)

        self.conv2 = FourierLayer(c1, c2, kernel_size, padding, stride=1, modes=modes)
        self.bn2 = nn.BatchNorm1d(num_features=c2)

        self.rb1 = ResidualBlockFNO(c2, kernel_size, padding, stride=1, modes=modes)

        self.conv3 = FourierLayer(c2, c3, kernel_size, padding, stride=1, modes=modes)
        self.bn3 = nn.BatchNorm1d(num_features=c3)

        self.conv4 = FourierLayer(c3, c4, kernel_size, padding, stride=1, modes=modes)
        self.bn4 = nn.BatchNorm1d(num_features=c4)

        self.rb2 = ResidualBlockFNO(c4, kernel_size, padding, stride=1, modes=modes)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=c4, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_segment_batch(x)
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb1(x)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb2(x)

        x = self.gap(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class CFResNet1DSequence(nn.Module):
    """
    MOSBNET-compatible CF-ResNet-1D wrapper.

    This is the class you should instantiate in your current training pipeline.

    Input from AudioSequenceDataset:
        [B, S, 1, T]

    Output expected by train_model/evaluate_model:
        [B, S, num_classes]

    Implementation statement for paper/code comments:
        We adapted the original CF-ResNet-1D segment classifier to our sequence
        pipeline by applying the same CF-ResNet-1D network independently to each
        0.5-second segment. The wrapper reshapes the batch from [B, S, 1, T] to
        [B*S, 1, T], performs segment-level classification, and restores the
        output to [B, S, C] so that the existing segment-level loss and metrics
        operate without changing the training loop.
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
        modes: int = 16,
    ):
        super().__init__()
        self.segment_model = CFResNet1DSegment(
            num_classes=num_classes,
            variant=variant,
            pool_size=pool_size,
            kernel_size=kernel_size,
            modes=modes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            b, s, c, t = x.shape
            if c != 1:
                raise ValueError(f"Expected channel dimension 1, got shape {tuple(x.shape)}")
            x_flat = x.reshape(b * s, c, t)
            logits = self.segment_model(x_flat)
            return logits.reshape(b, s, -1)

        # Also allow direct segment batches for debugging or standalone use.
        if x.ndim in (2, 3):
            return self.segment_model(x)

        raise ValueError(f"Expected [B, S, 1, T], [N, 1, T], or [N, T], got {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# Convenience aliases for clearer experiment names
# -----------------------------------------------------------------------------

class CFResNet1DSmall(CFResNet1DSequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11, modes: int = 16):
        super().__init__(num_classes, variant="small", pool_size=pool_size, kernel_size=kernel_size, modes=modes)


class CFResNet1DMedium(CFResNet1DSequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11, modes: int = 16):
        super().__init__(num_classes, variant="medium", pool_size=pool_size, kernel_size=kernel_size, modes=modes)


class CFResNet1DLarge(CFResNet1DSequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11, modes: int = 16):
        super().__init__(num_classes, variant="large", pool_size=pool_size, kernel_size=kernel_size, modes=modes)


class ResNet9Small(ResNet9Sequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11):
        super().__init__(num_classes, variant="small", pool_size=pool_size, kernel_size=kernel_size)


class ResNet9Medium(ResNet9Sequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11):
        super().__init__(num_classes, variant="medium", pool_size=pool_size, kernel_size=kernel_size)


class ResNet9Large(ResNet9Sequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11):
        super().__init__(num_classes, variant="large", pool_size=pool_size, kernel_size=kernel_size)

