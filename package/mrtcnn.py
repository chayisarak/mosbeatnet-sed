"""
mtrcnn.py

Purpose
-------
This file keeps the MTRCNN backbone as close as possible to the original code
snippet, while adding only the small adapter needed for the mosquito audio
segment pipeline.

Reference
---------
Original MTRCNN code structure adapted from:
    Yuanbo2020/MTRCNN
    Gesture_classification/framework/models_pytorch.py

Paper reference from the original repository:
    Hou, Yuanbo, Ren, Qiaoqiao, Wang, Wenwu, and Botteldooren, Dick.
    "Sound-Based Recognition of Touch Gestures and Emotions for Enhanced
    Human-Robot Interaction." ICASSP 2025.

What is original-like?
----------------------
The class named MTRCNN below follows the uploaded original snippet closely:
    input : [batch, seq_len, mel_bins]
    output: [batch, class_num]

What is added for your pipeline?
--------------------------------
Your mosquito pipeline gives raw waveform segments:
    [batch, n_segments, 1, segment_samples]

The added class MTRCNNSegmentLevel converts each waveform segment to log-mel,
runs the original-like MTRCNN on each segment, and returns:
    [batch, n_segments, n_outputs]

So for your thesis/report/code comments, you can say:
    The MTRCNN backbone is preserved. A wrapper was added only to convert raw
    waveform segments to log-mel features and reshape predictions for the
    existing segment-level training loop.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# STANDALONE SUPPORT CODE
# =============================================================================
# The original GitHub file imports config, init_layer, init_bn,
# ConvBlock_single_layer, and ConvBlock_dilation_single_layer from its own
# framework. Because this is a single standalone file, those helpers are defined
# here with the same names expected by MTRCNN.
#
# NOTE: This support code is not the conceptual model change. It only lets the
# original-like MTRCNN class run outside the original repository.


class config:
    """Minimal replacement for the original repository config object."""

    # Keep 64 because the original MTRCNN uses BatchNorm2d(config.mel_bins)
    # and later expects frequency_num = 6 after the convolution/pooling stack.
    mel_bins = 64


def init_layer(layer):
    """Initialize Conv2d or Linear layers."""

    nn.init.xavier_uniform_(layer.weight)

    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize BatchNorm2d layers like the common PANNs-style helper."""

    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock_single_layer(nn.Module):
    """
    Standalone replacement for the original repo's ConvBlock_single_layer.

    Interface kept the same as the original MTRCNN expects:
        block(x, pool_size=(2, 2), pool_type='avg')
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
    ):
        super(ConvBlock_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))

        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError("pool_type must be 'avg' or 'max'")

        return x


class ConvBlock_dilation_single_layer(nn.Module):
    """
    Standalone replacement for the original repo's dilated conv block.

    Interface kept the same as the original MTRCNN expects:
        block(x, pool_size=(2, 2), pool_type='avg')
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    ):
        super(ConvBlock_dilation_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))

        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError("pool_type must be 'avg' or 'max'")

        return x


# =============================================================================
# ORIGINAL-LIKE MTRCNN BACKBONE
# =============================================================================
# This section intentionally stays very close to the uploaded MTRCNN snippet.
# Main things preserved:
#   1. Three parallel branches with kernel sizes 3, 5, and 7.
#   2. Dilations (2, 1) and (3, 1) in deeper conv blocks.
#   3. frequency_num = 6 and k_*_freq_to_1 Linear layers.
#   4. mean_max pooling over the time dimension.
#   5. fc_embedding_event followed by fc_final_event.
#   6. The forward input remains log-mel features, not raw waveform.
#
# Important: This class alone is not responsible for waveform preprocessing.
# That is handled by MTRCNNSegmentLevel later in this file.


class MTRCNN(nn.Module):
    def __init__(self, class_num, batchnormal=True):

        super(MTRCNN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 6
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock_single_layer(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, padding=(0, 0), dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, padding=(0, 0), dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 2))
        self.conv_block2_kernel_5 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)


        scene_event_embedding_dim = 128
        self.fc_embedding_event = nn.Linear(64 * 3, scene_event_embedding_dim, bias=True)
        # -----------------------------------------------------------------------------------------------------------

        # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(scene_event_embedding_dim, class_num, bias=True)

        ##############################################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_layer(self.fc_embedding_event)

        # classification layer -------------------------------------------------------------------------------------
        init_layer(self.fc_final_event)

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input):
        # Original expected shape: [batch, seq_len, mel_bins].
        # It is intentionally NOT [batch, segments, 1, waveform_samples].
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''


        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        batch_x = x

        # print(x.size())  # torch.Size([32, 1, 1001, 64])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block2(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block3(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.mean_max(x_k_3)
        x_k_3_mel = F.relu_(self.k_3_freq_to_1(x_k_3))[:, :, 0]
        # print('x_k_3_mel: ', x_k_3_mel.size())  # x_k_3_mel:  torch.Size([32, 64])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 64, 1496, 64])

        x_k_5 = self.conv_block2_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 128, 740, 52])

        x_k_5 = self.conv_block3_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([8, 256, 358, 32])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        x_k_5_mel = F.relu_(self.k_5_freq_to_1(x_k_5))[:, :, 0]
        # print('x_k_5_mel: ', x_k_5_mel.size())  torch.Size([32, 64])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 64, 1494, 64])

        x_k_7 = self.conv_block2_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 128, 735, 48])

        x_k_7 = self.conv_block3_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([8, 256, 349, 20])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        x_k_7_mel = F.relu_(self.k_7_freq_to_1(x_k_7))[:, :, 0]

        event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel, x_k_7_mel], dim=-1)
        # print(event_embs_log_mel.size())  # torch.Size([32, 64*4])  (node_num, batch, edge_dim)

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.gelu(self.fc_embedding_event(event_embs_log_mel))
        # -------------------------------------------------------------------------------------------------------------

        event = self.fc_final_event(event_embeddings)

        return event


# =============================================================================
# ADDED FOR MOSQUITO SEGMENT PIPELINE
# =============================================================================
# Everything below this line is adapter code. It is NOT part of the original
# MTRCNN backbone. It exists only because your current Dataset/Training flow uses
# raw waveform segments instead of precomputed log-mel features.
#
#  pipeline:
#   AudioSequenceDataset returns       [B, S, 1, L]
#   labels have shape                  [B, S]
#   train/evaluate flattens outputs    [B*S, C]
#   train/evaluate flattens labels     [B*S]
#
# Adapter job:
#   [B, S, 1, L] waveform
#       -> [B*S, L]
#       -> log-mel [B*S, T, 64]
#       -> original-like MTRCNN [B*S, C]
#       -> reshape back [B, S, C]


def _hz_to_mel(freq_hz):
    """Convert Hz to mel using the HTK formula."""

    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mels):
    """Convert mel to Hz using the HTK formula."""

    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)


def _create_mel_filterbank(sample_rate, n_fft, n_mels, f_min=0.0, f_max=None):
    """
    Create a torch mel filterbank.

    Added for pipeline use because the original MTRCNN expects log-mel input.
    We avoid librosa inside forward so this can run on GPU and save cleanly in
    PyTorch checkpoints.
    """

    if f_max is None:
        f_max = float(sample_rate) / 2.0

    n_freqs = n_fft // 2 + 1
    freq_bins = torch.linspace(0.0, float(sample_rate) / 2.0, n_freqs)

    mel_min = _hz_to_mel(torch.tensor(float(f_min)))
    mel_max = _hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = torch.zeros(n_mels, n_freqs)

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        up_slope = (freq_bins - left) / torch.clamp(center - left, min=1e-12)
        down_slope = (right - freq_bins) / torch.clamp(right - center, min=1e-12)
        fb[i] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

    # Slaney-style area normalization. This only affects feature scale, not shape.
    enorm = 2.0 / torch.clamp(hz_points[2:n_mels + 2] - hz_points[:n_mels], min=1e-12)
    fb *= enorm.unsqueeze(1)

    return fb


class TorchLogMelSpectrogram(nn.Module):
    """
    Added feature extractor for your pipeline.

    Input:
        waveform [N, L] or [N, 1, L]

    Output:
        log_mel [N, frames, n_mels]

    Why hop_length defaults to 32:
        Your segment is 0.5 s at 8 kHz, so L = 4000 samples.
        With hop=128 there are too few time frames for the original MTRCNN
        kernel-7/dilated stack. hop=32 gives enough frames while keeping the
        MTRCNN backbone itself unchanged.
    """

    def __init__(
        self,
        sample_rate=8000,
        n_fft=256,
        win_length=256,
        hop_length=32,
        n_mels=64,
        f_min=0.0,
        f_max=None,
        eps=1e-6,
        per_segment_norm=True,
    ):
        super(TorchLogMelSpectrogram, self).__init__()

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.eps = float(eps)
        self.per_segment_norm = bool(per_segment_norm)

        window = torch.hann_window(self.win_length)
        mel_fb = _create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        self.register_buffer('window', window)
        self.register_buffer('mel_fb', mel_fb)

    def forward(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        elif waveform.dim() != 2:
            raise ValueError(
                'TorchLogMelSpectrogram expects [N, L], [N, 1, L], or [L], '
                f'but got shape {tuple(waveform.shape)}.'
            )

        # STFT is safest in float32, especially when your training loop uses AMP.
        amp_off = torch.amp.autocast("cuda",enabled=False) if waveform.is_cuda else contextlib.nullcontext()
        with amp_off:
            waveform = waveform.float()

            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode='reflect',
                return_complex=True,
            )

            power = spec.abs().pow(2.0)                  # [N, F, T]
            mel = torch.matmul(self.mel_fb, power)       # [N, n_mels, T]
            log_mel = torch.log(torch.clamp(mel, min=self.eps)).transpose(1, 2)

            if self.per_segment_norm:
                mean = log_mel.mean(dim=(1, 2), keepdim=True)
                std = log_mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
                log_mel = (log_mel - mean) / std

        return log_mel


class MTRCNNSegmentLevel(nn.Module):
    """
    Pipeline-compatible wrapper around the original-like MTRCNN.

    This class is what you should instantiate in your current training flow:
        model = MTRCNNSegmentLevel(n_outputs=len(label_map))

    This class is intentionally small:
        1. Convert raw segment waveform to log-mel.
        2. Call MTRCNN without changing its architecture.
        3. Reshape logits back to [batch, segments, classes].
    """

    def __init__(
        self,
        n_outputs,
        sample_rate=8000,
        n_fft=256,
        win_length=256,
        hop_length=32,
        n_mels=64,
        batchnormal=True,
        per_segment_norm=True,
    ):
        super(MTRCNNSegmentLevel, self).__init__()

        if n_mels != config.mel_bins:
            raise ValueError(
                f'MTRCNN expects {config.mel_bins} mel bins, but got {n_mels}. '
                'Keep n_mels=64 unless you also modify the original frequency_num logic.'
            )

        # ADDED: waveform -> log-mel feature extractor.
        self.feature_extractor = TorchLogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            per_segment_norm=per_segment_norm,
        )

        # ORIGINAL-LIKE BACKBONE: the MTRCNN class above.
        self.backbone = MTRCNN(
            class_num=n_outputs,
            batchnormal=batchnormal,
        )

    def forward(self, x):
        """
        Accepts either pipeline sequence input or direct segment input.

        Pipeline input:
            x [B, S, 1, L]
            returns [B, S, C]

        Direct segment input:
            x [N, 1, L] or [N, L]
            returns [N, C]
        """

        if x.dim() == 4:
            batch_size, n_segments, channels, length = x.shape

            if channels != 1:
                raise ValueError(
                    'MTRCNNSegmentLevel expects mono audio with channel dimension 1, '
                    f'but got shape {tuple(x.shape)}.'
                )

            x_flat = x.reshape(batch_size * n_segments, length)
            log_mel = self.feature_extractor(x_flat)
            logits = self.backbone(log_mel)
            return logits.reshape(batch_size, n_segments, -1)

        if x.dim() == 3 and x.shape[1] != 1:
            # Optional convenience input: [B, S, L]
            batch_size, n_segments, length = x.shape
            x_flat = x.reshape(batch_size * n_segments, length)
            log_mel = self.feature_extractor(x_flat)
            logits = self.backbone(log_mel)
            return logits.reshape(batch_size, n_segments, -1)

        # Direct segment input: [N, 1, L] or [N, L]
        log_mel = self.feature_extractor(x)
        return self.backbone(log_mel)



OriginalLikeMTRCNN = MTRCNN


# if __name__ == '__main__':
#     # Smoke test 1: original-like MTRCNN receives log-mel input.
#     class_num = 10
#     # Keep smoke-test sizes small so running this file is fast on CPU.
#     batch_size = 2
#     seq_len = 126
#     mel_bins = config.mel_bins

#     mtrcnn = MTRCNN(class_num=class_num)
#     logmel = torch.randn(batch_size, seq_len, mel_bins)
#     logits = mtrcnn(logmel)

#     print('Original-like MTRCNN input :', tuple(logmel.shape))
#     print('Original-like MTRCNN output:', tuple(logits.shape))

#     # Smoke test 2: your mosquito pipeline receives raw waveform segments.
#     segment_model = MTRCNNSegmentLevel(n_outputs=class_num)
#     audio_segments = torch.randn(1, 2, 1, 4000)
#     segment_logits = segment_model(audio_segments)

#     print('Pipeline input :', tuple(audio_segments.shape))
#     print('Pipeline output:', tuple(segment_logits.shape))
